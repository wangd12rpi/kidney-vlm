#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import base64
import mimetypes
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.vqa.eval_gpt import (
    build_mcq_prompt,
    collect_required_image_paths,
    compute_mcq_metrics,
    option_values,
    parse_mcq_response,
    select_eval_rows,
)
from kidney_vlm.vqa.schema import normalize_vqa_df, validate_vqa_df

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="07_vqa_evaluation/evaluate_vqa_gpt.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_repo_env_value(name: str) -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return ""


def _build_azure_client(azure_cfg: Any):
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai is required. Install it with: uv add openai"
        ) from exc

    api_key_env = str(azure_cfg.api_key_env).strip()
    api_key = os.getenv(api_key_env, "").strip() or _read_repo_env_value(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing Azure OpenAI key in env var: {api_key_env}")

    return AzureOpenAI(
        api_version=str(azure_cfg.api_version),
        azure_endpoint=str(azure_cfg.endpoint),
        api_key=api_key,
    )


def _extract_text_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content.strip()
    if isinstance(raw_content, list):
        chunks: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                if item.strip():
                    chunks.append(item.strip())
                continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                chunks.append(text_attr.strip())
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    chunks.append(text_value.strip())
        return "\n".join(chunks).strip()
    return str(raw_content or "").strip()


def _image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_user_content(
    user_prompt: str, image_paths: list[Path]
) -> str | list[dict[str, Any]]:
    if not image_paths:
        return user_prompt

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(image_path),
                },
            }
        )
    return content


def _call_azure_gpt(
    *,
    client: Any,
    azure_cfg: Any,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[Path],
) -> str:
    deployment = str(azure_cfg.deployment)
    retries = int(azure_cfg.get("max_retries", 3))
    retry_sleep_seconds = float(azure_cfg.get("retry_sleep_seconds", 2.0))
    reasoning_effort = str(azure_cfg.get("reasoning_effort", "")).strip()
    verbosity = str(azure_cfg.get("verbosity", "")).strip()
    max_tokens = int(azure_cfg.get("max_completion_tokens", 256))

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": _build_user_content(user_prompt, image_paths),
                    },
                ],
                "max_completion_tokens": max_tokens,
                "model": deployment,
            }
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
            if verbosity:
                request_kwargs["verbosity"] = verbosity
            response = client.chat.completions.create(**request_kwargs)
            text = _extract_text_content(response.choices[0].message.content)
            if not text:
                raise RuntimeError("GPT returned an empty response.")
            return text
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_seconds)
    raise RuntimeError(
        f"GPT evaluation call failed after {retries} attempts: {last_error}"
    )


def _existing_prediction_ids(predictions_path: Path) -> set[int]:
    if not predictions_path.exists():
        return set()
    existing = pd.read_parquet(predictions_path)
    if "question_id" not in existing.columns:
        return set()
    return {int(value) for value in existing["question_id"].dropna().tolist()}


def _write_predictions(
    predictions_path: Path, rows: list[dict[str, Any]], *, overwrite: bool
) -> pd.DataFrame:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    generated = pd.DataFrame(rows)
    if generated.empty:
        if predictions_path.exists() and not overwrite:
            return pd.read_parquet(predictions_path)
        generated.to_parquet(predictions_path, index=False)
        return generated

    if predictions_path.exists() and not overwrite:
        existing = pd.read_parquet(predictions_path)
        if not existing.empty:
            existing = existing[~existing["question_id"].isin(generated["question_id"])]
            final = pd.concat([existing, generated], ignore_index=True)
        else:
            final = generated
    else:
        final = generated
    final = final.sort_values(
        ["project_id", "case_id", "task_id", "question_id"]
    ).reset_index(drop=True)
    final.to_parquet(predictions_path, index=False)
    return final


def _write_metrics(metrics_path: Path, metrics: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    cfg = load_cfg()
    eval_cfg = cfg.vqa_evaluation
    generation_cfg = OmegaConf.to_container(eval_cfg, resolve=True)

    vqa_path = _resolve_path(eval_cfg.vqa_parquet_path)
    predictions_path = _resolve_path(eval_cfg.predictions_path)
    metrics_path = _resolve_path(eval_cfg.metrics_path)

    vqa_df = normalize_vqa_df(pd.read_parquet(vqa_path))
    validate_vqa_df(vqa_df)
    selected_df = select_eval_rows(vqa_df, generation_cfg)
    if selected_df.empty:
        raise RuntimeError(
            "No VQA rows selected for evaluation. Check filters in the YAML."
        )

    overwrite = bool(eval_cfg.get("overwrite_predictions", False))
    resume_existing = bool(eval_cfg.get("resume_existing", True))
    existing_ids = (
        _existing_prediction_ids(predictions_path)
        if resume_existing and not overwrite
        else set()
    )
    eval_df = selected_df[
        ~selected_df["question_id"].astype(int).isin(existing_ids)
    ].reset_index(drop=True)

    print(f"VQA parquet: {vqa_path}")
    print(f"Rows selected: {len(selected_df)}")
    print(f"Rows skipped from existing predictions: {len(selected_df) - len(eval_df)}")
    print(f"Rows to evaluate: {len(eval_df)}")
    print(f"Predictions path: {predictions_path}")
    print(f"Metrics path: {metrics_path}")

    if str(eval_cfg.backend).strip() != "azure_openai_gpt":
        raise ValueError(
            f"Unsupported backend for this v0 evaluator: {eval_cfg.backend}"
        )

    client = (
        None
        if bool(eval_cfg.get("dry_run", False))
        else _build_azure_client(eval_cfg.azure_openai)
    )
    prediction_rows: list[dict[str, Any]] = []
    api_error_rows: list[dict[str, Any]] = []
    evaluated_at = datetime.now(timezone.utc).isoformat()

    for row in tqdm(
        eval_df.to_dict(orient="records"),
        total=len(eval_df),
        desc="Evaluating VQA MCQs",
        unit="question",
    ):
        system_prompt, user_prompt = build_mcq_prompt(row, generation_cfg)
        image_paths = collect_required_image_paths(row, generation_cfg, repo_root=ROOT)
        if bool(eval_cfg.get("dry_run", False)):
            raw_response = '{"answer": "", "rationale": "dry run"}'
            parsed = {"predicted_answer": "", "parse_status": "dry_run"}
        else:
            try:
                raw_response = _call_azure_gpt(
                    client=client,
                    azure_cfg=eval_cfg.azure_openai,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_paths=image_paths,
                )
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                api_error_row = {
                    "question_id": int(row["question_id"]),
                    "case_id": str(row["case_id"]),
                    "project_id": str(row["project_id"]),
                    "task_id": str(row["task_id"]),
                    "error": error_message,
                }
                api_error_rows.append(api_error_row)
                print(
                    "Azure GPT error; skipping question "
                    f"question_id={api_error_row['question_id']} "
                    f"case_id={api_error_row['case_id']} "
                    f"project_id={api_error_row['project_id']} "
                    f"task_id={api_error_row['task_id']}: "
                    f"{error_message}",
                    file=sys.stderr,
                )
                continue
            parsed = parse_mcq_response(raw_response, option_values(row))

        predicted_answer = parsed["predicted_answer"]
        correct = bool(
            predicted_answer and predicted_answer == str(row["answer"]).strip()
        )
        prediction_row = {
            "question_id": int(row["question_id"]),
            "base_question_id": int(row["base_question_id"]),
            "case_id": str(row["case_id"]),
            "project_id": str(row["project_id"]),
            "split": str(row["split"]),
            "task_category": str(row["task_category"]),
            "task_id": str(row["task_id"]),
            "use_pathology": bool(row["use_pathology"]),
            "use_radiology": bool(row["use_radiology"]),
            "use_dnam": bool(row["use_dnam"]),
            "use_rna": bool(row["use_rna"]),
            "question": str(row["question"]),
            "option_a": str(row["option_a"]),
            "option_b": str(row["option_b"]),
            "option_c": str(row["option_c"]),
            "option_d": str(row["option_d"]),
            "answer": str(row["answer"]),
            "predicted_answer": predicted_answer,
            "parse_status": parsed["parse_status"],
            "correct": correct,
            "raw_response": raw_response,
            "image_paths": [str(path) for path in image_paths],
            "backend": str(eval_cfg.backend),
            "model": str(eval_cfg.azure_openai.deployment),
            "evaluated_at": evaluated_at,
        }
        if bool(eval_cfg.get("include_prompt_in_predictions", False)):
            prediction_row["system_prompt"] = system_prompt
            prediction_row["user_prompt"] = user_prompt
        prediction_rows.append(prediction_row)

    final_predictions = _write_predictions(
        predictions_path, prediction_rows, overwrite=overwrite
    )
    metrics = compute_mcq_metrics(final_predictions)
    metrics["vqa_parquet_path"] = str(vqa_path)
    metrics["predictions_path"] = str(predictions_path)
    metrics["filters"] = generation_cfg.get("filters", {})
    metrics["backend"] = str(eval_cfg.backend)
    metrics["model"] = str(eval_cfg.azure_openai.deployment)
    metrics["evaluated_at"] = evaluated_at
    metrics["api_error_count"] = len(api_error_rows)
    metrics["api_errors"] = api_error_rows
    _write_metrics(metrics_path, metrics)

    print(f"Final prediction rows: {len(final_predictions)}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Parse failed: {metrics['parse_failed']}")
    print(f"Azure GPT errors skipped: {len(api_error_rows)}")


if __name__ == "__main__":
    main()
