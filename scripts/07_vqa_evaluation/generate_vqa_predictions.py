#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import base64
import mimetypes
import os
import random
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.vqa.constants import MODALITIES
from kidney_vlm.vqa.data import (
    _coerce_token_ids,
    _find_placeholder_span,
    filter_rows_with_prefix_cache,
    load_row_cached_prefix_tensor,
    pad_optional_prefix_tensors,
)
from kidney_vlm.vqa.eval_gpt import (
    apply_group_sampling,
    build_eval_prompt,
    collect_required_image_paths,
    enabled_model_configs,
    parse_model_response,
    prompt_cfg_for_model,
    question_type_key,
    select_eval_rows,
)
from kidney_vlm.vqa.modeling import (
    OncoVLMVQASFTModel,
    build_language_model,
    build_tokenizer,
    generate_language_model_with_soft_prefix,
    move_batch_to_device,
)
from kidney_vlm.vqa.prompts import (
    build_vqa_prompt,
    prefix_placeholder_for_modality,
    row_modalities,
    row_uses_modality,
)
from kidney_vlm.vqa.schema import normalize_vqa_df, validate_vqa_df
from kidney_vlm.vqa.stage_config import as_bool, cfg_get, clean_text, resolve_torch_dtype

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)
ONCOVLM_CACHE_BACKENDS = {"oncovlm_projector", "oncovlm_lora"}


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="07_vqa_evaluation/generate_vqa_predictions.yaml",
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

    api_key_env = clean_text(cfg_get(azure_cfg, "api_key_env"))
    if not api_key_env:
        raise ValueError("Azure VQA model config must define azure_openai.api_key_env.")
    api_key = os.getenv(api_key_env, "").strip() or _read_repo_env_value(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing Azure OpenAI key in env var: {api_key_env}")

    return AzureOpenAI(
        api_version=clean_text(cfg_get(azure_cfg, "api_version")),
        azure_endpoint=clean_text(cfg_get(azure_cfg, "endpoint")),
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
    generation_kwargs: Mapping[str, Any],
) -> str:
    deployment = clean_text(cfg_get(azure_cfg, "deployment"))
    if not deployment:
        raise ValueError("Azure VQA model config must define azure_openai.deployment.")
    retries = int(cfg_get(azure_cfg, "max_retries", 3))
    retry_sleep_seconds = float(cfg_get(azure_cfg, "retry_sleep_seconds", 2.0))
    reasoning_effort = clean_text(cfg_get(azure_cfg, "reasoning_effort"))
    verbosity = clean_text(cfg_get(azure_cfg, "verbosity"))
    max_tokens = 1024

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
            if "temperature" in generation_kwargs:
                request_kwargs["temperature"] = float(generation_kwargs["temperature"])
            if "top_p" in generation_kwargs:
                request_kwargs["top_p"] = float(generation_kwargs["top_p"])
            response = client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            text = _extract_text_content(choice.message.content)
            if not text:
                raise RuntimeError(
                    "GPT returned an empty response. "
                    f"finish_reason={getattr(choice, 'finish_reason', None)!r}; "
                    f"usage={getattr(response, 'usage', None)!r}. "
                    "If completion_tokens_details.reasoning_tokens equals max_completion_tokens, "
                    "lower azure_openai.reasoning_effort or raise generation.max_new_tokens."
                )
            return text
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_seconds)
    raise RuntimeError(
        f"GPT evaluation call failed after {retries} attempts: {last_error}"
    )


def _existing_prediction_keys(predictions_path: Path, *, model_display_name: str) -> set[tuple[int, int]]:
    if not predictions_path.exists():
        return set()
    existing = pd.read_parquet(predictions_path)
    required_columns = {"question_id", "model_display_name", "repeat_id"}
    missing_columns = required_columns - set(existing.columns)
    if missing_columns:
        raise ValueError(f"Existing predictions parquet is missing repeat columns: {sorted(missing_columns)}")
    existing = existing[existing["model_display_name"].astype(str) == str(model_display_name)]
    return {
        (int(question_id), int(repeat_id))
        for question_id, repeat_id in zip(existing["question_id"], existing["repeat_id"], strict=True)
    }


def _prediction_keys(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
    if frame.empty:
        return set()
    return {
        (str(model_name), int(question_id), int(repeat_id))
        for model_name, question_id, repeat_id in zip(
            frame["model_display_name"],
            frame["question_id"],
            frame["repeat_id"],
            strict=True,
        )
    }


def _write_predictions(
    predictions_path: Path, rows: list[dict[str, Any]], *, resume_existing: bool
) -> pd.DataFrame:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    generated = pd.DataFrame(rows)
    if generated.empty:
        if predictions_path.exists() and resume_existing:
            return pd.read_parquet(predictions_path)
        _write_parquet_atomic(generated, predictions_path)
        return generated

    if predictions_path.exists() and resume_existing:
        existing = pd.read_parquet(predictions_path)
        if not existing.empty:
            generated_keys = _prediction_keys(generated)
            keep_mask = [
                (str(model_name), int(question_id), int(repeat_id)) not in generated_keys
                for model_name, question_id, repeat_id in zip(
                    existing["model_display_name"],
                    existing["question_id"],
                    existing["repeat_id"],
                    strict=True,
                )
            ]
            existing = existing.loc[keep_mask]
            final = pd.concat([existing, generated], ignore_index=True)
        else:
            final = generated
    else:
        final = generated
    final = final.sort_values(
        ["model_display_name", "repeat_id", "project_id", "case_id", "task_id", "question_id"]
    ).reset_index(drop=True)
    _write_parquet_atomic(final, predictions_path)
    return final


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _resolve_device(device_value: Any) -> torch.device:
    requested = clean_text(device_value) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is unavailable.")
    return torch.device(requested)


def _run_output_dir(eval_cfg: Mapping[str, Any]) -> Path:
    run_cfg = dict(eval_cfg.get("run") or {})
    run_name = clean_text(run_cfg.get("name"))
    if not run_name:
        raise ValueError("vqa_evaluation.run.name must be populated.")
    output_root = _resolve_path(run_cfg.get("output_root", "results/vqa_eval"))
    return output_root / run_name


def _run_filename(eval_cfg: Mapping[str, Any], key: str, default: str) -> str:
    run_cfg = dict(eval_cfg.get("run") or {})
    value = clean_text(run_cfg.get(key)) or default
    if "/" in value or "\\" in value:
        raise ValueError(f"vqa_evaluation.run.{key} must be a file name, got {value!r}.")
    return value


def _predictions_path(eval_cfg: Mapping[str, Any]) -> Path:
    return _run_output_dir(eval_cfg) / _run_filename(eval_cfg, "prediction_filename", "predictions.parquet")


def _model_name_or_path(model_cfg: Mapping[str, Any]) -> str:
    if clean_text(model_cfg.get("model_name_or_path")):
        return clean_text(model_cfg.get("model_name_or_path"))
    azure_cfg = dict(model_cfg.get("azure_openai") or {})
    if clean_text(azure_cfg.get("deployment")):
        return clean_text(azure_cfg.get("deployment"))
    return clean_text(model_cfg.get("display_name"))


def _generation_kwargs(eval_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    generation_cfg = dict(eval_cfg.get("generation") or {})
    model_generation_cfg = dict(model_cfg.get("generation") or {})
    generation_cfg.update(model_generation_cfg)
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 128)),
        "do_sample": bool(generation_cfg.get("do_sample", False)),
    }
    temperature = generation_cfg.get("temperature")
    if temperature not in (None, "", "null"):
        kwargs["temperature"] = float(temperature)
    top_p = generation_cfg.get("top_p")
    if top_p not in (None, "", "null"):
        kwargs["top_p"] = float(top_p)
    return kwargs


def _repeat_cfg(eval_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if "repeats" not in eval_cfg:
        raise ValueError("VQA generation config must define repeats.")
    repeats_cfg = dict(eval_cfg.get("repeats") or {})
    repeat_count = int(repeats_cfg.get("n", 1) or 1)
    if repeat_count < 1:
        raise ValueError(f"repeats.n must be >= 1, got {repeat_count}.")
    repeats_cfg["n"] = repeat_count
    repeats_cfg["seed"] = int(repeats_cfg.get("seed", 0) or 0)
    combos = [
        clean_text(item)
        for item in repeats_cfg.get("repeat_modality_combination_names", [])
        if clean_text(item)
    ]
    repeats_cfg["repeat_modality_combination_names"] = set(combos)
    return repeats_cfg


def _row_repeat_count(row: Mapping[str, Any], repeats_cfg: Mapping[str, Any]) -> int:
    repeat_count = int(repeats_cfg["n"])
    if repeat_count <= 1:
        return 1
    repeated_combos = set(repeats_cfg.get("repeat_modality_combination_names") or [])
    modality_combo = clean_text(row.get("modality_combination_name"))
    return repeat_count if modality_combo in repeated_combos else 1


def _modality_ablation_cfg(eval_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(eval_cfg.get("modality_ablation") or {})
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "modality_combination_names": set(), "model_display_names": set()}
    modality_names = {
        clean_text(item)
        for item in cfg.get("modality_combination_names", [])
        if clean_text(item)
    }
    model_names = {
        clean_text(item)
        for item in cfg.get("model_display_names", [])
        if clean_text(item)
    }
    if not modality_names:
        raise ValueError("modality_ablation.modality_combination_names must be populated when enabled.")
    if not model_names:
        raise ValueError("modality_ablation.model_display_names must be populated when enabled.")
    return {
        "enabled": True,
        "modality_combination_names": modality_names,
        "model_display_names": model_names,
    }


def _filter_model_modality_ablation_rows(
    frame: pd.DataFrame,
    *,
    model_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    cfg = _modality_ablation_cfg(eval_cfg)
    if not bool(cfg["enabled"]):
        return frame.reset_index(drop=True)
    model_name = clean_text(model_cfg.get("display_name"))
    if model_name in cfg["model_display_names"]:
        return frame.reset_index(drop=True)
    ablation_names = set(cfg["modality_combination_names"])
    return frame[
        ~frame["modality_combination_name"].astype(str).isin(ablation_names)
    ].reset_index(drop=True)


def _expand_repeat_rows(frame: pd.DataFrame, repeats_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    base_rows = frame.to_dict(orient="records")
    max_repeats = int(repeats_cfg["n"])
    expanded: list[dict[str, Any]] = []
    for repeat_id in range(max_repeats):
        for row in base_rows:
            if repeat_id >= _row_repeat_count(row, repeats_cfg):
                continue
            out = dict(row)
            out["repeat_id"] = repeat_id
            expanded.append(out)
    return expanded


def _set_repeat_seed(repeats_cfg: Mapping[str, Any], repeat_id: int) -> None:
    seed = int(repeats_cfg["seed"]) + int(repeat_id)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_batch_size(model_cfg: Mapping[str, Any]) -> int:
    raw_batch_size = model_cfg.get("batch_size", 1)
    if raw_batch_size in (None, "", "null"):
        raw_batch_size = 1
    batch_size = int(raw_batch_size)
    if batch_size < 1:
        raise ValueError(f"Model {model_cfg.get('display_name')} has invalid batch_size={batch_size}.")
    return batch_size


def _batched_records(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[start : start + batch_size] for start in range(0, len(records), batch_size)]


def _batched_records_by_repeat(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    repeat_ids = sorted({int(record["repeat_id"]) for record in records})
    for repeat_id in repeat_ids:
        repeat_records = [record for record in records if int(record["repeat_id"]) == repeat_id]
        batches.extend(_batched_records(repeat_records, batch_size))
    return batches


def _sort_generation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    type_order = {"qa": 0, "open_ended": 0, "open-ended": 0, "open ended": 0, "mcq": 1}
    out["_generation_question_type_order"] = (
        out["question_type"].fillna("").astype(str).str.strip().str.lower().map(type_order).fillna(2)
    )
    return (
        out.sort_values(
            [
                "_generation_question_type_order",
                "question_type",
                "generation_type",
                "task_category",
                "task_id",
                "project_id",
                "case_id",
                "question_id",
            ]
        )
        .drop(columns=["_generation_question_type_order"])
        .reset_index(drop=True)
    )


def _filter_missing_prefix_cache_rows(
    frame: pd.DataFrame,
    *,
    model_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    if clean_text(model_cfg.get("backend")) not in ONCOVLM_CACHE_BACKENDS:
        return frame
    stage_cfg = _oncovlm_stage_cfg(model_cfg, eval_cfg=eval_cfg)
    prefix_cfg = dict(stage_cfg["prefix_cache"])
    if not as_bool(prefix_cfg.get("scan_before_training", True)):
        return frame.reset_index(drop=True)

    filtered, skipped = filter_rows_with_prefix_cache(frame, root_dir=ROOT, stage_cfg=stage_cfg)
    if not skipped:
        return filtered

    max_examples = int(prefix_cfg.get("max_missing_examples", 10) or 0)
    for item in skipped[:max_examples]:
        print(
            "Warning: skipping VQA row "
            f"qid={item.get('question_id')} model={clean_text(model_cfg.get('display_name'))}: "
            f"missing cached prefix ({'; '.join(item.get('missing') or [])})."
        )
    if len(skipped) > max_examples:
        print(
            "Warning: skipped "
            f"{len(skipped) - max_examples} additional VQA rows for model={clean_text(model_cfg.get('display_name'))} "
            "because cached prefixes were missing."
        )
    if not as_bool(prefix_cfg.get("skip_missing_rows", True)):
        raise FileNotFoundError(
            f"Missing cached prefixes for {len(skipped)} rows for model={clean_text(model_cfg.get('display_name'))}."
        )
    return filtered


def _should_generate_row(row: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> bool:
    missing: list[str] = []
    if bool(row.get("use_dnam")) and not clean_text(row.get("dnam_text_summary")):
        missing.append("dnam_text_summary")
    if bool(row.get("use_rna")) and not clean_text(row.get("rna_text_summary")):
        missing.append("rna_text_summary")
    if not missing:
        return True
    print(
        "Warning: skipping VQA row "
        f"qid={row.get('question_id')} model={clean_text(model_cfg.get('display_name'))}: "
        f"enabled DNAm/RNA has empty fallback text ({', '.join(missing)})."
    )
    return False


def _print_first_n_outputs(run_cfg: Mapping[str, Any]) -> int:
    raw_value = run_cfg.get("print_first_n_outputs", 0)
    if raw_value in (None, "", "null"):
        return 0
    count = int(raw_value)
    if count < 0:
        raise ValueError(f"run.print_first_n_outputs must be non-negative, got {count}.")
    return count


def _save_every_n_predictions(run_cfg: Mapping[str, Any]) -> int:
    raw_value = run_cfg.get("save_every_n_predictions", 0)
    if raw_value in (None, "", "null"):
        return 0
    count = int(raw_value)
    if count < 0:
        raise ValueError(f"run.save_every_n_predictions must be non-negative, got {count}.")
    return count


def _print_output_preview(
    *,
    preview_index: int,
    preview_limit: int,
    row: Mapping[str, Any],
    raw_response: str,
    parsed: Mapping[str, str],
    model_cfg: Mapping[str, Any],
) -> None:
    answer_label = clean_text(row.get("answer_label"))
    predicted_label = clean_text(parsed.get("predicted_answer_label"))
    label_text = ""
    if answer_label or predicted_label:
        label_text = f" | labels gt={answer_label or '-'} pred={predicted_label or '-'}"
    print(
        f"\n[VQA preview {preview_index}/{preview_limit}] "
        f"model={clean_text(model_cfg.get('display_name'))} "
        f"qid={row.get('question_id')} repeat={row.get('repeat_id')}"
    )
    print(f"Q: {clean_text(row.get('question'))}")
    print(
        f"GT: {clean_text(row.get('answer'))} | "
        f"OUT: {clean_text(raw_response)} | "
        f"parsed: {clean_text(parsed.get('predicted_answer'))}"
        f"{label_text}"
    )


def _prompt_block_for_projector(row: Mapping[str, Any], prompt_cfg: Mapping[str, Any]) -> dict[str, str]:
    key = question_type_key(row.get("question_type", ""))
    prompts = prompt_cfg.get("prompts") or {}
    block = prompts.get(key)
    if not isinstance(block, Mapping):
        raise ValueError(f"vqa_evaluation.prompts.{key} must be defined.")
    system_prompt = clean_text(block.get("system_prompt"))
    response_instruction = clean_text(block.get("response_instruction"))
    if not system_prompt or not response_instruction:
        raise ValueError(f"vqa_evaluation.prompts.{key} must define system_prompt and response_instruction.")
    return {
        "system_prompt": system_prompt,
        "mcq_response_instruction": response_instruction,
        "open_response_instruction": response_instruction,
    }


def _prompt_token_ids(tokenizer: Any, prompt_text: str, *, enable_thinking: bool = False) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        try:
            token_ids = tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                    **kwargs,
                )
            except TypeError:
                token_ids = tokenizer.apply_chat_template(messages, **kwargs)
        return _coerce_token_ids(token_ids)
    return _coerce_token_ids(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])


def _hf_message_payload(system_prompt: str, user_prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
    from PIL import Image

    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


class AzureGPTBackend:
    def __init__(self, model_cfg: Mapping[str, Any], *, dry_run: bool):
        self.model_cfg = dict(model_cfg)
        self.azure_cfg = self.model_cfg.get("azure_openai")
        if self.azure_cfg is None:
            raise ValueError(f"Model {self.model_cfg.get('display_name')} is missing azure_openai config.")
        self.client = None if dry_run else _build_azure_client(self.azure_cfg)
        self.dry_run = dry_run

    def generate(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path], generation_kwargs: Mapping[str, Any]) -> str:
        return self.generate_batch(
            requests=[
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "image_paths": image_paths,
                }
            ],
            generation_kwargs=generation_kwargs,
        )[0]

    def generate_batch(self, *, requests: list[Mapping[str, Any]], generation_kwargs: Mapping[str, Any]) -> list[str]:
        if self.dry_run:
            return ['{"answer": "", "rationale": "dry run"}' for _ in requests]
        if not requests:
            return []

        max_workers = min(len(requests), int(self.model_cfg.get("batch_size", 1) or 1))

        def _generate_one(request: Mapping[str, Any]) -> str:
            return _call_azure_gpt(
                client=self.client,
                azure_cfg=self.azure_cfg,
                system_prompt=clean_text(request.get("system_prompt")),
                user_prompt=clean_text(request.get("user_prompt")),
                image_paths=list(request.get("image_paths", [])),
                generation_kwargs=generation_kwargs,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_generate_one, requests))


class HFImageTextBackend:
    def __init__(self, model_cfg: Mapping[str, Any], *, dry_run: bool):
        self.model_cfg = dict(model_cfg)
        self.dry_run = dry_run
        self.device = _resolve_device(self.model_cfg.get("device"))
        self.processor = None
        self.model = None
        if dry_run:
            return
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("transformers is required for HF image-text VQA evaluation.") from exc

        model_name = _model_name_or_path(self.model_cfg)
        dtype = resolve_torch_dtype(self.model_cfg.get("torch_dtype"))
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(self.model_cfg.get("trust_remote_code", True)),
        }
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if clean_text(self.model_cfg.get("attn_implementation")):
            model_kwargs["attn_implementation"] = clean_text(self.model_cfg.get("attn_implementation"))
        if self.model_cfg.get("device_map") not in (None, "", "null"):
            model_kwargs["device_map"] = self.model_cfg.get("device_map")
        load_in_8bit = bool(self.model_cfg.get("load_in_8bit", False))
        if load_in_8bit:
            if self.device.type != "cuda":
                raise RuntimeError("HF 8-bit image-text VLM evaluation requires a CUDA device.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            if "device_map" not in model_kwargs:
                model_kwargs["device_map"] = {"": self.device.index or 0}
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=bool(self.model_cfg.get("trust_remote_code", True)),
        )
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        if "device_map" not in model_kwargs and not load_in_8bit:
            self.model.to(self.device)
        self.model.eval()

    def generate(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path], generation_kwargs: Mapping[str, Any]) -> str:
        return self.generate_batch(
            requests=[
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "image_paths": image_paths,
                }
            ],
            generation_kwargs=generation_kwargs,
        )[0]

    def generate_batch(self, *, requests: list[Mapping[str, Any]], generation_kwargs: Mapping[str, Any]) -> list[str]:
        if self.dry_run:
            return ['{"answer": "", "rationale": "dry run"}' for _ in requests]
        messages = [
            _hf_message_payload(
                clean_text(request.get("system_prompt")),
                clean_text(request.get("user_prompt")),
                list(request.get("image_paths", [])),
            )
            for request in requests
        ]
        message_payload = messages[0] if len(messages) == 1 else messages
        inputs = self.processor.apply_chat_template(
            message_payload,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        inputs = inputs.to(getattr(self.model, "device", self.device))
        input_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **dict(generation_kwargs))
        generated = outputs[:, input_len:] if input_len and outputs.shape[1] > input_len else outputs
        return [text.strip() for text in self.processor.batch_decode(generated, skip_special_tokens=True)]


def _oncovlm_stage_cfg(model_cfg: Mapping[str, Any], *, eval_cfg: Mapping[str, Any]) -> dict[str, Any]:
    stage_cfg = dict(model_cfg)
    stage_cfg["enable_thinking"] = as_bool(stage_cfg.get("enable_thinking", False))
    prefix_cfg = dict(eval_cfg.get("prefix_cache") or {})
    if not prefix_cfg:
        raise ValueError("VQA generation config must define prefix_cache for OncoVLM cache backends.")
    if not as_bool(prefix_cfg.get("enabled", True)):
        raise ValueError("OncoVLM evaluation is cache-only; prefix_cache.enabled must be true.")
    prefix_cfg["enabled"] = True
    stage_cfg["prefix_cache"] = prefix_cfg
    stage_cfg.pop("projectors", None)
    return stage_cfg


def _load_lora_adapter(language_model: torch.nn.Module, stage_cfg: Mapping[str, Any]) -> torch.nn.Module:
    adapter_path_text = clean_text(stage_cfg.get("lora_adapter_path"))
    if not adapter_path_text:
        raise ValueError("oncovlm_lora model config must define lora_adapter_path.")
    adapter_path = _resolve_path(adapter_path_text)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_path}")
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("peft is required for oncovlm_lora evaluation.") from exc
    return PeftModel.from_pretrained(language_model, str(adapter_path), is_trainable=False)


def _pad_prompt_batch(tokenizer: Any, token_id_rows: list[list[int]]) -> dict[str, torch.Tensor]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for VQA evaluation.")
    if not token_id_rows:
        raise ValueError("Cannot pad an empty VQA prompt batch.")
    max_tokens = max(len(token_ids) for token_ids in token_id_rows)
    input_ids = torch.full((len(token_id_rows), max_tokens), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(token_id_rows), max_tokens), dtype=torch.long)
    for row_index, token_ids in enumerate(token_id_rows):
        token_count = len(token_ids)
        input_ids[row_index, :token_count] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[row_index, :token_count] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _cached_prefix_rows_batch(
    *,
    rows: list[Mapping[str, Any]],
    tokenizer: Any,
    eval_cfg: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not rows:
        raise ValueError("OncoVLM cached-prefix evaluation received an empty row batch.")

    token_id_rows: list[list[int]] = []
    prefix_spans: list[list[dict[str, int | str]]] = []
    prompt_texts: list[str] = []
    prefix_tensors: dict[str, list[torch.Tensor | None]] = {modality: [] for modality in MODALITIES}
    enable_thinking = as_bool(stage_cfg.get("enable_thinking", False))
    prefix_value_override = clean_text(dict(stage_cfg.get("prefix_cache") or {}).get("prefix_value_override")) or "cached"
    if prefix_value_override not in {"cached", "ones", "random"}:
        raise ValueError("prefix_cache.prefix_value_override must be cached, ones, or random.")

    for row in rows:
        prompt_text = build_vqa_prompt(row, _prompt_block_for_projector(row, eval_cfg))
        token_ids = _prompt_token_ids(tokenizer, prompt_text, enable_thinking=enable_thinking)
        spans: list[dict[str, int | str]] = []
        for modality in row_modalities(row):
            start, end = _find_placeholder_span(tokenizer, token_ids, prefix_placeholder_for_modality(modality))
            spans.append({"modality": modality, "start": start, "end": end})
        token_id_rows.append(token_ids)
        prefix_spans.append(spans)
        prompt_texts.append(prompt_text)

        for modality in MODALITIES:
            if row_uses_modality(row, modality):
                tensor = load_row_cached_prefix_tensor(ROOT, stage_cfg, row, modality)
                if prefix_value_override == "ones":
                    tensor = torch.ones_like(tensor)
                elif prefix_value_override == "random":
                    tensor = torch.randn_like(tensor)
                prefix_tensors[modality].append(tensor)
            else:
                prefix_tensors[modality].append(None)

    batch = _pad_prompt_batch(tokenizer, token_id_rows)
    batch["prefix_spans"] = prefix_spans
    for modality, tensors in prefix_tensors.items():
        batch.update(pad_optional_prefix_tensors(modality, tensors))
    return batch, prompt_texts


class OncoVLMProjectorBackend:
    def __init__(self, model_cfg: Mapping[str, Any], *, eval_cfg: Mapping[str, Any], dry_run: bool):
        self.model_cfg = dict(model_cfg)
        self.eval_cfg = eval_cfg
        self.stage_cfg = _oncovlm_stage_cfg(model_cfg, eval_cfg=eval_cfg)
        self.dry_run = dry_run
        self.device = _resolve_device(self.stage_cfg.get("device"))
        self.tokenizer = None
        self.model = None
        self.prefix_dtype = None
        self.autocast_dtype = resolve_torch_dtype(self.stage_cfg.get("autocast_dtype", "bfloat16")) or torch.bfloat16
        if dry_run:
            return
        self.tokenizer = build_tokenizer(
            _model_name_or_path(self.stage_cfg),
            trust_remote_code=bool(self.stage_cfg.get("trust_remote_code", True)),
        )
        language_model = build_language_model(self.stage_cfg, device=self.device)
        if clean_text(self.model_cfg.get("backend")) == "oncovlm_lora":
            language_model = _load_lora_adapter(language_model, self.stage_cfg)
        self.model = OncoVLMVQASFTModel(
            language_model=language_model,
        )
        if not bool(self.stage_cfg.get("load_in_8bit", False)):
            self.model.to(self.device)
        self.prefix_dtype = self.model.language_model.get_input_embeddings().weight.dtype
        self.model.eval()

    def generate(self, *, row: Mapping[str, Any], generation_kwargs: Mapping[str, Any]) -> tuple[str, str]:
        return self.generate_batch(rows=[row], generation_kwargs=generation_kwargs)[0]

    def generate_batch(self, *, rows: list[Mapping[str, Any]], generation_kwargs: Mapping[str, Any]) -> list[tuple[str, str]]:
        if self.dry_run:
            return [('{"answer": "", "rationale": "dry run"}', "") for _ in rows]
        batch, prompt_texts = _cached_prefix_rows_batch(
            rows=rows,
            tokenizer=self.tokenizer,
            eval_cfg=prompt_cfg_for_model(self.eval_cfg, self.model_cfg),
            stage_cfg=self.stage_cfg,
        )
        batch = move_batch_to_device(batch, self.device, floating_dtype=self.prefix_dtype)
        use_autocast = self.device.type == "cuda" and self.autocast_dtype != torch.float32
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=use_autocast):
            inputs = self.model.prepare_interleaved_generation_inputs(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pathology_prefix_embeddings=batch.get("pathology_prefix_embeddings"),
                pathology_prefix_mask=batch.get("pathology_prefix_mask"),
                radiology_prefix_embeddings=batch.get("radiology_prefix_embeddings"),
                radiology_prefix_mask=batch.get("radiology_prefix_mask"),
                dnam_prefix_embeddings=batch.get("dnam_prefix_embeddings"),
                dnam_prefix_mask=batch.get("dnam_prefix_mask"),
                rna_prefix_embeddings=batch.get("rna_prefix_embeddings"),
                rna_prefix_mask=batch.get("rna_prefix_mask"),
                prefix_spans=batch["prefix_spans"],
            )
            generate_kwargs = dict(generation_kwargs)
            generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            if self.tokenizer.eos_token_id is not None:
                generate_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
            generated = generate_language_model_with_soft_prefix(
                self.model.language_model,
                inputs=inputs,
                generation_kwargs=generate_kwargs,
            )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [(text.strip(), prompt_text) for text, prompt_text in zip(decoded, prompt_texts, strict=True)]


def _build_backend(model_cfg: Mapping[str, Any], *, eval_cfg: Mapping[str, Any], dry_run: bool):
    backend = clean_text(model_cfg.get("backend"))
    if backend == "azure_openai_gpt":
        return AzureGPTBackend(model_cfg, dry_run=dry_run)
    if backend == "hf_image_text_to_text":
        return HFImageTextBackend(model_cfg, dry_run=dry_run)
    if backend in ONCOVLM_CACHE_BACKENDS:
        return OncoVLMProjectorBackend(model_cfg, eval_cfg=eval_cfg, dry_run=dry_run)
    if backend in {"hf_causal_lm"}:
        raise NotImplementedError(f"VQA evaluation backend {backend!r} is intentionally not supported yet.")
    raise ValueError(f"Unsupported VQA evaluation backend: {backend!r}")


def _prediction_row(
    *,
    row: Mapping[str, Any],
    parsed: Mapping[str, str],
    raw_response: str,
    image_paths: list[Path],
    model_cfg: Mapping[str, Any],
    evaluated_at: str,
    system_prompt: str,
    user_prompt: str,
    include_prompt: bool,
) -> dict[str, Any]:
    predicted_answer = clean_text(parsed.get("predicted_answer"))
    correct = (
        bool(predicted_answer and predicted_answer == clean_text(row.get("answer")))
        if question_type_key(row.get("question_type", "")) == "mcq"
        else None
    )
    output = {
        "question_id": int(row["question_id"]),
        "repeat_id": int(row["repeat_id"]),
        "base_question_id": int(row["base_question_id"]),
        "case_id": clean_text(row.get("case_id")),
        "project_id": clean_text(row.get("project_id")),
        "split": clean_text(row.get("split")),
        "question_type": clean_text(row.get("question_type")),
        "generation_type": clean_text(row.get("generation_type")),
        "task_category": clean_text(row.get("task_category")),
        "task_id": clean_text(row.get("task_id")),
        "modality_combination_name": clean_text(row.get("modality_combination_name")),
        "use_pathology": bool(row.get("use_pathology")),
        "use_radiology": bool(row.get("use_radiology")),
        "use_dnam": bool(row.get("use_dnam")),
        "use_rna": bool(row.get("use_rna")),
        "question": clean_text(row.get("question")),
        "option_a": clean_text(row.get("option_a")),
        "option_b": clean_text(row.get("option_b")),
        "option_c": clean_text(row.get("option_c")),
        "option_d": clean_text(row.get("option_d")),
        "answer": clean_text(row.get("answer")),
        "answer_label": clean_text(row.get("answer_label")),
        "predicted_answer": predicted_answer,
        "predicted_answer_label": clean_text(parsed.get("predicted_answer_label")),
        "parse_status": clean_text(parsed.get("parse_status")),
        "correct": correct,
        "raw_response": raw_response,
        "image_paths": [str(path) for path in image_paths],
        "backend": clean_text(model_cfg.get("backend")),
        "model_display_name": clean_text(model_cfg.get("display_name")),
        "model_name_or_path": _model_name_or_path(model_cfg),
        "evaluated_at": evaluated_at,
    }
    if include_prompt:
        output["system_prompt"] = system_prompt
        output["user_prompt"] = user_prompt
    return output


def main() -> None:
    cfg = load_cfg()
    eval_cfg = cfg.vqa_evaluation
    eval_dict = OmegaConf.to_container(eval_cfg, resolve=True)
    if not isinstance(eval_dict, dict):
        raise TypeError("Resolved VQA evaluation config must be a mapping.")

    run_cfg = dict(eval_dict.get("run") or {})
    data_cfg = dict(eval_dict.get("data") or {})
    if not run_cfg or not data_cfg:
        raise ValueError("VQA evaluation config must use the v2 run/data/models/prompts sections.")

    vqa_path_value = clean_text(data_cfg.get("vqa_parquet_path"))
    if not vqa_path_value:
        raise ValueError("vqa_evaluation.data.vqa_parquet_path must be populated.")
    vqa_path = _resolve_path(vqa_path_value)

    vqa_df = normalize_vqa_df(pd.read_parquet(vqa_path))
    validate_vqa_df(vqa_df)
    selected_df = select_eval_rows(vqa_df, eval_dict)
    selected_df = apply_group_sampling(selected_df, eval_dict.get("sampling", {}))
    selected_df = _sort_generation_rows(selected_df)
    if selected_df.empty:
        raise RuntimeError(
            "No VQA rows selected for evaluation. Check filters in the YAML."
        )

    print(f"VQA parquet: {vqa_path}")
    print(f"Rows selected: {len(selected_df)}")
    enabled_models = enabled_model_configs(eval_dict.get("models") or {})
    print(f"Enabled models: {', '.join(model['display_name'] for model in enabled_models)}")

    resume_existing = bool(run_cfg.get("resume_existing", True))
    dry_run = bool(run_cfg.get("dry_run", False))
    include_prompt = bool(run_cfg.get("include_prompt_in_predictions", False))
    print_first_n_outputs = _print_first_n_outputs(run_cfg)
    save_every_n_predictions = _save_every_n_predictions(run_cfg)
    repeats_cfg = _repeat_cfg(eval_dict)
    modality_ablation_cfg = _modality_ablation_cfg(eval_dict)
    predictions_path = _predictions_path(eval_dict)
    print(f"Predictions path: {predictions_path}")
    print(
        "Repeats: "
        f"n={repeats_cfg['n']} seed={repeats_cfg['seed']} "
        f"multi-repeat combos={sorted(repeats_cfg['repeat_modality_combination_names'])}"
    )
    if modality_ablation_cfg["enabled"]:
        print(
            "Model-specific modality ablation: "
            f"combos={sorted(modality_ablation_cfg['modality_combination_names'])} "
            f"models={sorted(modality_ablation_cfg['model_display_names'])}"
    )
    if predictions_path.exists() and not resume_existing:
        predictions_path.unlink()

    for model_cfg in enabled_models:
        model_base_df = _filter_model_modality_ablation_rows(
            selected_df,
            model_cfg=model_cfg,
            eval_cfg=eval_dict,
        )
        model_selected_df = _filter_missing_prefix_cache_rows(
            model_base_df,
            model_cfg=model_cfg,
            eval_cfg=eval_dict,
        )
        existing_keys = (
            _existing_prediction_keys(predictions_path, model_display_name=model_cfg["display_name"])
            if resume_existing
            else set()
        )
        repeated_records = _expand_repeat_rows(model_selected_df, repeats_cfg)
        row_records = [
            row
            for row in repeated_records
            if (int(row["question_id"]), int(row["repeat_id"])) not in existing_keys
        ]
        evaluated_at = datetime.now(timezone.utc).isoformat()
        print("\nVQA generation model")
        print(f"  Model: {model_cfg['display_name']}")
        print(f"  Backend: {model_cfg['backend']}")
        print(f"  Rows skipped by model ablation filter: {len(selected_df) - len(model_base_df)}")
        print(f"  Rows skipped from missing cached prefixes: {len(model_base_df) - len(model_selected_df)}")
        print(f"  Prediction repeats selected: {len(repeated_records)}")
        print(f"  Rows skipped from existing predictions: {len(repeated_records) - len(row_records)}")
        print(f"  Rows to evaluate: {len(row_records)}")
        print(f"  Predictions path: {predictions_path}")

        backend = _build_backend(model_cfg, eval_cfg=eval_dict, dry_run=dry_run)
        prediction_rows: list[dict[str, Any]] = []
        unsaved_prediction_rows = 0
        generation_kwargs = _generation_kwargs(eval_dict, model_cfg)
        model_prompt_cfg = prompt_cfg_for_model(eval_dict, model_cfg)
        batch_size = _model_batch_size(model_cfg)
        print(f"  Batch size: {batch_size}")
        print(f"  Save every N predictions: {save_every_n_predictions or 'disabled'}")
        previews_printed = 0

        progress = tqdm(
            total=len(row_records),
            desc=f"Generating {model_cfg['display_name']}",
            unit="question",
        )
        for row_batch in _batched_records_by_repeat(row_records, batch_size):
            progress_count = len(row_batch)
            row_batch = [row for row in row_batch if _should_generate_row(row, model_cfg)]
            if row_batch:
                repeat_id = int(row_batch[0]["repeat_id"])
                _set_repeat_seed(repeats_cfg, repeat_id)
                if clean_text(model_cfg["backend"]) in ONCOVLM_CACHE_BACKENDS:
                    raw_outputs = backend.generate_batch(
                        rows=row_batch,
                        generation_kwargs=generation_kwargs,
                    )
                    for row, (raw_response, projector_prompt) in zip(row_batch, raw_outputs, strict=True):
                        system_prompt, user_prompt = build_eval_prompt(row, model_prompt_cfg)
                        if projector_prompt:
                            user_prompt = projector_prompt
                        parsed = parse_model_response(row, raw_response)
                        if dry_run:
                            parsed = {"predicted_answer": "", "predicted_answer_label": "", "parse_status": "dry_run"}
                        if previews_printed < print_first_n_outputs:
                            previews_printed += 1
                            _print_output_preview(
                                preview_index=previews_printed,
                                preview_limit=print_first_n_outputs,
                                row=row,
                                raw_response=raw_response,
                                parsed=parsed,
                                model_cfg=model_cfg,
                            )
                        prediction_rows.append(
                            _prediction_row(
                                row=row,
                                parsed=parsed,
                                raw_response=raw_response,
                                image_paths=[],
                                model_cfg=model_cfg,
                                evaluated_at=evaluated_at,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                include_prompt=include_prompt,
                            )
                        )
                        unsaved_prediction_rows += 1
                else:
                    requests: list[dict[str, Any]] = []
                    for row in row_batch:
                        system_prompt, user_prompt = build_eval_prompt(row, model_prompt_cfg)
                        image_paths = collect_required_image_paths(row, model_prompt_cfg, repo_root=ROOT)
                        requests.append(
                            {
                                "row": row,
                                "system_prompt": system_prompt,
                                "user_prompt": user_prompt,
                                "image_paths": image_paths,
                            }
                        )
                    raw_responses = backend.generate_batch(
                        requests=requests,
                        generation_kwargs=generation_kwargs,
                    )
                    for request, raw_response in zip(requests, raw_responses, strict=True):
                        row = request["row"]
                        parsed = parse_model_response(row, raw_response)
                        if dry_run:
                            parsed = {"predicted_answer": "", "predicted_answer_label": "", "parse_status": "dry_run"}
                        if previews_printed < print_first_n_outputs:
                            previews_printed += 1
                            _print_output_preview(
                                preview_index=previews_printed,
                                preview_limit=print_first_n_outputs,
                                row=row,
                                raw_response=raw_response,
                                parsed=parsed,
                                model_cfg=model_cfg,
                            )
                        prediction_rows.append(
                            _prediction_row(
                                row=row,
                                parsed=parsed,
                                raw_response=raw_response,
                                image_paths=request["image_paths"],
                                model_cfg=model_cfg,
                                evaluated_at=evaluated_at,
                                system_prompt=request["system_prompt"],
                                user_prompt=request["user_prompt"],
                                include_prompt=include_prompt,
                            )
                        )
                        unsaved_prediction_rows += 1
            if save_every_n_predictions and unsaved_prediction_rows >= save_every_n_predictions:
                saved = _write_predictions(
                    predictions_path, prediction_rows, resume_existing=True
                )
                print(f"  Saved {len(prediction_rows)} new prediction rows ({len(saved)} total).")
                unsaved_prediction_rows = 0
            progress.update(progress_count)
        progress.close()

        final_predictions = _write_predictions(
            predictions_path, prediction_rows, resume_existing=True
        )
        print(f"  Final prediction rows: {len(final_predictions)}")

        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
