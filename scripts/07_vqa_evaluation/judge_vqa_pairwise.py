#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha1
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
from kidney_vlm.vqa.stage_config import clean_text

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="07_vqa_evaluation/judge_vqa_pairwise.yaml",
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


def _build_azure_client(azure_cfg: dict[str, Any]):
    if clean_text(azure_cfg.get("api_style")) == "anthropic_messages":
        return None

    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError("openai is required for the LLM judge.") from exc

    api_key_env = clean_text(azure_cfg.get("api_key_env"))
    api_key = os.getenv(api_key_env, "").strip() or _read_repo_env_value(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing Azure OpenAI key in env var or .env: {api_key_env}")

    return AzureOpenAI(
        api_version=clean_text(azure_cfg.get("api_version")),
        azure_endpoint=clean_text(azure_cfg.get("endpoint")),
        api_key=api_key,
    )


def _extract_text_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content.strip()
    if isinstance(raw_content, list):
        chunks: list[str] = []
        for item in raw_content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        return "\n".join(chunks).strip()
    return str(raw_content or "").strip()


def _call_judge(client: Any, azure_cfg: dict[str, Any], *, system_prompt: str, user_prompt: str) -> str:
    if clean_text(azure_cfg.get("api_style")) == "anthropic_messages":
        return _call_anthropic_messages(azure_cfg, system_prompt=system_prompt, user_prompt=user_prompt)

    request_kwargs: dict[str, Any] = {
        "model": clean_text(azure_cfg["deployment"]),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": int(azure_cfg.get("max_completion_tokens", 700)),
    }
    if clean_text(azure_cfg.get("reasoning_effort")):
        request_kwargs["reasoning_effort"] = clean_text(azure_cfg.get("reasoning_effort"))
    if clean_text(azure_cfg.get("verbosity")):
        request_kwargs["verbosity"] = clean_text(azure_cfg.get("verbosity"))
    if "temperature" in azure_cfg:
        request_kwargs["temperature"] = float(azure_cfg["temperature"])
    if "top_p" in azure_cfg:
        request_kwargs["top_p"] = float(azure_cfg["top_p"])

    last_error: Exception | None = None
    for attempt in range(1, int(azure_cfg.get("max_retries", 2)) + 1):
        try:
            response = client.chat.completions.create(**request_kwargs)
            return _extract_text_content(response.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            if attempt < int(azure_cfg.get("max_retries", 2)):
                time.sleep(float(azure_cfg.get("retry_sleep_seconds", 1.0)))
    raise RuntimeError(f"Judge call failed: {last_error}")


def _call_anthropic_messages(
    azure_cfg: dict[str, Any],
    *,
    system_prompt: str,
    user_prompt: str,
) -> str:
    import requests

    api_key_env = clean_text(azure_cfg.get("api_key_env"))
    api_key = os.getenv(api_key_env, "").strip() or _read_repo_env_value(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing Azure OpenAI key in env var or .env: {api_key_env}")

    endpoint = clean_text(azure_cfg.get("endpoint")).rstrip("/")
    api_version = clean_text(azure_cfg.get("api_version"))
    url = f"{endpoint}/v1/messages?api-version={api_version}"
    body: dict[str, Any] = {
        "model": clean_text(azure_cfg["deployment"]),
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": int(azure_cfg.get("max_completion_tokens", 700)),
    }
    if "temperature" in azure_cfg:
        body["temperature"] = float(azure_cfg["temperature"])
    if "temperature" not in azure_cfg and "top_p" in azure_cfg:
        body["top_p"] = float(azure_cfg["top_p"])

    last_error: Exception | None = None
    for attempt in range(1, int(azure_cfg.get("max_retries", 2)) + 1):
        try:
            response = requests.post(
                url,
                headers={
                    "api-key": api_key,
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=120,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            data = response.json()
            if "choices" in data:
                return _extract_text_content(data["choices"][0]["message"]["content"])
            return _extract_text_content(data["content"])
        except Exception as exc:
            last_error = exc
            if attempt < int(azure_cfg.get("max_retries", 2)):
                time.sleep(float(azure_cfg.get("retry_sleep_seconds", 1.0)))
    raise RuntimeError(f"Judge call failed: {last_error}")


def _filter_predictions(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if clean_text(filters.get("project_id")):
        out = out[out["project_id"].astype(str) == clean_text(filters["project_id"])]
    if filters.get("repeat_id") is not None:
        out = out[out["repeat_id"].astype(int) == int(filters["repeat_id"])]
    for cfg_key, col in [
        ("question_types", "question_type"),
        ("generation_types", "generation_type"),
        ("task_categories", "task_category"),
    ]:
        values = [clean_text(value) for value in filters.get(cfg_key, []) if clean_text(value)]
        if values:
            out = out[out[col].astype(str).isin(values)]
    if clean_text(filters.get("modality_combination_name")):
        out = out[
            out["modality_combination_name"].astype(str)
            == clean_text(filters["modality_combination_name"])
        ]
    return out


def _pair_rows(predictions: pd.DataFrame, models: dict[str, Any]) -> pd.DataFrame:
    target_name = clean_text(models["target_model"])
    baseline_name = clean_text(models["baseline_model"])
    target = predictions[predictions["model_display_name"].astype(str) == target_name]
    baseline = predictions[predictions["model_display_name"].astype(str) == baseline_name]
    target = target.drop_duplicates("question_id")
    baseline = baseline.drop_duplicates("question_id")
    paired = target.merge(baseline, on="question_id", suffixes=("_target", "_baseline"))
    if paired.empty:
        raise RuntimeError(f"No paired rows found for {target_name} vs {baseline_name}.")
    return paired


def _attach_case_summary(paired: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    summaries = pd.read_parquet(summary_path)[["case_id", "caption"]].drop_duplicates("case_id")
    out = paired.merge(summaries, left_on="case_id_target", right_on="case_id", how="left")
    out = out[out["caption"].astype(str).str.strip().astype(bool)]
    if out.empty:
        raise RuntimeError("No paired rows have detailed case summaries after joining by case_id.")
    return out


def _stable_int(*parts: Any) -> int:
    text = "||".join(str(part) for part in parts)
    return int(sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def _sample_pairs(paired: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    max_pairs = int(filters.get("max_pairs_per_task") or 0)
    if max_pairs <= 0:
        return paired.reset_index(drop=True)

    sample_seed = int(filters.get("sample_seed", 42))
    pieces: list[pd.DataFrame] = []
    for task_id, group in paired.groupby("task_id_target", sort=True):
        if len(group) <= max_pairs:
            pieces.append(group)
        elif bool(filters.get("sample", True)):
            pieces.append(
                group.sample(
                    n=max_pairs,
                    random_state=_stable_int(sample_seed, task_id) % (2**32),
                )
            )
        else:
            pieces.append(group.head(max_pairs))
    return pd.concat(pieces, ignore_index=True).reset_index(drop=True)


def _parse_judge_response(text: str) -> tuple[str, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return "", text.strip()
    data = json.loads(text[start : end + 1])
    preference = clean_text(data.get("preference")).lower()
    if preference not in {"a", "b", "tie"}:
        preference = ""
    return preference, clean_text(data.get("reason"))


def _random_swap(question_id: Any, *, seed: int) -> bool:
    return _stable_int(seed, question_id) % 2 == 1


def _format_prompt(template: str, row: pd.Series, *, answer_a: str, answer_b: str) -> str:
    return template.format(
        case_summary=clean_text(row.get("caption")),
        question=clean_text(row.get("question_target")),
        reference_answer=clean_text(row.get("answer_target")),
        answer_a=answer_a,
        answer_b=answer_b,
    )


def _judge_metric_record(
    *,
    metric_group: str,
    task_id: str,
    counts: Counter[str],
    target_name: str,
    baseline_name: str,
) -> dict[str, Any]:
    target_wins = int(counts[target_name])
    baseline_wins = int(counts[baseline_name])
    ties = int(counts["tie"])
    parse_failed = int(counts["parse_failed"])
    judged = target_wins + baseline_wins + ties
    return {
        "metric_group": metric_group,
        "task_id": task_id,
        "n": judged,
        "target_model": target_name,
        "baseline_model": baseline_name,
        "target_wins": target_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "parse_failed": parse_failed,
        "target_win_rate_excluding_ties": target_wins / max(target_wins + baseline_wins, 1),
        "target_win_rate_ties_half": (target_wins + 0.5 * ties) / max(judged, 1),
    }


def _write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    cfg = load_cfg()
    judge_cfg = OmegaConf.to_container(cfg.vqa_evaluation, resolve=True)

    run_cfg = dict(judge_cfg["run"])
    filters = dict(judge_cfg["filters"])
    models = dict(judge_cfg["models"])
    prompt_cfg = dict(judge_cfg["prompt"])
    azure_cfg = dict(judge_cfg["azure"])

    predictions_path = _resolve_path(run_cfg["predictions_path"])
    summary_path = _resolve_path(run_cfg["detailed_case_summary_path"])
    predictions = pd.read_parquet(predictions_path)
    filtered = _filter_predictions(predictions, filters)
    paired = _pair_rows(filtered, models)
    paired = _attach_case_summary(paired, summary_path)
    paired = _sample_pairs(paired, filters)

    target_name = clean_text(models["target_model"])
    baseline_name = clean_text(models["baseline_model"])
    print(f"Predictions: {predictions_path}")
    print(f"Detailed summaries: {summary_path}")
    print(f"Target model: {target_name}")
    print(f"Baseline model: {baseline_name}")
    print(f"Paired rows to judge: {len(paired)}")
    print("Rows per task:")
    for task_id, count in paired["task_id_target"].value_counts().sort_index().items():
        print(f"  {task_id}: {count}")
    print(f"Filters: {filters}")

    if bool(run_cfg.get("dry_run", False)):
        print("Dry run enabled; not calling judge endpoint.")
        return

    client = _build_azure_client(azure_cfg)
    counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = {}
    pair_order_seed = int(filters.get("pair_order_seed", 123))

    for idx, row in enumerate(tqdm(paired.itertuples(index=False), total=len(paired), desc="LLM judge"), start=1):
        row_s = pd.Series(row._asdict())
        task_id = clean_text(row_s.get("task_id_target"))
        task_counts.setdefault(task_id, Counter())
        swap = _random_swap(row_s["question_id"], seed=pair_order_seed)
        target_answer = clean_text(row_s.get("raw_response_target")) or clean_text(row_s.get("predicted_answer_target"))
        baseline_answer = clean_text(row_s.get("raw_response_baseline")) or clean_text(row_s.get("predicted_answer_baseline"))
        answer_a, answer_b = (baseline_answer, target_answer) if swap else (target_answer, baseline_answer)
        user_prompt = _format_prompt(
            clean_text(prompt_cfg["user_template"]),
            row_s,
            answer_a=answer_a,
            answer_b=answer_b,
        )
        raw = _call_judge(
            client,
            azure_cfg,
            system_prompt=clean_text(prompt_cfg["system_prompt"]),
            user_prompt=user_prompt,
        )
        preference, reason = _parse_judge_response(raw)
        if preference == "tie":
            winner = "tie"
        elif preference == "a":
            winner = baseline_name if swap else target_name
        elif preference == "b":
            winner = target_name if swap else baseline_name
        else:
            winner = "parse_failed"
        counts[winner] += 1
        task_counts[task_id][winner] += 1

        if bool(run_cfg.get("print_each", True)):
            print(
                f"[{idx}/{len(paired)}] qid={row_s['question_id']} "
                f"task={task_id} winner={winner} reason={reason}"
            )

    overall = _judge_metric_record(
        metric_group="overall",
        task_id="ALL",
        counts=counts,
        target_name=target_name,
        baseline_name=baseline_name,
    )
    by_task = [
        _judge_metric_record(
            metric_group="by_task",
            task_id=task_id,
            counts=task_counts[task_id],
            target_name=target_name,
            baseline_name=baseline_name,
        )
        for task_id in sorted(task_counts)
    ]
    metrics_path = predictions_path.parent / clean_text(run_cfg.get("metric_filename", "llm_as_judge_metric.json"))
    payload = {
        "run": {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "predictions_path": str(predictions_path),
            "detailed_case_summary_path": str(summary_path),
            "judge_deployment": clean_text(azure_cfg.get("deployment")),
            "target_model": target_name,
            "baseline_model": baseline_name,
            "filters": filters,
        },
        "metrics": [overall, *by_task],
    }
    _write_metrics(metrics_path, payload)

    print("\nPairwise judge result")
    print(f"  Judged pairs: {overall['n']}")
    print(f"  {target_name} wins: {overall['target_wins']}")
    print(f"  {baseline_name} wins: {overall['baseline_wins']}")
    print(f"  Ties: {overall['ties']}")
    print(f"  Parse/API failures: {overall['parse_failed']}")
    print(f"  Win rate excluding ties: {overall['target_win_rate_excluding_ties']:.3f}")
    print(f"  Win rate with ties as 0.5: {overall['target_win_rate_ties_half']:.3f}")
    print("  By task:")
    for record in by_task:
        print(
            f"    {record['task_id']}: n={record['n']} "
            f"win_ex_tie={record['target_win_rate_excluding_ties']:.3f} "
            f"win_tie_half={record['target_win_rate_ties_half']:.3f} "
            f"wins={record['target_wins']} losses={record['baseline_wins']} ties={record['ties']}"
        )
    print(f"  Wrote: {metrics_path}")


if __name__ == "__main__":
    main()
