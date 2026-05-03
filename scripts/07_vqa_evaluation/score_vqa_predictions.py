#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.vqa.eval_gpt import (
    _base_metric_record,
    _metric_values,
    add_bertscore_columns,
    build_flat_metric_records,
    parse_model_response,
    question_type_key,
)
from kidney_vlm.vqa.stage_config import clean_text

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

MODALITY_ABLATION_COMBOS = ("all_available", "path_only", "radiology_only")
UNIFIED_PATH = ROOT / "data/registry/unified.parquet"

METRIC_ID_COLUMNS = [
    "metric_group",
    "model_display_name",
    "backend",
    "question_type",
    "generation_type",
    "task_category",
    "task_id",
    "modality_combination_name",
    "project_id",
]

COUNT_COLUMNS = {"n", "correct", "parse_failed"}
VALUE_COLUMNS = {
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "bertscore_precision_mean",
    "bertscore_recall_mean",
    "bertscore_f1_mean",
}


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="07_vqa_evaluation/score_vqa_predictions.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _run_root(score_cfg: Mapping[str, Any]) -> Path:
    run_cfg = dict(score_cfg.get("run") or {})
    run_name = clean_text(run_cfg.get("name"))
    if not run_name:
        raise ValueError("vqa_evaluation.run.name must be populated.")
    output_root = _resolve_path(run_cfg.get("output_root", "results"))
    return output_root / run_name


def _run_filename(score_cfg: Mapping[str, Any], key: str, default: str) -> str:
    run_cfg = dict(score_cfg.get("run") or {})
    value = clean_text(run_cfg.get(key)) or default
    if "/" in value or "\\" in value:
        raise ValueError(f"vqa_evaluation.run.{key} must be a file name, got {value!r}.")
    return value


def _write_metrics(metrics_path: Path, metrics: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(metrics_path, json.dumps(metrics, indent=2, sort_keys=True))


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _prediction_path(score_cfg: Mapping[str, Any], prediction_filename: str) -> Path:
    path = _run_root(score_cfg) / prediction_filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing VQA predictions parquet: {path}")
    return path


def _reparse_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if "raw_response" not in predictions_df.columns:
        raise ValueError("Predictions parquet must contain a raw_response column.")

    out = predictions_df.copy()
    predicted_answers: list[str] = []
    predicted_labels: list[str] = []
    parse_statuses: list[str] = []
    correct_values: list[bool | None] = []

    for row in out.to_dict(orient="records"):
        parsed = parse_model_response(row, clean_text(row.get("raw_response")))
        predicted_answer = clean_text(parsed.get("predicted_answer"))
        predicted_label = clean_text(parsed.get("predicted_answer_label"))
        parse_status = clean_text(parsed.get("parse_status"))
        predicted_answers.append(predicted_answer)
        predicted_labels.append(predicted_label)
        parse_statuses.append(parse_status)
        correct_values.append(
            bool(predicted_answer and predicted_answer == clean_text(row.get("answer")))
            if question_type_key(row.get("question_type", "")) == "mcq"
            else None
        )

    out["predicted_answer"] = predicted_answers
    out["predicted_answer_label"] = predicted_labels
    out["parse_status"] = parse_statuses
    out["correct"] = correct_values
    return out


def _artifact_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(clean_text(item) for item in value)
    if hasattr(value, "tolist"):
        return _artifact_populated(value.tolist())
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = clean_text(value)
    return bool(text) and text.lower() not in {"none", "nan", "null", "[]"}


def _path_radiology_patient_ids() -> set[str]:
    columns = ["patient_id", "pathology_tile_embedding_paths", "radiology_embedding_paths"]
    unified = pd.read_parquet(UNIFIED_PATH, columns=columns)
    mask = unified["pathology_tile_embedding_paths"].map(_artifact_populated) & unified[
        "radiology_embedding_paths"
    ].map(_artifact_populated)
    return set(unified.loc[mask, "patient_id"].astype(str))


def _modality_ablation_rows(predictions_df: pd.DataFrame, path_radiology_patient_ids: set[str]) -> pd.DataFrame:
    frame = predictions_df[
        predictions_df["case_id"].astype(str).isin(path_radiology_patient_ids)
        & predictions_df["modality_combination_name"].astype(str).isin(MODALITY_ABLATION_COMBOS)
    ].copy()
    frame["base_question_id"] = frame["base_question_id"].astype(str)
    frame["modality_combination_name"] = frame["modality_combination_name"].astype(str)

    matched_groups: list[pd.DataFrame] = []
    for _, group in frame.groupby(["question_type", "generation_type", "task_category"], dropna=False, sort=True):
        combo_sets = group.groupby("base_question_id")["modality_combination_name"].agg(set)
        matched_base_ids = combo_sets[
            combo_sets.map(lambda combos: set(MODALITY_ABLATION_COMBOS).issubset(combos))
        ].index
        if len(matched_base_ids):
            matched_groups.append(group[group["base_question_id"].isin(matched_base_ids)])
    if not matched_groups:
        return frame.iloc[0:0].copy()
    return pd.concat(matched_groups, ignore_index=True)


def build_modality_ablation_records(
    predictions_df: pd.DataFrame,
    *,
    model_display_name: str,
    backend: str,
    path_radiology_patient_ids: set[str],
) -> list[dict[str, Any]]:
    frame = _modality_ablation_rows(predictions_df, path_radiology_patient_ids)
    records: list[dict[str, Any]] = []
    group_columns = ["question_type", "generation_type", "task_category", "modality_combination_name"]
    for group_values, group in frame.groupby(group_columns, dropna=False, sort=True):
        dimension_values = {
            column: str(value)
            for column, value in zip(group_columns, group_values, strict=True)
        }
        record = _base_metric_record(
            metric_group="modality_ablation",
            model_display_name=model_display_name,
            backend=backend,
            values=dimension_values,
        )
        record.update(_metric_values(group))
        records.append(record)
    return records


def _run_payload(
    *,
    score_cfg: Mapping[str, Any],
    predictions_path: Path,
    metrics_path: Path,
    scored_at: str,
    model_count: int,
    repeat_count: int,
) -> dict[str, Any]:
    run_cfg = dict(score_cfg.get("run") or {})
    return {
        "name": clean_text(run_cfg.get("name")),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "scored_at": scored_at,
        "model_count": int(model_count),
        "repeat_count": int(repeat_count),
    }


def _stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return float((sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _clean_number(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _aggregate_repeat_records(per_repeat_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in per_repeat_records:
        key = tuple(clean_text(record.get(column)) for column in METRIC_ID_COLUMNS)
        grouped.setdefault(key, []).append(record)

    out: list[dict[str, Any]] = []
    for records in grouped.values():
        first = records[0]
        aggregate = {column: first[column] for column in METRIC_ID_COLUMNS}
        aggregate["num_repeat"] = int(len(records))

        for column in COUNT_COLUMNS:
            values = [_clean_number(record.get(column)) for record in records if record.get(column) is not None]
            if values:
                total = sum(values)
                aggregate[column] = int(total) if total.is_integer() else float(total)
                aggregate[f"{column}_stdev"] = _stdev(values)

        for column in VALUE_COLUMNS:
            values = [_clean_number(record.get(column)) for record in records if record.get(column) is not None]
            values = [value for value in values if value is not None]
            if not values:
                continue
            if column == "accuracy" and aggregate.get("n") and aggregate.get("correct") is not None:
                aggregate[column] = float(aggregate["correct"] / aggregate["n"])
            else:
                aggregate[column] = _mean(values)
            aggregate[f"{column}_stdev"] = _stdev(values)

        out.append(aggregate)
    return sorted(
        out,
        key=lambda record: tuple(clean_text(record.get(column)) for column in METRIC_ID_COLUMNS),
    )


def main() -> None:
    cfg = load_cfg()
    score_cfg = cfg.vqa_evaluation
    score_dict = OmegaConf.to_container(score_cfg, resolve=True)
    if not isinstance(score_dict, dict):
        raise TypeError("Resolved VQA scoring config must be a mapping.")

    prediction_filename = _run_filename(score_dict, "prediction_filename", "predictions.parquet")
    metrics_filename = _run_filename(score_dict, "metrics_filename", "metrics.json")
    predictions_path = _prediction_path(score_dict, prediction_filename)
    metrics_path = _run_root(score_dict) / metrics_filename

    print(f"Predictions path: {predictions_path}")
    print(f"Metrics path: {metrics_path}")

    predictions = pd.read_parquet(predictions_path)
    if "repeat_id" not in predictions.columns:
        raise ValueError("Predictions parquet must contain repeat_id. Regenerate predictions with the repeat-aware pipeline.")
    scored_at = datetime.now(timezone.utc).isoformat()
    scored_predictions = _reparse_predictions(predictions)
    scored_predictions = add_bertscore_columns(
        scored_predictions,
        dict(dict(score_dict.get("metrics") or {}).get("bert_score") or {}),
    )
    path_radiology_patient_ids = _path_radiology_patient_ids()

    per_repeat_records: list[dict[str, Any]] = []
    model_columns = ["model_display_name", "backend", "repeat_id"]
    for group_values, group in scored_predictions.groupby(model_columns, dropna=False, sort=True):
        model_display_name = clean_text(group_values[0])
        backend = clean_text(group_values[1])
        repeat_id = int(group_values[2])
        print("\nVQA scoring model")
        print(f"  Model: {model_display_name}")
        print(f"  Repeat: {repeat_id}")
        print(f"  Prediction rows: {len(group)}")
        records = build_flat_metric_records(
            group,
            model_display_name=model_display_name,
            backend=backend,
        )
        records.extend(
            build_modality_ablation_records(
                group,
                model_display_name=model_display_name,
                backend=backend,
                path_radiology_patient_ids=path_radiology_patient_ids,
            )
        )
        for record in records:
            record["repeat_id"] = repeat_id
        per_repeat_records.extend(records)
        overall = next((item for item in records if item["metric_group"] == "overall"), {})
        print(f"  Overall metrics: {overall}")

    metric_records = _aggregate_repeat_records(per_repeat_records)

    payload = {
        "run": _run_payload(
            score_cfg=score_dict,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            scored_at=scored_at,
            model_count=int(scored_predictions["model_display_name"].nunique()),
            repeat_count=int(scored_predictions["repeat_id"].nunique()),
        ),
        "metrics": metric_records,
        "errors": [],
    }
    _write_metrics(metrics_path, payload)


if __name__ == "__main__":
    main()
