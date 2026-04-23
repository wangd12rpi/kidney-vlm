from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

VQA_COLUMNS = [
    "case_id",
    "project_id",
    "question_id",
    "base_question_id",
    "split",
    "question_type",
    "generation_type",
    "task_category",
    "task_id",
    "use_pathology",
    "use_radiology",
    "use_dnam",
    "use_rna",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer",
    "caption_id",
    "ground_truth_source",
    "radiology_biomarker",
    "pathology_feature_paths",
    "radiology_feature_paths",
    "dnam_feature_path",
    "rna_feature_path",
    "pathology_roi_png_dir",
    "radiology_view_png_dir",
    "dnam_text_summary",
    "rna_text_summary",
]

ID_COLUMNS = ["question_id", "base_question_id"]
BOOL_COLUMNS = ["use_pathology", "use_radiology", "use_dnam", "use_rna"]
ARRAY_COLUMNS = ["pathology_feature_paths", "radiology_feature_paths"]
OPTION_COLUMNS = ["option_a", "option_b", "option_c", "option_d"]
TEXT_COLUMNS = [
    column
    for column in VQA_COLUMNS
    if column not in ID_COLUMNS and column not in BOOL_COLUMNS and column not in ARRAY_COLUMNS
]

QUESTION_TYPES = {"mcq", "qa"}
GENERATION_TYPES = {"from_ground_truth", "from_caption"}


def empty_vqa_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=VQA_COLUMNS)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _coerce_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value) != 0
    text = _coerce_text(value).lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"", "0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot coerce value to bool: {value!r}")


def _coerce_int_id(value: object) -> int | pd.NA:
    if _is_missing(value):
        return pd.NA
    if isinstance(value, bool):
        raise ValueError(f"Boolean value is not a valid integer ID: {value!r}")
    if isinstance(value, Integral):
        return int(value)
    text = _coerce_text(value)
    return int(float(text))


def _coerce_array(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        raw_items = value.tolist()
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raise ValueError(f"VQA array columns must contain one-layer list values, got {type(value).__name__}: {value!r}")

    items: list[str] = []
    for item in raw_items:
        if isinstance(item, (list, tuple, np.ndarray)):
            raise ValueError(f"VQA array columns must be one-layer lists, got nested value: {item!r}")
        text = _coerce_text(item)
        if text:
            items.append(text)
    return items


def _assert_exact_schema_columns(df: pd.DataFrame) -> None:
    columns = list(df.columns)
    missing = [column for column in VQA_COLUMNS if column not in columns]
    extra = [column for column in columns if column not in VQA_COLUMNS]
    if missing:
        raise ValueError(f"VQA frame is missing required columns: {missing}")
    if extra:
        raise ValueError(f"VQA frame has unexpected columns: {extra}")


def normalize_vqa_df(df: pd.DataFrame) -> pd.DataFrame:
    _assert_exact_schema_columns(df)
    out = df.copy()
    for column in TEXT_COLUMNS:
        out[column] = out[column].map(_coerce_text)
    for column in ARRAY_COLUMNS:
        out[column] = out[column].map(_coerce_array)
    for column in BOOL_COLUMNS:
        out[column] = out[column].map(_coerce_bool).astype(bool)
    for column in ID_COLUMNS:
        out[column] = out[column].map(_coerce_int_id).astype("Int64")
    return out[VQA_COLUMNS + [column for column in out.columns if column not in VQA_COLUMNS]]


def _option_values(row: pd.Series) -> list[str]:
    return [str(row[column]).strip() for column in OPTION_COLUMNS if str(row[column]).strip()]


def validate_vqa_df(df: pd.DataFrame, required_columns: Iterable[str] = VQA_COLUMNS) -> None:
    if tuple(required_columns) == tuple(VQA_COLUMNS):
        _assert_exact_schema_columns(df)
    else:
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"VQA frame is missing required columns: {missing}")

    if df.empty:
        return

    invalid_question_types = sorted(set(df["question_type"].astype(str)) - QUESTION_TYPES)
    if invalid_question_types:
        raise ValueError(f"Invalid question_type values: {invalid_question_types}")

    invalid_generation_types = sorted(set(df["generation_type"].astype(str)) - GENERATION_TYPES)
    if invalid_generation_types:
        raise ValueError(f"Invalid generation_type values: {invalid_generation_types}")

    for column in ID_COLUMNS:
        if df[column].isna().any():
            bad_indices = df.index[df[column].isna()].tolist()[:10]
            raise ValueError(f"Column '{column}' must be populated. Invalid row indices: {bad_indices}")

    duplicated_question_ids = df.loc[df["question_id"].duplicated(keep=False), "question_id"].tolist()
    if duplicated_question_ids:
        raise ValueError(f"question_id must be unique. Duplicates: {duplicated_question_ids[:10]}")

    for column in ["case_id", "project_id", "split", "task_category", "task_id", "question", "answer"]:
        empty_indices = df.index[df[column].astype(str).str.strip().eq("")].tolist()
        if empty_indices:
            raise ValueError(f"Column '{column}' must be populated. Invalid row indices: {empty_indices[:10]}")

    for row_index, row in df.iterrows():
        for column in ARRAY_COLUMNS:
            value = row[column]
            if not isinstance(value, list):
                raise ValueError(f"Column '{column}' must contain one-layer lists. Invalid row index: {row_index}")
            if any(isinstance(item, (list, tuple, np.ndarray)) for item in value):
                raise ValueError(f"Column '{column}' must contain one-layer lists. Invalid row index: {row_index}")

        question_type = str(row["question_type"]).strip()
        generation_type = str(row["generation_type"]).strip()
        if question_type == "mcq":
            options = _option_values(row)
            if len(options) < 2:
                raise ValueError(f"MCQ row {row_index} must have at least two non-empty options.")
            if str(row["answer"]).strip() not in options:
                raise ValueError(f"MCQ row {row_index} answer must exactly match one non-empty option.")
        if generation_type == "from_ground_truth" and not str(row["ground_truth_source"]).strip():
            raise ValueError(f"Ground-truth row {row_index} must set ground_truth_source.")
        if generation_type == "from_caption" and not str(row["caption_id"]).strip():
            raise ValueError(f"Caption row {row_index} must set caption_id.")


def read_vqa_parquet_or_empty(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return empty_vqa_frame()
    return normalize_vqa_df(pd.read_parquet(parquet_path))


def write_vqa_parquet(df: pd.DataFrame, path: str | Path, validate: bool = True) -> Path:
    out = normalize_vqa_df(df)
    if validate:
        validate_vqa_df(out)
    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(parquet_path, index=False)
    return parquet_path


def _key_tuples(df: pd.DataFrame, key_columns: tuple[str, ...]) -> list[tuple[object, ...]]:
    return [tuple(row[column] for column in key_columns) for _, row in df.iterrows()]


def upsert_vqa_rows(
    existing_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    *,
    key_columns: tuple[str, ...] = ("question_id",),
) -> pd.DataFrame:
    generated = normalize_vqa_df(generated_df)
    validate_vqa_df(generated)
    if generated.empty:
        return normalize_vqa_df(existing_df)

    existing = normalize_vqa_df(existing_df)
    for column in key_columns:
        if column not in VQA_COLUMNS:
            raise ValueError(f"Unsupported VQA upsert key column: {column}")

    generated_keys = set(_key_tuples(generated, key_columns))
    if existing.empty:
        final_df = generated
    else:
        existing = existing.drop_duplicates(subset=list(key_columns), keep="last").reset_index(drop=True)
        keep_mask = [key not in generated_keys for key in _key_tuples(existing, key_columns)]
        final_df = pd.concat([existing.loc[keep_mask], generated], ignore_index=True)

    final_df = normalize_vqa_df(final_df)
    validate_vqa_df(final_df)
    return final_df
