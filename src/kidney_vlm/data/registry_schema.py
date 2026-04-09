from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd

CORE_COLUMNS = [
    "sample_id",
    "source",
    "patient_id",
    "study_id",
    "split",
    "pathology_wsi_paths",
    "radiology_image_paths",
    "pathology_mask_paths",
    "radiology_mask_paths",
    "pathology_tile_embedding_paths",
    "pathology_slide_embedding_paths",
    "radiology_embedding_paths",
    "biomarkers_text",
    "question",
    "answer",
]

LIST_COLUMNS = [
    "pathology_wsi_paths",
    "radiology_image_paths",
    "pathology_mask_paths",
    "radiology_mask_paths",
    "pathology_tile_embedding_paths",
    "pathology_slide_embedding_paths",
    "radiology_embedding_paths",
]

TEXT_COLUMNS = [
    "sample_id",
    "source",
    "patient_id",
    "study_id",
    "split",
    "biomarkers_text",
    "question",
    "answer",
]


@dataclass(frozen=True)
class RegistrySchema:
    required_columns: tuple[str, ...] = tuple(CORE_COLUMNS)


def empty_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CORE_COLUMNS)


def _parse_serialized_list(text: str) -> list[str] | None:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None

    quoted_matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", stripped)
    recovered = [left or right for left, right in quoted_matches if (left or right).strip()]

    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        if recovered or not stripped[1:-1].strip():
            return recovered
        return None

    if isinstance(parsed, (list, tuple)):
        normalized: list[str] = []
        for item in parsed:
            normalized.extend(_normalize_list_value(item))
        if len(recovered) > len(normalized):
            return recovered
        return normalized
    return None


def _normalize_list_value(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            normalized.extend(_normalize_list_value(item))
        return normalized
    if isinstance(value, tuple):
        normalized: list[str] = []
        for item in value:
            normalized.extend(_normalize_list_value(item))
        return normalized
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            normalized: list[str] = []
            for item in converted:
                normalized.extend(_normalize_list_value(item))
            return normalized
    text = str(value).strip()
    if not text:
        return []
    parsed = _parse_serialized_list(text)
    if parsed is not None:
        return parsed
    return [text]


def _default_for_column(column: str) -> object:
    if column in LIST_COLUMNS:
        return []
    return ""


def ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in CORE_COLUMNS:
        if column not in out.columns:
            default_value = _default_for_column(column)
            if isinstance(default_value, list):
                out[column] = [list(default_value) for _ in range(len(out))]
            else:
                out[column] = default_value
    ordered = CORE_COLUMNS + [c for c in out.columns if c not in CORE_COLUMNS]
    return out[ordered]


def normalize_registry_df(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_core_columns(df)
    for column in LIST_COLUMNS:
        out[column] = out[column].map(_normalize_list_value)
    for column in TEXT_COLUMNS:
        out[column] = out[column].fillna("").map(str)
    return out


def validate_registry_df(df: pd.DataFrame, required_columns: Iterable[str] = CORE_COLUMNS) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Registry is missing required columns: {missing}")

    for column in LIST_COLUMNS:
        if column not in df.columns:
            continue
        invalid = [idx for idx, value in enumerate(df[column].tolist()) if not isinstance(value, list)]
        if invalid:
            raise ValueError(f"Column '{column}' must contain lists. Invalid row indices: {invalid[:10]}")
