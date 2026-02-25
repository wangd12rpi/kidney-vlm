from __future__ import annotations

from dataclasses import dataclass
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
    "pathology_feature_paths",
    "radiology_feature_paths",
    "biomarkers_text",
    "question",
    "answer",
]

LIST_COLUMNS = [
    "pathology_wsi_paths",
    "radiology_image_paths",
    "pathology_mask_paths",
    "radiology_mask_paths",
    "pathology_feature_paths",
    "radiology_feature_paths",
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


def _normalize_list_value(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


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
