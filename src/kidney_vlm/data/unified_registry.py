from __future__ import annotations

import pandas as pd

from .registry_schema import empty_registry_frame


def _value_is_effectively_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _series_is_effectively_empty(series: pd.Series) -> bool:
    return bool(series.map(_value_is_effectively_empty).all())


def replace_source_slice(unified_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    source_name = str(source_name)
    source_rows = source_df.copy()
    source_rows["source"] = source_name

    if unified_df.empty:
        return source_rows

    kept = unified_df[unified_df["source"] != source_name].copy()
    stale_columns = [
        column
        for column in kept.columns
        if column not in source_rows.columns and _series_is_effectively_empty(kept[column])
    ]
    if stale_columns:
        kept = kept.drop(columns=stale_columns)
    if kept.empty:
        return source_rows
    return pd.concat([kept, source_rows], ignore_index=True, sort=False)


def initialize_if_missing(unified_df: pd.DataFrame | None) -> pd.DataFrame:
    if unified_df is None:
        return empty_registry_frame()
    return unified_df
