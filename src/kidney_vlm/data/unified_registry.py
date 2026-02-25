from __future__ import annotations

import pandas as pd

from .registry_schema import empty_registry_frame, normalize_registry_df, validate_registry_df


def replace_source_slice(unified_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    source_name = str(source_name)
    normalized_source = normalize_registry_df(source_df)
    normalized_source["source"] = source_name
    validate_registry_df(normalized_source)

    if unified_df.empty:
        return normalize_registry_df(normalized_source)

    normalized_unified = normalize_registry_df(unified_df)
    kept = normalized_unified[normalized_unified["source"] != source_name]
    merged = pd.concat([kept, normalized_source], ignore_index=True)
    merged = normalize_registry_df(merged)
    validate_registry_df(merged)
    return merged


def initialize_if_missing(unified_df: pd.DataFrame | None) -> pd.DataFrame:
    if unified_df is None:
        return empty_registry_frame()
    return normalize_registry_df(unified_df)
