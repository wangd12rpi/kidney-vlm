from __future__ import annotations

import pandas as pd

from .registry_schema import empty_registry_frame


def replace_source_slice(unified_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    source_name = str(source_name)
    source_rows = source_df.copy()
    source_rows["source"] = source_name

    if unified_df.empty:
        return source_rows

    kept = unified_df[unified_df["source"] != source_name]
    return pd.concat([kept, source_rows], ignore_index=True)


def initialize_if_missing(unified_df: pd.DataFrame | None) -> pd.DataFrame:
    if unified_df is None:
        return empty_registry_frame()
    return unified_df
