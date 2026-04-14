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


def upsert_source_slice(
    unified_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    source_name: str,
    key_columns: list[str] | tuple[str, ...] = ("sample_id",),
) -> pd.DataFrame:
    source_name = str(source_name)
    source_rows = source_df.copy()
    source_rows["source"] = source_name

    if unified_df.empty:
        return source_rows

    source_existing = unified_df[unified_df["source"] == source_name].copy()
    non_source_rows = unified_df[unified_df["source"] != source_name].copy()
    if source_existing.empty:
        return pd.concat([non_source_rows, source_rows], ignore_index=True)

    keys = [str(column).strip() for column in key_columns if str(column).strip()]
    if not keys:
        raise ValueError("At least one key column is required for source-slice upserts.")

    missing = [column for column in keys if column not in source_rows.columns or column not in source_existing.columns]
    if missing:
        raise ValueError(f"Cannot upsert source slice without key columns present in both frames: {missing}")

    source_rows = source_rows.drop_duplicates(subset=keys, keep="last")
    incoming_keys = source_rows[keys].drop_duplicates()
    existing_to_keep = source_existing.merge(incoming_keys, on=keys, how="left", indicator=True)
    existing_to_keep = existing_to_keep[existing_to_keep["_merge"] == "left_only"].drop(columns=["_merge"])
    return pd.concat([non_source_rows, existing_to_keep, source_rows], ignore_index=True)


def initialize_if_missing(unified_df: pd.DataFrame | None) -> pd.DataFrame:
    if unified_df is None:
        return empty_registry_frame()
    return unified_df
