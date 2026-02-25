from __future__ import annotations

from pathlib import Path

import pandas as pd

from .registry_schema import empty_registry_frame, normalize_registry_df, validate_registry_df


def read_parquet_or_empty(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return empty_registry_frame()
    frame = pd.read_parquet(parquet_path)
    return normalize_registry_df(frame)


def write_registry_parquet(df: pd.DataFrame, path: str | Path, validate: bool = True) -> Path:
    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_registry_df(df)
    if validate:
        validate_registry_df(normalized)
    normalized.to_parquet(parquet_path, index=False)
    return parquet_path
