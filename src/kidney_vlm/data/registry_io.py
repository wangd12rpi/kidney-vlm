from __future__ import annotations

from pathlib import Path

import pandas as pd

from .registry_schema import empty_registry_frame, validate_registry_df


def read_parquet_or_empty(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return empty_registry_frame()
    return pd.read_parquet(parquet_path)


def write_registry_parquet(df: pd.DataFrame, path: str | Path, validate: bool = True) -> Path:
    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if validate:
        validate_registry_df(df)
    df.to_parquet(parquet_path, index=False)
    return parquet_path
