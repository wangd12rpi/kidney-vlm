from __future__ import annotations

from pathlib import Path

import pandas as pd

from .registry_schema import normalize_registry_df


def load_hf_dataset_from_registry(registry_path: str | Path, split_filter: str | None = None):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("datasets is not installed. Install project dependencies first.") from exc

    frame = pd.read_parquet(Path(registry_path))
    frame = normalize_registry_df(frame)
    if split_filter is not None:
        frame = frame[frame["split"] == split_filter].reset_index(drop=True)
    return Dataset.from_pandas(frame, preserve_index=False)


def load_hf_streaming_dataset(registry_path: str | Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is not installed. Install project dependencies first.") from exc

    return load_dataset("parquet", data_files=str(Path(registry_path)), split="train", streaming=True)
