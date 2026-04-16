from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def resolve_path(root: Path, path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def normalize_string_list(values: Any) -> list[str]:
    items: list[str] = []
    for value in list(values or []):
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def normalize_split_value(value: Any, default_split_name: str) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return str(default_split_name).strip().lower() or "train"
    return text


def build_dataset_for_push(
    frame: pd.DataFrame,
    *,
    split_column: str | None,
    default_split_name: str,
    allowed_split_names: list[str],
):
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError("datasets is required for HF dataset upload.") from exc

    if frame.empty:
        raise RuntimeError("Cannot upload an empty parquet.")

    normalized_split_column = str(split_column or "").strip()
    normalized_allowed_splits = [value.lower() for value in allowed_split_names if value.strip()]

    if normalized_split_column and normalized_split_column in frame.columns:
        split_values = frame[normalized_split_column].map(
            lambda value: normalize_split_value(value, default_split_name=default_split_name)
        )
        split_order = normalized_allowed_splits or split_values.drop_duplicates().tolist()

        split_datasets: dict[str, Any] = {}
        for split_name in split_order:
            split_frame = frame.loc[split_values == split_name].reset_index(drop=True)
            if split_frame.empty:
                continue
            split_datasets[split_name] = Dataset.from_pandas(split_frame, preserve_index=False)

        if split_datasets:
            return DatasetDict(split_datasets)

    return Dataset.from_pandas(frame.reset_index(drop=True), preserve_index=False)


def describe_dataset_payload(dataset_payload: Any) -> str:
    try:
        from datasets import DatasetDict
    except ImportError as exc:
        raise RuntimeError("datasets is required for HF dataset upload.") from exc

    if isinstance(dataset_payload, DatasetDict):
        parts = [f"{split_name}={dataset_payload[split_name].num_rows}" for split_name in dataset_payload.keys()]
        return ", ".join(parts)
    return f"rows={dataset_payload.num_rows}"
