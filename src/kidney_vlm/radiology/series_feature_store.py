from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


SERIES_TOKEN = "::series="


@dataclass(frozen=True)
class SeriesFeatureRef:
    store_path: str
    series_path: str


def looks_like_series_feature_ref(value: str | Path | None) -> bool:
    if value is None:
        return False
    return SERIES_TOKEN in str(value).strip()


def parse_series_feature_ref(value: str | Path) -> SeriesFeatureRef:
    raw_value = str(value).strip()
    parts = raw_value.split("::")
    store_path = parts[0].strip()
    series_path = ""

    for part in parts[1:]:
        if part.startswith("series="):
            series_path = part.removeprefix("series=").strip()

    if not store_path:
        raise ValueError(f"Series feature ref is missing store path: {value}")
    if not series_path:
        raise ValueError(f"Series feature ref is missing series path: {value}")
    return SeriesFeatureRef(store_path=store_path, series_path=series_path)


def _resolve_store_path(root_dir: Path, store_path: str | Path) -> Path:
    path = Path(str(store_path)).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


def _normalize_relpath(text: str | Path) -> str:
    raw_text = str(text).strip().replace("\\", "/").strip("/")
    return Path(raw_text).as_posix().strip("/")


@lru_cache(maxsize=4)
def _get_series_store_handle(store_path: Path):
    import h5py

    return h5py.File(str(store_path), "r")


def _decode_string_array(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def load_series_feature_array(root_dir: Path, embedding_ref: str | Path) -> np.ndarray:
    parsed = parse_series_feature_ref(embedding_ref)
    store_path = _resolve_store_path(root_dir, parsed.store_path)
    handle = _get_series_store_handle(store_path)
    if "features" not in handle:
        raise RuntimeError(f"Series feature store is missing 'features': {store_path}")
    if "png_relpaths" not in handle:
        raise RuntimeError(f"Series feature store is missing 'png_relpaths': {store_path}")

    features = handle["features"]
    png_relpaths = _decode_string_array(handle["png_relpaths"][:])
    target_series_path = _normalize_relpath(parsed.series_path)
    matching_indices = [
        idx
        for idx, png_relpath in enumerate(png_relpaths)
        if _normalize_relpath(Path(png_relpath).parent) == target_series_path
    ]
    if not matching_indices:
        raise FileNotFoundError(
            "No radiology slice features matched the requested series ref: "
            f"{embedding_ref}"
        )

    series_features = np.asarray(features[matching_indices], dtype=np.float32)
    if series_features.ndim != 2:
        raise ValueError(
            f"Expected 2D radiology feature matrix for {embedding_ref}, got shape {tuple(series_features.shape)}"
        )
    return series_features
