from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


PMC_OA_SAMPLE_TOKEN = "::sample="
PMC_OA_IMAGE_TOKEN = "::image="


@dataclass(frozen=True)
class PmcOaFeatureRef:
    store_path: str
    sample_id: str
    image_key: str


def _to_registry_relative_path(root_dir: Path, path_value: str | Path) -> str:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        return path.as_posix().lstrip("/")
    resolved = path.resolve()
    try:
        return resolved.relative_to(root_dir).as_posix()
    except ValueError:
        return resolved.as_posix().lstrip("/")


def _resolve_store_path(root_dir: Path, store_path: str | Path) -> Path:
    path = Path(str(store_path)).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


def looks_like_pmc_oa_feature_ref(value: str | Path | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return PMC_OA_SAMPLE_TOKEN in text and PMC_OA_IMAGE_TOKEN in text


def format_pmc_oa_feature_ref(
    *,
    root_dir: Path,
    store_path: str | Path,
    sample_id: str,
    image_key: str = "image_0",
) -> str:
    store_rel = _to_registry_relative_path(root_dir, store_path)
    return f"{store_rel}::sample={str(sample_id).strip()}::image={str(image_key).strip()}"


def parse_pmc_oa_feature_ref(value: str | Path) -> PmcOaFeatureRef:
    raw_value = str(value).strip()
    parts = raw_value.split("::")
    store_path = parts[0].strip()
    sample_id = ""
    image_key = "image_0"

    for part in parts[1:]:
        if part.startswith("sample="):
            sample_id = part.removeprefix("sample=").strip()
        elif part.startswith("image="):
            image_key = part.removeprefix("image=").strip() or "image_0"

    if not store_path:
        raise ValueError(f"PMC-OA feature ref is missing store path: {value}")
    if not sample_id:
        raise ValueError(f"PMC-OA feature ref is missing sample id: {value}")
    return PmcOaFeatureRef(store_path=store_path, sample_id=sample_id, image_key=image_key)


def load_pmc_oa_feature_array(root_dir: Path, embedding_ref: str | Path) -> np.ndarray:
    parsed = parse_pmc_oa_feature_ref(embedding_ref)
    store_path = _resolve_store_path(root_dir, parsed.store_path)
    handle = _get_pmc_oa_store_handle(store_path)
    dataset = handle["samples"][parsed.sample_id][parsed.image_key]
    array = np.asarray(dataset, dtype=np.float32)

    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D PMC-OA embedding array, got shape {tuple(array.shape)} from {embedding_ref}"
        )
    return array


@lru_cache(maxsize=4)
def _get_pmc_oa_store_handle(store_path: Path):
    import h5py

    return h5py.File(str(store_path), "r")


def build_pmc_oa_feature_index(
    *,
    root_dir: Path,
    store_path: str | Path,
) -> pd.DataFrame:
    import h5py

    resolved_store_path = _resolve_store_path(root_dir, store_path)
    rows: list[dict[str, object]] = []

    with h5py.File(resolved_store_path, "r") as handle:
        samples = handle["samples"]
        for sample_id in samples:
            sample_group = samples[sample_id]
            for image_key in sample_group:
                dataset = sample_group[image_key]
                image_path = str(dataset.attrs.get("image_path", "")).strip()
                image_name = Path(image_path).name if image_path else ""
                rows.append(
                    {
                        "sample_id": str(dataset.attrs.get("sample_id", sample_id)).strip() or str(sample_id).strip(),
                        "image_key": str(image_key).strip(),
                        "image_path": image_path,
                        "image_name": image_name,
                        "image_stem": Path(image_name).stem if image_name else "",
                        "feature_type": str(dataset.attrs.get("feature_type", "")).strip(),
                        "model_name": str(dataset.attrs.get("model_name", "")).strip(),
                        "embedding_dim": int(dataset.shape[-1]) if getattr(dataset, "shape", ()) else 0,
                        "store_path": _to_registry_relative_path(root_dir, resolved_store_path),
                        "embedding_ref": format_pmc_oa_feature_ref(
                            root_dir=root_dir,
                            store_path=resolved_store_path,
                            sample_id=str(dataset.attrs.get("sample_id", sample_id)).strip() or str(sample_id).strip(),
                            image_key=str(image_key).strip(),
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["image_name", "sample_id", "image_key"]).reset_index(drop=True)
    return frame


def read_or_build_pmc_oa_feature_index(
    *,
    root_dir: Path,
    store_path: str | Path,
    index_path: str | Path,
    rebuild: bool = False,
) -> pd.DataFrame:
    resolved_index_path = _resolve_store_path(root_dir, index_path)
    if resolved_index_path.exists() and not rebuild:
        return pd.read_parquet(resolved_index_path)

    frame = build_pmc_oa_feature_index(root_dir=root_dir, store_path=store_path)
    resolved_index_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(resolved_index_path, index=False)
    return frame


def build_pmc_oa_lookup_by_image_name(feature_index: pd.DataFrame) -> dict[str, dict[str, object]]:
    if feature_index.empty:
        return {}

    image_names = feature_index["image_name"].fillna("").astype(str).str.strip()
    duplicates = feature_index.loc[image_names.duplicated(keep=False) & image_names.ne("")]
    if not duplicates.empty:
        duplicate_names = duplicates["image_name"].drop_duplicates().tolist()
        raise RuntimeError(
            "PMC-OA feature index contains duplicate image names and cannot be joined safely: "
            f"{duplicate_names[:10]}"
        )

    out: dict[str, dict[str, object]] = {}
    for _, row in feature_index.iterrows():
        image_name = str(row.get("image_name", "")).strip()
        if not image_name:
            continue
        out[image_name] = row.to_dict()
    return out
