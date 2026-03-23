from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _resolve_path(path_value: str | Path, root_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root_dir / path).resolve()


def _read_h5_features(path: Path, dataset_name: str) -> torch.Tensor:
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            available = ", ".join(sorted(handle.keys()))
            raise KeyError(
                f"Dataset '{dataset_name}' not found in '{path}'. Available keys: [{available}]"
            )
        features = handle[dataset_name][()]
    tensor = torch.as_tensor(features, dtype=torch.float32)
    if tensor.ndim == 1:
        # The PMC-OA extractor saves one pooled MedSigLIP embedding per image.
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor in '{path}', got shape {tuple(tensor.shape)}")
    return tensor


@dataclass(frozen=True)
class PMCOACaptionSample:
    sample_id: str
    split: str
    feature_path: Path
    caption_text: str
    image_path: str


class PMCOACaptionDataset(Dataset):
    """Load PMC-OA caption rows backed by the separate PMC-OA HDF5 feature registry."""

    def __init__(
        self,
        registry_path: str | Path,
        *,
        split: str | None = None,
        dataset_name: str = "patch_features",
        feature_field: str = "radiology_embedding_paths",
        caption_field: str = "caption_text",
        fallback_caption_fields: Iterable[str] = ("biomarkers_text",),
        root_dir: str | Path | None = None,
        max_rows: int | None = None,
        require_feature_paths: bool = True,
    ) -> None:
        super().__init__()
        self.registry_path = Path(registry_path).expanduser().resolve()
        self.dataset_name = str(dataset_name).strip() or "patch_features"
        self.feature_field = str(feature_field).strip() or "radiology_embedding_paths"
        self.caption_field = str(caption_field).strip() or "caption_text"
        self.fallback_caption_fields = tuple(
            str(field).strip() for field in fallback_caption_fields if str(field).strip()
        )
        self.root_dir = (
            Path(root_dir).expanduser().resolve()
            if root_dir is not None
            else Path.cwd().resolve()
        )

        frame = pd.read_parquet(self.registry_path)
        if split is not None:
            split_name = str(split).strip()
            frame = frame[frame["split"].fillna("").map(str) == split_name]

        records: list[PMCOACaptionSample] = []
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            feature_paths = _normalize_list(row_dict.get(self.feature_field))
            if require_feature_paths and not feature_paths:
                continue

            caption_text = _normalize_text(row_dict.get(self.caption_field))
            if not caption_text:
                for field_name in self.fallback_caption_fields:
                    caption_text = _normalize_text(row_dict.get(field_name))
                    if caption_text:
                        break
            if not caption_text:
                continue

            feature_path = _resolve_path(feature_paths[0], self.root_dir) if feature_paths else None
            if feature_path is None:
                continue
            if not feature_path.exists():
                raise FileNotFoundError(
                    f"Feature file referenced by '{self.registry_path}' does not exist: {feature_path}"
                )

            image_paths = _normalize_list(row_dict.get("radiology_image_paths"))
            records.append(
                PMCOACaptionSample(
                    sample_id=_normalize_text(row_dict.get("sample_id")),
                    split=_normalize_text(row_dict.get("split")),
                    feature_path=feature_path,
                    caption_text=caption_text,
                    image_path=image_paths[0] if image_paths else "",
                )
            )

        if max_rows is not None:
            if max_rows < 0:
                raise ValueError("max_rows must be >= 0")
            records = records[:max_rows]

        if not records:
            split_label = "all" if split is None else str(split)
            raise ValueError(
                f"No PMC-OA caption samples were found in '{self.registry_path}' for split='{split_label}'."
            )

        self.samples = records
        
        # Eagerly cache all features in memory to avoid per-item HDF5 open/close.
        self._feature_cache: dict[str, torch.Tensor] = {}
        for sample in self.samples:
            path_key = str(sample.feature_path)
            if path_key not in self._feature_cache:
                self._feature_cache[path_key] = _read_h5_features(
                    sample.feature_path, self.dataset_name
                )
        
        example = self._feature_cache[str(self.samples[0].feature_path)]
        
        # example = _read_h5_features(self.samples[0].feature_path, self.dataset_name)
        self.feature_dim = int(example.shape[-1])
        self.feature_token_count = int(example.shape[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # sample = self.samples[index]
        # visual_features = _read_h5_features(sample.feature_path, self.dataset_name)
        # return {
        #     "sample_id": sample.sample_id,
        #     "split": sample.split,
        #     "feature_path": str(sample.feature_path),
        #     "image_path": sample.image_path,
        #     "caption_text": sample.caption_text,
        #     "visual_features": visual_features,
        # }
        
        sample = self.samples[index]
        visual_features = self._feature_cache[str(sample.feature_path)]
        return {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "feature_path": str(sample.feature_path),
            "image_path": sample.image_path,
            "caption_text": sample.caption_text,
            "visual_features": visual_features,
        }


def build_pmc_oa_caption_datasets(
    registry_path: str | Path,
    *,
    dataset_name: str = "patch_features",
    root_dir: str | Path | None = None,
    train_split: str = "train",
    val_split: str | None = "val",
    test_split: str | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_test_rows: int | None = None,
) -> dict[str, PMCOACaptionDataset]:
    datasets = {
        "train": PMCOACaptionDataset(
            registry_path,
            split=train_split,
            dataset_name=dataset_name,
            root_dir=root_dir,
            max_rows=max_train_rows,
        )
    }
    if val_split:
        datasets["val"] = PMCOACaptionDataset(
            registry_path,
            split=val_split,
            dataset_name=dataset_name,
            root_dir=root_dir,
            max_rows=max_val_rows,
        )
    if test_split:
        datasets["test"] = PMCOACaptionDataset(
            registry_path,
            split=test_split,
            dataset_name=dataset_name,
            root_dir=root_dir,
            max_rows=max_test_rows,
        )
    return datasets
