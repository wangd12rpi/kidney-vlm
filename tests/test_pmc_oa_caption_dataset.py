from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import torch

from kidney_vlm.data.pmc_oa_caption_dataset import PMCOACaptionDataset


def _create_test_registry(tmp_path: Path) -> tuple[Path, Path]:
    """Shared setup: create a minimal HDF5 feature file and parquet registry."""
    features_path = tmp_path / "sample.h5"
    with h5py.File(features_path, "w") as handle:
        handle.create_dataset("patch_features", data=[1.0, 2.0, 3.0, 4.0])

    registry_path = tmp_path / "pmc.parquet"
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "source": "pmc_oa",
                "patient_id": "p1",
                "study_id": "st1",
                "split": "train",
                "pathology_wsi_paths": [],
                "radiology_image_paths": ["images/sample.jpg"],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
                "pathology_tile_embedding_paths": [],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": [features_path.name],
                "biomarkers_text": "",
                "question": "",
                "answer": "",
                "caption_text": "renal ct caption",
            }
        ]
    )
    frame.to_parquet(registry_path, index=False)
    return registry_path, features_path


def test_pmc_oa_caption_dataset_loads_h5_features(tmp_path: Path) -> None:
    registry_path, _ = _create_test_registry(tmp_path)

    dataset = PMCOACaptionDataset(
        registry_path,
        split="train",
        dataset_name="patch_features",
        root_dir=tmp_path,
    )

    sample = dataset[0]
    assert len(dataset) == 1
    assert dataset.feature_dim == 4
    assert dataset.feature_token_count == 1
    assert sample["sample_id"] == "sample-1"
    assert sample["caption_text"] == "renal ct caption"
    assert torch.equal(
        sample["visual_features"],
        torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
    )


def test_pmc_oa_caption_dataset_caches_features(tmp_path: Path) -> None:
    registry_path, _ = _create_test_registry(tmp_path)

    dataset = PMCOACaptionDataset(
        registry_path,
        split="train",
        dataset_name="patch_features",
        root_dir=tmp_path,
    )

    assert hasattr(dataset, "_feature_cache")
    assert len(dataset._feature_cache) == 1
    # Same tensor object returned each time (not a fresh HDF5 read).
    assert dataset[0]["visual_features"] is dataset[0]["visual_features"]