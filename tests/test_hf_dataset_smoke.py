from __future__ import annotations

import pandas as pd
import pytest

from kidney_vlm.data.hf_dataset import load_hf_dataset_from_registry


def test_hf_dataset_smoke(tmp_path) -> None:
    pytest.importorskip("datasets")

    frame = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "source": "tcga",
                "patient_id": "p1",
                "study_id": "st1",
                "split": "train",
                "pathology_wsi_paths": ["/tmp/a.svs"],
                "radiology_image_paths": [],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
                "pathology_tile_embedding_paths": [],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": [],
                "biomarkers_text": "marker: high",
                "question": "q",
                "answer": "a",
            }
        ]
    )
    parquet_path = tmp_path / "registry.parquet"
    frame.to_parquet(parquet_path, index=False)

    dataset = load_hf_dataset_from_registry(parquet_path, split_filter="train")
    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "s1"
