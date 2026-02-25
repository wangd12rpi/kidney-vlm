from __future__ import annotations

import pandas as pd

from kidney_vlm.data.registry_schema import CORE_COLUMNS
from kidney_vlm.data.unified_registry import replace_source_slice


def _row(sample_id: str, source: str) -> dict:
    base = {
        "sample_id": sample_id,
        "source": source,
        "patient_id": "p",
        "study_id": "s",
        "split": "train",
        "pathology_wsi_paths": [],
        "radiology_image_paths": [],
        "pathology_mask_paths": [],
        "radiology_mask_paths": [],
        "pathology_feature_paths": [],
        "radiology_feature_paths": [],
        "biomarkers_text": "",
        "question": "",
        "answer": "",
    }
    return base


def test_replace_source_slice_without_duplicates() -> None:
    unified = pd.DataFrame([
        _row("tcga-old", "tcga"),
        _row("other-1", "other"),
    ])
    source = pd.DataFrame([
        _row("tcga-new-1", "tcga"),
        _row("tcga-new-2", "tcga"),
    ])

    merged = replace_source_slice(unified, source, source_name="tcga")

    tcga_rows = merged[merged["source"] == "tcga"]
    assert len(tcga_rows) == 2
    assert set(tcga_rows["sample_id"].tolist()) == {"tcga-new-1", "tcga-new-2"}
    assert "other-1" in merged["sample_id"].tolist()
    assert all(column in merged.columns for column in CORE_COLUMNS)
