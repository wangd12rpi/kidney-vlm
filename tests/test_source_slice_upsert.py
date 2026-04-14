from __future__ import annotations

import pandas as pd

from kidney_vlm.data.unified_registry import upsert_source_slice


def test_upsert_source_slice_replaces_only_matching_source_keys() -> None:
    unified = pd.DataFrame(
        [
            {"sample_id": "tcga-a", "source": "tcga", "patient_id": "A"},
            {"sample_id": "tcga-b", "source": "tcga", "patient_id": "B"},
            {"sample_id": "other-c", "source": "other", "patient_id": "C"},
        ]
    )
    source = pd.DataFrame(
        [
            {"sample_id": "tcga-b", "source": "tcga", "patient_id": "B-new"},
            {"sample_id": "tcga-d", "source": "tcga", "patient_id": "D"},
        ]
    )

    merged = upsert_source_slice(unified, source, source_name="tcga", key_columns=("sample_id",))

    tcga_rows = merged[merged["source"] == "tcga"].sort_values("sample_id").reset_index(drop=True)
    assert tcga_rows["sample_id"].tolist() == ["tcga-a", "tcga-b", "tcga-d"]
    assert tcga_rows["patient_id"].tolist() == ["A", "B-new", "D"]
    other_rows = merged[merged["source"] == "other"]
    assert other_rows["sample_id"].tolist() == ["other-c"]
