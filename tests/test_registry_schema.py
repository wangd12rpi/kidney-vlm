from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kidney_vlm.data.registry_schema import CORE_COLUMNS, empty_registry_frame, normalize_registry_df, validate_registry_df


def test_missing_core_column_raises() -> None:
    frame = empty_registry_frame().drop(columns=["sample_id"])
    with pytest.raises(ValueError):
        validate_registry_df(frame)


def test_normalize_fills_core_columns() -> None:
    frame = pd.DataFrame([{"sample_id": "a", "source": "tcga"}])
    normalized = normalize_registry_df(frame)
    assert all(column in normalized.columns for column in CORE_COLUMNS)
    validate_registry_df(normalized)


def test_normalize_recovers_numpy_backed_and_stringified_list_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "source": "tcga",
                "pathology_wsi_paths": np.array(["slide-1.svs", "slide-2.svs"], dtype=object),
                "radiology_image_paths": np.array([], dtype=object),
            },
            {
                "sample_id": "b",
                "source": "tcga",
                "pathology_wsi_paths": np.array(["['slide-3.svs'\n 'slide-4.svs']"], dtype=object),
                "radiology_image_paths": np.array(["[]"], dtype=object),
            },
        ]
    )

    normalized = normalize_registry_df(frame)

    assert normalized.at[0, "pathology_wsi_paths"] == ["slide-1.svs", "slide-2.svs"]
    assert normalized.at[0, "radiology_image_paths"] == []
    assert normalized.at[1, "pathology_wsi_paths"] == ["slide-3.svs", "slide-4.svs"]
    assert normalized.at[1, "radiology_image_paths"] == []
