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
    assert normalized.at[0, "genomics_cnv_gene_paths"] == []
    assert normalized.at[0, "genomics_cnv_segment_paths"] == []
    assert normalized.at[0, "genomics_available_modalities"] == []
    assert normalized.at[0, "genomics_json_path"] == ""
    assert normalized.at[0, "genomics_llm_input_text_path"] == ""


def test_normalize_recovers_numpy_backed_and_stringified_list_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "source": "tcga",
                "pathology_wsi_paths": np.array(["slide-1.svs", "slide-2.svs"], dtype=object),
                "radiology_image_paths": np.array([], dtype=object),
                "radiology_image_modalities": np.array(["CT"], dtype=object),
                "radiology_series_slice_counts": np.array([12], dtype=object),
            },
            {
                "sample_id": "b",
                "source": "tcga",
                "pathology_wsi_paths": np.array(["['slide-3.svs'\n 'slide-4.svs']"], dtype=object),
                "radiology_image_paths": np.array(["[]"], dtype=object),
                "radiology_image_modalities": np.array(["['CT|MR']"], dtype=object),
                "radiology_series_slice_counts": np.array(["[7, 9]"], dtype=object),
            },
        ]
    )

    normalized = normalize_registry_df(frame)

    assert normalized.at[0, "pathology_wsi_paths"] == ["slide-1.svs", "slide-2.svs"]
    assert normalized.at[0, "radiology_image_paths"] == []
    assert normalized.at[0, "radiology_image_modalities"] == ["CT"]
    assert normalized.at[0, "radiology_series_slice_counts"] == [12]
    assert normalized.at[1, "pathology_wsi_paths"] == ["slide-3.svs", "slide-4.svs"]
    assert normalized.at[1, "radiology_image_paths"] == []
    assert normalized.at[1, "radiology_image_modalities"] == ["CT|MR"]
    assert normalized.at[1, "radiology_series_slice_counts"] == [7, 9]


def test_normalize_recovers_extra_genomics_optional_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "source": "tcga",
                "genomics_cnv_gene_paths": "['gene.tsv']",
                "genomics_cnv_segment_paths": ["segment.tsv"],
                "genomics_mirna_paths": np.array(["mirna.tsv"], dtype=object),
                "genomics_available_modalities": "['copy_number_gene', 'mutation_maf']",
                "genomics_json_path": None,
            }
        ]
    )

    normalized = normalize_registry_df(frame)

    assert normalized.at[0, "genomics_cnv_gene_paths"] == ["gene.tsv"]
    assert normalized.at[0, "genomics_cnv_segment_paths"] == ["segment.tsv"]
    assert normalized.at[0, "genomics_mirna_paths"] == ["mirna.tsv"]
    assert normalized.at[0, "genomics_available_modalities"] == [
        "copy_number_gene",
        "mutation_maf",
    ]
    assert normalized.at[0, "genomics_json_path"] == ""
