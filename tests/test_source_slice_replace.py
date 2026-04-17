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
        "genomics_rna_bulk_paths": [],
        "genomics_rna_bulk_feature_path": "",
        "genomics_rna_bulk_file_ids": [],
        "genomics_rna_bulk_file_names": [],
        "genomics_rna_bulk_sample_types": [],
        "genomics_rna_bulk_workflow_types": [],
        "genomics_rna_bulk_molecular_subtype": "",
        "genomics_rna_bulk_immune_subtype": "",
        "genomics_rna_bulk_leukocyte_fraction": "",
        "genomics_rna_bulk_tumor_purity": "",
        "genomics_rna_bulk_top_immune_cell_types": [],
        "genomics_rna_bulk_top_immune_cell_fractions": [],
        "genomics_dna_methylation_paths": [],
        "genomics_dna_methylation_feature_path": "",
        "genomics_cnv_paths": [],
        "genomics_cnv_feature_path": "",
        "pathology_wsi_paths": [],
        "radiology_image_paths": [],
        "radiology_image_modalities": [],
        "radiology_report_download_paths": [],
        "radiology_report_uri_paths": [],
        "radiology_report_series_descriptions": [],
        "pathology_mask_paths": [],
        "pathology_segmentation_slide_image_paths": [],
        "pathology_segmentation_overlay_paths": [],
        "pathology_segmentation_metadata_paths": [],
        "radiology_mask_paths": [],
        "radiology_mask_manifest_paths": [],
        "pathology_tile_embedding_paths": [],
        "pathology_slide_embedding_paths": [],
        "radiology_embedding_paths": [],
        "radiology_png_dirs": [],
        "radiology_series_slice_counts": [],
        "radiology_download_paths": [],
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


def test_replace_source_slice_drops_stale_source_specific_columns() -> None:
    unified = pd.DataFrame([
        {**_row("tcga-old", "tcga"), "has_mutation_vhl": True},
        _row("other-1", "other"),
    ])
    source = pd.DataFrame([
        {**_row("tcga-new-1", "tcga"), "mutation_vhl": True},
    ])

    merged = replace_source_slice(unified, source, source_name="tcga")

    assert "mutation_vhl" in merged.columns
    assert "has_mutation_vhl" not in merged.columns
