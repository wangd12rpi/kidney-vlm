from __future__ import annotations

import pandas as pd

from kidney_vlm.data.unified_registry import merge_case_level_rna_feature_paths


def _registry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "tcga-1",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "split": "train",
                "split_group_id": "tcga:TCGA-KIRC:TCGA-AA-0001",
                "genomics_rna_bulk_feature_path": "",
                "genomics_dna_methylation_feature_path": "data/features/dnam.pt",
                "pathology_wsi_paths": ["slide.svs"],
                "radiology_image_paths": ["scan.zip"],
            },
            {
                "sample_id": "tcga-2",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0002",
                "split": "val",
                "split_group_id": "tcga:TCGA-KIRC:TCGA-AA-0002",
                "genomics_rna_bulk_feature_path": "data/features/old.pt",
                "genomics_dna_methylation_feature_path": "data/features/dnam2.pt",
                "pathology_wsi_paths": ["slide2.svs"],
                "radiology_image_paths": ["scan2.zip"],
            },
            {
                "sample_id": "pmc-1",
                "source": "pmc_oa",
                "project_id": "PMC",
                "patient_id": "PMC-1",
                "split": "test",
                "split_group_id": "pmc:PMC:PMC-1",
                "genomics_rna_bulk_feature_path": "",
                "genomics_dna_methylation_feature_path": "",
                "pathology_wsi_paths": [],
                "radiology_image_paths": [],
            },
        ]
    )


def test_merge_case_level_rna_feature_paths_updates_only_rna_feature_column() -> None:
    registry_df = _registry_frame()
    assignments_df = pd.DataFrame(
        [
            {
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "genomics_rna_bulk_feature_path": "data/features/features_bulkformer_rna/TCGA-KIRC/tcga1.pt",
            },
            {
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0002",
                "genomics_rna_bulk_feature_path": "data/features/features_bulkformer_rna/TCGA-KIRC/tcga2.pt",
            },
        ]
    )

    merged_df, report = merge_case_level_rna_feature_paths(registry_df, assignments_df)

    assert report.matched_registry_rows == 2
    assert report.updated_feature_rows == 2
    assert merged_df.loc[0, "genomics_rna_bulk_feature_path"].endswith("tcga1.pt")
    assert merged_df.loc[1, "genomics_rna_bulk_feature_path"].endswith("tcga2.pt")

    protected_columns = [column for column in registry_df.columns if column != "genomics_rna_bulk_feature_path"]
    assert merged_df[protected_columns].equals(registry_df[protected_columns])


def test_merge_case_level_rna_feature_paths_can_preserve_existing_paths() -> None:
    registry_df = _registry_frame()
    assignments_df = pd.DataFrame(
        [
            {
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0002",
                "genomics_rna_bulk_feature_path": "data/features/features_bulkformer_rna/TCGA-KIRC/new.pt",
            }
        ]
    )

    merged_df, report = merge_case_level_rna_feature_paths(
        registry_df,
        assignments_df,
        overwrite_existing=False,
    )

    assert report.matched_registry_rows == 1
    assert report.skipped_existing_feature_rows == 1
    assert report.updated_feature_rows == 0
    assert merged_df.loc[1, "genomics_rna_bulk_feature_path"] == "data/features/old.pt"
