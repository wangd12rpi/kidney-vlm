from __future__ import annotations

from pathlib import Path

import pandas as pd

from kidney_vlm.data.sources.tcga import APIQueryError, TCIAClient, assign_split, build_tcga_registry_rows


def test_build_tcga_registry_rows_multimodal_lists() -> None:
    cases = [
        {
            "case_id": "case-1",
            "submitter_id": "TCGA-AA-0001",
            "project": {"project_id": "TCGA-KIRC"},
            "primary_site": "Kidney",
            "disease_type": "A",
            "diagnoses": [
                {
                    "primary_diagnosis": "Renal cell carcinoma",
                    "tumor_grade": "G2",
                    "tumor_stage": "Stage II",
                    "ajcc_pathologic_stage": "Stage II",
                    "ajcc_pathologic_t": "T2",
                    "ajcc_pathologic_n": "N0",
                    "ajcc_pathologic_m": "M0",
                    "age_at_diagnosis": 22000,
                    "morphology": "8310/3",
                    "last_known_disease_status": "disease free",
                    "days_to_last_known_disease_status": 120,
                    "days_to_recurrence": "",
                    "vital_status": "Alive",
                    "days_to_last_follow_up": 100,
                    "days_to_death": None,
                }
            ],
            "demographic": {
                "gender": "female",
                "race": "white",
                "ethnicity": "not hispanic or latino",
                "year_of_birth": 1962,
            },
        },
        {
            "case_id": "case-2",
            "submitter_id": "TCGA-BB-0002",
            "project": {"project_id": "TCGA-KIRP"},
            "primary_site": "Kidney",
            "disease_type": "B",
            "diagnoses": [{"primary_diagnosis": "Papillary renal cell carcinoma"}],
            "demographic": {"gender": "male", "vital_status": "Dead", "days_to_death": 55},
        },
    ]

    pathology_files = [
        {
            "file_id": "f1",
            "file_name": "slide1.svs",
            "cases": [{"case_id": "case-1", "submitter_id": "TCGA-AA-0001", "project": {"project_id": "TCGA-KIRC"}}],
        },
        {
            "file_id": "f2",
            "file_name": "slide2.svs",
            "cases": [{"case_id": "case-1", "submitter_id": "TCGA-AA-0001", "project": {"project_id": "TCGA-KIRC"}}],
        },
    ]
    rna_bulk_files = [
        {
            "file_id": "rna1",
            "file_name": "TCGA-AA-0001_rna_counts.tsv",
            "analysis": {"workflow_type": "STAR - Counts"},
            "cases": [
                {
                    "case_id": "case-1",
                    "submitter_id": "TCGA-AA-0001",
                    "project": {"project_id": "TCGA-KIRC"},
                    "samples": [
                        {
                            "submitter_id": "TCGA-AA-0001-01A",
                            "sample_type": "Primary Tumor",
                        }
                    ],
                }
            ],
        },
        {
            "file_id": "rna2",
            "file_name": "TCGA-BB-0002_rna_counts.tsv",
            "analysis": {"workflow_type": "HTSeq - Counts"},
            "cases": [
                {
                    "case_id": "case-2",
                    "submitter_id": "TCGA-BB-0002",
                    "project": {"project_id": "TCGA-KIRP"},
                    "samples": [
                        {
                            "submitter_id": "TCGA-BB-0002-11A",
                            "sample_type": "Solid Tissue Normal",
                        }
                    ],
                }
            ],
        },
    ]

    tcia_studies_by_patient = {
        "TCGA-AA-0001": [
            {
                "collection": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "study_instance_uid": "1.2.3",
                "study_date": "2017-04-20",
                "study_description": "baseline",
                "modalities_in_study": ["CT"],
            },
            {
                "collection": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "study_instance_uid": "1.2.4",
                "study_date": "2017-04-24",
                "study_description": "followup",
                "modalities_in_study": ["CT", "MR"],
            },
        ],
        "TCGA-BB-0002": [
            {
                "collection": "TCGA-KIRP",
                "patient_id": "TCGA-BB-0002",
                "study_instance_uid": "2.3.4",
                "modalities_in_study": ["MR"],
            },
        ],
    }

    tcia_series_by_patient = {
        "TCGA-AA-0001": [
            {
                "collection": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "study_instance_uid": "1.2.3",
                "series_instance_uid": "1.2.3.5",
                "modality": "CT",
                "body_part_examined": "ABDOMEN",
                "series_description": "abdomen venous",
            },
            {
                "collection": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "study_instance_uid": "1.2.4",
                "series_instance_uid": "1.2.4.8",
                "modality": "MR",
                "body_part_examined": "ABDOMEN",
                "series_description": "t2 axial",
            },
        ],
    }

    ssm_mutations_by_case_id = {
        "case-1": [
            {
                "ssm_id": "ssm-1",
                "mutation_type": "Missense_Mutation",
                "gene_symbols": ["VHL"],
                "consequence_terms": ["missense_variant"],
            },
            {
                "ssm_id": "ssm-2",
                "mutation_type": "Frame_Shift_Del",
                "gene_symbols": ["PBRM1"],
                "consequence_terms": ["frameshift_variant"],
            },
        ]
    }
    ssm_mutations_by_patient_id = {
        "TCGA-BB-0002": [
            {
                "ssm_id": "ssm-3",
                "mutation_type": "Nonsense_Mutation",
                "gene_symbols": ["TP53"],
                "consequence_terms": ["stop_gained"],
            }
        ]
    }
    rna_bulk_metadata_by_patient_id = {
        "TCGA-AA-0001": {
            "genomics_rna_bulk_molecular_subtype": "ccA",
            "genomics_rna_bulk_subtype_mrna": "mRNA-ccA",
            "genomics_dna_methylation_subtype": "CIMP-high",
            "genomics_integrative_subtype": "iCluster-2",
            "genomics_msi_status": "MSI",
            "genomics_rna_bulk_leukocyte_fraction": "0.143",
            "genomics_rna_bulk_tumor_purity": "0.82",
            "genomics_aneuploidy_score": "14",
            "genomics_hrd_score": "27",
            "genomics_rna_bulk_top_immune_cell_types": ["T cells CD8", "Macrophages M2"],
            "genomics_rna_bulk_top_immune_cell_fractions": ["0.331", "0.214"],
        }
    }

    frame = build_tcga_registry_rows(
        cases=cases,
        pathology_files=pathology_files,
        rna_bulk_files=rna_bulk_files,
        rna_bulk_metadata_by_patient_id=rna_bulk_metadata_by_patient_id,
        tcia_studies_by_patient=tcia_studies_by_patient,
        tcia_series_by_patient=tcia_series_by_patient,
        ssm_mutations_by_case_id=ssm_mutations_by_case_id,
        ssm_mutations_by_patient_id=ssm_mutations_by_patient_id,
        mutation_gene_panel=["VHL", "PBRM1", "TP53"],
        mutation_panel_version="test_panel_v1",
        project_driver_gene_panel_by_project={
            "TCGA-KIRC": ["VHL", "PBRM1"],
            "TCGA-KIRP": ["TP53"],
        },
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
    )

    assert len(frame) == 2

    row1 = frame[frame["patient_id"] == "TCGA-AA-0001"].iloc[0]
    assert len(row1["pathology_wsi_paths"]) == 2
    assert row1["radiology_image_paths"] == []
    assert all(not str(path).startswith("/") for path in row1["pathology_wsi_paths"])
    assert bool(row1["has_pathology"]) is True
    assert bool(row1["has_radiology"]) is True
    assert row1["tumor_stage"] == "Stage II"
    assert row1["ajcc_pathologic_t"] == "T2"
    assert row1["year_of_birth"] == "1962"
    assert set(row1["tcia_series_uids"]) == {"1.2.3.5", "1.2.4.8"}
    assert set(row1["tcia_modalities"]) == {"CT", "MR"}
    assert set(row1["tcia_body_parts"]) == {"ABDOMEN"}
    assert set(row1["tcia_study_dates"]) == {"2017-04-20", "2017-04-24"}
    assert row1["radiology_image_modalities"] == []
    assert row1["radiology_report_download_paths"] == []
    assert row1["radiology_report_uri_paths"] == []
    assert row1["radiology_report_series_descriptions"] == []
    assert set(row1["mutated_gene_symbols"]) == {"PBRM1", "VHL"}
    assert set(row1["project_driver_gene_mutations"]) == {"PBRM1", "VHL"}
    assert row1["split_group_id"] == "tcga:TCGA-KIRC:TCGA-AA-0001"
    assert row1["split_scheme_version"] == "tcga_project_patient_hash_v1"
    assert bool(row1["mutation_query_succeeded"]) is True
    assert row1["mutation_panel_version"] == "test_panel_v1"
    assert bool(row1["mutation_panel_observed"]) is True
    assert row1["mutation_event_count"] == 2
    assert bool(row1["mutation_vhl"]) is True
    assert bool(row1["mutation_tp53"]) is False
    assert isinstance(row1["pathology_tile_embedding_paths"], list)
    assert isinstance(row1["pathology_slide_embedding_paths"], list)
    assert isinstance(row1["radiology_embedding_paths"], list)
    assert row1["genomics_rna_bulk_paths"] == ["raw/tcga/rna_bulk/TCGA-KIRC/TCGA-AA-0001/TCGA-AA-0001_rna_counts.tsv"]
    assert row1["genomics_rna_bulk_file_ids"] == ["rna1"]
    assert row1["genomics_rna_bulk_file_names"] == ["TCGA-AA-0001_rna_counts.tsv"]
    assert row1["genomics_rna_bulk_sample_types"] == ["Primary Tumor"]
    assert row1["genomics_rna_bulk_workflow_types"] == ["STAR - Counts"]
    assert row1["genomics_rna_bulk_molecular_subtype"] == "ccA"
    assert row1["genomics_rna_bulk_subtype_mrna"] == "mRNA-ccA"
    assert row1["genomics_dna_methylation_subtype"] == "CIMP-high"
    assert row1["genomics_integrative_subtype"] == "iCluster-2"
    assert row1["genomics_msi_status"] == "MSI"
    assert row1["genomics_rna_bulk_leukocyte_fraction"] == "0.143"
    assert row1["genomics_rna_bulk_tumor_purity"] == "0.82"
    assert row1["genomics_aneuploidy_score"] == "14"
    assert row1["genomics_hrd_score"] == "27"
    assert row1["genomics_rna_bulk_top_immune_cell_types"] == ["T cells CD8", "Macrophages M2"]
    assert row1["genomics_rna_bulk_top_immune_cell_fractions"] == ["0.331", "0.214"]

    row2 = frame[frame["patient_id"] == "TCGA-BB-0002"].iloc[0]
    assert len(row2["pathology_wsi_paths"]) == 0
    assert row2["radiology_image_paths"] == []
    assert row2["radiology_report_download_paths"] == []
    assert row2["radiology_report_uri_paths"] == []
    assert row2["radiology_report_series_descriptions"] == []
    assert bool(row2["has_pathology"]) is False
    assert bool(row2["has_radiology"]) is True
    assert row2["vital_status"] == "Dead"
    assert row2["days_to_death"] == "55"
    assert set(row2["mutated_gene_symbols"]) == {"TP53"}
    assert bool(row2["mutation_tp53"]) is True
    assert bool(row2["task_survival_event"]) is True
    assert row2["genomics_rna_bulk_paths"] == ["raw/tcga/rna_bulk/TCGA-KIRP/TCGA-BB-0002/TCGA-BB-0002_rna_counts.tsv"]
    assert row2["genomics_rna_bulk_sample_types"] == ["Solid Tissue Normal"]
    assert row2["genomics_rna_bulk_workflow_types"] == ["HTSeq - Counts"]
    assert row2["genomics_rna_bulk_molecular_subtype"] == ""
    assert row2["genomics_rna_bulk_subtype_mrna"] == ""
    assert row2["genomics_dna_methylation_subtype"] == ""
    assert row2["genomics_integrative_subtype"] == ""
    assert row2["genomics_msi_status"] == ""
    assert row2["genomics_aneuploidy_score"] == ""
    assert row2["genomics_hrd_score"] == ""
    assert row2["genomics_rna_bulk_top_immune_cell_types"] == []
    assert row2["genomics_rna_bulk_top_immune_cell_fractions"] == []


def test_build_tcga_registry_rows_keeps_all_tcga_slide_kinds() -> None:
    frame = build_tcga_registry_rows(
        cases=[
            {
                "case_id": "case-slide-kinds",
                "submitter_id": "TCGA-EE-0005",
                "project": {"project_id": "TCGA-KIRC"},
                "diagnoses": [{}],
                "demographic": {},
            }
        ],
        pathology_files=[
            {
                "file_id": "dx",
                "file_name": "TCGA-EE-0005-01Z-00-DX1.svs",
                "cases": [{"case_id": "case-slide-kinds", "submitter_id": "TCGA-EE-0005", "project": {"project_id": "TCGA-KIRC"}}],
            },
            {
                "file_id": "ts",
                "file_name": "TCGA-EE-0005-01Z-00-TS1.svs",
                "cases": [{"case_id": "case-slide-kinds", "submitter_id": "TCGA-EE-0005", "project": {"project_id": "TCGA-KIRC"}}],
            },
            {
                "file_id": "bs",
                "file_name": "TCGA-EE-0005-01Z-00-BS1.svs",
                "cases": [{"case_id": "case-slide-kinds", "submitter_id": "TCGA-EE-0005", "project": {"project_id": "TCGA-KIRC"}}],
            },
        ],
        tcia_studies_by_patient={},
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 1.0},
    )

    row = frame.iloc[0]
    slide_names = {Path(path).name for path in row["pathology_wsi_paths"]}
    assert slide_names == {
        "TCGA-EE-0005-01Z-00-DX1.svs",
        "TCGA-EE-0005-01Z-00-TS1.svs",
        "TCGA-EE-0005-01Z-00-BS1.svs",
    }


def test_build_tcga_registry_rows_tracks_multimodality_study_fallback_paths() -> None:
    frame = build_tcga_registry_rows(
        cases=[
            {
                "case_id": "case-rad-fallback",
                "submitter_id": "TCGA-FF-0006",
                "project": {"project_id": "TCGA-KIRC"},
                "diagnoses": [{}],
                "demographic": {},
            }
        ],
        pathology_files=[],
        tcia_studies_by_patient={
            "TCGA-FF-0006": [
                {
                    "collection": "TCGA-KIRC",
                    "patient_id": "TCGA-FF-0006",
                    "study_instance_uid": "7.8.9",
                    "modalities_in_study": ["CT", "MRI"],
                }
            ]
        },
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 1.0},
    )

    row = frame.iloc[0]
    assert row["radiology_image_paths"] == []
    assert row["radiology_image_modalities"] == []
    assert row["radiology_report_download_paths"] == []
    assert row["radiology_report_uri_paths"] == []
    assert row["radiology_report_series_descriptions"] == []
    assert row["tcia_modalities"] == ["CT", "MR"]


def test_build_tcga_registry_rows_prefers_qc_passed_downloaded_series_dirs() -> None:
    frame = build_tcga_registry_rows(
        cases=[
            {
                "case_id": "case-rad-download",
                "submitter_id": "TCGA-GG-0007",
                "project": {"project_id": "TCGA-KIRC"},
                "diagnoses": [{}],
                "demographic": {},
            }
        ],
        pathology_files=[],
        tcia_studies_by_patient={
            "TCGA-GG-0007": [
                {
                    "collection": "TCGA-KIRC",
                    "patient_id": "TCGA-GG-0007",
                    "study_instance_uid": "9.8.7",
                    "modalities_in_study": ["CT", "MR"],
                }
            ]
        },
        tcia_series_by_patient={
            "TCGA-GG-0007": [
                {
                    "collection": "TCGA-KIRC",
                    "patient_id": "TCGA-GG-0007",
                    "study_instance_uid": "9.8.7",
                    "series_instance_uid": "9.8.7.1",
                    "modality": "CT",
                },
                {
                    "collection": "TCGA-KIRC",
                    "patient_id": "TCGA-GG-0007",
                    "study_instance_uid": "9.8.7",
                    "series_instance_uid": "9.8.7.2",
                    "modality": "MR",
                },
            ]
        },
        downloaded_tcia_series_by_patient={
            "TCGA-GG-0007": [
                {
                    "collection": "TCGA-KIRC",
                    "patient_id": "TCGA-GG-0007",
                    "study_instance_uid": "9.8.7",
                    "series_instance_uid": "9.8.7.1",
                    "modality": "CT",
                    "local_path": "/tmp/raw/tcga/radiology/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.1.zip",
                    "accepted": True,
                    "extracted_path": "/tmp/raw/tcga/radiology/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.1",
                    "png_dir": "/tmp/data/processes/radiology/chunk1/pngs/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.1",
                    "embedding_ref": "data/processes/radiology/chunk1/features_medsiglip448/chunk1.h5::series=data/processes/radiology/chunk1/pngs/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.1",
                    "slice_count": 12,
                },
                {
                    "collection": "TCGA-KIRC",
                    "patient_id": "TCGA-GG-0007",
                    "study_instance_uid": "9.8.7",
                    "series_instance_uid": "9.8.7.2",
                    "modality": "MR",
                    "local_path": "/tmp/raw/tcga/radiology/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.2.zip",
                    "accepted": False,
                    "extracted_path": "",
                    "reject_reason": "series_is_localizer_or_scout",
                },
            ]
        },
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 1.0},
        downloaded_radiology_only=True,
    )

    row = frame.iloc[0]
    assert row["radiology_image_paths"] == [
        "data/processes/radiology/chunk1/pngs/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.1"
    ]
    assert row["radiology_image_modalities"] == ["CT"]
    assert row["radiology_embedding_paths"] == [
        "data/processes/radiology/chunk1/features_medsiglip448/chunk1.h5::series=data/processes/radiology/chunk1/pngs/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.1"
    ]
    assert row["radiology_download_paths"] == [
        "raw/tcga/radiology/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.1.zip",
        "raw/tcga/radiology/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.2.zip",
    ]
    assert row["radiology_report_download_paths"] == []
    assert row["radiology_report_uri_paths"] == []
    assert row["radiology_report_series_descriptions"] == []
    assert row["radiology_png_dirs"] == [
        "data/processes/radiology/chunk1/pngs/TCGA-KIRC/TCGA-GG-0007/9.8.7/9.8.7.1"
    ]
    assert row["radiology_series_slice_counts"] == [12]
    assert bool(row["has_radiology"]) is True


def test_build_tcga_registry_rows_downloaded_radiology_only_does_not_fall_back_after_qc_rejection() -> None:
    frame = build_tcga_registry_rows(
        cases=[
            {
                "case_id": "case-rad-rejected",
                "submitter_id": "TCGA-HH-0008",
                "project": {"project_id": "TCGA-KIRC"},
                "diagnoses": [{}],
                "demographic": {},
            }
        ],
        pathology_files=[],
        tcia_studies_by_patient={
            "TCGA-HH-0008": [
                {
                    "collection": "TCGA-KIRC",
                    "patient_id": "TCGA-HH-0008",
                    "study_instance_uid": "4.5.6",
                    "modalities_in_study": ["CT"],
                }
            ]
        },
        tcia_series_by_patient={
            "TCGA-HH-0008": [
                {
                    "collection": "TCGA-KIRC",
                    "patient_id": "TCGA-HH-0008",
                    "study_instance_uid": "4.5.6",
                    "series_instance_uid": "4.5.6.1",
                    "modality": "CT",
                }
            ]
        },
        downloaded_tcia_series_by_patient={
            "TCGA-HH-0008": [
                {
                    "collection": "TCGA-KIRC",
                    "patient_id": "TCGA-HH-0008",
                    "study_instance_uid": "4.5.6",
                    "series_instance_uid": "4.5.6.1",
                    "modality": "CT",
                    "local_path": "/tmp/raw/tcga/radiology/TCGA-KIRC/TCGA-HH-0008/4.5.6/4.5.6.1.zip",
                    "accepted": False,
                    "extracted_path": "",
                    "reject_reason": "too_few_usable_images",
                }
            ]
        },
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 1.0},
        downloaded_radiology_only=True,
    )

    row = frame.iloc[0]
    assert row["radiology_image_paths"] == []
    assert row["radiology_image_modalities"] == []
    assert row["radiology_download_paths"] == [
        "raw/tcga/radiology/TCGA-KIRC/TCGA-HH-0008/4.5.6/4.5.6.1.zip"
    ]
    assert row["radiology_report_download_paths"] == []
    assert row["radiology_report_uri_paths"] == []
    assert row["radiology_report_series_descriptions"] == []
    assert bool(row["has_radiology"]) is False


def test_build_tcga_registry_rows_tracks_sr_series_as_radiology_reports() -> None:
    frame = build_tcga_registry_rows(
        cases=[
            {
                "case_id": "case-rad-report",
                "submitter_id": "TCGA-II-0009",
                "project": {"project_id": "TCGA-OV"},
                "diagnoses": [{}],
                "demographic": {},
            }
        ],
        pathology_files=[],
        tcia_studies_by_patient={
            "TCGA-II-0009": [
                {
                    "collection": "TCGA-OV",
                    "patient_id": "TCGA-II-0009",
                    "study_instance_uid": "5.6.7",
                    "modalities_in_study": ["CT", "SR"],
                }
            ]
        },
        tcia_series_by_patient={
            "TCGA-II-0009": [
                {
                    "collection": "TCGA-OV",
                    "patient_id": "TCGA-II-0009",
                    "study_instance_uid": "5.6.7",
                    "series_instance_uid": "5.6.7.1",
                    "modality": "CT",
                    "series_description": "AXIAL CHEST",
                },
                {
                    "collection": "TCGA-OV",
                    "patient_id": "TCGA-II-0009",
                    "study_instance_uid": "5.6.7",
                    "series_instance_uid": "5.6.7.2",
                    "modality": "SR",
                    "series_description": "Imaging Report",
                },
            ]
        },
        downloaded_tcia_series_by_patient={
            "TCGA-II-0009": [
                {
                    "collection": "TCGA-OV",
                    "patient_id": "TCGA-II-0009",
                    "study_instance_uid": "5.6.7",
                    "series_instance_uid": "5.6.7.2",
                    "modality": "SR",
                    "local_path": "/tmp/raw/tcga/radiology/TCGA-OV/TCGA-II-0009/5.6.7/5.6.7.2.zip",
                    "accepted": False,
                    "extracted_path": "",
                }
            ]
        },
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 1.0},
    )

    row = frame.iloc[0]
    assert row["radiology_report_download_paths"] == [
        "raw/tcga/radiology/TCGA-OV/TCGA-II-0009/5.6.7/5.6.7.2.zip"
    ]
    assert row["radiology_report_uri_paths"] == [
        "tcia://TCGA-OV/TCGA-II-0009/5.6.7/5.6.7.2"
    ]
    assert row["radiology_report_series_descriptions"] == ["Imaging Report"]
    assert set(row["tcia_modalities"]) == {"CT", "SR"}
    assert bool(row["has_radiology"]) is True


def test_assign_split_is_deterministic() -> None:
    split_1 = assign_split("tcga:TCGA-KIRC:TCGA-AA-0001", {"train": 0.85, "val": 0.05, "test": 0.1})
    split_2 = assign_split("tcga:TCGA-KIRC:TCGA-AA-0001", {"train": 0.85, "val": 0.05, "test": 0.1})
    assert split_1 == split_2
    assert split_1 in {"train", "val", "test"}


def test_task_survival_days_is_numeric_or_null() -> None:
    frame = build_tcga_registry_rows(
        cases=[
            {
                "case_id": "case-3",
                "submitter_id": "TCGA-CC-0003",
                "project": {"project_id": "TCGA-KICH"},
                "diagnoses": [{}],
                "demographic": {},
            }
        ],
        pathology_files=[],
        tcia_studies_by_patient={},
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 1.0},
    )
    row = frame.iloc[0]
    assert pd.isna(row["task_survival_days"])


def test_mutation_flags_are_null_when_mutation_query_unavailable() -> None:
    frame = build_tcga_registry_rows(
        cases=[
            {
                "case_id": "case-4",
                "submitter_id": "TCGA-DD-0004",
                "project": {"project_id": "TCGA-KIRC"},
                "diagnoses": [{}],
                "demographic": {},
            }
        ],
        pathology_files=[],
        tcia_studies_by_patient={},
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 1.0},
        mutation_query_succeeded=False,
        mutation_gene_panel=["VHL"],
    )
    row = frame.iloc[0]
    assert pd.isna(row["mutation_vhl"])
    assert pd.isna(row["mutation_panel_observed"])
    assert bool(row["mutation_query_succeeded"]) is False


def test_tcia_fetch_studies_by_patient_skips_bad_collection(monkeypatch) -> None:
    client = TCIAClient()

    def fake_fetch_patient_studies(collection: str, max_studies: int | None = None):
        if collection == "TCGA-BAD":
            raise APIQueryError("non-json response")
        return [
            {
                "PatientID": "TCGA-AA-0001",
                "StudyInstanceUID": "1.2.3",
                "StudyDate": "2020-01-01",
            }
        ]

    monkeypatch.setattr(client, "fetch_patient_studies", fake_fetch_patient_studies)

    studies_by_patient = client.fetch_studies_by_patient(
        collections=["TCGA-BAD", "TCGA-KIRC"],
        max_studies_per_collection=None,
    )

    assert set(studies_by_patient) == {"TCGA-AA-0001"}
    assert studies_by_patient["TCGA-AA-0001"][0]["collection"] == "TCGA-KIRC"


def test_tcia_empty_body_is_treated_as_empty_result(monkeypatch) -> None:
    client = TCIAClient()

    class _FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self):
            raise ValueError("empty body")

    monkeypatch.setattr("kidney_vlm.data.sources.tcga.requests.get", lambda *args, **kwargs: _FakeResponse())

    payload = client._get_json("getPatientStudy", {"Collection": "TCGA-ACC", "format": "json"})

    assert payload == []
