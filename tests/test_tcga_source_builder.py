from __future__ import annotations

from pathlib import Path

import pandas as pd

from kidney_vlm.data.sources.tcga import (
    APIQueryError,
    TCIAClient,
    assign_split,
    build_tcga_registry_rows,
    normalize_tcia_modality,
    select_tcia_radiology_cohort,
)


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

    frame = build_tcga_registry_rows(
        cases=cases,
        pathology_files=pathology_files,
        tcia_studies_by_patient=tcia_studies_by_patient,
        tcia_series_by_patient=tcia_series_by_patient,
        ssm_mutations_by_case_id=ssm_mutations_by_case_id,
        ssm_mutations_by_patient_id=ssm_mutations_by_patient_id,
        mutation_gene_panel=["VHL", "PBRM1", "TP53"],
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
    )

    assert len(frame) == 2

    row1 = frame[frame["patient_id"] == "TCGA-AA-0001"].iloc[0]
    assert len(row1["pathology_wsi_paths"]) == 2
    assert len(row1["radiology_image_paths"]) == 2
    assert all(not str(path).startswith("/") for path in row1["pathology_wsi_paths"])
    assert all(not str(path).startswith("/") for path in row1["radiology_image_paths"])
    assert bool(row1["has_pathology"]) is True
    assert bool(row1["has_radiology"]) is True
    assert row1["tumor_stage"] == "Stage II"
    assert row1["ajcc_pathologic_t"] == "T2"
    assert row1["year_of_birth"] == "1962"
    assert set(row1["tcia_series_uids"]) == {"1.2.3.5", "1.2.4.8"}
    assert set(row1["tcia_modalities"]) == {"CT", "MR"}
    assert set(row1["tcia_body_parts"]) == {"ABDOMEN"}
    assert set(row1["tcia_study_dates"]) == {"2017-04-20", "2017-04-24"}
    assert row1["radiology_image_modalities"] == ["CT", "MR"]
    assert set(row1["mutated_gene_symbols"]) == {"PBRM1", "VHL"}
    assert set(row1["kidney_driver_gene_mutations"]) == {"PBRM1", "VHL"}
    assert row1["mutation_event_count"] == 2
    assert bool(row1["has_mutation_vhl"]) is True
    assert bool(row1["has_mutation_tp53"]) is False
    assert isinstance(row1["pathology_tile_embedding_paths"], list)
    assert isinstance(row1["pathology_slide_embedding_paths"], list)
    assert isinstance(row1["radiology_embedding_paths"], list)

    row2 = frame[frame["patient_id"] == "TCGA-BB-0002"].iloc[0]
    assert len(row2["pathology_wsi_paths"]) == 0
    assert len(row2["radiology_image_paths"]) == 1
    assert row2["radiology_image_modalities"] == ["MR"]
    assert all(not str(path).startswith("/") for path in row2["radiology_image_paths"])
    assert bool(row2["has_pathology"]) is False
    assert bool(row2["has_radiology"]) is True
    assert row2["vital_status"] == "Dead"
    assert row2["days_to_death"] == "55"
    assert set(row2["mutated_gene_symbols"]) == {"TP53"}
    assert bool(row2["has_mutation_tp53"]) is True
    assert bool(row2["task_survival_event"]) is True


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
    assert row["radiology_image_paths"] == ["raw/tcga/radiology/TCGA-KIRC/TCGA-FF-0006/7.8.9"]
    assert row["radiology_image_modalities"] == ["CT|MR"]
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
    assert bool(row["has_radiology"]) is False


def test_build_tcga_registry_rows_includes_genomics_text() -> None:
    frame = build_tcga_registry_rows(
        cases=[
            {
                "case_id": "case-genomics",
                "submitter_id": "TCGA-ZZ-9999",
                "project": {"project_id": "TCGA-KIRC"},
                "diagnoses": [{}],
                "demographic": {},
            }
        ],
        pathology_files=[],
        tcia_studies_by_patient={},
        genomics_by_patient_id={
            "TCGA-ZZ-9999": {
                "patient_id": "TCGA-ZZ-9999",
                "cancer_code": "KIRC",
                "genomics_text": "[GENOMICS]\ncancer_type: KIRC\n[/GENOMICS]",
            }
        },
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 1.0},
    )

    row = frame.iloc[0]
    assert row["genomics_text"] == "[GENOMICS]\ncancer_type: KIRC\n[/GENOMICS]"


def test_build_tcga_registry_rows_includes_genomics_download_paths() -> None:
    frame = build_tcga_registry_rows(
        cases=[
            {
                "case_id": "case-genomics-files",
                "submitter_id": "TCGA-YY-1111",
                "project": {"project_id": "TCGA-KIRC"},
                "diagnoses": [{}],
                "demographic": {},
            }
        ],
        pathology_files=[],
        tcia_studies_by_patient={},
        downloaded_genomics_by_patient_id={
            "TCGA-YY-1111": [
                {
                    "payload_key": "genomics_text",
                    "file_id": "",
                    "file_name": "TCGA-YY-1111.txt",
                    "data_type": "Patient Genomics Text",
                    "local_path": "/tmp/staging/tcga_genomics_text/TCGA-YY-1111.txt",
                },
            ]
        },
        genomics_download_manifest_path="data/staging/tcga_genomics_downloads.jsonl",
        raw_root=Path("/tmp/raw"),
        source_name="tcga",
        split_ratios={"train": 1.0},
    )

    row = frame.iloc[0]
    assert row["genomics_download_manifest_path"] == "data/staging/tcga_genomics_downloads.jsonl"
    assert row["genomics_file_ids"] == []
    assert row["genomics_payload_keys"] == ["genomics_text"]
    assert row["genomics_data_types"] == ["Patient Genomics Text"]
    assert row["genomics_file_paths"] == [
        "staging/tcga_genomics_text/TCGA-YY-1111.txt",
    ]


def test_assign_split_is_deterministic() -> None:
    split_1 = assign_split("TCGA-AA-0001", {"train": 0.9, "test": 0.1})
    split_2 = assign_split("TCGA-AA-0001", {"train": 0.9, "test": 0.1})
    assert split_1 == split_2
    assert split_1 in {"train", "test"}


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
    assert pd.isna(row["has_mutation_vhl"])


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


def test_tcia_fetch_collection_values_parses_records(monkeypatch) -> None:
    client = TCIAClient()

    def fake_get_json(endpoint: str, params: dict[str, str]):
        assert endpoint == "getCollectionValues"
        assert params["format"] == "json"
        return [
            {"Collection": "TCGA-KIRC"},
            {"Collection": "TCGA-BRCA"},
            {"Collection": "TCGA-KIRC"},
            {},
        ]

    monkeypatch.setattr(client, "_get_json", fake_get_json)

    assert client.fetch_collection_values() == ["TCGA-BRCA", "TCGA-KIRC"]


def test_select_tcia_radiology_cohort_filters_to_qualifying_modalities() -> None:
    studies_by_patient = {
        "TCGA-AA-0001": [
            {
                "collection": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "study_instance_uid": "1.2.3",
                "modalities_in_study": ["CT", "SEG"],
            }
        ],
        "TCGA-BB-0002": [
            {
                "collection": "TCGA-BRCA",
                "patient_id": "TCGA-BB-0002",
                "study_instance_uid": "2.3.4",
                "modalities_in_study": ["MG"],
            }
        ],
        "TCGA-CC-0003": [
            {
                "collection": "TCGA-LUAD",
                "patient_id": "TCGA-CC-0003",
                "study_instance_uid": "3.4.5",
                "modalities_in_study": ["SR"],
            }
        ],
    }
    series_by_patient = {
        "TCGA-AA-0001": [
            {
                "collection": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "study_instance_uid": "1.2.3",
                "series_instance_uid": "1.2.3.1",
                "modality": "CT",
                "body_part_examined": "ABDOMEN",
                "series_description": "abdomen",
            },
            {
                "collection": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "study_instance_uid": "1.2.3",
                "series_instance_uid": "1.2.3.2",
                "modality": "SEG",
                "body_part_examined": "ABDOMEN",
                "series_description": "segmentation",
            },
        ],
        "TCGA-CC-0003": [
            {
                "collection": "TCGA-LUAD",
                "patient_id": "TCGA-CC-0003",
                "study_instance_uid": "3.4.5",
                "series_instance_uid": "3.4.5.1",
                "modality": "SR",
                "body_part_examined": "CHEST",
                "series_description": "structured report",
            }
        ],
    }

    eligible_patients, filtered_studies, filtered_series = select_tcia_radiology_cohort(
        studies_by_patient,
        series_by_patient=series_by_patient,
        qualifying_modalities=["CT", "mammography"],
    )

    assert eligible_patients == {"TCGA-AA-0001", "TCGA-BB-0002"}
    assert set(filtered_studies) == {"TCGA-AA-0001", "TCGA-BB-0002"}
    assert set(filtered_series) == {"TCGA-AA-0001"}
    assert filtered_series["TCGA-AA-0001"][0]["modality"] == "CT"
    assert filtered_studies["TCGA-BB-0002"][0]["modalities_in_study"] == ["MG"]


def test_normalize_tcia_modality_supports_common_aliases() -> None:
    assert normalize_tcia_modality("MRI") == "MR"
    assert normalize_tcia_modality("mammography") == "MG"
    assert normalize_tcia_modality("PET") == "PT"
