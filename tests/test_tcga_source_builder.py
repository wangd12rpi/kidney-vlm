from __future__ import annotations

from pathlib import Path

import pandas as pd

from kidney_vlm.data.sources.tcga import assign_split, build_tcga_registry_rows


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
    assert bool(row1["has_pathology"]) is True
    assert bool(row1["has_radiology"]) is True
    assert row1["tumor_stage"] == "Stage II"
    assert row1["ajcc_pathologic_t"] == "T2"
    assert row1["year_of_birth"] == "1962"
    assert set(row1["tcia_series_uids"]) == {"1.2.3.5", "1.2.4.8"}
    assert set(row1["tcia_modalities"]) == {"CT", "MR"}
    assert set(row1["tcia_body_parts"]) == {"ABDOMEN"}
    assert set(row1["tcia_study_dates"]) == {"2017-04-20", "2017-04-24"}
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
    assert bool(row2["has_pathology"]) is False
    assert bool(row2["has_radiology"]) is True
    assert row2["vital_status"] == "Dead"
    assert row2["days_to_death"] == "55"
    assert set(row2["mutated_gene_symbols"]) == {"TP53"}
    assert bool(row2["has_mutation_tp53"]) is True
    assert bool(row2["task_survival_event"]) is True


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
