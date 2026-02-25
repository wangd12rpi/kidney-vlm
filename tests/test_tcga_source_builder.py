from __future__ import annotations

from pathlib import Path

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
                    "ajcc_pathologic_stage": "Stage II",
                    "vital_status": "Alive",
                    "days_to_last_follow_up": 100,
                    "days_to_death": None,
                }
            ],
            "demographic": {"gender": "female", "race": "white", "ethnicity": "not hispanic or latino"},
        },
        {
            "case_id": "case-2",
            "submitter_id": "TCGA-BB-0002",
            "project": {"project_id": "TCGA-KIRP"},
            "primary_site": "Kidney",
            "disease_type": "B",
            "diagnoses": [{"primary_diagnosis": "Papillary renal cell carcinoma"}],
            "demographic": {"gender": "male"},
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
            {"collection": "TCGA-KIRC", "patient_id": "TCGA-AA-0001", "study_instance_uid": "1.2.3"},
            {"collection": "TCGA-KIRC", "patient_id": "TCGA-AA-0001", "study_instance_uid": "1.2.4"},
        ],
        "TCGA-BB-0002": [
            {"collection": "TCGA-KIRP", "patient_id": "TCGA-BB-0002", "study_instance_uid": "2.3.4"},
        ],
    }

    frame = build_tcga_registry_rows(
        cases=cases,
        pathology_files=pathology_files,
        tcia_studies_by_patient=tcia_studies_by_patient,
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

    row2 = frame[frame["patient_id"] == "TCGA-BB-0002"].iloc[0]
    assert len(row2["pathology_wsi_paths"]) == 0
    assert len(row2["radiology_image_paths"]) == 1
    assert bool(row2["has_pathology"]) is False
    assert bool(row2["has_radiology"]) is True


def test_assign_split_is_deterministic() -> None:
    split_1 = assign_split("TCGA-AA-0001", {"train": 0.9, "test": 0.1})
    split_2 = assign_split("TCGA-AA-0001", {"train": 0.9, "test": 0.1})
    assert split_1 == split_2
    assert split_1 in {"train", "test"}
