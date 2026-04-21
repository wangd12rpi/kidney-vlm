from __future__ import annotations

from pathlib import Path

from kidney_vlm.data.sources.cptac import (
    build_cptac_registry_rows,
    cptac_file_filter,
    resolve_existing_cptac_files,
    stable_file_output_path,
)


def test_cptac_file_filter_pushes_sample_type_into_gdc_query() -> None:
    payload = cptac_file_filter(
        primary_sites=["Kidney"],
        data_categories=["Transcriptome Profiling"],
        data_types=["Gene Expression Quantification"],
        data_formats=["TSV"],
        experimental_strategies=["RNA-Seq"],
        workflow_types=["STAR - Counts"],
        access=["open"],
        sample_types=["Primary Tumor"],
    )

    clauses = payload["content"]
    field_values = {
        clause["content"]["field"]: tuple(clause["content"]["value"])
        for clause in clauses
    }
    assert field_values["cases.project.program.name"] == ("CPTAC",)
    assert field_values["cases.primary_site"] == ("Kidney",)
    assert field_values["cases.samples.sample_type"] == ("Primary Tumor",)
    assert field_values["analysis.workflow_type"] == ("STAR - Counts",)


def test_build_cptac_registry_rows_keeps_metadata_without_fake_download_paths(tmp_path: Path) -> None:
    cases = [
        {
            "case_id": "case-kidney-1",
            "submitter_id": "C3N-00001",
            "project": {"project_id": "CPTAC-3"},
            "primary_site": "Kidney",
            "disease_type": "Clear cell renal cell carcinoma",
            "diagnoses": [
                {
                    "primary_diagnosis": "Clear cell adenocarcinoma, NOS",
                    "tumor_grade": "G3",
                    "ajcc_pathologic_stage": "Stage III",
                    "ajcc_pathologic_t": "T3a",
                    "ajcc_pathologic_n": "N0",
                    "ajcc_pathologic_m": "M0",
                    "age_at_diagnosis": 22345,
                    "vital_status": "Dead",
                    "days_to_last_follow_up": "",
                    "days_to_death": 420,
                }
            ],
            "demographic": {
                "gender": "female",
                "race": "white",
                "ethnicity": "not hispanic or latino",
                "vital_status": "Dead",
                "year_of_birth": 1950,
            },
        }
    ]
    cancer_groups = [
        {
            "name": "kidney",
            "primary_sites": ["Kidney"],
            "tcia_collections": ["CPTAC-CCRCC"],
        }
    ]
    linked_case = {
        "case_id": "case-kidney-1",
        "submitter_id": "C3N-00001",
        "project": {"project_id": "CPTAC-3"},
        "samples": [{"submitter_id": "C3N-00001-01", "sample_type": "Primary Tumor"}],
    }
    rna_bulk_files = [
        {
            "file_id": "rna-1",
            "file_name": "rna-counts.tsv",
            "analysis": {"workflow_type": "STAR - Counts"},
            "cases": [linked_case],
        }
    ]
    dnam_files = [
        {
            "file_id": "dnam-1",
            "file_name": "dnam-betas.txt",
            "analysis": {"workflow_type": "SeSAMe Methylation Beta Estimation"},
            "cases": [linked_case],
        }
    ]
    mutation_files = [
        {
            "file_id": "maf-1",
            "file_name": "mutations.maf.gz",
            "analysis": {"workflow_type": "Aliquot Ensemble Somatic Variant Merging and Masking"},
            "cases": [linked_case],
        }
    ]
    tcia_studies_by_patient = {
        "C3N-00001": [
            {
                "collection": "CPTAC-CCRCC",
                "patient_id": "C3N-00001",
                "study_instance_uid": "1.2.840.study",
                "study_date": "20180102",
                "study_description": "baseline abdomen",
                "modalities_in_study": ["CT", "MR"],
            }
        ]
    }
    tcia_series_by_patient = {
        "C3N-00001": [
            {
                "collection": "CPTAC-CCRCC",
                "patient_id": "C3N-00001",
                "study_instance_uid": "1.2.840.study",
                "series_instance_uid": "1.2.840.series.ct",
                "modality": "CT",
                "body_part_examined": "ABDOMEN",
                "series_description": "venous phase",
            },
            {
                "collection": "CPTAC-CCRCC",
                "patient_id": "C3N-00001",
                "study_instance_uid": "1.2.840.study",
                "series_instance_uid": "1.2.840.series.mr",
                "modality": "MR",
                "body_part_examined": "ABDOMEN",
                "series_description": "t2 axial",
            },
        ]
    }
    downloaded_rna = {
        "rna-1": str(tmp_path / "data" / "raw" / "cptac" / "rna_bulk" / "CPTAC-3" / "C3N-00001" / "rna-counts.tsv")
    }

    frame = build_cptac_registry_rows(
        cases=cases,
        cancer_groups=cancer_groups,
        rna_bulk_files=rna_bulk_files,
        dnam_files=dnam_files,
        mutation_files=mutation_files,
        report_files=[],
        tcia_studies_by_patient=tcia_studies_by_patient,
        tcia_series_by_patient=tcia_series_by_patient,
        downloaded_rna_bulk_by_file_id=downloaded_rna,
        downloaded_dnam_by_file_id={},
        downloaded_mutation_by_file_id={},
        project_root=tmp_path,
        source_name="cptac",
        split_name="cptac_external_test",
        show_progress=False,
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["source"] == "cptac"
    assert row["split"] == "cptac_external_test"
    assert row["split_scheme_version"] == "cptac_external_test_v1"
    assert row["cptac_cancer_group"] == "kidney"
    assert row["cptac_tcia_collections"] == ["CPTAC-CCRCC"]
    assert row["primary_diagnosis"] == "Clear cell adenocarcinoma, NOS"
    assert row["task_grade_label"] == "G3"
    assert row["task_stage_label"] == "Stage III"
    assert bool(row["task_survival_event"]) is True
    assert row["task_survival_days"] == 420.0
    assert row["genomics_rna_bulk_paths"] == [
        "data/raw/cptac/rna_bulk/CPTAC-3/C3N-00001/rna-counts.tsv"
    ]
    assert row["genomics_rna_bulk_file_ids"] == ["rna-1"]
    assert row["genomics_rna_bulk_workflow_types"] == ["STAR - Counts"]
    assert row["genomics_dna_methylation_paths"] == []
    assert row["genomics_dna_methylation_file_ids"] == ["dnam-1"]
    assert row["genomics_dna_methylation_workflow_types"] == ["SeSAMe Methylation Beta Estimation"]
    assert row["genomics_mutation_paths"] == []
    assert row["genomics_mutation_file_ids"] == ["maf-1"]
    assert row["report_pdf_paths"] == []
    assert row["report_file_ids"] == []
    assert row["radiology_image_paths"] == []
    assert row["radiology_download_paths"] == []
    assert row["radiology_uri_paths"] == ["tcia://CPTAC-CCRCC/C3N-00001/1.2.840.study"]
    assert set(row["tcia_modalities"]) == {"CT", "MR"}
    assert set(row["tcia_series_uids"]) == {"1.2.840.series.ct", "1.2.840.series.mr"}
    assert "has_pathology" not in frame.columns


def test_resolve_existing_cptac_files_uses_stable_download_path(tmp_path: Path) -> None:
    file_hit = {
        "file_id": "dnam-1",
        "file_name": "methylation_array.sesame.level3betas.txt",
        "cases": [
            {
                "submitter_id": "C3N-00001",
                "project": {"project_id": "CPTAC-3"},
            }
        ],
    }
    expected_path = stable_file_output_path(
        raw_root=tmp_path,
        source_name="cptac",
        subfolder="dna_methylation",
        file_hit=file_hit,
    )
    assert expected_path is not None
    expected_path.parent.mkdir(parents=True)
    expected_path.write_text("cg00000029\t0.5\n", encoding="utf-8")

    resolved = resolve_existing_cptac_files(
        [file_hit],
        raw_root=tmp_path,
        source_name="cptac",
        subfolder="dna_methylation",
    )

    assert resolved == {"dnam-1": str(expected_path)}
