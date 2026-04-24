from __future__ import annotations

from pathlib import Path

import pandas as pd

from kidney_vlm.genomics.registry_integration import (
    update_registry_with_extra_genomics_manifest,
    update_registry_with_genomics_json_manifest,
    update_registry_with_llm_input_context_manifest,
)


def test_update_registry_with_extra_genomics_manifest_adds_paths_and_metadata(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    mutation_path = repo_root / "data/raw/tcga/mutation_maf/TCGA-KIRC/TCGA-AA-0001/case.maf"
    gene_cna_path = repo_root / "data/raw/tcga/copy_number_gene/TCGA-KIRC/TCGA-AA-0001/gene.tsv"
    segment_cna_path = repo_root / "data/raw/tcga/copy_number_segment/TCGA-KIRC/TCGA-AA-0001/seg.tsv"
    for path in (mutation_path, gene_cna_path, segment_cna_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
            }
        ]
    )
    manifest_df = pd.DataFrame(
        [
            {
                "modality": "mutation_maf",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "output_path": str(mutation_path),
                "file_id": "maf-file",
                "file_name": "case.maf",
                "sample_submitter_id": "TCGA-AA-0001-01A",
                "workflow_type": "Mutect2 Variant Aggregation and Masking",
            },
            {
                "modality": "copy_number_gene",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "output_path": str(gene_cna_path),
                "file_id": "gene-file",
                "file_name": "gene.tsv",
                "sample_submitter_id": "TCGA-AA-0001-01A",
                "workflow_type": "ASCAT3",
            },
            {
                "modality": "copy_number_segment",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "output_path": str(segment_cna_path),
                "file_id": "seg-file",
                "file_name": "seg.tsv",
                "sample_submitter_id": "TCGA-AA-0001-01A",
                "workflow_type": "ASCAT3",
            },
        ]
    )

    updated, stats = update_registry_with_extra_genomics_manifest(
        registry_df,
        manifest_df,
        repo_root=repo_root,
        source_name="tcga",
    )

    row = updated.iloc[0]
    assert stats.matched_registry_rows == 1
    assert stats.updated_registry_rows == 1
    assert stats.unmatched_manifest_cases == 0
    assert row["genomics_mutation_paths"] == [
        "data/raw/tcga/mutation_maf/TCGA-KIRC/TCGA-AA-0001/case.maf"
    ]
    assert row["genomics_cnv_gene_paths"] == [
        "data/raw/tcga/copy_number_gene/TCGA-KIRC/TCGA-AA-0001/gene.tsv"
    ]
    assert row["genomics_cnv_segment_paths"] == [
        "data/raw/tcga/copy_number_segment/TCGA-KIRC/TCGA-AA-0001/seg.tsv"
    ]
    assert row["genomics_cnv_paths"] == [
        "data/raw/tcga/copy_number_gene/TCGA-KIRC/TCGA-AA-0001/gene.tsv",
        "data/raw/tcga/copy_number_segment/TCGA-KIRC/TCGA-AA-0001/seg.tsv",
    ]
    assert row["genomics_mutation_file_ids"] == ["maf-file"]
    assert row["genomics_cnv_gene_workflow_types"] == ["ASCAT3"]
    assert row["genomics_available_modalities"] == [
        "copy_number_gene",
        "copy_number_segment",
        "mutation_maf",
    ]


def test_update_registry_with_genomics_json_manifest_adds_text_context_paths(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    features_json = repo_root / "data/features/llm_input_contexts/tcga/TCGA-KIRC/TCGA-AA-0001/llm_input.json"
    teacher_text = features_json.with_name("teacher.txt")
    student_text = features_json.with_name("student.txt")
    for path in (features_json, teacher_text, student_text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
            }
        ]
    )
    manifest_df = pd.DataFrame(
        [
            {
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "genomics_json_path": str(features_json),
                "teacher_text_path": str(teacher_text),
                "student_text_path": str(student_text),
                "available_modalities": ["mutation_maf"],
                "errors": "",
            }
        ]
    )

    updated, stats = update_registry_with_genomics_json_manifest(
        registry_df,
        manifest_df,
        repo_root=repo_root,
        source_name="tcga",
    )

    row = updated.iloc[0]
    assert stats.updated_registry_rows == 1
    assert row["genomics_json_path"] == (
        "data/features/llm_input_contexts/tcga/TCGA-KIRC/TCGA-AA-0001/llm_input.json"
    )
    assert row["genomics_teacher_text_path"] == (
        "data/features/llm_input_contexts/tcga/TCGA-KIRC/TCGA-AA-0001/teacher.txt"
    )
    assert row["genomics_student_text_path"] == (
        "data/features/llm_input_contexts/tcga/TCGA-KIRC/TCGA-AA-0001/student.txt"
    )
    assert row["genomics_available_modalities"] == ["mutation_maf"]


def test_update_registry_with_llm_input_context_manifest_adds_prompt_paths(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    case_dir = repo_root / "data/features/llm_input_contexts/tcga/TCGA-KIRC/TCGA-AA-0001"
    clinical_path = case_dir / "clinical.txt"
    gdisc_path = case_dir / "gdisc.txt"
    llm_input_path = case_dir / "llm_input.txt"
    json_path = case_dir / "llm_input.json"
    for path in (clinical_path, gdisc_path, llm_input_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
            }
        ]
    )
    manifest_df = pd.DataFrame(
        [
            {
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "clinical_text_path": str(clinical_path),
                "gdisc_text_path": str(gdisc_path),
                "llm_input_text_path": str(llm_input_path),
                "llm_input_json_path": str(json_path),
                "mutation_available": True,
                "copy_number_gene_available": True,
                "copy_number_segment_available": False,
                "errors": "",
            }
        ]
    )

    updated, stats = update_registry_with_llm_input_context_manifest(
        registry_df,
        manifest_df,
        repo_root=repo_root,
        source_name="tcga",
    )

    row = updated.iloc[0]
    assert stats.updated_registry_rows == 1
    assert row["genomics_clinical_text_path"] == (
        "data/features/llm_input_contexts/tcga/TCGA-KIRC/TCGA-AA-0001/clinical.txt"
    )
    assert row["genomics_gdisc_text_path"] == (
        "data/features/llm_input_contexts/tcga/TCGA-KIRC/TCGA-AA-0001/gdisc.txt"
    )
    assert row["genomics_llm_input_text_path"] == (
        "data/features/llm_input_contexts/tcga/TCGA-KIRC/TCGA-AA-0001/llm_input.txt"
    )
    assert row["genomics_llm_input_json_path"] == (
        "data/features/llm_input_contexts/tcga/TCGA-KIRC/TCGA-AA-0001/llm_input.json"
    )
    assert row["genomics_available_modalities"] == ["mutation_maf", "copy_number_gene"]
