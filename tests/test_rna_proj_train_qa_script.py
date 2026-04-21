from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "04_rna_proj" / "03_build_rna_proj_train_qa.py"
    spec = importlib.util.spec_from_file_location("rna_proj_train_qa_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_rna_qa_row_id_uses_one_based_caption_variant() -> None:
    module = _load_script_module()

    assert module._build_qa_row_id("sample-a", 0) == "sample-a::rna-qa-1"
    assert module._build_qa_row_id("sample-a", 2) == "sample-a::rna-qa-3"


def test_build_training_rows_inner_joins_registry_and_captions(tmp_path: Path) -> None:
    module = _load_script_module()
    feature_path = tmp_path / "rna.pt"
    feature_path.write_bytes(b"feature")
    missing_case_feature_path = tmp_path / "other.pt"
    missing_case_feature_path.write_bytes(b"feature")

    registry_rows = [
        {
            "sample_id": "sample-a",
            "source": "tcga",
            "project_id": "TCGA-KIRC",
            "patient_id": "TCGA-AA-0001",
            "study_id": "study-a",
            "split": "train",
            "genomics_rna_bulk_paths": ["raw-a.tsv"],
            "genomics_rna_bulk_feature_path": str(feature_path),
        },
        {
            "sample_id": "sample-without-caption",
            "source": "tcga",
            "project_id": "TCGA-KIRC",
            "patient_id": "TCGA-AA-0002",
            "study_id": "study-b",
            "split": "val",
            "genomics_rna_bulk_paths": ["raw-b.tsv"],
            "genomics_rna_bulk_feature_path": str(missing_case_feature_path),
        },
    ]
    caption_rows = [
        {
            "rna_caption_row_id": "sample-a::rna-caption-1",
            "sample_id": "sample-a",
            "source": "tcga",
            "caption_variant_index": 0,
            "caption_prompt_variant": "variant",
            "caption_length_instruction": "Write 4-5 sentences.",
            "instruction": "Describe RNA.",
            "question": "Describe RNA.",
            "caption": "caption one",
            "answer": "caption one",
            "caption_model": "gpt-test",
            "caption_api_version": "2024-12-01-preview",
            "selected_rna_sample_id": "TCGA-AA-0001",
            "selected_rna_sample_type": "primary tumor",
            "selected_rna_tsv_path": "raw-a.tsv",
            "selected_rna_feature_path": str(feature_path),
        },
        {
            "rna_caption_row_id": "sample-a::rna-caption-2",
            "sample_id": "sample-a",
            "source": "tcga",
            "caption_variant_index": 1,
            "instruction": "Describe RNA.",
            "question": "Describe RNA.",
            "caption": "",
            "answer": "",
            "caption_model": "gpt-test",
            "selected_rna_sample_id": "TCGA-AA-0001",
            "selected_rna_tsv_path": "raw-a.tsv",
            "selected_rna_feature_path": str(feature_path),
        },
    ]

    training_rows, stats = module._build_training_rows(
        registry_rows,
        caption_rows,
        default_instruction="Describe the bulk RNA-seq expression profile.",
    )

    assert len(training_rows) == 1
    assert training_rows[0]["qa_row_id"] == "sample-a::rna-qa-1"
    assert training_rows[0]["rna_caption_row_id"] == "sample-a::rna-caption-1"
    assert training_rows[0]["genomics_rna_bulk_paths"] == ["raw-a.tsv"]
    assert training_rows[0]["caption_api_version"] == "2024-12-01-preview"
    assert stats["registry_rows_without_caption"] == 1
    assert stats["skipped_blank_caption_rows"] == 1


def test_build_training_rows_counts_feature_path_mismatches(tmp_path: Path) -> None:
    module = _load_script_module()
    registry_feature_path = tmp_path / "registry.pt"
    selected_feature_path = tmp_path / "selected.pt"
    registry_feature_path.write_bytes(b"feature")
    selected_feature_path.write_bytes(b"feature")

    registry_rows = [
        {
            "sample_id": "sample-a",
            "source": "tcga",
            "project_id": "TCGA-KIRC",
            "patient_id": "TCGA-AA-0001",
            "study_id": "study-a",
            "split": "train",
            "genomics_rna_bulk_paths": ["raw-a.tsv"],
            "genomics_rna_bulk_feature_path": str(registry_feature_path),
        }
    ]
    caption_rows = [
        {
            "rna_caption_row_id": "sample-a::rna-caption-1",
            "sample_id": "sample-a",
            "source": "tcga",
            "caption_variant_index": 0,
            "caption": "caption",
            "answer": "caption",
            "caption_model": "gpt-test",
            "selected_rna_sample_id": "TCGA-AA-0001",
            "selected_rna_tsv_path": "raw-a.tsv",
            "selected_rna_feature_path": str(selected_feature_path),
        }
    ]

    training_rows, stats = module._build_training_rows(
        registry_rows,
        caption_rows,
        default_instruction="Describe RNA.",
        require_matching_selected_rna_feature_path=True,
    )

    assert training_rows == []
    assert stats["feature_path_mismatch_rows"] == 1


def test_build_output_frame_deduplicates_on_qa_row_id() -> None:
    module = _load_script_module()
    existing_output = pd.DataFrame(
        [
            {"qa_row_id": "row-1", "caption": "old"},
            {"qa_row_id": "row-2", "caption": "keep"},
        ]
    )
    generated_rows = [
        {"qa_row_id": "row-1", "caption": "new"},
        {"qa_row_id": "row-3", "caption": "added"},
    ]

    final_df = module._build_output_frame(
        existing_output=existing_output,
        generated_rows=generated_rows,
        overwrite_output=False,
    )

    assert final_df["qa_row_id"].tolist() == ["row-2", "row-1", "row-3"]
    assert final_df["caption"].tolist() == ["keep", "new", "added"]


def test_assert_output_sanity_requires_all_splits_and_existing_features(tmp_path: Path) -> None:
    module = _load_script_module()
    feature_paths = []
    rows = []
    for split in ("train", "val", "test"):
        feature_path = tmp_path / f"{split}.pt"
        feature_path.write_bytes(b"feature")
        feature_paths.append(feature_path)
        rows.append(
            {
                "qa_row_id": f"sample-{split}::rna-qa-1",
                "rna_caption_row_id": f"sample-{split}::rna-caption-1",
                "sample_id": f"sample-{split}",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": f"TCGA-{split}",
                "study_id": f"study-{split}",
                "split": split,
                "caption_variant_index": 0,
                "genomics_rna_bulk_paths": [f"{split}.tsv"],
                "genomics_rna_bulk_feature_path": str(feature_path),
                "instruction": "Describe RNA.",
                "question": "Describe RNA.",
                "caption": "caption",
                "answer": "caption",
                "caption_model": "gpt-test",
                "selected_rna_sample_id": f"TCGA-{split}",
                "selected_rna_tsv_path": f"{split}.tsv",
                "selected_rna_feature_path": str(feature_path),
            }
        )

    module._assert_output_sanity(pd.DataFrame(rows))

    missing_split_df = pd.DataFrame(rows[:2])
    with pytest.raises(RuntimeError, match="missing split"):
        module._assert_output_sanity(missing_split_df)
