from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kidney_vlm.data.dnam_feature_import import (
    build_case_level_dnam_assignments,
    build_cpgpt_feature_filename,
    build_cpgpt_hash_index,
    build_cpgpt_output_path,
    cpgpt_cache_hash_for_beta_path,
    normalize_cpgpt_cache_key_with_prefix,
    parse_cpgpt_index_row,
)


def test_normalize_cpgpt_cache_key_rewrites_anvil_prefix() -> None:
    original = "/anvil/projects/x-cis250966/dna/tcga/file.txt"
    assert normalize_cpgpt_cache_key_with_prefix(original, local_prefix="/tmp/local_root") == "/tmp/local_root/tcga/file.txt"


def test_parse_cpgpt_index_row_resolves_relative_beta_path(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    row = {
        "project_id": "TCGA-BRCA",
        "case_submitter_id": "TCGA-LL-A440",
        "sample_submitter_id": "TCGA-LL-A440-01A",
        "methylation_beta": {
            "file_id": "4532b645-4510-4cbe-bfd1-ef6bfa032125",
            "file_name": "37659589-9724-4724-aec8-e9537d5b9de8.methylation_array.sesame.level3betas.txt",
            "local_path": "TCGA-LL-A440-01A/37659589-9724-4724-aec8-e9537d5b9de8.methylation_array.sesame.level3betas.txt",
        },
    }

    record = parse_cpgpt_index_row(row, raw_root=raw_root, index_name="index_full.jsonl")

    assert record is not None
    assert record.project_id == "TCGA-BRCA"
    assert record.sample_submitter_id == "TCGA-LL-A440-01A"
    assert record.beta_path == str(raw_root / row["methylation_beta"]["local_path"])
    assert record.cache_hash == cpgpt_cache_hash_for_beta_path(record.beta_path)


def test_build_cpgpt_hash_index_merges_duplicate_index_sources(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    row = {
        "project_id": "TCGA-BRCA",
        "case_submitter_id": "TCGA-LL-A440",
        "sample_submitter_id": "TCGA-LL-A440-01A",
        "methylation_beta": {
            "file_id": "4532b645-4510-4cbe-bfd1-ef6bfa032125",
            "file_name": "37659589-9724-4724-aec8-e9537d5b9de8.methylation_array.sesame.level3betas.txt",
            "local_path": "TCGA-LL-A440-01A/37659589-9724-4724-aec8-e9537d5b9de8.methylation_array.sesame.level3betas.txt",
        },
    }
    first = tmp_path / "index_train.jsonl"
    second = tmp_path / "index_test.jsonl"
    first.write_text(json.dumps(row) + "\n", encoding="utf-8")
    second.write_text(json.dumps(row) + "\n", encoding="utf-8")

    records = build_cpgpt_hash_index([first, second], raw_root=raw_root)

    assert len(records) == 1
    only_record = next(iter(records.values()))
    assert only_record.source_index_files == ("index_test.jsonl", "index_train.jsonl")


def test_build_cpgpt_output_path_uses_project_and_tcga_readable_filename(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    row = {
        "project_id": "TCGA-LGG",
        "case_submitter_id": "TCGA-HT-A614",
        "sample_submitter_id": "TCGA-HT-A614-01A",
        "methylation_beta": {
            "file_id": "abc123",
            "file_name": "1d77e688-28f9-4fa4-970f-2fc48284c6b7.methylation_array.sesame.level3betas.txt",
            "local_path": "TCGA-HT-A614-01A/1d77e688-28f9-4fa4-970f-2fc48284c6b7.methylation_array.sesame.level3betas.txt",
        },
    }
    record = parse_cpgpt_index_row(row, raw_root=raw_root, index_name="index_full.jsonl")
    assert record is not None

    filename = build_cpgpt_feature_filename(record)
    output_path = build_cpgpt_output_path(tmp_path / "features_cpgpt_dnam", record)

    assert filename == "TCGA-HT-A614-01A__abc123.pt"
    assert output_path == tmp_path / "features_cpgpt_dnam" / "TCGA-LGG" / filename


def test_build_case_level_dnam_assignments_chooses_primary_tumor_feature() -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "project_id": "TCGA-BRCA",
                "case_submitter_id": "TCGA-LL-A440",
                "sample_submitter_id": "TCGA-LL-A440-11A",
                "beta_path": "../hescapedna/raw/TCGA-LL-A440-11A/normal.txt",
                "feature_path": "data/features/features_cpgpt_dnam/TCGA-BRCA/TCGA-LL-A440-11A__normal.pt",
                "beta_file_id": "normal",
            },
            {
                "project_id": "TCGA-BRCA",
                "case_submitter_id": "TCGA-LL-A440",
                "sample_submitter_id": "TCGA-LL-A440-01A",
                "beta_path": "../hescapedna/raw/TCGA-LL-A440-01A/tumor.txt",
                "feature_path": "data/features/features_cpgpt_dnam/TCGA-BRCA/TCGA-LL-A440-01A__tumor.pt",
                "beta_file_id": "tumor",
            },
        ]
    )

    assignments = build_case_level_dnam_assignments(manifest_df)

    assert len(assignments) == 1
    row = assignments.iloc[0]
    assert row["patient_id"] == "TCGA-LL-A440"
    assert row["selected_sample_submitter_id"] == "TCGA-LL-A440-01A"
    assert row["genomics_dna_methylation_feature_path"].endswith("TCGA-LL-A440-01A__tumor.pt")
    assert row["genomics_dna_methylation_paths"] == [
        "../hescapedna/raw/TCGA-LL-A440-01A/tumor.txt",
        "../hescapedna/raw/TCGA-LL-A440-11A/normal.txt",
    ]
