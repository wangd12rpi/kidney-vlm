from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kidney_vlm.data.rna_feature_import import (
    RnaFeatureRecord,
    align_to_bulkformer_vocab,
    build_case_level_rna_assignments,
    build_rna_feature_filename,
    build_rna_output_path,
    build_rna_records_from_raw_tree_limited,
    infer_rna_file_id_from_name,
    read_tcga_star_log_tpm,
    select_case_level_rna_records,
)


def test_infer_rna_file_id_from_star_gene_counts_name() -> None:
    file_name = "835763c1-a525-401c-bb56-90db5918a621.rna_seq.augmented_star_gene_counts.tsv"

    assert infer_rna_file_id_from_name(file_name) == "835763c1-a525-401c-bb56-90db5918a621"


def test_read_tcga_star_log_tpm_filters_star_summary_and_dedupes_ensembl_versions(tmp_path: Path) -> None:
    tsv = tmp_path / "sample.rna_seq.augmented_star_gene_counts.tsv"
    tsv.write_text(
        "\n".join(
            [
                "# gene-model: GENCODE v36",
                "gene_id\tgene_name\tgene_type\tunstranded\ttpm_unstranded",
                "N_unmapped\t\t\t1\t",
                "ENSG000001.1\tGENE1\tprotein_coding\t10\t3.0",
                "ENSG000001.2\tGENE1\tprotein_coding\t12\t8.0",
                "ENSG000002.1\tGENE2\tlncRNA\t1\t20.0",
                "ENSG000003.1\tGENE3\tprotein_coding\t1\t0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = read_tcga_star_log_tpm(tsv)

    assert list(row.columns) == ["ENSG000001", "ENSG000003"]
    assert np.isclose(float(row.loc[tsv.stem, "ENSG000001"]), np.log1p(8.0))
    assert np.isclose(float(row.loc[tsv.stem, "ENSG000003"]), 0.0)


def test_align_to_bulkformer_vocab_pads_missing_genes_with_minus_ten() -> None:
    row = pd.DataFrame([[1.5, 2.5]], columns=["ENSG000001", "ENSG000003"], index=["sample"])

    aligned, mask_prob = align_to_bulkformer_vocab(row, ["ENSG000003", "ENSG000002", "ENSG000001"])

    assert aligned.shape == (1, 3)
    assert aligned.columns.tolist() == ["ENSG000003", "ENSG000002", "ENSG000001"]
    assert aligned.iloc[0].tolist() == [2.5, -10.0, 1.5]
    assert mask_prob == 1 / 3


def test_select_case_level_rna_records_prefers_primary_tumor() -> None:
    records = [
        RnaFeatureRecord(
            project_id="TCGA-BRCA",
            case_submitter_id="TCGA-LL-A440",
            sample_submitter_id="TCGA-LL-A440-11A",
            rna_file_id="normal",
            rna_file_name="normal.tsv",
            rna_tsv_path="normal.tsv",
            sample_type="Solid Tissue Normal",
        ),
        RnaFeatureRecord(
            project_id="TCGA-BRCA",
            case_submitter_id="TCGA-LL-A440",
            sample_submitter_id="TCGA-LL-A440-01A",
            rna_file_id="tumor",
            rna_file_name="tumor.tsv",
            rna_tsv_path="tumor.tsv",
            sample_type="Primary Tumor",
        ),
    ]

    selected = select_case_level_rna_records(records)

    assert len(selected) == 1
    assert selected[0].sample_submitter_id == "TCGA-LL-A440-01A"
    assert selected[0].rna_file_id == "tumor"


def test_build_rna_records_from_raw_tree_limited_stops_after_requested_cases(tmp_path: Path) -> None:
    first = tmp_path / "data" / "raw" / "tcga" / "rna_bulk" / "TCGA-ACC" / "TCGA-OR-A5J1"
    second = tmp_path / "data" / "raw" / "tcga" / "rna_bulk" / "TCGA-ACC" / "TCGA-OR-A5J2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "file1.rna_seq.augmented_star_gene_counts.tsv").write_text("", encoding="utf-8")
    (second / "file2.rna_seq.augmented_star_gene_counts.tsv").write_text("", encoding="utf-8")

    records = build_rna_records_from_raw_tree_limited(
        tmp_path / "data" / "raw" / "tcga" / "rna_bulk",
        repo_root=tmp_path,
        max_cases=1,
    )

    assert len(records) == 1
    assert records[0].project_id == "TCGA-ACC"
    assert records[0].case_submitter_id == "TCGA-OR-A5J1"
    assert records[0].rna_tsv_path.endswith("file1.rna_seq.augmented_star_gene_counts.tsv")


def test_build_rna_output_path_uses_project_and_tcga_readable_filename(tmp_path: Path) -> None:
    record = RnaFeatureRecord(
        project_id="TCGA-LGG",
        case_submitter_id="TCGA-HT-A614",
        sample_submitter_id="TCGA-HT-A614-01A",
        rna_file_id="abc123",
        rna_file_name="abc123.rna_seq.augmented_star_gene_counts.tsv",
        rna_tsv_path="data/raw/tcga/rna_bulk/TCGA-LGG/TCGA-HT-A614/abc123.tsv",
    )

    filename = build_rna_feature_filename(record)
    output_path = build_rna_output_path(tmp_path / "features_bulkformer_rna", record)

    assert filename == "TCGA-HT-A614-01A__abc123.pt"
    assert output_path == tmp_path / "features_bulkformer_rna" / "TCGA-LGG" / filename


def test_build_case_level_rna_assignments_chooses_primary_tumor_feature() -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "project_id": "TCGA-BRCA",
                "case_submitter_id": "TCGA-LL-A440",
                "sample_submitter_id": "TCGA-LL-A440-11A",
                "rna_tsv_path": "normal.tsv",
                "feature_path": "data/features/features_bulkformer_rna/TCGA-BRCA/normal.pt",
                "rna_file_id": "normal",
                "sample_type": "Solid Tissue Normal",
            },
            {
                "project_id": "TCGA-BRCA",
                "case_submitter_id": "TCGA-LL-A440",
                "sample_submitter_id": "TCGA-LL-A440-01A",
                "rna_tsv_path": "tumor.tsv",
                "feature_path": "data/features/features_bulkformer_rna/TCGA-BRCA/tumor.pt",
                "rna_file_id": "tumor",
                "sample_type": "Primary Tumor",
            },
        ]
    )

    assignments = build_case_level_rna_assignments(manifest_df)

    assert len(assignments) == 1
    row = assignments.iloc[0]
    assert row["patient_id"] == "TCGA-LL-A440"
    assert row["selected_sample_submitter_id"] == "TCGA-LL-A440-01A"
    assert row["genomics_rna_bulk_feature_path"].endswith("tumor.pt")
    assert row["genomics_rna_bulk_paths"] == ["normal.tsv", "tumor.tsv"]
