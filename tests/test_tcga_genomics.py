from __future__ import annotations

from pathlib import Path

import pandas as pd

from kidney_vlm.data.sources.tcga_genomics import DEFAULT_PANCAN_KEYS, build_tcga_genomics_by_patient_id


def _write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def test_build_tcga_genomics_by_patient_id_from_mock_pancan_cache(tmp_path: Path) -> None:
    data_dir = tmp_path / "pancan"

    _write_tsv(
        data_dir / "TCGASubtype.20170308.tsv",
        pd.DataFrame(
            [
                {
                    "sampleID": "TCGA-AA-0001-01A",
                    "Subtype_Selected": "ccB",
                    "Immune_Subtype": "C2",
                }
            ]
        ),
    )
    _write_tsv(
        data_dir / "mc3.v0.2.8.PUBLIC.maf",
        pd.DataFrame(
            [
                {
                    "Hugo_Symbol": "VHL",
                    "Variant_Classification": "Missense_Mutation",
                    "Variant_Type": "SNP",
                    "HGVSp_Short": "p.R167Q",
                    "Tumor_Sample_Barcode": "TCGA-AA-0001-01A",
                    "IMPACT": "MODERATE",
                    "Chromosome": "3",
                    "Start_Position": 101,
                    "End_Position": 101,
                }
            ]
        ),
    )
    _write_tsv(
        data_dir / "all_thresholded.by_genes_whitelisted.tsv",
        pd.DataFrame(
            [
                {"Gene Symbol": "CDKN2A", "TCGA-AA-0001-01A": -2},
                {"Gene Symbol": "MYC", "TCGA-AA-0001-01A": 1},
            ]
        ),
    )
    _write_tsv(
        data_dir / "TCGA_all_leuk_estimate.masked.20170107.tsv",
        pd.DataFrame(
            [
                {
                    "sampleID": "TCGA-AA-0001-01A",
                    "leukocyte_fraction": 0.142,
                }
            ]
        ),
    )
    _write_tsv(
        data_dir / "TCGA.Kallisto.fullIDs.cibersort.relative.tsv",
        pd.DataFrame(
            [
                {
                    "sampleID": "TCGA-AA-0001-01A",
                    "Macrophages M2": 0.21,
                    "T cells CD8": 0.15,
                    "T cells CD4 memory resting": 0.12,
                    "P-value": 0.03,
                    "Correlation": 0.8,
                    "RMSE": 0.1,
                }
            ]
        ),
    )
    _write_tsv(
        data_dir / "mutation-load_updated.txt",
        pd.DataFrame(
            [
                {
                    "Tumor_Sample_Barcode": "TCGA-AA-0001-01A",
                    "Non-silent Mutation Count": 36,
                }
            ]
        ),
    )
    _write_tsv(
        data_dir / "ABSOLUTE_scores.tsv",
        pd.DataFrame(
            [
                {
                    "sampleID": "TCGA-AA-0001-01A",
                    "aneuploidy_score": 8,
                    "wgd": 0,
                    "purity": 0.72,
                }
            ]
        ),
    )
    _write_tsv(
        data_dir / "TCGA.HRD_withSampleID.txt",
        pd.DataFrame(
            [
                {
                    "sampleID": "TCGA-AA-0001-01A",
                    "HRD score": 12,
                }
            ]
        ),
    )
    _write_tsv(
        data_dir / "PANCAN_ArmCallsAndAneuploidyScore_092817.txt",
        pd.DataFrame(
            [
                {
                    "sampleID": "TCGA-AA-0001-01A",
                    "3p": -1,
                }
            ]
        ),
    )
    _write_tsv(
        data_dir / "TCGA_mastercalls.abs_tables_JSedit.fixed.txt",
        pd.DataFrame([{"sampleID": "TCGA-AA-0001-01A", "purity": 0.72}]),
    )

    genomics_by_patient = build_tcga_genomics_by_patient_id(
        cases=[
            {
                "case_id": "case-1",
                "submitter_id": "TCGA-AA-0001",
                "project": {"project_id": "TCGA-KIRC"},
            }
        ],
        data_dir=data_dir,
    )

    record = genomics_by_patient["TCGA-AA-0001"]
    genomics_text = record["genomics_text"]

    assert record["cancer_code"] == "KIRC"
    assert "molecular_subtype: ccB" in genomics_text
    assert "immune_subtype: C2" in genomics_text
    assert "tmb: 1.2 mut/Mb (low)" in genomics_text
    assert "key_somatic_mutations: VHL p.R167Q (missense)" in genomics_text
    assert "key_copy_number_alterations: CDKN2A: deep_deletion, MYC: gain" in genomics_text
    assert "aneuploidy_score: 8" in genomics_text
    assert "whole_genome_doubling: no" in genomics_text
    assert "tumor_purity: 0.72" in genomics_text
    assert "hrd_score: 12" in genomics_text
    assert "arm_level_cna: chr3p_loss: yes" in genomics_text
    assert "vhl_status: mutated" in genomics_text
    assert "leukocyte_fraction: 0.142" in genomics_text
    assert "dominant_immune_cells: Macrophages M2 (0.21), T cells CD8 (0.15), T cells CD4 memory resting (0.12)" in genomics_text
    assert record["genomics_fields"]["arm_level_events"] == {"chr3p_loss": "yes"}
    assert record["genomics_fields"]["cna_by_gene"] == {"CDKN2A": "deep_deletion", "MYC": "gain"}


def test_build_tcga_genomics_by_patient_id_uses_optional_viral_annotations_for_hnsc(tmp_path: Path) -> None:
    data_dir = tmp_path / "pancan"

    _write_tsv(
        data_dir / "TCGASubtype.20170308.tsv",
        pd.DataFrame([{"sampleID": "TCGA-BB-0002-01A", "Subtype_Selected": "atypical"}]),
    )
    _write_tsv(
        data_dir / "mc3.v0.2.8.PUBLIC.maf",
        pd.DataFrame(
            [
                {
                    "Hugo_Symbol": "TP53",
                    "Variant_Classification": "Missense_Mutation",
                    "Variant_Type": "SNP",
                    "HGVSp_Short": "p.R248Q",
                    "Tumor_Sample_Barcode": "TCGA-BB-0002-01A",
                    "IMPACT": "MODERATE",
                    "Chromosome": "17",
                    "Start_Position": 101,
                    "End_Position": 101,
                }
            ]
        ),
    )
    _write_tsv(
        data_dir / "all_thresholded.by_genes_whitelisted.tsv",
        pd.DataFrame([{"Gene Symbol": "EGFR", "TCGA-BB-0002-01A": 0}]),
    )
    _write_tsv(
        data_dir / "TCGA_all_leuk_estimate.masked.20170107.tsv",
        pd.DataFrame([{"sampleID": "TCGA-BB-0002-01A", "leukocyte_fraction": 0.05}]),
    )
    _write_tsv(
        data_dir / "TCGA.Kallisto.fullIDs.cibersort.relative.tsv",
        pd.DataFrame([{"sampleID": "TCGA-BB-0002-01A", "T cells CD8": 0.2}]),
    )
    _write_tsv(
        data_dir / "mutation-load_updated.txt",
        pd.DataFrame([{"Tumor_Sample_Barcode": "TCGA-BB-0002-01A", "Non-silent Mutation Count": 12}]),
    )
    _write_tsv(
        data_dir / "ABSOLUTE_scores.tsv",
        pd.DataFrame([{"sampleID": "TCGA-BB-0002-01A", "aneuploidy_score": 6, "wgd": 1, "purity": 0.61}]),
    )
    _write_tsv(
        data_dir / "TCGA.HRD_withSampleID.txt",
        pd.DataFrame([{"sampleID": "TCGA-BB-0002-01A", "HRD score": 3}]),
    )
    _write_tsv(
        data_dir / "viral.tsv",
        pd.DataFrame([{"sampleID": "TCGA-BB-0002-01A", "HPV16": 22, "EBV": 0}]),
    )

    genomics_by_patient = build_tcga_genomics_by_patient_id(
        cases=[
            {
                "case_id": "case-2",
                "submitter_id": "TCGA-BB-0002",
                "project": {"project_id": "TCGA-HNSC"},
                "diagnoses": [{"primary_diagnosis": "Squamous cell carcinoma"}],
            }
        ],
        data_dir=data_dir,
    )

    genomics_text = genomics_by_patient["TCGA-BB-0002"]["genomics_text"]
    assert "hpv_status: HPV_positive" in genomics_text


def test_default_pancan_keys_include_tcga_clinical_data_resource() -> None:
    assert "tcga_clinical_data_resource" in DEFAULT_PANCAN_KEYS
