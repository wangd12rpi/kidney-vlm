from __future__ import annotations

from pathlib import Path

from kidney_vlm.data.sources.tcga_rna_metadata import build_tcga_rna_metadata_by_patient_id


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_tcga_rna_metadata_by_patient_id_reads_structured_annotations(tmp_path: Path) -> None:
    _write(
        tmp_path / "TCGASubtype.20170308.tsv",
        "\n".join(
            [
                "sampleID\tSubtype_Selected\tSubtype_mRNA\tSubtype_DNAmeth\tSubtype_Integrative",
                "TCGA-AA-0001-01A\tUCEC.MSI\tBasal\tCIMP-high\tiCluster-1",
                "TCGA-AA-0001-11A\tNormal-like\tNormal-like\tNormal\tNormal",
                "TCGA-BB-0002-01A\tLumA\tLumA\tLuminal\tIntClust-3",
            ]
        ),
    )
    _write(
        tmp_path / "TCGA_all_leuk_estimate.masked.20170107.tsv",
        "\n".join(
            [
                "ACC\tTCGA-AA-0001-01A\t0.2451",
                "BRCA\tTCGA-BB-0002-01A\t0.1012",
            ]
        ),
    )
    _write(
        tmp_path / "ABSOLUTE_scores.tsv",
        "\n".join(
            [
                "SampleID\tAS",
                "TCGA-AA-0001-01A\t18",
                "TCGA-BB-0002-01A\t7",
            ]
        ),
    )
    _write(
        tmp_path / "TCGA_mastercalls.abs_tables_JSedit.fixed.txt",
        "\n".join(
            [
                "Sample\tpurity",
                "TCGA-AA-0001-01A\t0.78",
                "TCGA-BB-0002-01A\t0.66",
            ]
        ),
    )
    _write(
        tmp_path / "TCGA.HRD_withSampleID.txt",
        "\n".join(
            [
                "SampleID\tHRD",
                "TCGA-AA-0001-01A\t42",
                "TCGA-BB-0002-01A\t11",
            ]
        ),
    )
    _write(
        tmp_path / "TCGA.Kallisto.fullIDs.cibersort.relative.tsv",
        "\n".join(
            [
                "SampleID\tT.cells.CD8\tMacrophages.M2\tB.cells.naive\tP-value",
                "TCGA.AA.0001.01A\t0.300\t0.200\t0.100\t0.05",
                "TCGA.BB.0002.01A\t0.050\t0.250\t0.400\t0.02",
            ]
        ),
    )

    metadata = build_tcga_rna_metadata_by_patient_id(
        cases=[
            {"submitter_id": "TCGA-AA-0001"},
            {"submitter_id": "TCGA-BB-0002"},
        ],
        data_dir=tmp_path,
    )

    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_molecular_subtype"] == "UCEC.MSI"
    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_subtype_mrna"] == "Basal"
    assert metadata["TCGA-AA-0001"]["genomics_dna_methylation_subtype"] == "CIMP-high"
    assert metadata["TCGA-AA-0001"]["genomics_integrative_subtype"] == "iCluster-1"
    assert metadata["TCGA-AA-0001"]["genomics_msi_status"] == "MSI"
    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_leukocyte_fraction"] == "0.245"
    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_tumor_purity"] == "0.78"
    assert metadata["TCGA-AA-0001"]["genomics_aneuploidy_score"] == "18"
    assert metadata["TCGA-AA-0001"]["genomics_hrd_score"] == "42"
    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_top_immune_cell_types"] == [
        "T cells CD8",
        "Macrophages M2",
        "B cells naive",
    ]
    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_top_immune_cell_fractions"] == [
        "0.300",
        "0.200",
        "0.100",
    ]
    assert metadata["TCGA-BB-0002"]["genomics_rna_bulk_molecular_subtype"] == "LumA"
    assert metadata["TCGA-BB-0002"]["genomics_rna_bulk_subtype_mrna"] == "LumA"
    assert metadata["TCGA-BB-0002"]["genomics_dna_methylation_subtype"] == "Luminal"
    assert metadata["TCGA-BB-0002"]["genomics_integrative_subtype"] == "IntClust-3"
    assert metadata["TCGA-BB-0002"]["genomics_aneuploidy_score"] == "7"
    assert metadata["TCGA-BB-0002"]["genomics_hrd_score"] == "11"
    assert metadata["TCGA-BB-0002"]["genomics_rna_bulk_top_immune_cell_types"] == [
        "B cells naive",
        "Macrophages M2",
        "T cells CD8",
    ]
