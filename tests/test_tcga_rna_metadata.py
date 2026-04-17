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
                "sampleID\tSubtype_Selected\tImmune_Subtype",
                "TCGA-AA-0001-01A\tBasal\tC2",
                "TCGA-AA-0001-11A\tNormal-like\tC1",
                "TCGA-BB-0002-01A\tLumA\tC3",
            ]
        ),
    )
    _write(
        tmp_path / "TCGA_all_leuk_estimate.masked.20170107.tsv",
        "\n".join(
            [
                "SampleID\tleukocyte_fraction",
                "TCGA-AA-0001-01A\t0.2451",
                "TCGA-BB-0002-01A\t0.1012",
            ]
        ),
    )
    _write(
        tmp_path / "ABSOLUTE_scores.tsv",
        "\n".join(
            [
                "SampleID\tpurity",
                "TCGA-AA-0001-01A\t0.78",
                "TCGA-BB-0002-01A\t0.66",
            ]
        ),
    )
    _write(
        tmp_path / "TCGA.Kallisto.fullIDs.cibersort.relative.tsv",
        "\n".join(
            [
                "SampleID\tT cells CD8\tMacrophages M2\tB cells naive\tP-value",
                "TCGA-AA-0001-01A\t0.300\t0.200\t0.100\t0.05",
                "TCGA-BB-0002-01A\t0.050\t0.250\t0.400\t0.02",
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

    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_molecular_subtype"] == "Basal"
    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_immune_subtype"] == "C2"
    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_leukocyte_fraction"] == "0.245"
    assert metadata["TCGA-AA-0001"]["genomics_rna_bulk_tumor_purity"] == "0.78"
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
    assert metadata["TCGA-BB-0002"]["genomics_rna_bulk_top_immune_cell_types"] == [
        "B cells naive",
        "Macrophages M2",
        "T cells CD8",
    ]
