from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def test_cptac_source_defaults_are_external_validation_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "conf" / "data" / "sources" / "cptac.yaml"
    cfg = OmegaConf.load(cfg_path)

    assert (repo_root / "scripts" / "data" / "02_upsert_cptac_registry_rows.py").exists()
    assert not (repo_root / "scripts" / "data" / "01_upsert_cptac_registry_rows.py").exists()

    assert str(cfg.data.source.name) == "cptac"
    assert str(cfg.data.source.cptac.split_name) == "cptac_external_test"
    assert isinstance(bool(cfg.data.source.download.enabled), bool)
    assert set(cfg.data.source.download.include.keys()) == {
        "rna_bulk",
        "dna_methylation",
        "mutation",
        "radiology",
    }

    groups = list(cfg.data.source.cptac.cancer_groups)
    assert [str(group.name) for group in groups] == ["kidney", "lung", "uterus"]
    assert list(groups[0].primary_sites) == ["Kidney"]
    assert list(groups[1].primary_sites) == ["Bronchus and lung"]
    assert list(groups[2].primary_sites) == ["Uterus, NOS"]
    assert list(groups[0].tcia_collections) == ["CPTAC-CCRCC"]
    assert list(groups[1].tcia_collections) == ["CPTAC-LUAD", "CPTAC-LSCC"]
    assert list(groups[2].tcia_collections) == ["CPTAC-UCEC"]

    assert int(cfg.data.source.cptac.gdc.max_retries) == 1
    assert int(cfg.data.source.cptac.tcia.max_retries) == 1
    assert list(cfg.data.source.cptac.gdc.sample_types) == ["Primary Tumor"]
    assert list(cfg.data.source.cptac.gdc.rna_bulk.workflow_types) == ["STAR - Counts"]
    assert list(cfg.data.source.cptac.gdc.dna_methylation.workflow_types) == [
        "SeSAMe Methylation Beta Estimation"
    ]
    assert list(cfg.data.source.cptac.gdc.mutation.workflow_types) == [
        "Aliquot Ensemble Somatic Variant Merging and Masking"
    ]
    assert list(cfg.data.source.cptac.gdc.reports.data_types) == [
        "Pathology Report",
        "Clinical Supplement",
    ]
    assert list(cfg.data.source.cptac.tcia.download_modalities) == ["CT", "MR", "PT"]
