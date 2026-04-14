from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def test_tcga_project_defaults_exclude_none_and_keep_bulk_fetch_defaults() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "conf" / "data" / "sources" / "tcga.yaml"
    cfg = OmegaConf.load(cfg_path)
    exclude_project_ids = list(cfg.data.source.tcga.exclude_project_ids)
    assert exclude_project_ids == []
    assert str(cfg.data.source.tcga.upsert_mode) == "replace"
    assert list(cfg.data.source.tcga.patient_subset_ids) == []
    assert int(cfg.data.source.tcga.patient_chunk.index) == 0
    assert cfg.data.source.tcga.patient_chunk.size is None
    assert float(cfg.data.source.tcga.split_ratios.train) == 0.9
    assert float(cfg.data.source.tcga.split_ratios.test) == 0.1
    assert int(cfg.data.source.tcga.gdc.page_size) == 1000
    assert bool(cfg.data.source.tcga.tcia.enabled) is True
    assert bool(cfg.data.source.tcga.tcia.restrict_to_radiology_cases) is True
    assert bool(cfg.data.source.tcga.tcia.fetch_series_metadata) is True
    assert "CT" in list(cfg.data.source.tcga.tcia.qualifying_modalities)
    assert bool(cfg.data.source.tcga.gdc.fetch_ssm_mutations) is True
    assert "VHL" in list(cfg.data.source.tcga.gdc.mutation_gene_panel)
    assert bool(cfg.data.source.tcga.genomics.enabled) is True
    assert bool(cfg.data.source.tcga.genomics.download_raw) is True
    assert bool(cfg.data.source.tcga.genomics.build_text_from_pancan) is True
    assert bool(cfg.data.source.tcga.genomics.cleanup_temp_cache) is False
    assert bool(cfg.data.source.tcga.genomics.write_download_manifest) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.masked_somatic_mutation) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.gene_expression_quantification) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.copy_number_segments) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.masked_copy_number_segments) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.gene_level_copy_number) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.mirna_expression_quantification) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.clinical_supplement) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.biospecimen_supplement) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.msisensor_scores) is True
    assert bool(cfg.data.source.tcga.genomics.raw_payloads.methylation_beta_value) is True
    assert bool(cfg.data.source.tcga.genomics.write_jsonl_sidecar) is True
    assert bool(cfg.data.source.tcga.genomics.write_text_files) is True
