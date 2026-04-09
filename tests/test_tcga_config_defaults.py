from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def test_tcga_project_defaults_exclude_none_and_keep_bulk_fetch_defaults() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "conf" / "data" / "sources" / "tcga.yaml"
    cfg = OmegaConf.load(cfg_path)
    exclude_project_ids = list(cfg.data.source.tcga.exclude_project_ids)
    assert exclude_project_ids == []
    assert float(cfg.data.source.tcga.split_ratios.train) == 0.9
    assert float(cfg.data.source.tcga.split_ratios.test) == 0.1
    assert int(cfg.data.source.tcga.gdc.page_size) == 1000
    assert bool(cfg.data.source.tcga.tcia.enabled) is True
    assert bool(cfg.data.source.tcga.tcia.fetch_series_metadata) is True
    assert bool(cfg.data.source.tcga.gdc.fetch_ssm_mutations) is True
    assert "VHL" in list(cfg.data.source.tcga.gdc.mutation_gene_panel)
