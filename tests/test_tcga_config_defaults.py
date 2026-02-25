from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def test_tcga_project_defaults_include_kidney_projects() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "conf" / "data" / "sources" / "tcga.yaml"
    cfg = OmegaConf.load(cfg_path)
    project_ids = list(cfg.data.source.tcga.project_ids)
    assert project_ids == ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"]
    assert float(cfg.data.source.tcga.split_ratios.train) == 0.9
    assert float(cfg.data.source.tcga.split_ratios.test) == 0.1
