from __future__ import annotations

from pathlib import Path

import pytest


def test_hydra_compose_root_config() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    repo_root = Path(__file__).resolve().parents[1]
    conf_dir = repo_root / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config")

    assert str(cfg.project.name) == "kidney-vlm"
    assert str(cfg.embeding_extraction.pathology.name) == "trident"
    assert str(cfg.vlm_train.name) == "medgemma_hf"
