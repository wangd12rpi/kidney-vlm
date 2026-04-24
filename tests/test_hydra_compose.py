from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_hydra_compose_root_config() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    repo_root = Path(__file__).resolve().parents[1]
    conf_dir = repo_root / "conf"
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config")

    assert str(cfg.project.name) == "kidney-vlm"
    assert str(cfg.pathology_features.name) == "trident"
    assert str(cfg.pathology_png.name) == "pathology_png"
    assert str(cfg.dnam_features.name) == "cpgpt_dnam_features"
    assert str(cfg.radiology_proj.modality_tag) == "radiology"
    assert str(cfg.vlm_train.name) == "medgemma_hf"
    assert str(cfg.dnam_proj.modality_tag) == "dnam"
    assert str(cfg.rna_proj.modality_tag) == "rna"


def test_radiology_projector_cfg_accepts_gemma4_override() -> None:
    pytest.importorskip("hydra")

    from kidney_vlm.script_config import load_script_cfg

    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)

    cfg = load_script_cfg(
        repo_root=repo_root,
        config_relative_path="02_radiology_proj/03_train_radiology_projectors.yaml",
        overrides=["radiology_proj.model_name_or_path=google/gemma-4-E4B-it"],
    )

    assert str(cfg.radiology_proj.model_name_or_path) == "google/gemma-4-E4B-it"
