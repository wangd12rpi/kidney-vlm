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
    assert str(cfg.vqa_train.name) == "vqa_lora_sft"
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


def test_vqa_train_script_cfg_wraps_under_stage_package() -> None:
    pytest.importorskip("hydra")

    from kidney_vlm.script_config import load_script_cfg

    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)

    cfg = load_script_cfg(
        repo_root=repo_root,
        config_relative_path="06_vqa_train/vqa_lora_sft.yaml",
        overrides=["vqa_train.dataset.max_train_samples=4"],
    )

    assert str(cfg.vqa_train.name) == "vqa_lora_sft"
    assert int(cfg.vqa_train.dataset.max_train_samples) == 4


def test_vqa_prefix_cache_script_cfg_wraps_under_stage_package() -> None:
    pytest.importorskip("hydra")

    from kidney_vlm.script_config import load_script_cfg

    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)

    cfg = load_script_cfg(
        repo_root=repo_root,
        config_relative_path="06_vqa_train/cache_vqa_prefixes.yaml",
        overrides=["vqa_train.batch_size.pathology=2"],
    )

    assert str(cfg.vqa_train.name) == "vqa_prefix_cache"
    assert str(cfg.vqa_train.dataset.unified_parquet_path) == "data/registry/unified.parquet"
    assert int(cfg.vqa_train.batch_size.pathology) == 2


def test_vqa_evaluation_script_cfg_uses_stage_07_path() -> None:
    pytest.importorskip("hydra")

    from kidney_vlm.script_config import load_script_cfg

    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)

    cfg = load_script_cfg(
        repo_root=repo_root,
        config_relative_path="07_vqa_evaluation/evaluate_vqa_gpt.yaml",
    )

    assert str(cfg.vqa_evaluation.name) == "vqa_eval_v2"
    assert str(cfg.vqa_evaluation.run.name) == "smoke_test"
    assert int(cfg.vqa_evaluation.run.print_first_n_outputs) == 0
    assert bool(cfg.vqa_evaluation.models.gpt_5_1.enabled) is False
    assert int(cfg.vqa_evaluation.models.gpt_5_1.batch_size) == 1
    assert "display_name" not in cfg.vqa_evaluation.models.gpt_5_1
    assert bool(cfg.vqa_evaluation.models.medgemma_4b_it.load_in_8bit) is True
    assert int(cfg.vqa_evaluation.models.medgemma_4b_it.batch_size) == 1
    assert bool(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.enabled) is True
    assert int(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.batch_size) == 1
    assert bool(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.enable_thinking) is False
    assert "display_name" not in cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune
    assert "enabled" not in cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.projectors.pathology
