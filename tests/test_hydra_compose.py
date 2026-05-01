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


def test_vqa_generation_script_cfg_uses_stage_07_path() -> None:
    pytest.importorskip("hydra")

    from kidney_vlm.script_config import load_script_cfg

    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)

    cfg = load_script_cfg(
        repo_root=repo_root,
        config_relative_path="07_vqa_evaluation/generate_vqa_predictions.yaml",
    )

    assert str(cfg.vqa_evaluation.name) == "vqa_generate_predictions"
    assert str(cfg.vqa_evaluation.run.name) == "smoke_test"
    assert int(cfg.vqa_evaluation.run.print_first_n_outputs) == 0
    assert bool(cfg.vqa_evaluation.run.include_prompt_in_predictions) is True
    assert str(cfg.vqa_evaluation.run.prediction_filename) == "predictions.parquet"
    assert int(cfg.vqa_evaluation.run.save_every_n_predictions) == 50
    assert bool(cfg.vqa_evaluation.prefix_cache.enabled) is True
    assert str(cfg.vqa_evaluation.prefix_cache.cache_root) == "data/vqa/prefix_cache"
    assert int(cfg.vqa_evaluation.prefix_cache.max_prefix_tokens.pathology) == 512
    assert int(cfg.vqa_evaluation.prefix_cache.max_prefix_tokens.radiology) == 256
    assert bool(cfg.vqa_evaluation.models.gpt_5_4_mini.enabled) is False
    assert str(cfg.vqa_evaluation.models.gpt_5_4_mini.prompt_profile) == "baseline"
    assert int(cfg.vqa_evaluation.models.gpt_5_4_mini.batch_size) == 1
    assert "display_name" not in cfg.vqa_evaluation.models.gpt_5_4_mini
    assert bool(cfg.vqa_evaluation.models["medgemma-4b-it"].load_in_8bit) is True
    assert str(cfg.vqa_evaluation.models["medgemma-4b-it"].prompt_profile) == "baseline"
    assert int(cfg.vqa_evaluation.models["medgemma-4b-it"].batch_size) == 16
    assert bool(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.enabled) is True
    assert str(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.prompt_profile) == "baseline"
    assert int(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.batch_size) == 8
    assert bool(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.enable_thinking) is False
    assert "display_name" not in cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune
    assert "projectors" not in cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune
    assert bool(cfg.vqa_evaluation.models.oncovlm_qwen_lora.enabled) is True
    assert str(cfg.vqa_evaluation.models.oncovlm_qwen_lora.backend) == "oncovlm_lora"
    assert str(cfg.vqa_evaluation.models.oncovlm_qwen_lora.prompt_profile) == "tuned_oncovlm"
    assert str(cfg.vqa_evaluation.models.oncovlm_qwen_lora.lora_adapter_path).endswith("/lora_adapter")
    assert "projectors" not in cfg.vqa_evaluation.models.oncovlm_qwen_lora
    assert bool(cfg.vqa_evaluation.models.oncovlm_gemma4_no_finetune.enabled) is True
    assert str(cfg.vqa_evaluation.models.oncovlm_gemma4_no_finetune.model_name_or_path) == "google/gemma-4-E4B-it"
    assert str(cfg.vqa_evaluation.models.oncovlm_gemma4_no_finetune.prompt_profile) == "baseline"
    assert int(cfg.vqa_evaluation.models.oncovlm_gemma4_no_finetune.batch_size) == 8
    assert bool(cfg.vqa_evaluation.models.oncovlm_gemma4_no_finetune.enable_thinking) is False
    assert "display_name" not in cfg.vqa_evaluation.models.oncovlm_gemma4_no_finetune
    assert "projectors" not in cfg.vqa_evaluation.models.oncovlm_gemma4_no_finetune
    assert "60 to 120 words" in str(cfg.vqa_evaluation.prompts.baseline.qa.response_instruction)
    assert str(cfg.vqa_evaluation.prompts.tuned_oncovlm.qa.response_instruction) == "Answer concisely."


def test_vqa_scoring_script_cfg_uses_stage_07_path() -> None:
    pytest.importorskip("hydra")

    from kidney_vlm.script_config import load_script_cfg

    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)

    cfg = load_script_cfg(
        repo_root=repo_root,
        config_relative_path="07_vqa_evaluation/score_vqa_predictions.yaml",
    )

    assert str(cfg.vqa_evaluation.name) == "vqa_score_predictions"
    assert str(cfg.vqa_evaluation.run.prediction_filename) == "predictions.parquet"
    assert "scored_prediction_filename" not in cfg.vqa_evaluation.run
    assert "models" not in cfg.vqa_evaluation
    assert (
        str(cfg.vqa_evaluation.metrics.bert_score.model_type)
        == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    assert int(cfg.vqa_evaluation.metrics.bert_score.num_layers) == 9
