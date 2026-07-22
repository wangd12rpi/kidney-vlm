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
    assert str(cfg.vqa_train.name) == "vqa_sft"
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


def test_vqa_train_script_cfg_merges_common_and_method() -> None:
    pytest.importorskip("hydra")

    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)
    script_path = repo_root / "scripts" / "06_vqa_train" / "train_vqa_lora.py"
    spec = importlib.util.spec_from_file_location("train_vqa_lora_script_for_cfg_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.load_cfg_from_overrides(
        [
            "method=grpo",
            "vqa_train.dataset.max_train_samples=4",
        ]
    )

    assert str(cfg.vqa_train.name) == "vqa_grpo"
    assert str(cfg.vqa_train.post_train_method) == "grpo"
    assert str(cfg.vqa_train.dataset.vqa_parquet_path).endswith("data/vqa/merged_vqa.parquet")
    assert list(cfg.vqa_train.dataset.modality_combination_names) == ["all_available"]
    assert list(cfg.vqa_train.dataset.task_ids) == ["pathology_findings"]
    assert str(cfg.vqa_train.grpo.init_lora_adapter_path).endswith(
        "outputs/oncovlm/sft/"
        "qwen3_5_9b_sft_caption_mcq_all_available_pathology_findings_image_step_cot_"
        "n2673_r8_projfrozen_20260715_020332_est/best/lora_adapter"
    )
    assert bool(cfg.vqa_train.grpo.visual_judge.enabled) is False
    assert float(cfg.vqa_train.grpo.visual_judge.observation_weight) == 0.10
    assert float(cfg.vqa_train.grpo.visual_judge.reasoning_weight) == 0.20
    assert int(cfg.vqa_train.dataset.max_train_samples) == 4


def test_vqa_sft_cfg_uses_image_grounded_cot_and_nocot_warm_start() -> None:
    pytest.importorskip("hydra")

    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)
    script_path = repo_root / "scripts" / "06_vqa_train" / "train_vqa_lora.py"
    spec = importlib.util.spec_from_file_location("train_vqa_lora_script_for_sft_cfg_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.load_cfg_from_overrides(["method=sft"])

    assert bool(cfg.vqa_train.cot.enabled)
    assert str(cfg.vqa_train.dataset.vqa_parquet_path).endswith(
        "data/21_cot_rationale_gen/caption_mcq_all_available_pathology_findings_image_step_cot.parquet"
    )
    assert str(cfg.vqa_train.sft.init_lora_adapter_path).endswith(
        "outputs/oncovlm/kidneyvlm_nocot_qwen35_9b_2ep_20260616_223227/best/lora_adapter"
    )
    assert "Step 1 — Observation:" in str(cfg.vqa_train.prompt.cot_mcq_response_instruction)
    assert str(cfg.vqa_train.autocast_dtype) == "bfloat16"
    assert int(cfg.vqa_train.num_epochs) == 2
    assert float(cfg.vqa_train.learning_rate) == pytest.approx(2e-5)


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
    assert str(cfg.vqa_evaluation.run.name) == "image_step_cot_vs_nocot_pathology_findings_test"
    assert int(cfg.vqa_evaluation.run.print_first_n_outputs) == 0
    assert bool(cfg.vqa_evaluation.run.include_prompt_in_predictions) is True
    assert str(cfg.vqa_evaluation.run.prediction_filename) == "predictions.parquet"
    assert int(cfg.vqa_evaluation.run.save_every_n_predictions) == 10
    assert bool(cfg.vqa_evaluation.prefix_cache.enabled) is True
    assert str(cfg.vqa_evaluation.prefix_cache.cache_root) == "data/vqa/prefix_cache"
    assert int(cfg.vqa_evaluation.prefix_cache.max_prefix_tokens.pathology) == 512
    assert int(cfg.vqa_evaluation.prefix_cache.max_prefix_tokens.radiology) == 256
    assert bool(cfg.vqa_evaluation.filters.row_limit.enabled) is True
    assert int(cfg.vqa_evaluation.filters.row_limit.max_rows) == 16
    assert int(cfg.vqa_evaluation.filters.row_limit.sample_seed) == 410
    assert bool(cfg.vqa_evaluation.models.gpt_5_4.enabled) is False
    assert str(cfg.vqa_evaluation.models.gpt_5_4.prompt_profile) == "baseline"
    assert int(cfg.vqa_evaluation.models.gpt_5_4.batch_size) == 8
    assert "display_name" not in cfg.vqa_evaluation.models.gpt_5_4
    assert bool(cfg.vqa_evaluation.models["medgemma-4b-it"].load_in_8bit) is False
    assert str(cfg.vqa_evaluation.models["medgemma-4b-it"].prompt_profile) == "baseline"
    assert int(cfg.vqa_evaluation.models["medgemma-4b-it"].batch_size) == 10
    assert bool(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.enabled) is False
    assert str(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.prompt_profile) == "baseline"
    assert int(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.batch_size) == 16
    assert bool(cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune.enable_thinking) is False
    assert "display_name" not in cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune
    assert "projectors" not in cfg.vqa_evaluation.models.oncovlm_qwen_no_finetune
    cot_model = cfg.vqa_evaluation.models.oncovlm_qwen_lora_image_step_cot
    assert bool(cot_model.enabled) is True
    assert str(cot_model.backend) == "oncovlm_lora"
    assert str(cot_model.prompt_profile) == "tuned_oncovlm"
    assert str(cot_model.lora_adapter_path).endswith("/best/lora_adapter")
    assert bool(cot_model.cot.enabled) is True
    assert "projectors" not in cot_model
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
