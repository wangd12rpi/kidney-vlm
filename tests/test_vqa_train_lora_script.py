from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

pytest.importorskip("torch", exc_type=ImportError)
pytest.importorskip("omegaconf")


def _base_row(**overrides):
    row = {
        "case_id": "case-1",
        "project_id": "TCGA-KIRC",
        "question_id": 1,
        "split": "train",
        "question_type": "mcq",
        "generation_type": "from_ground_truth",
        "task_category": "mutation",
        "task_id": "mutation:VHL",
        "modality_combination_name": "all_available",
        "use_pathology": True,
        "use_radiology": False,
        "use_dnam": True,
        "use_rna": False,
        "question": "What is the VHL status?",
        "option_a": "mutated",
        "option_b": "wild type",
        "option_c": "",
        "option_d": "",
        "answer": "mutated",
        "answer_label": "A",
        "pathology_feature_paths": ["data/features/path.h5"],
        "radiology_feature_paths": [],
        "dnam_feature_path": "data/features/dnam.pt",
        "rna_feature_path": "",
    }
    row.update(overrides)
    return row


def test_mcq_prompt_injects_choices_and_projector_evidence() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.prompts import build_vqa_prompt, build_vqa_prompt_preview

    row = _base_row()
    prompt = build_vqa_prompt(
        row,
        OmegaConf.create(
            {
                "system_prompt": "Use the features.",
                "mcq_response_instruction": "Answer with the exact choice text.",
                "open_response_instruction": "Answer concisely.",
            }
        ),
    )
    preview = build_vqa_prompt_preview(
        _base_row(),
        OmegaConf.create(
            {
                "system_prompt": "Use the features.",
                "mcq_response_instruction": "Answer with the exact choice text.",
                "open_response_instruction": "Answer concisely.",
            }
        ),
    )

    assert "<choices>" in prompt
    assert "- mutated" in prompt
    assert "- wild type" in prompt
    assert "A." not in prompt
    assert "option_a" not in prompt
    assert "<pathology_features>" in prompt
    assert "<|oncovlm_pathology_prefix|>" in prompt
    assert "<dnam_features>" in prompt
    assert "<|oncovlm_dnam_prefix|>" in prompt
    assert "[PREFIX:pathology soft tokens]" in preview
    assert "[PREFIX:dnam soft tokens]" in preview


def test_cot_mcq_prompt_keeps_unlabeled_choice_surface() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.prompts import build_vqa_prompt

    prompt = build_vqa_prompt(
        _base_row(),
        OmegaConf.create(
            {
                "system_prompt": "Use the features.",
                "mcq_response_instruction": "Answer with the exact choice text.",
                "cot_mcq_response_instruction": "Reason in tags, then answer with exact choice text.",
                "open_response_instruction": "Answer concisely.",
                "use_cot": True,
            }
        ),
    )

    assert "- mutated" in prompt
    assert "- wild type" in prompt
    assert "A. mutated" not in prompt
    assert "B. wild type" not in prompt


def test_cot_training_target_requires_full_answer_text() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.data import build_vqa_training_target

    cfg = OmegaConf.create({"cot": {"enabled": True, "rationale_column": "rationale"}})
    valid = _base_row(
        rationale="<think>The mutation evidence supports this choice.</think><answer>mutated</answer>"
    )
    invalid = _base_row(
        rationale="<think>The mutation evidence supports this choice.</think><answer>A</answer>"
    )

    assert build_vqa_training_target(valid, cfg) == valid["rationale"]
    with pytest.raises(ValueError, match="does not match answer"):
        build_vqa_training_target(invalid, cfg)


def test_open_ended_prompt_leaves_choices_out() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.prompts import build_vqa_prompt

    prompt = build_vqa_prompt(
        _base_row(
            question_type="open_ended",
            option_a="",
            option_b="",
            answer="The tumor has a VHL alteration.",
        ),
        OmegaConf.create(
            {
                "system_prompt": "Use the features.",
                "mcq_response_instruction": "Answer with the exact choice text.",
                "open_response_instruction": "Answer concisely.",
            }
        ),
    )

    assert "<choices>" not in prompt
    assert "Answer concisely." in prompt


def test_radiology_prompt_includes_biomarker_text_when_present() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.prompts import build_vqa_prompt

    prompt = build_vqa_prompt(
        _base_row(
            use_radiology=True,
            radiology_feature_paths=["data/features/rad.h5::series=abc"],
            radiology_biomarker="Radiology report biomarker: enhancing renal mass.",
        ),
        OmegaConf.create(
            {
                "system_prompt": "Use the features.",
                "mcq_response_instruction": "Answer with the exact choice text.",
                "open_response_instruction": "Answer concisely.",
            }
        ),
    )

    assert "<radiology_features>" in prompt
    assert "<radiology_biomarker>" in prompt
    assert "Radiology report biomarker: enhancing renal mass." in prompt


def test_select_train_rows_keeps_modality_dropout_and_skips_disabled_modalities() -> (
    None
):
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.data import select_vqa_rows

    rows = [
        _base_row(question_id=1, modality_combination_name="all_available"),
        _base_row(
            question_id=2,
            modality_combination_name="path_only",
            use_dnam=False,
            dnam_feature_path="",
        ),
        _base_row(question_id=3, split="val"),
        _base_row(question_id=4, use_rna=True, rna_feature_path="data/features/rna.pt"),
    ]
    cfg = OmegaConf.create(
        {
            "dataset": {
                "question_types": ["mcq", "open_ended"],
                "generation_types": [],
                "modality_combination_names": [],
                "project_ids": [],
                "task_categories": [],
                "task_ids": [],
                "sample_seed": 42,
            },
            "projectors": {
                "pathology": {"enabled": True},
                "radiology": {"enabled": False},
                "dnam": {"enabled": True},
                "rna": {"enabled": False},
            },
        }
    )

    selected = select_vqa_rows(
        pd.DataFrame(rows),
        cfg,
        split="train",
        max_samples_key="max_train_samples",
        sample_key="sample_train",
    )

    assert selected["question_id"].tolist() == [1, 2]


def test_caption_mcq_filter_keeps_all_task_ids() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.data import select_vqa_rows

    rows = [
        _base_row(
            question_id=1, generation_type="from_caption", task_id="pathology_findings"
        ),
        _base_row(
            question_id=2, generation_type="from_caption", task_id="genomic_findings"
        ),
        _base_row(
            question_id=3,
            generation_type="from_ground_truth",
            task_id="pathology_findings",
        ),
        _base_row(
            question_id=4, generation_type="from_caption", question_type="open_ended"
        ),
    ]
    cfg = OmegaConf.create(
        {
            "dataset": {
                "question_types": ["mcq"],
                "generation_types": ["from_caption"],
                "modality_combination_names": [],
                "project_ids": [],
                "task_categories": [],
                "task_ids": [],
                "sample_seed": 42,
            },
            "projectors": {
                "pathology": {"enabled": True},
                "radiology": {"enabled": True},
                "dnam": {"enabled": True},
                "rna": {"enabled": True},
            },
        }
    )

    selected = select_vqa_rows(
        pd.DataFrame(rows),
        cfg,
        split="train",
        max_samples_key="max_train_samples",
        sample_key="sample_train",
    )

    assert selected["question_id"].tolist() == [2, 1]


def test_select_rows_filters_modality_combination_and_task_id() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.data import select_vqa_rows

    rows = [
        _base_row(
            question_id=1, generation_type="from_caption", task_id="pathology_findings"
        ),
        _base_row(
            question_id=2,
            generation_type="from_caption",
            task_id="pathology_findings",
            modality_combination_name="path_only",
            use_dnam=False,
            dnam_feature_path="",
        ),
        _base_row(
            question_id=3, generation_type="from_caption", task_id="genomic_findings"
        ),
    ]
    cfg = OmegaConf.create(
        {
            "dataset": {
                "question_types": ["mcq"],
                "generation_types": ["from_caption"],
                "modality_combination_names": ["all_available"],
                "project_ids": [],
                "task_categories": [],
                "task_ids": ["pathology_findings"],
                "sample_seed": 42,
            },
            "projectors": {
                "pathology": {"enabled": True},
                "radiology": {"enabled": True},
                "dnam": {"enabled": True},
                "rna": {"enabled": True},
            },
        }
    )

    selected = select_vqa_rows(
        pd.DataFrame(rows),
        cfg,
        split="train",
        max_samples_key="max_train_samples",
        sample_key="sample_train",
    )

    assert selected["question_id"].tolist() == [1]


def _load_vqa_train_script(name: str):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "06_vqa_train" / "train_vqa_lora.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grpo_reward_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "reward_weights": {
                "correctness": 1.0,
                "valid_choice": 0.0,
                "format": 0.03,
                "observation": 0.04,
                "reasoning": 0.03,
            },
            "observation_min_words": 5,
            "observation_max_words": 40,
            "reasoning_min_words": 5,
            "reasoning_max_words": 40,
            "think_min_words": 10,
            "think_max_words": 90,
        }
    )


def _two_step_completion(answer: str = "mutated", *, explicit_open: bool = True) -> str:
    opening = "<think>" if explicit_open else ""
    return (
        f"{opening}Step 1 — Observation: Dense cells form compact nests with pink stroma and small vessels. "
        "Step 2 — Reasoning: The nested architecture and stromal pattern fit the diagnosis; the closest "
        f"alternative lacks this combination.</think><answer>{answer}</answer>"
    )


def test_grpo_reward_requires_exact_complete_answer_tag() -> None:
    module = _load_vqa_train_script("train_vqa_lora_script_for_reward_test")
    reward_cfg = _grpo_reward_cfg()
    full_score = module.score_grpo_completion(
        completion=_two_step_completion(),
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )
    letter_score = module.score_grpo_completion(
        completion=_two_step_completion("A"),
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )
    truncated_score = module.score_grpo_completion(
        completion=_two_step_completion().replace(
            "<answer>mutated</answer>", "<answer>mut"
        ),
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )
    bare_score = module.score_grpo_completion(
        completion="mutated",
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )

    assert full_score["correct"] == 1.0
    assert full_score["valid_choice"] == 1.0
    assert letter_score["correct"] == 0.0
    assert truncated_score["correct"] == 0.0
    assert bare_score["correct"] == 0.0


def test_grpo_reward_requires_clean_two_step_format() -> None:
    module = _load_vqa_train_script("train_vqa_lora_script_for_reward_format_test")
    reward_cfg = _grpo_reward_cfg()
    explicit_score = module.score_grpo_completion(
        completion=_two_step_completion(),
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )
    implicit_score = module.score_grpo_completion(
        completion=_two_step_completion(explicit_open=False),
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )
    junk_between_tags = _two_step_completion().replace(
        "</think><answer>", "</think>junk<answer>"
    )
    wrong_heading = _two_step_completion().replace(
        "Step 1 — Observation:", "Step 1 - Observation:"
    )

    assert explicit_score["format"] == 1.0
    assert explicit_score["two_step"] == 1.0
    assert implicit_score["format"] == 1.0
    assert implicit_score["think_format"] == "implicit_open"
    assert (
        module.score_grpo_completion(
            completion=junk_between_tags,
            answer="mutated",
            choices=["mutated", "wild type"],
            reward_cfg=reward_cfg,
        )["format"]
        == 0.0
    )
    assert (
        module.score_grpo_completion(
            completion=wrong_heading,
            answer="mutated",
            choices=["mutated", "wild type"],
            reward_cfg=reward_cfg,
        )["two_step"]
        == 0.0
    )


def test_grpo_reward_rejects_choice_directed_observation_and_walkthrough() -> None:
    module = _load_vqa_train_script("train_vqa_lora_script_for_reward_focus_test")
    reward_cfg = _grpo_reward_cfg()
    good = _two_step_completion()
    choice_directed = good.replace(
        "Dense cells form compact nests with pink stroma and small vessels.",
        "The cells support this answer rather than the other diagnosis in the choices.",
    )
    absence_directed = good.replace(
        "Dense cells form compact nests with pink stroma and small vessels.",
        "Dense cells form compact nests with pink stroma. No necrosis is identified.",
    )
    walkthrough = good.replace(
        "The nested architecture and stromal pattern fit the diagnosis; the closest alternative lacks this combination.",
        "The first choice matches the nests while the second choice lacks them and the other choices fail.",
    )

    good_score = module.score_grpo_completion(
        completion=good,
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )
    observation_score = module.score_grpo_completion(
        completion=choice_directed,
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )
    absence_score = module.score_grpo_completion(
        completion=absence_directed,
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )
    walkthrough_score = module.score_grpo_completion(
        completion=walkthrough,
        answer="mutated",
        choices=["mutated", "wild type"],
        reward_cfg=reward_cfg,
    )

    assert good_score["observation"] == 1.0
    assert good_score["observation_presence_only"] == 1.0
    assert good_score["reasoning"] == 1.0
    assert observation_score["observation"] == 0.0
    assert absence_score["observation"] == 0.0
    assert absence_score["observation_presence_only"] == 0.0
    assert walkthrough_score["walkthrough"] == 1.0
    assert walkthrough_score["reasoning"] == 0.0


def test_grpo_loss_weights_each_completion_equally() -> None:
    module = _load_vqa_train_script("train_vqa_lora_script_for_loss_test")
    current = torch.zeros((2, 4), requires_grad=True)
    old = torch.zeros_like(current)
    token_mask = torch.tensor([[True, False, False, False], [True, True, True, True]])
    advantages = torch.tensor([1.0, -1.0])

    loss = module.grpo_loss(
        current_logprobs=current,
        old_logprobs=old,
        token_mask=token_mask,
        advantages=advantages,
        clip_range=0.2,
    )

    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-7)


def test_centered_judge_rewards_keep_the_configured_small_scale() -> None:
    module = _load_vqa_train_script("train_vqa_lora_script_for_centered_reward_test")
    rewards = torch.tensor([0.0, 0.15, 0.10, 0.05])

    advantages = module.centered_group_rewards(rewards, num_generations=4)

    assert advantages.tolist() == pytest.approx([-0.075, 0.075, 0.025, -0.025])
    assert float(advantages.abs().max()) < 0.15


def test_completion_span_mask_selects_only_text_between_required_markers() -> None:
    module = _load_vqa_train_script("train_vqa_lora_script_for_span_mask_test")
    completion_ids = torch.tensor([[9, 10, 1, 2, 7, 8, 3], [9, 10, 1, 2, 3, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]])

    mask = module.completion_span_mask(
        completion_ids=completion_ids,
        completion_attention_mask=attention_mask,
        start_ids=[9, 10],
        end_ids=[7, 8],
    )

    assert mask.tolist() == [
        [False, False, True, True, False, False, False],
        [False, False, False, False, False, False, False],
    ]


def test_auxiliary_judge_advantage_has_no_answer_token_gradient() -> None:
    module = _load_vqa_train_script("train_vqa_lora_script_for_auxiliary_loss_test")
    current = torch.zeros((1, 4), requires_grad=True)
    old = torch.zeros_like(current)

    loss = module.grpo_loss(
        current_logprobs=current,
        old_logprobs=old,
        token_mask=torch.ones_like(current, dtype=torch.bool),
        advantages=torch.zeros(1),
        clip_range=0.2,
        auxiliary_terms=(
            (
                torch.tensor([0.1]),
                torch.tensor([[True, True, False, False]]),
            ),
        ),
    )
    loss.backward()

    assert current.grad is not None
    assert current.grad[0, :2].tolist() == pytest.approx([-0.05, -0.05])
    assert current.grad[0, 2:].tolist() == pytest.approx([0.0, 0.0])


def test_pathology_judge_bonus_uses_separate_observation_and_reasoning_weights() -> (
    None
):
    module = _load_vqa_train_script("train_vqa_lora_script_for_judge_reward_test")
    score_rows = [
        {
            "reward": 1.03,
            "correct": 1.0,
            "format": 1.0,
            "two_step": 1.0,
            "observation": 1.0,
            "reasoning": 1.0,
        },
        {
            "reward": 0.0,
            "correct": 0.0,
            "format": 1.0,
            "two_step": 1.0,
            "observation": 1.0,
            "reasoning": 1.0,
        },
    ]
    result = module.PathologyJudgeResult(
        scores=(0.75, 0.25),
        observation_scores=(1.0, 1.0),
        reasoning_scores=(0.5, 0.25),
        observation_support=(4, 2),
        observation_salience=(4, 1),
        reasoning_validity=(3, 1),
        reasoning_answer_alignment=(2, 1),
        issues=("", "unsupported architecture"),
        image_inventory=("Compact nests", "Fibrous stroma", "Small vessels"),
        cache_key="a" * 64,
        cache_hit=False,
        raw_inventory_response="{}",
        raw_response="{}",
    )

    module._apply_pathology_judge_result(
        score_rows=score_rows,
        result=result,
        observation_weight=0.10,
        reasoning_weight=0.05,
        observation_min_score=0.75,
        reasoning_min_score=0.5,
        latency_seconds=2.5,
    )

    assert score_rows[0]["reward"] == pytest.approx(1.155)
    assert score_rows[1]["reward"] == 0.0
    assert score_rows[0]["judge_success"] == 1.0
    assert score_rows[0]["judge_observation_support"] == 1.0
    assert score_rows[0]["judge_reasoning_answer_alignment"] == 0.5
    assert score_rows[0]["judge_observation_reward"] == pytest.approx(0.1)
    assert score_rows[0]["judge_reasoning_reward"] == pytest.approx(0.025)
    assert score_rows[1]["judge_observation_reward"] == 0.0
    assert score_rows[1]["judge_observation_reward_eligible"] == 0.0
    assert score_rows[1]["judge_reasoning_reward"] == 0.0


def test_pathology_judge_bonus_requires_strict_format() -> None:
    module = _load_vqa_train_script("train_vqa_lora_script_for_judge_format_gate_test")
    score_rows = [
        {
            "reward": 0.0,
            "correct": 1.0,
            "format": 0.0,
            "two_step": 1.0,
            "observation": 1.0,
            "reasoning": 1.0,
        }
    ]
    result = module.PathologyJudgeResult(
        scores=(1.0,),
        observation_scores=(1.0,),
        reasoning_scores=(1.0,),
        observation_support=(4,),
        observation_salience=(4,),
        reasoning_validity=(4,),
        reasoning_answer_alignment=(4,),
        issues=("",),
        image_inventory=("Compact nests", "Fibrous stroma", "Small vessels"),
        cache_key="a" * 64,
        cache_hit=False,
        raw_inventory_response="{}",
        raw_response="{}",
    )

    module._apply_pathology_judge_result(
        score_rows=score_rows,
        result=result,
        observation_weight=0.10,
        reasoning_weight=0.05,
        observation_min_score=0.75,
        reasoning_min_score=0.75,
        latency_seconds=1.0,
    )

    assert score_rows[0]["reward"] == 0.0
    assert score_rows[0]["judge_reward"] == 0.0
    assert score_rows[0]["judge_reward_eligible"] == 0.0


def test_pathology_judge_bonus_cannot_match_correctness_reward() -> None:
    from omegaconf import OmegaConf

    module = _load_vqa_train_script("train_vqa_lora_script_for_judge_weight_test")
    cfg = OmegaConf.create(
        {
            "reward_weights": {"correctness": 1.0},
            "visual_judge": {
                "enabled": True,
                "observation_weight": 0.7,
                "reasoning_weight": 0.3,
            },
        }
    )

    with pytest.raises(ValueError, match="smaller than the correctness reward"):
        module._pathology_judge_weights(cfg)


def test_text_pair_records_interleaved_prefix_spans() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.data import build_text_pair_with_prefix_spans

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(character) for character in str(text)]}

    input_ids, labels, spans, _ = build_text_pair_with_prefix_spans(
        FakeTokenizer(),
        row=_base_row(),
        prompt_cfg=OmegaConf.create(
            {
                "system_prompt": "Use the features.",
                "mcq_response_instruction": "Answer with the exact choice text.",
                "open_response_instruction": "Answer concisely.",
            }
        ),
        answer_text="mutated",
        max_text_length=4096,
    )

    assert [span["modality"] for span in spans] == ["pathology", "dnam"]
    assert all(
        labels[index] == -100
        for span in spans
        for index in range(int(span["start"]), int(span["end"]))
    )
    assert len(input_ids) == len(labels)


def test_generated_run_name_contains_basic_config_and_est_timestamp() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.stage_config import EST, generate_run_name

    cfg = OmegaConf.create(
        {
            "model_name_or_path": "google/gemma-4-E4B-it",
            "dataset": {"vqa_parquet_path": "/tmp/gt_mcq_questions.parquet"},
            "lora": {"r": 32},
            "projectors": {
                "pathology": {"enabled": True, "trainable": False},
                "radiology": {"enabled": True, "trainable": False},
                "dnam": {"enabled": True, "trainable": True},
                "rna": {"enabled": False, "trainable": False},
            },
        }
    )

    run_name = generate_run_name(
        cfg,
        train_rows=128,
        now=datetime(2026, 4, 27, 12, 30, 0, tzinfo=EST),
    )

    assert (
        run_name
        == "gemma_4_e4b_it_sft_gt_mcq_questions_n128_r32_projft_dnam_20260427_123000_EST"
    )


def test_prefix_cache_path_is_readable_and_project_relative(tmp_path) -> None:
    from kidney_vlm.vqa.prefix_cache import prefix_cache_path

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    cache_path = prefix_cache_path(
        repo_root=repo_root,
        cache_root="data/vqa/prefix_cache",
        model_name_or_path="Qwen/Qwen3.5-9B",
        modality="pathology",
        checkpoint_path="outputs/projectors/qwen3_5_9b/pathology/path_resampler_20260417_180638_EST/best.ckpt",
        feature_ref="data/features/features_uni/slide-a.h5",
    )

    assert cache_path.relative_to(repo_root).as_posix() == (
        "data/vqa/prefix_cache/qwen3_5_9b/"
        "pathology__path_resampler_20260417_180638_EST__best.ckpt/"
        "data__features__features_uni__slide-a.h5.pt"
    )


def test_prefix_cache_path_handles_readable_radiology_refs(tmp_path) -> None:
    from kidney_vlm.vqa.prefix_cache import prefix_cache_path

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    cache_path = prefix_cache_path(
        repo_root=repo_root,
        cache_root="data/vqa/prefix_cache",
        model_name_or_path="Qwen/Qwen3.5-9B",
        modality="radiology",
        checkpoint_path="outputs/projectors/qwen3_5_9b/radiology/radiology_remote_weights_20260423_000000_EST/best.ckpt",
        feature_ref="data/features/features_radiology/radiology_features.h5::series=data/processes/radiology/series-a",
    )

    assert cache_path.relative_to(repo_root).as_posix() == (
        "data/vqa/prefix_cache/qwen3_5_9b/"
        "radiology__radiology_remote_weights_20260423_000000_EST__best.ckpt/"
        "data__features__features_radiology__radiology_features.h5__ref/"
        "series=data__processes__radiology__series-a.pt"
    )


def test_prefix_cache_filter_skips_rows_with_missing_cache(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.data import filter_rows_with_prefix_cache, row_prefix_cache_path

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    cfg = OmegaConf.create(
        {
            "model_name_or_path": "Qwen/Qwen3.5-9B",
            "prefix_cache": {"enabled": True, "cache_root": "data/vqa/prefix_cache"},
            "projectors": {
                "pathology": {
                    "enabled": True,
                    "checkpoint_path": "outputs/projectors/qwen3_5_9b/pathology/path_resampler_20260417_180638_EST/best.ckpt",
                },
                "radiology": {"enabled": False, "checkpoint_path": "unused.ckpt"},
                "dnam": {"enabled": False, "checkpoint_path": "unused.ckpt"},
                "rna": {"enabled": False, "checkpoint_path": "unused.ckpt"},
            },
        }
    )
    present_row = _base_row(question_id=1, use_dnam=False, dnam_feature_path="")
    missing_row = _base_row(
        question_id=2,
        use_dnam=False,
        dnam_feature_path="",
        pathology_feature_paths=["data/features/missing.h5"],
    )
    cache_path = row_prefix_cache_path(
        repo_root, cfg, "pathology", "data/features/path.h5"
    )
    cache_path.parent.mkdir(parents=True)
    torch.save(torch.ones((2, 4), dtype=torch.float16), cache_path)

    filtered, skipped = filter_rows_with_prefix_cache(
        pd.DataFrame([present_row, missing_row]),
        root_dir=repo_root,
        stage_cfg=cfg,
    )

    assert filtered["question_id"].tolist() == [1]
    assert skipped[0]["question_id"] == 2
    assert "missing.h5.pt" in skipped[0]["missing"][0]


def test_train_script_can_skip_prefix_cache_prescan() -> None:
    import importlib.util

    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "06_vqa_train" / "train_vqa_lora.py"
    spec = importlib.util.spec_from_file_location(
        "train_vqa_lora_script_for_test", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frame = pd.DataFrame([_base_row(use_dnam=False, dnam_feature_path="")])
    cfg = OmegaConf.create(
        {
            "prefix_cache": {
                "enabled": True,
                "scan_before_training": False,
            }
        }
    )

    out = module._filter_missing_prefix_cache_rows(
        stage_cfg=cfg, frame=frame, split_label="train"
    )

    assert out.equals(frame)


def test_train_script_resolves_default_device_without_yaml_rank() -> None:
    import importlib.util

    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "06_vqa_train" / "train_vqa_lora.py"
    spec = importlib.util.spec_from_file_location(
        "train_vqa_lora_script_for_test_device", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = OmegaConf.create({"ddp": False})
    ddp_state = {
        "requested": False,
        "initialized": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "is_main": True,
    }

    device = module._resolve_training_device(cfg, ddp_state)

    if torch.cuda.is_available():
        assert str(device) == "cuda:0"
    else:
        assert str(device) == "cpu"


def test_train_script_ddp_requires_torchrun_env(monkeypatch) -> None:
    import importlib.util

    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "06_vqa_train" / "train_vqa_lora.py"
    spec = importlib.util.spec_from_file_location(
        "train_vqa_lora_script_for_test_ddp", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="torchrun"):
        module._init_ddp(OmegaConf.create({"ddp": True}))


def test_vqa_collator_loads_cached_prefixes_without_raw_features(
    tmp_path: Path,
) -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.data import VQATrainingCollator, row_prefix_cache_path

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(character) for character in str(text)]}

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    cfg = OmegaConf.create(
        {
            "model_name_or_path": "Qwen/Qwen3.5-9B",
            "max_text_length": 4096,
            "prefix_cache": {"enabled": True, "cache_root": "data/vqa/prefix_cache"},
            "prompt": {
                "system_prompt": "Use the features.",
                "mcq_response_instruction": "Answer with the exact choice text.",
                "open_response_instruction": "Answer concisely.",
            },
            "projectors": {
                "pathology": {
                    "enabled": True,
                    "checkpoint_path": "outputs/projectors/qwen3_5_9b/pathology/path_resampler_20260417_180638_EST/best.ckpt",
                },
                "radiology": {"enabled": False, "checkpoint_path": "unused.ckpt"},
                "dnam": {
                    "enabled": True,
                    "checkpoint_path": "outputs/projectors/qwen3_5_9b/dnam/dnam_mlp_20260420_025027_EST/best.ckpt",
                },
                "rna": {"enabled": False, "checkpoint_path": "unused.ckpt"},
            },
        }
    )
    for modality, ref, tensor in [
        ("pathology", "data/features/path.h5", torch.ones((2, 4), dtype=torch.float16)),
        ("dnam", "data/features/dnam.pt", torch.ones((3, 4), dtype=torch.float16)),
    ]:
        cache_path = row_prefix_cache_path(repo_root, cfg, modality, ref)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, cache_path)

    batch = VQATrainingCollator(
        tokenizer=FakeTokenizer(), root_dir=repo_root, stage_cfg=cfg
    )([_base_row()])

    assert "pathology_features" not in batch
    assert tuple(batch["pathology_prefix_embeddings"].shape) == (1, 2, 4)
    assert tuple(batch["dnam_prefix_embeddings"].shape) == (1, 3, 4)
    assert batch["pathology_prefix_embeddings"].dtype == torch.float16


def test_vqa_collator_caps_cached_pathology_prefix_tokens(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.data import VQATrainingCollator, row_prefix_cache_path

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(character) for character in str(text)]}

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    cfg = OmegaConf.create(
        {
            "model_name_or_path": "Qwen/Qwen3.5-9B",
            "max_text_length": 4096,
            "prefix_cache": {
                "enabled": True,
                "cache_root": "data/vqa/prefix_cache",
                "max_prefix_tokens": {"pathology": 3},
            },
            "prompt": {
                "system_prompt": "Use the features.",
                "mcq_response_instruction": "Answer with the exact choice text.",
                "open_response_instruction": "Answer concisely.",
            },
            "projectors": {
                "pathology": {
                    "enabled": True,
                    "checkpoint_path": "outputs/projectors/qwen3_5_9b/pathology/path_resampler_20260417_180638_EST/best.ckpt",
                },
                "radiology": {"enabled": False, "checkpoint_path": "unused.ckpt"},
                "dnam": {"enabled": False, "checkpoint_path": "unused.ckpt"},
                "rna": {"enabled": False, "checkpoint_path": "unused.ckpt"},
            },
        }
    )
    refs = ["data/features/path-a.h5", "data/features/path-b.h5"]
    for index, ref in enumerate(refs):
        cache_path = row_prefix_cache_path(repo_root, cfg, "pathology", ref)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            torch.full((2, 4), float(index + 1), dtype=torch.float16), cache_path
        )

    batch = VQATrainingCollator(
        tokenizer=FakeTokenizer(), root_dir=repo_root, stage_cfg=cfg
    )(
        [
            _base_row(
                use_dnam=False,
                dnam_feature_path="",
                pathology_feature_paths=refs,
            )
        ]
    )

    assert tuple(batch["pathology_prefix_embeddings"].shape) == (1, 3, 4)
    assert batch["pathology_prefix_mask"].tolist() == [[1, 1, 1]]


def test_vqa_model_replaces_placeholder_span_with_prefix_tokens() -> None:
    from types import SimpleNamespace

    from kidney_vlm.vqa.modeling import OncoVLMVQASFTModel

    class DummyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, vocab_size=16)
            self.embedding = nn.Embedding(256, 4)
            self.last_inputs_embeds_shape = None

        def get_input_embeddings(self):
            return self.embedding

        def forward(
            self, *, inputs_embeds, attention_mask, position_ids, labels=None, **kwargs
        ):
            self.last_inputs_embeds_shape = tuple(inputs_embeds.shape)
            return SimpleNamespace(
                loss=inputs_embeds.sum() * 0.0,
                logits=torch.zeros((*inputs_embeds.shape[:2], 16)),
            )

    class IdentityPrefixProjector(nn.Module):
        projector_type = "mlp"

        def forward(self, x, mask=None):
            return x, x.mean(dim=1)

        def build_output_mask(self, mask, *, batch_size, output_length, device, dtype):
            return mask.to(device=device, dtype=dtype)

    language_model = DummyLM()
    model = OncoVLMVQASFTModel(
        language_model=language_model,
        projectors={
            "pathology": nn.ModuleDict({"pathology": IdentityPrefixProjector()})
        },
        projector_metadata={"pathology": {"trainable": False}},
    )

    outputs = model(
        input_ids=torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        labels=torch.full((1, 5), -100, dtype=torch.long),
        pathology_features=torch.ones((1, 2, 4), dtype=torch.float32),
        pathology_feature_mask=torch.ones((1, 2), dtype=torch.long),
        prefix_spans=[[{"modality": "pathology", "start": 2, "end": 3}]],
    )

    assert outputs.loss is not None
    assert language_model.last_inputs_embeds_shape == (1, 6, 4)
    generation_inputs = model.prepare_interleaved_generation_inputs(
        input_ids=torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        pathology_features=torch.ones((1, 2, 4), dtype=torch.float32),
        pathology_feature_mask=torch.ones((1, 2), dtype=torch.long),
        prefix_spans=[[{"modality": "pathology", "start": 2, "end": 3}]],
    )

    assert tuple(generation_inputs["inputs_embeds"].shape) == (1, 6, 4)
    assert generation_inputs["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1]]
    assert generation_inputs["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5]]


def test_vqa_model_uses_cached_prefixes_without_projectors() -> None:
    from types import SimpleNamespace

    from kidney_vlm.vqa.modeling import OncoVLMVQASFTModel

    class DummyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, vocab_size=16)
            self.embedding = nn.Embedding(256, 4)
            self.last_inputs_embeds_shape = None

        def get_input_embeddings(self):
            return self.embedding

        def forward(
            self, *, inputs_embeds, attention_mask, position_ids, labels=None, **kwargs
        ):
            self.last_inputs_embeds_shape = tuple(inputs_embeds.shape)
            return SimpleNamespace(
                loss=inputs_embeds.sum() * 0.0,
                logits=torch.zeros((*inputs_embeds.shape[:2], 16)),
            )

    language_model = DummyLM()
    model = OncoVLMVQASFTModel(
        language_model=language_model, projectors={}, projector_metadata={}
    )

    outputs = model(
        input_ids=torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        labels=torch.full((1, 5), -100, dtype=torch.long),
        pathology_prefix_embeddings=torch.ones((1, 2, 4), dtype=torch.float32),
        pathology_prefix_mask=torch.ones((1, 2), dtype=torch.long),
        prefix_spans=[[{"modality": "pathology", "start": 2, "end": 3}]],
    )

    assert outputs.loss is not None
    assert language_model.last_inputs_embeds_shape == (1, 6, 4)
    generation_inputs = model.prepare_interleaved_generation_inputs(
        input_ids=torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        pathology_prefix_embeddings=torch.ones((1, 2, 4), dtype=torch.float32),
        pathology_prefix_mask=torch.ones((1, 2), dtype=torch.long),
        prefix_spans=[[{"modality": "pathology", "start": 2, "end": 3}]],
    )

    assert tuple(generation_inputs["inputs_embeds"].shape) == (1, 6, 4)


def test_vqa_generation_inputs_left_pad_batched_decoder_prompts() -> None:
    from types import SimpleNamespace

    from kidney_vlm.vqa.modeling import OncoVLMVQASFTModel

    class DummyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, vocab_size=16)
            self.embedding = nn.Embedding(256, 4)

        def get_input_embeddings(self):
            return self.embedding

    model = OncoVLMVQASFTModel(
        language_model=DummyLM(), projectors={}, projector_metadata={}
    )

    generation_inputs = model.prepare_interleaved_generation_inputs(
        input_ids=torch.tensor(
            [
                [11, 12, 13, 14, 15],
                [21, 22, 23, 0, 0],
            ],
            dtype=torch.long,
        ),
        attention_mask=torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.long,
        ),
        pathology_prefix_embeddings=torch.ones((2, 2, 4), dtype=torch.float32),
        pathology_prefix_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
        prefix_spans=[
            [{"modality": "pathology", "start": 2, "end": 3}],
            [{"modality": "pathology", "start": 1, "end": 2}],
        ],
    )

    assert tuple(generation_inputs["inputs_embeds"].shape) == (2, 6, 4)
    assert generation_inputs["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ]
    assert generation_inputs["position_ids"].tolist() == [
        [0, 1, 2, 3, 4, 5],
        [0, 0, 0, 0, 1, 2],
    ]


def test_vqa_generation_inputs_include_gemma4_per_layer_inputs() -> None:
    from types import SimpleNamespace

    from kidney_vlm.vqa.modeling import OncoVLMVQASFTModel

    class DummyGemma4LikeLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=4,
                vocab_size=32,
                hidden_size_per_layer_input=3,
                num_hidden_layers=2,
            )
            self.hidden_size_per_layer_input = 3
            self.embedding = nn.Embedding(256, 4)

        def get_input_embeddings(self):
            return self.embedding

        def get_per_layer_inputs(self, input_ids, inputs_embeds):
            assert inputs_embeds is None
            base = input_ids.to(dtype=torch.float32).view(*input_ids.shape, 1, 1)
            return base.expand(*input_ids.shape, 2, 3)

    model = OncoVLMVQASFTModel(
        language_model=DummyGemma4LikeLM(), projectors={}, projector_metadata={}
    )

    generation_inputs = model.prepare_interleaved_generation_inputs(
        input_ids=torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        pathology_prefix_embeddings=torch.ones((1, 2, 4), dtype=torch.float32),
        pathology_prefix_mask=torch.ones((1, 2), dtype=torch.long),
        prefix_spans=[[{"modality": "pathology", "start": 2, "end": 3}]],
    )

    assert tuple(generation_inputs["inputs_embeds"].shape) == (1, 5, 4)
    assert tuple(generation_inputs["per_layer_inputs"].shape) == (1, 5, 2, 3)
    assert torch.equal(generation_inputs["per_layer_inputs"][0, 2], torch.zeros((2, 3)))
    assert torch.equal(
        generation_inputs["per_layer_inputs"][0, 0], torch.full((2, 3), 11.0)
    )


def test_soft_prefix_generation_uses_gemma4_inner_text_model() -> None:
    from types import SimpleNamespace

    from kidney_vlm.vqa.modeling import generate_language_model_with_soft_prefix

    class DummyTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(
            self,
            *,
            input_ids=None,
            inputs_embeds=None,
            per_layer_inputs=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            return_dict=None,
        ):
            self.calls.append(
                {
                    "input_ids": None if input_ids is None else tuple(input_ids.shape),
                    "inputs_embeds": None
                    if inputs_embeds is None
                    else tuple(inputs_embeds.shape),
                    "per_layer_inputs": None
                    if per_layer_inputs is None
                    else tuple(per_layer_inputs.shape),
                    "attention_mask": tuple(attention_mask.shape),
                }
            )
            batch_size = attention_mask.shape[0]
            seq_len = (
                inputs_embeds.shape[1]
                if inputs_embeds is not None
                else input_ids.shape[1]
            )
            hidden = torch.zeros((batch_size, seq_len, 4))
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=object())

    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            token = 5 if self.calls == 0 else 2
            self.calls += 1
            logits = torch.zeros((*hidden_states.shape[:2], 8))
            logits[..., token] = 10.0
            return logits

    class DummyConditionalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, text_config=SimpleNamespace())
            self.model = SimpleNamespace(language_model=DummyTextModel())
            self.lm_head = DummyHead()

    language_model = DummyConditionalLM()
    generated = generate_language_model_with_soft_prefix(
        language_model,
        inputs={
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "inputs_embeds": torch.zeros((1, 3, 4)),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "per_layer_inputs": torch.zeros((1, 3, 2, 3)),
        },
        generation_kwargs={
            "max_new_tokens": 2,
            "do_sample": False,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
    )

    assert generated.tolist() == [[5, 2]]
    text_calls = language_model.model.language_model.calls
    assert text_calls[0]["input_ids"] is None
    assert text_calls[0]["inputs_embeds"] == (1, 3, 4)
    assert text_calls[0]["per_layer_inputs"] == (1, 3, 2, 3)
    assert text_calls[1]["input_ids"] == (1, 1)
    assert text_calls[1]["inputs_embeds"] is None
    assert text_calls[1]["per_layer_inputs"] is None
