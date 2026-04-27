from __future__ import annotations

from datetime import datetime

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


def test_select_train_rows_keeps_modality_dropout_and_skips_disabled_modalities() -> None:
    from omegaconf import OmegaConf

    from kidney_vlm.vqa.data import select_vqa_rows

    rows = [
        _base_row(question_id=1, modality_combination_name="all_available"),
        _base_row(question_id=2, modality_combination_name="path_only", use_dnam=False, dnam_feature_path=""),
        _base_row(question_id=3, split="val"),
        _base_row(question_id=4, use_rna=True, rna_feature_path="data/features/rna.pt"),
    ]
    cfg = OmegaConf.create(
        {
            "dataset": {
                "question_types": ["mcq", "open_ended"],
                "generation_types": [],
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
    assert all(labels[index] == -100 for span in spans for index in range(int(span["start"]), int(span["end"])))
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

    assert run_name == "gemma_4_e4b_it_gt_mcq_questions_n128_r32_projft_dnam_20260427_123000_EST"


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

        def forward(self, *, inputs_embeds, attention_mask, position_ids, labels=None, **kwargs):
            self.last_inputs_embeds_shape = tuple(inputs_embeds.shape)
            return SimpleNamespace(loss=inputs_embeds.sum() * 0.0, logits=torch.zeros((*inputs_embeds.shape[:2], 16)))

    class IdentityPrefixProjector(nn.Module):
        projector_type = "mlp"

        def forward(self, x, mask=None):
            return x, x.mean(dim=1)

        def build_output_mask(self, mask, *, batch_size, output_length, device, dtype):
            return mask.to(device=device, dtype=dtype)

    language_model = DummyLM()
    model = OncoVLMVQASFTModel(
        language_model=language_model,
        projectors={"pathology": nn.ModuleDict({"pathology": IdentityPrefixProjector()})},
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
