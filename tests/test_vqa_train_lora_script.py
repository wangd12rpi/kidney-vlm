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
    cache_path = row_prefix_cache_path(repo_root, cfg, "pathology", "data/features/path.h5")
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
    spec = importlib.util.spec_from_file_location("train_vqa_lora_script_for_test", script_path)
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

    out = module._filter_missing_prefix_cache_rows(stage_cfg=cfg, frame=frame, split_label="train")

    assert out.equals(frame)


def test_train_script_resolves_default_device_without_yaml_rank() -> None:
    import importlib.util

    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "06_vqa_train" / "train_vqa_lora.py"
    spec = importlib.util.spec_from_file_location("train_vqa_lora_script_for_test_device", script_path)
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
    spec = importlib.util.spec_from_file_location("train_vqa_lora_script_for_test_ddp", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="torchrun"):
        module._init_ddp(OmegaConf.create({"ddp": True}))


def test_vqa_collator_loads_cached_prefixes_without_raw_features(tmp_path: Path) -> None:
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

    batch = VQATrainingCollator(tokenizer=FakeTokenizer(), root_dir=repo_root, stage_cfg=cfg)([_base_row()])

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
        torch.save(torch.full((2, 4), float(index + 1), dtype=torch.float16), cache_path)

    batch = VQATrainingCollator(tokenizer=FakeTokenizer(), root_dir=repo_root, stage_cfg=cfg)(
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

        def forward(self, *, inputs_embeds, attention_mask, position_ids, labels=None, **kwargs):
            self.last_inputs_embeds_shape = tuple(inputs_embeds.shape)
            return SimpleNamespace(loss=inputs_embeds.sum() * 0.0, logits=torch.zeros((*inputs_embeds.shape[:2], 16)))

    language_model = DummyLM()
    model = OncoVLMVQASFTModel(language_model=language_model, projectors={}, projector_metadata={})

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

    model = OncoVLMVQASFTModel(language_model=DummyLM(), projectors={}, projector_metadata={})

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
