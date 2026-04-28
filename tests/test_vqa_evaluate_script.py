from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _load_eval_script():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "07_vqa_evaluation"
        / "evaluate_vqa.py"
    )
    spec = importlib.util.spec_from_file_location(
        "kidney_vlm_test_evaluate_vqa_script", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prompt_cfg() -> dict[str, object]:
    return {
        "prompts": {
            "mcq": {
                "system_prompt": "You are OncoVLM. Use only the provided projector features.",
                "response_instruction": "Answer with the exact choice text.",
            },
            "qa": {
                "system_prompt": "You are OncoVLM. Use only the provided projector features.",
                "response_instruction": "Answer concisely.",
            },
        }
    }


def _projector_model_cfg() -> dict[str, object]:
    return {
        "backend": "oncovlm_projector",
        "model_name_or_path": "Qwen/Qwen3.5-9B",
        "device": "cpu",
        "projectors": {
            "pathology": {"checkpoint_path": "pathology/best.ckpt"},
            "radiology": {"checkpoint_path": "radiology/best.ckpt"},
            "dnam": {"checkpoint_path": "dnam/best.ckpt"},
            "rna": {"checkpoint_path": "rna/best.ckpt"},
        },
    }


def test_azure_backend_dry_run_smoke() -> None:
    module = _load_eval_script()
    backend = module._build_backend(
        {
            "backend": "azure_openai_gpt",
            "azure_openai": {"api_key_env": "MISSING_KEY", "deployment": "gpt-5.1"},
        },
        eval_cfg=_prompt_cfg(),
        dry_run=True,
    )

    assert (
        backend.generate(
            system_prompt="system",
            user_prompt="user",
            image_paths=[],
            generation_kwargs={},
        )
        == '{"answer": "", "rationale": "dry run"}'
    )
    assert backend.generate_batch(
        requests=[{"system_prompt": "system", "user_prompt": "user", "image_paths": []}, {}],
        generation_kwargs={},
    ) == [
        '{"answer": "", "rationale": "dry run"}',
        '{"answer": "", "rationale": "dry run"}',
    ]


def test_hf_image_text_backend_dry_run_smoke() -> None:
    module = _load_eval_script()
    backend = module._build_backend(
        {
            "backend": "hf_image_text_to_text",
            "model_name_or_path": "google/medgemma-4b-it",
            "device": "cpu",
        },
        eval_cfg=_prompt_cfg(),
        dry_run=True,
    )

    assert (
        backend.generate(
            system_prompt="system",
            user_prompt="user",
            image_paths=[],
            generation_kwargs={},
        )
        == '{"answer": "", "rationale": "dry run"}'
    )
    assert backend.generate_batch(
        requests=[{"system_prompt": "system", "user_prompt": "user", "image_paths": []}, {}],
        generation_kwargs={},
    ) == [
        '{"answer": "", "rationale": "dry run"}',
        '{"answer": "", "rationale": "dry run"}',
    ]


def test_hf_image_text_backend_passes_8bit_load_kwargs(monkeypatch) -> None:
    module = _load_eval_script()
    calls: dict[str, object] = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["processor_model_name"] = model_name
            calls["processor_kwargs"] = kwargs
            return cls()

    class FakeModel:
        device = module.torch.device("cuda:0")

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["model_model_name"] = model_name
            calls["model_kwargs"] = kwargs
            return cls()

        def to(self, device):
            calls["to_device"] = device

        def eval(self):
            calls["eval"] = True

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForImageTextToText=FakeModel,
            AutoProcessor=FakeProcessor,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
        ),
    )

    module.HFImageTextBackend(
        {
            "backend": "hf_image_text_to_text",
            "model_name_or_path": "google/medgemma-4b-it",
            "device": "cuda:0",
            "load_in_8bit": True,
            "torch_dtype": "bfloat16",
        },
        dry_run=False,
    )

    assert calls["model_model_name"] == "google/medgemma-4b-it"
    quantization_config = calls["model_kwargs"].pop("quantization_config")
    assert isinstance(quantization_config, FakeBitsAndBytesConfig)
    assert quantization_config.kwargs == {"load_in_8bit": True}
    assert calls["model_kwargs"] == {
        "trust_remote_code": True,
        "dtype": module.torch.bfloat16,
        "device_map": {"": 0},
    }
    assert "to_device" not in calls
    assert calls["eval"] is True


def test_oncovlm_projector_backend_dry_run_requires_all_projectors() -> None:
    module = _load_eval_script()
    backend = module._build_backend(
        _projector_model_cfg(),
        eval_cfg=_prompt_cfg(),
        dry_run=True,
    )

    raw_response, prompt = backend.generate(row={}, generation_kwargs={})

    assert raw_response == '{"answer": "", "rationale": "dry run"}'
    assert prompt == ""
    batched = backend.generate_batch(rows=[{}, {}], generation_kwargs={})
    assert batched == [
        ('{"answer": "", "rationale": "dry run"}', ""),
        ('{"answer": "", "rationale": "dry run"}', ""),
    ]


def test_eval_model_batch_size_helpers() -> None:
    module = _load_eval_script()

    assert module._model_batch_size({"display_name": "m", "batch_size": 3}) == 3
    assert module._print_first_n_outputs({"print_first_n_outputs": 2}) == 2
    assert module._batched_records(
        [{"i": 1}, {"i": 2}, {"i": 3}, {"i": 4}, {"i": 5}],
        2,
    ) == [[{"i": 1}, {"i": 2}], [{"i": 3}, {"i": 4}], [{"i": 5}]]

    with pytest.raises(ValueError, match="batch_size=0"):
        module._model_batch_size({"display_name": "m", "batch_size": 0})
    with pytest.raises(ValueError, match="print_first_n_outputs"):
        module._print_first_n_outputs({"print_first_n_outputs": -1})


def test_write_predictions_uses_resume_existing_as_single_switch(tmp_path) -> None:
    module = _load_eval_script()
    predictions_path = tmp_path / "predictions.parquet"
    existing = pd.DataFrame(
        [
            {"question_id": 1, "project_id": "TCGA-A", "case_id": "case-a", "task_id": "old", "raw_response": "old"},
        ]
    )
    existing.to_parquet(predictions_path, index=False)

    resumed = module._write_predictions(
        predictions_path,
        [
            {"question_id": 2, "project_id": "TCGA-A", "case_id": "case-b", "task_id": "new", "raw_response": "new"},
        ],
        resume_existing=True,
    )

    assert resumed["question_id"].tolist() == [1, 2]

    replaced = module._write_predictions(
        predictions_path,
        [
            {"question_id": 3, "project_id": "TCGA-A", "case_id": "case-c", "task_id": "rerun", "raw_response": "rerun"},
        ],
        resume_existing=False,
    )

    assert replaced["question_id"].tolist() == [3]
    assert pd.read_parquet(predictions_path)["question_id"].tolist() == [3]


def test_prompt_token_ids_passes_explicit_thinking_flag() -> None:
    module = _load_eval_script()

    class DummyTokenizer:
        def __init__(self):
            self.enable_thinking = None

        def apply_chat_template(self, messages, *, enable_thinking, **kwargs):
            self.enable_thinking = enable_thinking
            return [1, 2, 3]

    tokenizer = DummyTokenizer()

    assert module._prompt_token_ids(tokenizer, "prompt", enable_thinking=False) == [1, 2, 3]
    assert tokenizer.enable_thinking is False


def test_output_preview_prints_model_output_and_ground_truth(capsys) -> None:
    module = _load_eval_script()

    module._print_output_preview(
        preview_index=1,
        preview_limit=2,
        row={
            "question_id": 7,
            "question": "What is the answer?",
            "answer": "ground truth answer",
            "answer_label": "A",
        },
        raw_response="raw model output",
        parsed={"predicted_answer": "parsed output", "predicted_answer_label": "B"},
        model_cfg={"display_name": "oncovlm_qwen_no_finetune"},
    )

    output = capsys.readouterr().out
    assert "[VQA preview 1/2]" in output
    assert "model=oncovlm_qwen_no_finetune" in output
    assert "Q: What is the answer?" in output
    assert "GT: ground truth answer" in output
    assert "OUT: raw model output" in output
    assert "parsed: parsed output" in output
    assert "labels gt=A pred=B" in output


def test_hf_text_only_backend_is_intentionally_unsupported() -> None:
    module = _load_eval_script()

    with pytest.raises(NotImplementedError, match="intentionally not supported"):
        module._build_backend(
            {
                "backend": "hf_causal_lm",
                "model_name_or_path": "google/gemma",
                "device": "cpu",
            },
            eval_cfg=_prompt_cfg(),
            dry_run=True,
        )
