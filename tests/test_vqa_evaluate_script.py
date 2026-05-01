from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _load_generate_script():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "07_vqa_evaluation"
        / "generate_vqa_predictions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "kidney_vlm_test_generate_vqa_predictions_script", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_score_script():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "07_vqa_evaluation"
        / "score_vqa_predictions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "kidney_vlm_test_score_vqa_predictions_script", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prompt_cfg() -> dict[str, object]:
    return {
        "prefix_cache": {
            "enabled": True,
            "cache_root": "data/vqa/prefix_cache",
            "scan_before_training": True,
            "skip_missing_rows": True,
            "max_missing_examples": 10,
            "max_prefix_tokens": {"pathology": 512, "radiology": 256, "dnam": None, "rna": None},
        },
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
    }


def test_azure_backend_dry_run_smoke() -> None:
    module = _load_generate_script()
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
    module = _load_generate_script()
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
    module = _load_generate_script()
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


def test_hf_image_text_backend_passes_padding_in_processor_kwargs(monkeypatch) -> None:
    module = _load_generate_script()
    calls: dict[str, object] = {}

    class FakeInputs(dict):
        def to(self, device):
            calls["inputs_to_device"] = device
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls["chat_template_kwargs"] = kwargs
            return FakeInputs({"input_ids": module.torch.tensor([[1, 2]])})

        def batch_decode(self, generated, skip_special_tokens):
            calls["generated"] = generated.tolist()
            return ["answer"]

    class FakeModel:
        device = module.torch.device("cpu")

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()

        def to(self, device):
            self.device = device

        def eval(self):
            return None

        def generate(self, **inputs):
            return module.torch.tensor([[1, 2, 3]])

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForImageTextToText=FakeModel,
            AutoProcessor=FakeProcessor,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
        ),
    )

    backend = module.HFImageTextBackend(
        {
            "backend": "hf_image_text_to_text",
            "model_name_or_path": "google/medgemma-4b-it",
            "device": "cpu",
        },
        dry_run=False,
    )
    assert backend.generate_batch(
        requests=[{"system_prompt": "system", "user_prompt": "user", "image_paths": []}],
        generation_kwargs={},
    ) == ["answer"]

    assert calls["chat_template_kwargs"]["processor_kwargs"] == {"padding": True}
    assert "padding" not in calls["chat_template_kwargs"]


def test_oncovlm_projector_backend_dry_run_is_cache_only() -> None:
    module = _load_generate_script()
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


def test_oncovlm_lora_backend_dry_run_uses_cache_backend() -> None:
    module = _load_generate_script()
    model_cfg = _projector_model_cfg()
    model_cfg["backend"] = "oncovlm_lora"
    model_cfg["lora_adapter_path"] = "outputs/oncovlm/qwen/lora_adapter"

    backend = module._build_backend(
        model_cfg,
        eval_cfg=_prompt_cfg(),
        dry_run=True,
    )

    raw_response, prompt = backend.generate(row={}, generation_kwargs={})

    assert raw_response == '{"answer": "", "rationale": "dry run"}'
    assert prompt == ""


def test_eval_model_batch_size_helpers() -> None:
    module = _load_generate_script()

    assert module._model_batch_size({"display_name": "m", "batch_size": 3}) == 3
    assert module._print_first_n_outputs({"print_first_n_outputs": 2}) == 2
    assert module._save_every_n_predictions({"save_every_n_predictions": 100}) == 100
    assert module._save_every_n_predictions({"save_every_n_predictions": 0}) == 0
    assert module._batched_records(
        [{"i": 1}, {"i": 2}, {"i": 3}, {"i": 4}, {"i": 5}],
        2,
    ) == [[{"i": 1}, {"i": 2}], [{"i": 3}, {"i": 4}], [{"i": 5}]]

    with pytest.raises(ValueError, match="batch_size=0"):
        module._model_batch_size({"display_name": "m", "batch_size": 0})
    with pytest.raises(ValueError, match="print_first_n_outputs"):
        module._print_first_n_outputs({"print_first_n_outputs": -1})
    with pytest.raises(ValueError, match="save_every_n_predictions"):
        module._save_every_n_predictions({"save_every_n_predictions": -1})


def test_sort_generation_rows_puts_open_ended_before_mcq() -> None:
    module = _load_generate_script()
    frame = pd.DataFrame(
        [
            {
                "question_id": 3,
                "question_type": "mcq",
                "generation_type": "from_ground_truth",
                "task_category": "mutation",
                "task_id": "mutation_tp53",
                "project_id": "TCGA-BRCA",
                "case_id": "case-b",
            },
            {
                "question_id": 2,
                "question_type": "qa",
                "generation_type": "from_caption",
                "task_category": "pathology_description",
                "task_id": "CO2",
                "project_id": "TCGA-BRCA",
                "case_id": "case-a",
            },
            {
                "question_id": 1,
                "question_type": "mcq",
                "generation_type": "from_caption",
                "task_category": "caption_mcq",
                "task_id": "CO1",
                "project_id": "TCGA-BRCA",
                "case_id": "case-a",
            },
        ]
    )

    sorted_frame = module._sort_generation_rows(frame)

    assert sorted_frame["question_id"].tolist() == [2, 1, 3]


def test_write_predictions_uses_resume_existing_as_single_switch(tmp_path) -> None:
    module = _load_generate_script()
    predictions_path = tmp_path / "predictions.parquet"
    existing = pd.DataFrame(
        [
            {
                "model_display_name": "model_a",
                "question_id": 1,
                "repeat_id": 0,
                "project_id": "TCGA-A",
                "case_id": "case-a",
                "task_id": "old",
                "raw_response": "old",
            },
        ]
    )
    existing.to_parquet(predictions_path, index=False)

    resumed = module._write_predictions(
        predictions_path,
        [
            {
                "model_display_name": "model_b",
                "question_id": 1,
                "repeat_id": 0,
                "project_id": "TCGA-A",
                "case_id": "case-b",
                "task_id": "new",
                "raw_response": "new",
            },
        ],
        resume_existing=True,
    )

    assert resumed[["model_display_name", "question_id", "repeat_id"]].values.tolist() == [
        ["model_a", 1, 0],
        ["model_b", 1, 0],
    ]

    replaced = module._write_predictions(
        predictions_path,
        [
            {
                "model_display_name": "model_a",
                "question_id": 3,
                "repeat_id": 0,
                "project_id": "TCGA-A",
                "case_id": "case-c",
                "task_id": "rerun",
                "raw_response": "rerun",
            },
        ],
        resume_existing=False,
    )

    assert replaced["question_id"].tolist() == [3]
    assert pd.read_parquet(predictions_path)["question_id"].tolist() == [3]


def test_should_generate_row_skips_missing_genomics_text_with_one_line_warning(capsys) -> None:
    module = _load_generate_script()

    keep = module._should_generate_row(
        {"question_id": 1, "use_dnam": True, "dnam_text_summary": "DNA text", "use_rna": False},
        {"display_name": "m"},
    )
    skip = module._should_generate_row(
        {"question_id": 2, "use_dnam": True, "dnam_text_summary": "", "use_rna": False},
        {"display_name": "m"},
    )

    output = capsys.readouterr().out.strip().splitlines()
    assert keep is True
    assert skip is False
    assert output == [
        "Warning: skipping VQA row qid=2 model=m: enabled DNAm/RNA has empty fallback text (dnam_text_summary)."
    ]


def test_prediction_row_leaves_open_ended_correct_empty() -> None:
    module = _load_generate_script()
    row = {
        "question_id": 1,
        "repeat_id": 0,
        "base_question_id": 1,
        "case_id": "case-a",
        "project_id": "TCGA-A",
        "split": "test",
        "question_type": "qa",
        "generation_type": "from_caption",
        "task_category": "pathology_description",
        "task_id": "CO2",
        "modality_combination_name": "path_only",
        "use_pathology": True,
        "use_radiology": False,
        "use_dnam": False,
        "use_rna": False,
        "question": "Describe the pathology.",
        "option_a": "",
        "option_b": "",
        "option_c": "",
        "option_d": "",
        "answer": "reference answer",
        "answer_label": "",
    }

    output = module._prediction_row(
        row=row,
        parsed={"predicted_answer": "reference answer", "predicted_answer_label": "", "parse_status": "raw"},
        raw_response="reference answer",
        image_paths=[],
        model_cfg={"backend": "hf_image_text_to_text", "display_name": "m", "model_name_or_path": "m"},
        evaluated_at="now",
        system_prompt="system",
        user_prompt="user",
        include_prompt=False,
    )

    assert output["correct"] is None


def test_prompt_token_ids_passes_explicit_thinking_flag() -> None:
    module = _load_generate_script()

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
    module = _load_generate_script()

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
    module = _load_generate_script()

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


def test_score_script_uses_one_run_level_prediction_parquet(tmp_path) -> None:
    module = _load_score_script()
    run_root = tmp_path / "results" / "smoke"
    run_root.mkdir(parents=True)
    predictions_path = run_root / "predictions.parquet"
    pd.DataFrame(
        [
            {
                "model_display_name": "oncovlm_qwen_no_finetune",
                "backend": "oncovlm_projector",
                "model_name_or_path": "Qwen/Qwen3.5-9B",
                "raw_response": "answer",
            }
        ]
    ).to_parquet(predictions_path, index=False)

    cfg = {
        "run": {
            "name": "smoke",
            "output_root": str(tmp_path / "results"),
        }
    }

    assert module._prediction_path(cfg, "predictions.parquet") == predictions_path
