from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _load_rationale_script():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "21_cot_rationale_gen"
        / "generate_vqa_rationales.py"
    )
    spec = importlib.util.spec_from_file_location(
        "kidney_vlm_test_generate_vqa_rationales_script", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(**overrides):
    row = {
        "case_id": "case-1",
        "project_id": "TCGA-KIRC",
        "question_id": 1,
        "question_type": "mcq",
        "generation_type": "from_caption",
        "task_category": "caption",
        "task_id": "pathology_findings",
        "modality_combination_name": "all_available",
        "use_pathology": True,
        "use_radiology": False,
        "use_dnam": False,
        "use_rna": False,
        "question": "Which pathology finding is best supported?",
        "option_a": "clear cell morphology",
        "option_b": "papillary architecture",
        "option_c": "squamous differentiation",
        "option_d": "mucinous glands",
        "answer": "clear cell morphology",
        "answer_label": "A",
    }
    row.update(overrides)
    return row


def test_real_cot_rationale_prompt_is_image_only_and_hides_gold_answer() -> None:
    module = _load_rationale_script()
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1]
        / "conf"
        / "21_cot_rationale_gen"
        / "generate_vqa_rationales.yaml"
    )

    _, user_prompt = module._build_prompt(
        row=_row(),
        prompts_cfg=OmegaConf.to_container(cfg.prompts, resolve=False),
    )

    assert "- clear cell morphology" in user_prompt
    assert "- papillary architecture" in user_prompt
    assert "A. clear cell morphology" not in user_prompt
    assert "Target answer:" not in user_prompt
    assert "<answer>\nclear cell morphology\n</answer>" not in user_prompt
    assert "Step 1 — Observation" in user_prompt
    assert "Step 2 — Reasoning" in user_prompt
    assert "caption" in user_prompt.lower()
    assert "{{answer_label}}" not in user_prompt


def test_cot_rationale_validation_requires_two_steps_and_exact_answer() -> None:
    module = _load_rationale_script()
    cfg = {
        "enabled": True,
        "min_think_words": 1,
        "max_think_words": 80,
        "require_two_steps": True,
    }
    valid = (
        "<think>Step 1 — Observation: Clear cytoplasm and nested architecture are visible. "
        "Step 2 — Reasoning: Their combination supports a clear-cell growth pattern.</think>"
        "<answer>clear cell morphology</answer>"
    )
    choices = [
        "clear cell morphology",
        "papillary architecture",
        "squamous differentiation",
        "mucinous glands",
    ]

    assert module._validation_error(valid, "clear cell morphology", cfg, choices) == ""
    mismatch = valid.replace(
        "<answer>clear cell morphology</answer>",
        "<answer>papillary architecture</answer>",
    )
    assert module._validation_error(mismatch, "clear cell morphology", cfg, choices) == "answer_mismatch"
    assert "clear cell morphology" not in module._validation_error(
        mismatch,
        "clear cell morphology",
        cfg,
        choices,
    )


def test_retry_attempts_are_independent_and_keep_first_correct_response(monkeypatch) -> None:
    module = _load_rationale_script()
    wrong = (
        "<think>Step 1 — Observation: Clear cytoplasm and nested architecture are visible. "
        "Step 2 — Reasoning: The architecture is interpreted as papillary.</think>"
        "<answer>papillary architecture</answer>"
    )
    correct = wrong.replace(
        "<answer>papillary architecture</answer>",
        "<answer>clear cell morphology</answer>",
    )
    responses = [wrong, correct]
    prompts = []

    def fake_call(**kwargs):
        prompts.append(kwargs["user_prompt"])
        return responses.pop(0)

    monkeypatch.setattr(module, "_call_azure_gpt", fake_call)
    request = module.RationaleRequest(
        row_index=0,
        question_id=1,
        prompt_row=_row(),
        image_paths=[],
        expected_answer="clear cell morphology",
    )
    prompt_cfg = {
        "mcq": {
            "system_prompt": "Solve from images only.",
            "user_template": (
                "{{question}}\n- {{option_a}}\n- {{option_b}}\n- {{option_c}}\n- {{option_d}}"
            ),
        }
    }
    result = module._generate_rationale_for_request(
        request,
        client=object(),
        deployment="test",
        azure_cfg={},
        validation_cfg={
            "enabled": True,
            "min_think_words": 1,
            "max_think_words": 80,
            "require_two_steps": True,
        },
        image_cfg={},
        prompts_cfg=prompt_cfg,
        teacher_attempts_cfg={"max_attempts": 3},
    )

    assert result.rationale == correct
    assert len(result.failed_attempts) == 1
    assert result.failed_attempts[0]["error_type"] == "answer_mismatch"
    assert result.failed_attempts[0]["teacher_answer"] == "papillary architecture"
    assert len(prompts) == 2
    assert prompts[0] != prompts[1]


def test_failed_attempt_path_is_below_dataset_directory() -> None:
    module = _load_rationale_script()
    output = Path("/tmp/rationales/pilot.parquet")
    assert module._error_attempts_path(output, {"error_attempts_subdir": "errors"}) == Path(
        "/tmp/rationales/errors/pilot_failed_attempts.parquet"
    )


def test_rationale_output_rows_omits_failed_generation_rows() -> None:
    module = _load_rationale_script()

    completed = _row(question_id=1)
    completed["rationale"] = "<think>usable rationale</think><answer>clear cell morphology</answer>"
    failed = _row(question_id=2)
    failed["rationale"] = ""

    output_rows = module._rationale_output_rows([completed, failed])

    assert [row["question_id"] for row in output_rows] == [1]
