from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from kidney_vlm.vqa.eval_gpt import (
    _patch_bert_score_tokenizer_max_length,
    add_bertscore_columns,
    apply_group_sampling,
    build_mcq_prompt,
    build_flat_metric_records,
    collect_required_image_paths,
    enabled_model_configs,
    parse_model_response,
    parse_mcq_response,
    prompt_cfg_for_model,
    select_eval_rows,
)


def _row(**overrides):
    row = {
        "case_id": "TCGA-AA-0001",
        "project_id": "TCGA-BRCA",
        "question_id": 1,
        "base_question_id": 1,
        "split": "test",
        "question_type": "mcq",
        "generation_type": "from_ground_truth",
        "task_category": "mutation",
        "task_id": "mutation_tp53",
        "modality_combination_name": "dnam_rna",
        "use_pathology": False,
        "use_radiology": False,
        "use_dnam": True,
        "use_rna": True,
        "question": "What is the TP53 mutation status for this case?",
        "option_a": "TP53 mutation present",
        "option_b": "TP53 mutation absent",
        "option_c": "",
        "option_d": "",
        "answer": "TP53 mutation present",
        "answer_label": "A",
        "caption_id": "",
        "ground_truth_source": "mutation_tp53",
        "radiology_biomarker": "",
        "pathology_feature_paths": [],
        "radiology_feature_paths": [],
        "dnam_feature_path": "dnam.pt",
        "rna_feature_path": "rna.pt",
        "pathology_roi_png_dir": "",
        "radiology_view_png_dir": "",
        "dnam_text_summary": "DNA methylation raw beta summary: mean beta 0.4.",
        "rna_text_summary": "RNA-seq raw expression summary: median TPM 3.",
    }
    row.update(overrides)
    return row


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


def test_build_mcq_prompt_uses_semantic_options_without_letters() -> None:
    _, user_prompt = build_mcq_prompt(
        _row(),
        _prompt_cfg(),
    )

    assert "- TP53 mutation present" in user_prompt
    assert "- TP53 mutation absent" in user_prompt
    assert "<choices>" in user_prompt
    assert "A. TP53" not in user_prompt
    assert "answer_letter" not in user_prompt


def test_build_mcq_prompt_marks_required_images_without_fake_text() -> None:
    _, user_prompt = build_mcq_prompt(
        _row(
            use_pathology=True,
            use_radiology=True,
            use_dnam=False,
            use_rna=False,
            radiology_biomarker="Longest diameter 24 mm; heterogeneity present.",
        ),
        _prompt_cfg(),
    )

    assert "<pathology_images>attached</pathology_images>" in user_prompt
    assert "<radiology_images>attached</radiology_images>" in user_prompt
    assert "[not provided]" not in user_prompt


def test_build_mcq_prompt_can_ignore_missing_radiology_biomarker() -> None:
    cfg = _prompt_cfg()
    cfg["image_inputs"] = {"ignore_radiology_biomarker": True}

    _, user_prompt = build_mcq_prompt(
        _row(
            use_pathology=True,
            use_radiology=True,
            use_dnam=False,
            use_rna=False,
            radiology_biomarker="",
        ),
        cfg,
    )

    assert "<radiology_images>attached</radiology_images>" in user_prompt
    assert "<radiology_biomarker>" not in user_prompt


def test_build_mcq_prompt_fails_on_missing_required_text() -> None:
    with pytest.raises(ValueError, match="dnam_text_summary"):
        build_mcq_prompt(
            _row(dnam_text_summary=""),
            _prompt_cfg(),
        )


def test_build_mcq_prompt_fails_when_no_modalities_are_enabled() -> None:
    with pytest.raises(ValueError, match="no enabled modalities"):
        build_mcq_prompt(
            _row(use_dnam=False, use_rna=False),
            _prompt_cfg(),
        )


def test_parse_mcq_response_requires_semantic_answer_not_letter() -> None:
    options = ["TP53 mutation present", "TP53 mutation absent"]

    assert parse_mcq_response('{"answer": "TP53 mutation absent"}', options) == {
        "predicted_answer": "TP53 mutation absent",
        "parse_status": "exact",
    }
    assert parse_mcq_response('{"answer": "A"}', options) == {
        "predicted_answer": "",
        "parse_status": "failed",
    }
    assert parse_mcq_response(
        '{"answer": "A", "rationale": "TP53 mutation absent"}', options
    ) == {
        "predicted_answer": "",
        "parse_status": "failed",
    }


def test_select_eval_rows_uses_explicit_enabled_filter_blocks() -> None:
    frame = pd.DataFrame(
        [
            _row(question_id=1, task_id="mutation_tp53"),
            _row(question_id=2, task_id="mutation_tp53", use_pathology=True),
            _row(question_id=3, task_id="mutation_egfr", project_id="TCGA-LUAD"),
            _row(question_id=4, task_id="mutation_tp53", dnam_text_summary=""),
        ]
    )
    cfg = {
        "filters": {
            "split": {"enabled": True, "value": "test"},
            "question_types": {"enabled": True, "values": ["mcq"]},
            "generation_types": {"enabled": True, "values": ["from_ground_truth"]},
            "project_ids": {"enabled": True, "values": ["TCGA-BRCA"]},
            "task_ids": {"enabled": True, "values": ["mutation_tp53"]},
            "task_categories": {"enabled": False, "values": []},
            "row_limit": {"enabled": False, "max_rows": 1},
        }
    }

    selected = select_eval_rows(frame, cfg)

    assert selected["question_id"].tolist() == [1, 2, 4]


def test_select_eval_rows_filters_allowed_modality_combos() -> None:
    frame = pd.DataFrame(
        [
            _row(question_id=1, use_pathology=False, use_radiology=False),
            _row(question_id=2, use_pathology=True, use_radiology=False),
            _row(
                question_id=3,
                use_pathology=True,
                use_radiology=True,
                radiology_biomarker="Longest diameter 24 mm.",
            ),
        ]
    )
    cfg = {
        "filters": {
            "allowed_modality_combo": {
                "enabled": True,
                "values": [
                    {"path": False, "rad": False, "dnam": True, "rna": True},
                    {"path": True, "rad": True, "dnam": True, "rna": True},
                ],
            }
        }
    }

    selected = select_eval_rows(frame, cfg)

    assert selected["question_id"].tolist() == [1, 3]


def test_select_eval_rows_returns_empty_for_unmatched_modality_combo() -> None:
    selected = select_eval_rows(
        pd.DataFrame(
            [
                _row(question_id=1, use_pathology=False, use_radiology=False),
                _row(question_id=2, use_pathology=True, use_radiology=False),
            ]
        ),
        {
            "filters": {
                "allowed_modality_combo": {
                    "enabled": True,
                    "values": [{"path": False, "rad": True, "dnam": True, "rna": True}],
                }
            }
        },
    )

    assert selected.empty


def test_select_eval_rows_disabled_filters_do_not_filter() -> None:
    frame = pd.DataFrame(
        [
            _row(question_id=1, split="test", project_id="TCGA-BRCA"),
            _row(question_id=2, split="train", project_id="TCGA-LUAD"),
        ]
    )
    cfg = {
        "filters": {
            "split": {"enabled": False, "value": "test"},
            "question_types": {"enabled": False, "values": ["mcq"]},
            "project_ids": {"enabled": False, "values": ["TCGA-BRCA"]},
            "task_ids": {"enabled": False, "values": ["mutation_tp53"]},
            "task_categories": {"enabled": False, "values": []},
            "allowed_modality_combo": {
                "enabled": False,
                "values": [{"path": True, "rad": True, "dnam": True, "rna": True}],
            },
            "row_limit": {"enabled": False, "max_rows": 1},
        }
    }

    selected = select_eval_rows(frame, cfg)

    assert selected["question_id"].tolist() == [1, 2]


def test_select_eval_rows_rejects_enabled_empty_value_filter() -> None:
    with pytest.raises(ValueError, match="project_ids"):
        select_eval_rows(
            pd.DataFrame([_row()]),
            {"filters": {"project_ids": {"enabled": True, "values": []}}},
        )


def test_build_prompt_selects_qa_prompt_without_options() -> None:
    _, user_prompt = build_mcq_prompt(
        _row(
            question_type="qa",
            option_a="",
            option_b="",
            answer="A concise free-text answer.",
            answer_label="",
        ),
        _prompt_cfg(),
    )

    assert "Answer concisely." in user_prompt
    assert "<choices>" not in user_prompt


def test_prompt_cfg_for_model_selects_model_prompt_profile() -> None:
    cfg = {
        "image_inputs": {"enabled": True},
        "prompts": {
            "baseline": {
                "mcq": {"system_prompt": "baseline system", "response_instruction": "baseline mcq"},
                "qa": {"system_prompt": "baseline system", "response_instruction": "baseline qa"},
            },
            "tuned_oncovlm": {
                "mcq": {"system_prompt": "tuned system", "response_instruction": "tuned mcq"},
                "qa": {"system_prompt": "tuned system", "response_instruction": "Answer concisely."},
            },
        },
    }

    selected = prompt_cfg_for_model(cfg, {"prompt_profile": "tuned_oncovlm"})
    system_prompt, user_prompt = build_mcq_prompt(
        _row(question_type="qa", option_a="", option_b="", answer="free text"),
        selected,
    )

    assert system_prompt == "tuned system"
    assert "Answer concisely." in user_prompt
    assert selected["image_inputs"] == {"enabled": True}


def test_enabled_model_configs_returns_all_enabled_models() -> None:
    selected = enabled_model_configs(
        {
            "gpt_5_1": {
                "enabled": True,
                "backend": "azure_openai_gpt",
            },
            "medgemma_4b_it": {
                "enabled": False,
                "backend": "hf_image_text_to_text",
            },
            "oncovlm_qwen_no_finetune": {
                "enabled": True,
                "backend": "oncovlm_projector",
            },
        }
    )

    assert [model["display_name"] for model in selected] == [
        "gpt_5_1",
        "oncovlm_qwen_no_finetune",
    ]


def test_enabled_model_configs_rejects_no_enabled_models() -> None:
    with pytest.raises(RuntimeError, match="No enabled"):
        enabled_model_configs({"medgemma_4b_it": {"enabled": False}})


def test_enabled_model_configs_accepts_list_style_models() -> None:
    selected = enabled_model_configs(
        [
            {
                "enabled": True,
                "display_name": "medgemma_4b_it",
                "backend": "hf_image_text_to_text",
            },
            {
                "enabled": False,
                "display_name": "gpt_5_1",
                "backend": "azure_openai_gpt",
            },
        ]
    )

    assert selected[0]["model_key"] == "medgemma_4b_it"
    assert selected[0]["display_name"] == "medgemma_4b_it"


def test_apply_group_sampling_uses_ratio_and_minimum_per_group() -> None:
    frame = pd.DataFrame(
        [
            _row(question_id=1, task_id="mutation_tp53"),
            _row(question_id=2, task_id="mutation_tp53"),
            _row(question_id=3, task_id="mutation_tp53"),
            _row(question_id=4, task_id="mutation_tp53"),
            _row(question_id=10, task_id="mutation_egfr"),
        ]
    )

    sampled = apply_group_sampling(
        frame,
        {
            "enabled": True,
            "ratio": 0.25,
            "min_per_group": 2,
            "seed": 7,
            "group_by": [
                "question_type",
                "generation_type",
                "task_category",
                "task_id",
                "modality_combination_name",
            ],
        },
    )

    assert sampled.groupby("task_id").size().to_dict() == {
        "mutation_egfr": 1,
        "mutation_tp53": 2,
    }


def test_apply_group_sampling_protects_radiology_available_test_cases() -> None:
    frame = pd.DataFrame(
        [
            _row(
                question_id=1,
                case_id="case-rad",
                modality_combination_name="path_only",
                use_pathology=True,
                use_dnam=False,
                use_rna=False,
                task_id="caption_integrated",
            ),
            _row(
                question_id=2,
                case_id="case-no-rad-a",
                modality_combination_name="path_only",
                use_pathology=True,
                use_dnam=False,
                use_rna=False,
                task_id="caption_integrated",
            ),
            _row(
                question_id=3,
                case_id="case-no-rad-b",
                modality_combination_name="path_only",
                use_pathology=True,
                use_dnam=False,
                use_rna=False,
                task_id="caption_integrated",
            ),
            _row(
                question_id=4,
                case_id="case-no-rad-c",
                modality_combination_name="path_only",
                use_pathology=True,
                use_dnam=False,
                use_rna=False,
                task_id="caption_integrated",
            ),
            _row(
                question_id=10,
                case_id="case-rad",
                modality_combination_name="radiology_only",
                use_radiology=True,
                use_dnam=False,
                use_rna=False,
                task_id="caption_integrated",
                radiology_feature_paths=["rad.pt"],
            ),
        ]
    )

    sampled = apply_group_sampling(
        frame,
        {
            "enabled": True,
            "ratio": 0.25,
            "min_per_group": 0,
            "seed": 7,
            "protect_radiology_available_cases": True,
            "protect_split": "test",
            "group_by": [
                "question_type",
                "generation_type",
                "task_category",
                "task_id",
                "modality_combination_name",
            ],
        },
    )

    path_only = sampled[sampled["modality_combination_name"].eq("path_only")]
    assert path_only["case_id"].tolist() == ["case-rad"]


def test_parse_model_response_derives_answer_label_for_mcq() -> None:
    parsed = parse_model_response(_row(), '{"answer": "TP53 mutation absent"}')

    assert parsed == {
        "predicted_answer": "TP53 mutation absent",
        "predicted_answer_label": "B",
        "parse_status": "exact",
    }


def test_build_flat_metric_records_includes_mcq_accuracy_and_f1() -> None:
    predictions = pd.DataFrame(
        [
            {
                **_row(question_id=1, answer_label="A"),
                "predicted_answer": "TP53 mutation present",
                "predicted_answer_label": "A",
                "parse_status": "exact",
                "correct": True,
            },
            {
                **_row(question_id=2, answer_label="A"),
                "predicted_answer": "TP53 mutation absent",
                "predicted_answer_label": "B",
                "parse_status": "exact",
                "correct": False,
            },
            {
                **_row(
                    question_id=3,
                    answer="TP53 mutation absent",
                    answer_label="B",
                ),
                "predicted_answer": "",
                "predicted_answer_label": "",
                "parse_status": "failed",
                "correct": False,
            },
        ]
    )

    records = build_flat_metric_records(
        predictions,
        model_display_name="gpt_5_1",
        backend="azure_openai_gpt",
    )
    overall = next(record for record in records if record["metric_group"] == "overall")

    assert overall["question_type"] == "ALL"
    assert overall["n"] == 3
    assert overall["accuracy"] == pytest.approx(1 / 3)
    assert overall["f1_macro"] == pytest.approx(1 / 3)
    assert overall["f1_weighted"] == pytest.approx(4 / 9)
    assert overall["parse_failed"] == 1


def test_add_bertscore_columns_scores_open_ended_rows(monkeypatch) -> None:
    def fake_score(candidates, references, **kwargs):
        assert candidates == ["predicted free-text"]
        assert references == ["reference free-text"]
        assert kwargs["model_type"] == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        assert kwargs["num_layers"] == 9
        assert kwargs["rescale_with_baseline"] is False
        return torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.3])

    monkeypatch.setitem(sys.modules, "bert_score", SimpleNamespace(score=fake_score))
    predictions = pd.DataFrame(
        [
            {
                **_row(
                    question_id=1,
                    question_type="qa",
                    option_a="",
                    option_b="",
                    answer="reference free-text",
                    answer_label="",
                ),
                "predicted_answer": "predicted free-text",
                "predicted_answer_label": "",
                "parse_status": "exact",
            },
            {
                **_row(question_id=2),
                "predicted_answer": "TP53 mutation present",
                "predicted_answer_label": "A",
                "parse_status": "exact",
            },
        ]
    )

    scored = add_bertscore_columns(
        predictions,
        {
            "enabled": True,
            "model_type": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            "num_layers": 9,
            "lang": "en",
            "max_length": 512,
            "rescale_with_baseline": False,
        },
    )

    assert scored.loc[0, "bertscore_precision"] == pytest.approx(0.1)
    assert scored.loc[0, "bertscore_recall"] == pytest.approx(0.2)
    assert scored.loc[0, "bertscore_f1"] == pytest.approx(0.3)
    assert scored.loc[1, "bertscore_f1"] is None


def test_patch_bert_score_tokenizer_max_length(monkeypatch) -> None:
    calls = []

    def fake_get_tokenizer(model_type, use_fast=False):
        calls.append((model_type, use_fast))
        return SimpleNamespace(model_max_length=10**30)

    fake_score_module = SimpleNamespace(get_tokenizer=fake_get_tokenizer)
    monkeypatch.setitem(sys.modules, "bert_score.score", fake_score_module)

    restore = _patch_bert_score_tokenizer_max_length(512)
    tokenizer = fake_score_module.get_tokenizer("medical-model", use_fast=False)

    assert tokenizer.model_max_length == 512
    assert calls == [("medical-model", False)]

    restore()
    assert fake_score_module.get_tokenizer is fake_get_tokenizer


def test_select_eval_rows_rejects_partial_modality_combo() -> None:
    with pytest.raises(ValueError, match="allowed_modality_combo"):
        select_eval_rows(
            pd.DataFrame([_row()]),
            {
                "filters": {
                    "allowed_modality_combo": {
                        "enabled": True,
                        "values": [{"path": False, "dnam": True, "rna": True}],
                    }
                }
            },
        )


def test_collect_required_image_paths_reads_real_pathology_images(tmp_path) -> None:
    roi_dir = tmp_path / "roi"
    roi_dir.mkdir()
    (roi_dir / "b.png").write_bytes(b"png")
    (roi_dir / "a.jpg").write_bytes(b"jpg")
    (roi_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    paths = collect_required_image_paths(
        _row(
            use_pathology=True,
            use_dnam=False,
            use_rna=False,
            pathology_roi_png_dir=str(roi_dir),
        ),
        {
            "image_inputs": {
                "enabled": True,
                "max_pathology_images": 1,
                "allowed_extensions": [".png", ".jpg"],
            }
        },
        repo_root=tmp_path,
    )

    assert [path.name for path in paths] == ["a.jpg"]


def test_collect_required_image_paths_fails_when_images_disabled(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="requires image modalities"):
        collect_required_image_paths(
            _row(use_pathology=True, use_dnam=False, use_rna=False),
            {"image_inputs": {"enabled": False}},
            repo_root=tmp_path,
        )


def test_collect_required_image_paths_fails_on_missing_required_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        collect_required_image_paths(
            _row(
                use_pathology=True,
                use_dnam=False,
                use_rna=False,
                pathology_roi_png_dir="missing-roi-dir",
            ),
            {"image_inputs": {"enabled": True}},
            repo_root=tmp_path,
        )
