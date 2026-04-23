from __future__ import annotations

import pandas as pd
import pytest

from kidney_vlm.vqa.eval_gpt import (
    build_mcq_prompt,
    collect_required_image_paths,
    parse_mcq_response,
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


def test_build_mcq_prompt_uses_semantic_options_without_letters() -> None:
    _, user_prompt = build_mcq_prompt(
        _row(),
        {
            "system_prompt": "Use the evidence.",
            "response_instruction": "Return JSON with a semantic answer.",
        },
    )

    assert "- TP53 mutation present" in user_prompt
    assert "- TP53 mutation absent" in user_prompt
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
        {
            "system_prompt": "Use the evidence.",
            "response_instruction": "Return JSON with a semantic answer.",
        },
    )

    assert "<pathology_images>attached</pathology_images>" in user_prompt
    assert "<radiology_images>attached</radiology_images>" in user_prompt
    assert "[not provided]" not in user_prompt


def test_build_mcq_prompt_fails_on_missing_required_text() -> None:
    with pytest.raises(ValueError, match="dnam_text_summary"):
        build_mcq_prompt(
            _row(dnam_text_summary=""),
            {
                "system_prompt": "Use the evidence.",
                "response_instruction": "Return JSON with a semantic answer.",
            },
        )


def test_build_mcq_prompt_fails_when_no_modalities_are_enabled() -> None:
    with pytest.raises(ValueError, match="no enabled modalities"):
        build_mcq_prompt(
            _row(use_dnam=False, use_rna=False),
            {
                "system_prompt": "Use the evidence.",
                "response_instruction": "Return JSON with a semantic answer.",
            },
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
            "project_ids": {"enabled": True, "values": ["TCGA-BRCA"]},
            "task_ids": {"enabled": True, "values": ["mutation_tp53"]},
            "task_categories": {"enabled": False, "values": []},
            "nonempty_enabled_genomics_text": {"enabled": True},
            "row_limit": {"enabled": False, "max_rows": 1},
        }
    }

    selected = select_eval_rows(frame, cfg)

    assert selected["question_id"].tolist() == [1, 2]


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
            "nonempty_enabled_genomics_text": {"enabled": False},
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
