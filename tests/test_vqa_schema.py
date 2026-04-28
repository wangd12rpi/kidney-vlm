from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from kidney_vlm.vqa.schema import normalize_vqa_df, upsert_vqa_rows, validate_vqa_df


def _row(question_id: int, *, answer: str = "Stage II", question: str = "What is the stage?") -> dict[str, object]:
    answer_labels = {
        "Stage I": "A",
        "Stage II": "B",
        "Stage III": "C",
        "Stage IV": "D",
    }
    return {
        "case_id": "TCGA-AA-0001",
        "project_id": "TCGA-TEST",
        "question_id": question_id,
        "base_question_id": 100,
        "split": "train",
        "question_type": "mcq",
        "generation_type": "from_ground_truth",
        "task_category": "stage",
        "task_id": "pathologic_stage",
        "modality_combination_name": "path_only",
        "use_pathology": True,
        "use_radiology": False,
        "use_dnam": False,
        "use_rna": False,
        "question": question,
        "option_a": "Stage I",
        "option_b": "Stage II",
        "option_c": "Stage III",
        "option_d": "Stage IV",
        "answer": answer,
        "answer_label": answer_labels.get(answer, ""),
        "caption_id": "",
        "ground_truth_source": "task_stage_label",
        "radiology_biomarker": "",
        "pathology_feature_paths": ["features/path-a.h5"],
        "radiology_feature_paths": [],
        "dnam_feature_path": "",
        "rna_feature_path": "",
        "pathology_roi_png_dir": "",
        "radiology_view_png_dir": "",
        "dnam_text_summary": "",
        "rna_text_summary": "",
    }


def test_normalize_and_validate_vqa_frame_preserves_strict_schema() -> None:
    frame = pd.DataFrame([_row(1)])

    normalized = normalize_vqa_df(frame)
    validate_vqa_df(normalized)

    assert normalized.at[0, "caption_id"] == ""
    assert normalized.at[0, "pathology_feature_paths"] == ["features/path-a.h5"]
    assert normalized.at[0, "radiology_feature_paths"] == []
    assert bool(normalized.at[0, "use_pathology"]) is True
    assert int(normalized.at[0, "question_id"]) == 1


def test_normalize_vqa_frame_rejects_old_singular_feature_columns() -> None:
    row = _row(1)
    row["pathology_feature_path"] = row.pop("pathology_feature_paths")[0]
    row["radiology_feature_path"] = ""
    row.pop("radiology_feature_paths")
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="missing required columns"):
        normalize_vqa_df(frame)


def test_normalize_vqa_frame_rejects_extra_columns() -> None:
    row = _row(1)
    row["surprise_column"] = "not part of schema"
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="unexpected columns"):
        normalize_vqa_df(frame)


def test_normalize_vqa_frame_rejects_scalar_array_values_even_when_empty() -> None:
    row = _row(1)
    row["pathology_feature_paths"] = ""
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="one-layer list values"):
        normalize_vqa_df(frame)


def test_normalize_vqa_frame_rejects_null_array_values() -> None:
    row = _row(1)
    row["radiology_feature_paths"] = None
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="one-layer list values"):
        normalize_vqa_df(frame)


def test_normalize_vqa_frame_rejects_serialized_list_strings() -> None:
    row = _row(1)
    row["radiology_feature_paths"] = "['features/rad-a.h5::series=a']"
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="one-layer list values"):
        normalize_vqa_df(frame)


def test_normalize_vqa_frame_accepts_tuple_and_numpy_array_values_as_one_layer_arrays() -> None:
    row = _row(1)
    row["pathology_feature_paths"] = ("features/path-a.h5", " ", "features/path-b.h5")
    row["radiology_feature_paths"] = np.array(["features/rad-a.h5::series=a"], dtype=object)
    frame = pd.DataFrame([row])

    normalized = normalize_vqa_df(frame)
    validate_vqa_df(normalized)

    assert normalized.at[0, "pathology_feature_paths"] == ["features/path-a.h5", "features/path-b.h5"]
    assert normalized.at[0, "radiology_feature_paths"] == ["features/rad-a.h5::series=a"]


def test_normalize_vqa_frame_rejects_nested_array_values() -> None:
    row = _row(1)
    row["pathology_feature_paths"] = [["features/path-a.h5"]]
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="one-layer lists"):
        normalize_vqa_df(frame)


def test_normalize_vqa_frame_rejects_old_radiology_mask_column() -> None:
    row = _row(1)
    row["radiology_segmentation_mask_paths"] = []
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="unexpected columns"):
        normalize_vqa_df(frame)


def test_validate_vqa_frame_requires_mcq_answer_to_match_an_option() -> None:
    frame = normalize_vqa_df(pd.DataFrame([_row(1, answer="Stage X")]))

    with pytest.raises(ValueError, match="answer must exactly match"):
        validate_vqa_df(frame)


def test_validate_vqa_frame_requires_mcq_answer_label_to_match_answer_option() -> None:
    row = _row(1)
    row["answer_label"] = "A"
    frame = normalize_vqa_df(pd.DataFrame([row]))

    with pytest.raises(ValueError, match="answer_label must be B"):
        validate_vqa_df(frame)


def test_upsert_vqa_rows_replaces_matching_question_id_only() -> None:
    existing = pd.DataFrame(
        [
            _row(1, answer="Stage I", question="old question"),
            _row(2, answer="Stage II", question="keep question"),
        ]
    )
    generated = pd.DataFrame([_row(1, answer="Stage II", question="new question")])

    final = upsert_vqa_rows(existing, generated)

    assert final["question_id"].astype(int).tolist() == [2, 1]
    assert final["question"].tolist() == ["keep question", "new question"]
    assert final["answer"].tolist() == ["Stage II", "Stage II"]
