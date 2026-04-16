from __future__ import annotations

import pandas as pd

from kidney_vlm.hf_integration import build_dataset_for_push


def test_build_dataset_for_push_preserves_train_test_splits() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "a", "split": "train", "caption": "train 1"},
            {"sample_id": "b", "split": "train", "caption": "train 2"},
            {"sample_id": "c", "split": "test", "caption": "test 1"},
        ]
    )

    dataset_payload = build_dataset_for_push(
        frame,
        split_column="split",
        default_split_name="train",
        allowed_split_names=["train", "test"],
    )

    assert set(dataset_payload.keys()) == {"train", "test"}
    assert dataset_payload["train"].num_rows == 2
    assert dataset_payload["test"].num_rows == 1


def test_build_dataset_for_push_defaults_blank_splits_to_train() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "a", "split": "", "caption": "row 1"},
            {"sample_id": "b", "split": None, "caption": "row 2"},
            {"sample_id": "c", "split": "test", "caption": "row 3"},
        ]
    )

    dataset_payload = build_dataset_for_push(
        frame,
        split_column="split",
        default_split_name="train",
        allowed_split_names=["train", "test"],
    )

    assert dataset_payload["train"].num_rows == 2
    assert dataset_payload["test"].num_rows == 1
