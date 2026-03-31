from __future__ import annotations

import numpy as np
import pytest

from kidney_vlm.training.collator import ProjectorQACollator, QACollator


class _DummyTokenizer:
    def __call__(self, texts, **_kwargs):
        return {"input_ids": [0 for _ in texts]}


def test_collator_exposes_new_embedding_fields_and_legacy_aliases() -> None:
    collator = QACollator(tokenizer=_DummyTokenizer(), max_length=32)
    batch = collator(
        [
            {
                "question": "q",
                "answer": "a",
                "pathology_tile_embedding_paths": np.array(["/tmp/tile.npy"], dtype=object),
                "pathology_slide_embedding_paths": np.array(["/tmp/slide.npy"], dtype=object),
                "radiology_embedding_paths": np.array(["/tmp/rad.npy"], dtype=object),
                "pathology_wsi_paths": [],
                "radiology_image_paths": [],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
            }
        ]
    )

    assert batch["pathology_tile_embedding_paths"] == [["/tmp/tile.npy"]]
    assert batch["pathology_slide_embedding_paths"] == [["/tmp/slide.npy"]]
    assert batch["radiology_embedding_paths"] == [["/tmp/rad.npy"]]


class _ProjectorTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in str(text)]}


def test_projector_collator_uses_hardcoded_prompt_texts_not_parquet_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_prompt = "Write a detailed pathology description for this image.\nCaption:"
    monkeypatch.setattr(
        "kidney_vlm.training.collator.random.choice",
        lambda options: selected_prompt,
    )
    collator = ProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=".",
    )

    input_ids, labels = collator._build_text_pair(
        {
            "instruction": "THIS SHOULD BE IGNORED",
            "answer": "Clear cell renal neoplasm.",
        }
    )

    expected_prompt_ids = [ord(char) for char in selected_prompt]
    expected_answer_ids = [ord(char) for char in f" Clear cell renal neoplasm.<eos>"]

    assert input_ids == expected_prompt_ids + expected_answer_ids
    assert labels == ([-100] * len(expected_prompt_ids)) + expected_answer_ids


def test_projector_collator_has_ten_default_prompt_texts() -> None:
    collator = ProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=".",
    )

    assert len(collator.prompt_texts) == 10
