from __future__ import annotations

import numpy as np
import pytest
import torch

from kidney_vlm.training.collator import PathologyProjectorQACollator, QACollator, _apply_patch_token_dropout


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
                "radiology_image_modalities": np.array(["CT"], dtype=object),
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
            }
        ]
    )

    assert batch["pathology_tile_embedding_paths"] == [["/tmp/tile.npy"]]
    assert batch["pathology_slide_embedding_paths"] == [["/tmp/slide.npy"]]
    assert batch["radiology_embedding_paths"] == [["/tmp/rad.npy"]]
    assert batch["radiology_image_modalities"] == [["CT"]]


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
    collator = PathologyProjectorQACollator(
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
    collator = PathologyProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=".",
    )

    assert len(collator.prompt_texts) == 10


def test_apply_patch_token_dropout_keeps_selected_tokens_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kidney_vlm.training.collator.torch.rand",
        lambda size, device=None: torch.tensor([0.9, 0.1, 0.8, 0.2], device=device),
    )
    patch_tensor = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    dropped = _apply_patch_token_dropout(patch_tensor, dropout_prob=0.5)

    assert torch.equal(
        dropped,
        torch.tensor(
            [
                [1.0, 10.0],
                [3.0, 30.0],
            ]
        ),
    )


def test_apply_patch_token_dropout_keeps_one_token_when_all_would_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kidney_vlm.training.collator.torch.rand",
        lambda size, device=None: torch.tensor([0.1, 0.4, 0.3], device=device),
    )
    patch_tensor = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    dropped = _apply_patch_token_dropout(patch_tensor, dropout_prob=0.9)

    assert torch.equal(dropped, torch.tensor([[2.0, 20.0]]))
