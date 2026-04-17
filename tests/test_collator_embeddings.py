from __future__ import annotations

from collections import UserDict
import numpy as np
import pytest
import torch

from kidney_vlm.training.collator import (
    DNAMProjectorQACollator,
    PathologyProjectorQACollator,
    QACollator,
    _apply_patch_token_dropout,
)


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

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, chat_template_kwargs=None):
        assert tokenize is True
        assert chat_template_kwargs == {"enable_thinking": False}
        pieces = []
        for message in messages:
            role = str(message["role"]).upper()
            pieces.append(f"<{role}>{message['content']}")
        if add_generation_prompt:
            pieces.append("<ASSISTANT>")
        return [ord(char) for char in "".join(pieces)]


def test_projector_collator_uses_chat_template_with_hardcoded_prompt_text(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_prompt = "Write a pathology caption."
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

    expected_prompt = f"<USER>{selected_prompt}<ASSISTANT>"
    expected_full = f"{expected_prompt}Clear cell renal neoplasm."
    expected_prompt_ids = [ord(char) for char in expected_prompt]
    expected_full_ids = [ord(char) for char in expected_full]

    assert input_ids == expected_full_ids
    assert labels == ([-100] * len(expected_prompt_ids)) + expected_full_ids[len(expected_prompt_ids) :]


def test_projector_collator_has_five_default_prompt_texts() -> None:
    collator = PathologyProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=".",
    )

    assert len(collator.prompt_texts) == 5


def test_projector_collator_accepts_batch_encoding_like_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_prompt = "Describe the pathology image."
    monkeypatch.setattr(
        "kidney_vlm.training.collator.random.choice",
        lambda options: selected_prompt,
    )

    class _BatchEncodingLikeTokenizer(_ProjectorTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, chat_template_kwargs=None):
            encoded = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs,
            )
            return {"input_ids": torch.tensor([encoded], dtype=torch.long)}

    collator = PathologyProjectorQACollator(
        tokenizer=_BatchEncodingLikeTokenizer(),
        root_dir=".",
    )

    input_ids, labels = collator._build_text_pair(
        {
            "instruction": "THIS SHOULD BE IGNORED",
            "answer": "Example pathology caption.",
        }
    )

    expected_prompt = f"<USER>{selected_prompt}<ASSISTANT>"
    expected_full = f"{expected_prompt}Example pathology caption."
    expected_prompt_ids = [ord(char) for char in expected_prompt]
    expected_full_ids = [ord(char) for char in expected_full]

    assert input_ids == expected_full_ids
    assert labels == ([-100] * len(expected_prompt_ids)) + expected_full_ids[len(expected_prompt_ids) :]


def test_projector_collator_accepts_mapping_like_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_prompt = "Describe the pathology image."
    monkeypatch.setattr(
        "kidney_vlm.training.collator.random.choice",
        lambda options: selected_prompt,
    )

    class _MappingLikeTokenizer(_ProjectorTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, chat_template_kwargs=None):
            encoded = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs,
            )
            return UserDict({"input_ids": encoded})

    collator = PathologyProjectorQACollator(
        tokenizer=_MappingLikeTokenizer(),
        root_dir=".",
    )

    input_ids, labels = collator._build_text_pair(
        {
            "instruction": "THIS SHOULD BE IGNORED",
            "answer": "Example pathology caption.",
        }
    )

    expected_prompt = f"<USER>{selected_prompt}<ASSISTANT>"
    expected_full = f"{expected_prompt}Example pathology caption."
    expected_prompt_ids = [ord(char) for char in expected_prompt]
    expected_full_ids = [ord(char) for char in expected_full]

    assert input_ids == expected_full_ids
    assert labels == ([-100] * len(expected_prompt_ids)) + expected_full_ids[len(expected_prompt_ids) :]


def test_dnam_projector_collator_loads_pt_features(tmp_path) -> None:
    feature_path = tmp_path / "sample.pt"
    torch.save(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32), feature_path)

    collator = DNAMProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=tmp_path,
    )
    batch = collator(
        [
            {
                "sample_id": "tcga-1",
                "project_id": "TCGA-GBM",
                "source": "tcga",
                "answer": "Example DNAm caption.",
                "genomics_dna_methylation_feature_path": feature_path.name,
            }
        ]
    )

    assert batch["dnam_features"].shape == (1, 1, 3)
    assert batch["dnam_feature_mask"].tolist() == [[1]]


def test_dnam_projector_collator_has_default_prompt_texts() -> None:
    collator = DNAMProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=".",
    )

    assert len(collator.prompt_texts) == 5


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
