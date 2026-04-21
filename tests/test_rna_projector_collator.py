from __future__ import annotations

from pathlib import Path

import pytest


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


def test_rna_projector_collator_loads_single_token_bulkformer_features(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)

    from kidney_vlm.training.collator import RNAProjectorQACollator

    first_feature_path = tmp_path / "rna-a.pt"
    second_feature_path = tmp_path / "rna-b.pt"
    torch.save(torch.ones(512, dtype=torch.float32), first_feature_path)
    torch.save(torch.zeros(1, 512, dtype=torch.float32), second_feature_path)

    collator = RNAProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=tmp_path,
        max_rna_tokens=1,
    )
    batch = collator(
        [
            {
                "sample_id": "sample-a",
                "project_id": "TCGA-KIRC",
                "source": "tcga",
                "patient_id": "TCGA-AA-0001",
                "study_id": "study-a",
                "answer": "Example RNA caption.",
                "genomics_rna_bulk_feature_path": first_feature_path.name,
            },
            {
                "sample_id": "sample-b",
                "project_id": "TCGA-KIRP",
                "source": "tcga",
                "patient_id": "TCGA-BB-0002",
                "study_id": "study-b",
                "answer": "Another RNA caption.",
                "genomics_rna_bulk_feature_path": second_feature_path.name,
            },
        ]
    )

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["rna_features"].shape == (2, 1, 512)
    assert batch["rna_feature_mask"].tolist() == [[1], [1]]
    assert batch["sample_id"] == ["sample-a", "sample-b"]


def test_rna_projector_collator_has_default_prompt_texts() -> None:
    pytest.importorskip("torch", exc_type=ImportError)

    from kidney_vlm.training.collator import RNAProjectorQACollator

    collator = RNAProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=".",
    )

    assert len(collator.prompt_texts) == 5
