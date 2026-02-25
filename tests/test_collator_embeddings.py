from __future__ import annotations

import numpy as np

from kidney_vlm.training.collator import QACollator


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
