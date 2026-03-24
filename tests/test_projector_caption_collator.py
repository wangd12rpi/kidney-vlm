from __future__ import annotations

import torch

from kidney_vlm.training.collator import ProjectorCaptionCollator


class _DummyTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        tokens = [index + 1 for index, _ in enumerate(str(text).split())]
        return {"input_ids": tokens}


def test_projector_caption_collator_pads_features_and_targets() -> None:
    collator = ProjectorCaptionCollator(
        tokenizer=_DummyTokenizer(),
        prompt_text="caption this ct",
        prompt_max_length=8,
        target_max_length=6,
        use_chatml=False,
    )
    batch = collator(
        [
            {
                "sample_id": "s1",
                "caption_text": "one two",
                "image_path": "i1.jpg",
                "feature_path": "f1.h5",
                "visual_features": torch.tensor(
                    [[1.0, 1.0]],
                    dtype=torch.float32,
                ),
            },
            {
                "sample_id": "s2",
                "caption_text": "three",
                "image_path": "i2.jpg",
                "feature_path": "f2.h5",
                "visual_features": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
            },
        ]
    )

    assert batch["visual_features"].shape == (2, 1, 2)
    assert batch["visual_attention_mask"].tolist() == [[1], [1]]
    assert batch["prompt_input_ids"].shape[0] == 2
    assert batch["prompt_attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert batch["target_input_ids"].tolist()[0][-1] == 102
    assert batch["target_attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
