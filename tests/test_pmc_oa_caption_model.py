from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_pmc_oa_caption_model_forward_masks_prompt_loss() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    F = torch.nn.functional

    from kidney_vlm.modeling.pmc_oa_caption import (
        PMCCaptionProjector,
        PMCCaptionProjectorConfig,
        PMCOACaptionProjectorModel,
    )

    class _DummyTokenizer:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2

    class _DummyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(32, 8)
            self.lm_head = nn.Linear(8, 32)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, *, inputs_embeds, attention_mask=None, labels=None, return_dict=True):
            logits = self.lm_head(inputs_embeds)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(loss=loss, logits=logits)

    projector = PMCCaptionProjector(
        PMCCaptionProjectorConfig(
            input_dim=4,
            output_dim=8,
        )
    )
    
    # Verify absence token exists for Stage 2 checkpoint compatibility.
    assert hasattr(projector, "absence_token")
    assert projector.absence_token.shape == (1, 1, 8)
    
    model = PMCOACaptionProjectorModel(
        llm=_DummyLM(),
        tokenizer=_DummyTokenizer(),
        projector=projector,
        llm_model_name_or_path="dummy",
        freeze_llm=True,
    )

    outputs = model(
        visual_features=torch.randn(2, 1, 4),
        visual_attention_mask=torch.tensor([[1], [1]]),
        prompt_input_ids=torch.tensor([[1, 5], [1, 5]]),
        prompt_attention_mask=torch.tensor([[1, 1], [1, 1]]),
        target_input_ids=torch.tensor([[6, 7, 2], [8, 2, 0]]),
        target_attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
    )

    assert torch.isfinite(outputs["loss"])
    assert outputs["logits"].shape[0] == 2
    assert outputs["labels"].shape[0] == 2
    assert outputs["labels"][0, 0].item() == -100
    assert outputs["labels"][0, 1].item() == -100
    assert outputs["labels"][0, 2].item() == -100
    assert outputs["labels"][0, 3].item() == 6
    assert model.count_trainable_parameters() == sum(
        parameter.numel() for parameter in model.projector.parameters()
    )
