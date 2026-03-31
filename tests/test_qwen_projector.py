from __future__ import annotations

import pytest


def test_pathology_qwen_projector_forward_and_freeze() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from kidney_vlm.modeling.qwen_projector import PathologyQwenProjectorLM

    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    backbone = transformers.GPT2LMHeadModel(config)
    model = PathologyQwenProjectorLM(language_model=backbone, pathology_in_dim=8)

    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    pathology_features = torch.randn(2, 3, 8)
    pathology_feature_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pathology_features=pathology_features,
        pathology_feature_mask=pathology_feature_mask,
    )

    assert output.loss is not None
    assert output.logits.shape == (2, 8, config.vocab_size)
    assert all(parameter.requires_grad is False for parameter in model.language_model.parameters())
    assert any(parameter.requires_grad is True for parameter in model.projectors.parameters())
