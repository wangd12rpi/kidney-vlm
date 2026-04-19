from __future__ import annotations

import pytest


def test_dnam_qwen_projector_expands_single_feature_to_prefix_tokens() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from kidney_vlm.modeling.dnam_qwen_projector import DnamQwenProjectorLM

    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    backbone = transformers.GPT2LMHeadModel(config)
    model = DnamQwenProjectorLM(
        language_model=backbone,
        dnam_in_dim=8,
        projector_type="mlp",
        dnam_prefix_tokens=8,
    )
    expander = model.projectors["dnam_prefix_expander"]

    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    dnam_features = torch.randn(2, 1, 8)
    dnam_feature_mask = torch.ones(2, 1, dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        dnam_features=dnam_features,
        dnam_feature_mask=dnam_feature_mask,
    )

    assert output.loss is not None
    assert output.logits.shape == (2, 13, config.vocab_size)
    assert expander.token_embeddings.shape == (8, config.n_embd)
    assert any(parameter.requires_grad is True for parameter in model.projectors.parameters())
    assert all(parameter.requires_grad is False for parameter in model.language_model.parameters())


def test_dnam_qwen_projector_can_disable_prefix_expansion() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from kidney_vlm.modeling.dnam_qwen_projector import DnamQwenProjectorLM

    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    backbone = transformers.GPT2LMHeadModel(config)
    model = DnamQwenProjectorLM(
        language_model=backbone,
        dnam_in_dim=8,
        projector_type="mlp",
        dnam_prefix_tokens=0,
    )

    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    dnam_features = torch.randn(2, 1, 8)
    dnam_feature_mask = torch.ones(2, 1, dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        dnam_features=dnam_features,
        dnam_feature_mask=dnam_feature_mask,
    )

    assert output.loss is not None
    assert output.logits.shape == (2, 6, config.vocab_size)
