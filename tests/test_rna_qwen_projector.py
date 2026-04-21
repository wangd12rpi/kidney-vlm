from __future__ import annotations

import pytest


def test_rna_qwen_projector_expands_single_feature_to_prefix_tokens() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    transformers = pytest.importorskip("transformers")

    from kidney_vlm.modeling.rna_qwen_projector import RnaQwenProjectorLM

    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    backbone = transformers.GPT2LMHeadModel(config)
    model = RnaQwenProjectorLM(
        language_model=backbone,
        rna_in_dim=512,
        projector_type="mlp",
        rna_prefix_tokens=8,
    )
    expander = model.projectors["rna_prefix_expander"]

    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    rna_features = torch.randn(2, 1, 512)
    rna_feature_mask = torch.ones(2, 1, dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        rna_features=rna_features,
        rna_feature_mask=rna_feature_mask,
    )

    assert output.loss is not None
    assert output.logits.shape == (2, 13, config.vocab_size)
    assert expander.token_embeddings.shape == (8, config.n_embd)
    assert any(parameter.requires_grad is True for parameter in model.projectors.parameters())
    assert all(parameter.requires_grad is False for parameter in model.language_model.parameters())


def test_rna_qwen_projector_can_disable_prefix_expansion() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    transformers = pytest.importorskip("transformers")

    from kidney_vlm.modeling.rna_qwen_projector import RnaQwenProjectorLM

    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    backbone = transformers.GPT2LMHeadModel(config)
    model = RnaQwenProjectorLM(
        language_model=backbone,
        rna_in_dim=512,
        projector_type="mlp",
        rna_prefix_tokens=0,
    )

    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    rna_features = torch.randn(2, 1, 512)
    rna_feature_mask = torch.ones(2, 1, dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        rna_features=rna_features,
        rna_feature_mask=rna_feature_mask,
    )

    assert output.loss is not None
    assert output.logits.shape == (2, 6, config.vocab_size)
