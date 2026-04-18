from __future__ import annotations

import pytest


def test_radiology_qwen_projector_forward_and_freeze() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from kidney_vlm.modeling.radiology_qwen_projector import RadiologyQwenProjectorLM

    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    backbone = transformers.GPT2LMHeadModel(config)
    model = RadiologyQwenProjectorLM(language_model=backbone, radiology_in_dim=8)

    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    radiology_features = torch.randn(2, 3, 8)
    radiology_feature_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        radiology_features=radiology_features,
        radiology_feature_mask=radiology_feature_mask,
    )

    assert output.loss is not None
    assert output.logits.shape == (2, 8, config.vocab_size)
    assert all(parameter.requires_grad is False for parameter in model.language_model.parameters())
    assert any(parameter.requires_grad is True for parameter in model.rad_projectors.parameters())


def test_radiology_qwen_projector_resampler_forward() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from kidney_vlm.modeling.radiology_qwen_projector import RadiologyQwenProjectorLM

    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    backbone = transformers.GPT2LMHeadModel(config)
    model = RadiologyQwenProjectorLM(
        language_model=backbone,
        radiology_in_dim=8,
        projector_type="resampler",
        projector_num_latents=4,
        projector_depth=1,
        projector_num_heads=4,
    )

    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    radiology_features = torch.randn(2, 3, 8)
    radiology_feature_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        radiology_features=radiology_features,
        radiology_feature_mask=radiology_feature_mask,
    )

    assert output.loss is not None
    assert output.logits.shape == (2, 9, config.vocab_size)
