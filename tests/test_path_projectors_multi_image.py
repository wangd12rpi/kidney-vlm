from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_resolve_language_model_hidden_size_supports_nested_text_config() -> None:
    torch = pytest.importorskip("torch")

    from kidney_vlm.modeling.path_projectors import resolve_language_model_hidden_size

    language_model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=1234)),
        get_input_embeddings=lambda: torch.nn.Embedding(8, 4321),
    )

    assert resolve_language_model_hidden_size(language_model) == 1234


def test_build_soft_prefix_per_layer_inputs_prepends_zero_prefix() -> None:
    torch = pytest.importorskip("torch")

    from kidney_vlm.modeling.path_projectors import build_soft_prefix_per_layer_inputs

    class TinyTextModel:
        hidden_size_per_layer_input = 3

        @staticmethod
        def get_per_layer_inputs(input_ids, inputs_embeds):
            assert inputs_embeds is None
            batch, seq_len = input_ids.shape
            values = torch.arange(batch * seq_len * 2 * 3, dtype=torch.float32)
            return values.reshape(batch, seq_len, 2, 3)

    language_model = SimpleNamespace(model=TinyTextModel())
    input_ids = torch.ones((2, 4), dtype=torch.long)

    per_layer_inputs = build_soft_prefix_per_layer_inputs(
        language_model,
        input_ids=input_ids,
        prefix_length=5,
        device=input_ids.device,
        dtype=torch.float32,
    )

    assert per_layer_inputs is not None
    assert per_layer_inputs.shape == (2, 9, 2, 3)
    assert torch.equal(per_layer_inputs[:, :5], torch.zeros((2, 5, 2, 3)))
    assert torch.equal(per_layer_inputs[:, 5:], TinyTextModel.get_per_layer_inputs(input_ids, None))


def test_forward_language_model_with_soft_prefix_bypasses_conditional_outer_forward() -> None:
    torch = pytest.importorskip("torch")

    from kidney_vlm.modeling.path_projectors import forward_language_model_with_soft_prefix

    class TinyTextModel(torch.nn.Module):
        hidden_size_per_layer_input = 2

        def __init__(self):
            super().__init__()
            self.forward_called = False

        @staticmethod
        def get_per_layer_inputs(input_ids, inputs_embeds):
            assert inputs_embeds is None
            return torch.ones((input_ids.shape[0], input_ids.shape[1], 2, 2), dtype=torch.float32)

        def forward(self, *, per_layer_inputs, inputs_embeds, **kwargs):
            self.forward_called = True
            assert per_layer_inputs.shape == (inputs_embeds.shape[0], inputs_embeds.shape[1], 2, 2)
            return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=None, hidden_states=None, attentions=None)

    class TinyOuterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = TinyTextModel()

    class TinyConditionalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyOuterModel()
            self.lm_head = torch.nn.Linear(4, 7)
            self.config = SimpleNamespace(
                get_text_config=lambda: SimpleNamespace(final_logit_softcapping=None, vocab_size=7)
            )

        def forward(self, **kwargs):
            raise AssertionError("Outer conditional forward should not be called for soft-prefix training.")

    language_model = TinyConditionalModel()
    input_ids = torch.ones((2, 3), dtype=torch.long)
    inputs_embeds = torch.randn(2, 5, 4)
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
    labels = torch.ones((2, 5), dtype=torch.long)

    output = forward_language_model_with_soft_prefix(
        language_model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        prefix_length=2,
    )

    assert language_model.model.language_model.forward_called is True
    assert output.loss is not None
    assert output.logits.shape == (2, 5, 7)


def test_multimodal_projectors_support_multi_image_inputs() -> None:
    torch = pytest.importorskip("torch")

    from kidney_vlm.modeling.path_projectors import MultiModalProjectors

    batch = 2
    n_pathology = 3
    n_radiology = 2
    pathology_in = 8
    radiology_in = 6
    vlm_dim = 4

    module = MultiModalProjectors(
        pathology_in_dim=pathology_in,
        radiology_in_dim=radiology_in,
        vlm_dim=vlm_dim,
    )

    pathology_x = torch.randn(batch, n_pathology, pathology_in)
    radiology_x = torch.randn(batch, n_radiology, radiology_in)
    pathology_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    radiology_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)

    out = module(
        pathology_x=pathology_x,
        radiology_x=radiology_x,
        pathology_mask=pathology_mask,
        radiology_mask=radiology_mask,
    )

    assert out["pathology_projected"].shape == (batch, n_pathology, vlm_dim)
    assert out["radiology_projected"].shape == (batch, n_radiology, vlm_dim)
    assert out["pathology_pooled"].shape == (batch, vlm_dim)
    assert out["radiology_pooled"].shape == (batch, vlm_dim)


def test_modality_projector_resampler_supports_fixed_latent_output() -> None:
    torch = pytest.importorskip("torch")

    from kidney_vlm.modeling.path_projectors import ModalityProjector

    module = ModalityProjector(
        in_dim=8,
        out_dim=16,
        projector_type="resampler",
        num_latents=4,
        depth=1,
        num_heads=4,
    )

    x = torch.randn(2, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)

    projected, pooled = module(x, mask)
    output_mask = module.build_output_mask(
        mask,
        batch_size=projected.shape[0],
        output_length=projected.shape[1],
        device=projected.device,
        dtype=torch.float32,
    )

    assert projected.shape == (2, 4, 16)
    assert pooled.shape == (2, 16)
    assert output_mask.shape == (2, 4)


def test_modality_projector_mlp_preserves_token_count_and_mask() -> None:
    torch = pytest.importorskip("torch")

    from kidney_vlm.modeling.path_projectors import ModalityProjector

    module = ModalityProjector(
        in_dim=8,
        out_dim=16,
        projector_type="mlp",
        dropout=0.1,
    )

    x = torch.randn(2, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)

    projected, pooled = module(x, mask)
    output_mask = module.build_output_mask(
        mask,
        batch_size=projected.shape[0],
        output_length=projected.shape[1],
        device=projected.device,
        dtype=torch.float32,
    )

    assert projected.shape == (2, 6, 16)
    assert pooled.shape == (2, 16)
    assert torch.equal(output_mask, mask)
