from __future__ import annotations

import pytest


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
