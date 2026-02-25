from __future__ import annotations

import pytest


def test_multimodal_projectors_support_multi_image_inputs() -> None:
    torch = pytest.importorskip("torch")

    from kidney_vlm.modeling.projectors import MultiModalProjectors

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
