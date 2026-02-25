from __future__ import annotations

import torch
from torch import nn


class FeatureProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class ModalityProjector(nn.Module):
    """Project per-image embeddings and pool variable-length image sets."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projector = FeatureProjector(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, n_images, in_dim] or [batch, in_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        projected = self.projector(x)  # [batch, n_images, out_dim]
        pooled = masked_mean_pool(projected, mask=mask)
        return projected, pooled


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pool across image dimension with optional validity mask."""
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor [batch, n_items, dim], got shape {tuple(x.shape)}")

    if mask is None:
        return x.mean(dim=1)

    if mask.dim() != 2:
        raise ValueError(f"Expected 2D mask [batch, n_items], got shape {tuple(mask.shape)}")

    mask = mask.to(dtype=x.dtype).unsqueeze(-1)  # [batch, n_items, 1]
    masked_x = x * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked_x.sum(dim=1) / denom


class MultiModalProjectors(nn.Module):
    """Separate projectors for pathology and radiology embeddings."""

    def __init__(self, pathology_in_dim: int, radiology_in_dim: int, vlm_dim: int):
        super().__init__()
        self.pathology_projector = ModalityProjector(pathology_in_dim, vlm_dim)
        self.radiology_projector = ModalityProjector(radiology_in_dim, vlm_dim)

    def forward(
        self,
        pathology_x: torch.Tensor | None,
        radiology_x: torch.Tensor | None,
        pathology_mask: torch.Tensor | None = None,
        radiology_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        pathology_projected = None
        pathology_pooled = None
        radiology_projected = None
        radiology_pooled = None

        if pathology_x is not None:
            pathology_projected, pathology_pooled = self.pathology_projector(pathology_x, pathology_mask)

        if radiology_x is not None:
            radiology_projected, radiology_pooled = self.radiology_projector(radiology_x, radiology_mask)

        return {
            "pathology_projected": pathology_projected,
            "pathology_pooled": pathology_pooled,
            "radiology_projected": radiology_projected,
            "radiology_pooled": radiology_pooled,
        }
