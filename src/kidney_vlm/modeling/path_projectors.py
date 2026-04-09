from __future__ import annotations

import torch
from torch import nn


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = max(dim, int(round(dim * float(mlp_ratio))))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResamplerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, latents: torch.Tensor, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        cross_query = self.cross_attn_norm(latents)
        cross_out, _ = self.cross_attn(
            query=cross_query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = latents + cross_out

        self_query = self.self_attn_norm(latents)
        self_out, _ = self.self_attn(
            query=self_query,
            key=self_query,
            value=self_query,
            need_weights=False,
        )
        latents = latents + self_out
        latents = latents + self.ff(latents)
        return latents


class ResamplerProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_latents: int = 64,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_latents <= 0:
            raise ValueError("num_latents must be positive for a resampler projector.")
        if depth <= 0:
            raise ValueError("depth must be positive for a resampler projector.")
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by num_heads={num_heads}.")

        self.token_projector = MLPProjector(in_dim, out_dim, dropout=dropout)
        self.latents = nn.Parameter(torch.randn(num_latents, out_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                ResamplerBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.token_projector(x)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask <= 0
            if key_padding_mask.any():
                key_padding_mask = key_padding_mask.to(dtype=torch.bool, device=tokens.device)
                empty_rows = key_padding_mask.all(dim=1)
                if empty_rows.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[empty_rows, 0] = False
                    tokens = tokens.clone()
                    tokens[empty_rows] = 0

        latents = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, tokens, key_padding_mask=key_padding_mask)
        return self.output_norm(latents)


class ModalityProjector(nn.Module):
    """Project per-image embeddings and pool variable-length image sets."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        projector_type: str = "mlp",
        num_latents: int = 64,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.projector_type = str(projector_type).strip().lower() or "mlp"
        if self.projector_type == "mlp":
            self.projector = MLPProjector(
                in_dim=in_dim,
                out_dim=out_dim,
                dropout=dropout,
            )
        elif self.projector_type == "resampler":
            self.projector = ResamplerProjector(
                in_dim=in_dim,
                out_dim=out_dim,
                num_latents=num_latents,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")

    def build_output_mask(
        self,
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        output_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones((batch_size, output_length), device=device, dtype=dtype)
        if self.projector_type == "mlp":
            return mask.to(device=device, dtype=dtype)
        active = mask.to(device=device).sum(dim=1) > 0
        return active.to(dtype=dtype).unsqueeze(1).expand(batch_size, output_length)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, n_images, in_dim] or [batch, in_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.projector_type == "mlp":
            projected = self.projector(x)  # [batch, n_images, out_dim]
        else:
            projected = self.projector(x, mask=mask)
        output_mask = self.build_output_mask(
            mask,
            batch_size=projected.shape[0],
            output_length=projected.shape[1],
            device=projected.device,
            dtype=projected.dtype,
        )
        pooled = masked_mean_pool(projected, mask=output_mask)
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
