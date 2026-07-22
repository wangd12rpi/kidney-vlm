from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


def resolve_language_model_hidden_size(language_model: nn.Module) -> int:
    """Resolve LM embedding width across common HF config layouts."""
    config = getattr(language_model, "config", None)
    config_candidates: list[Any] = []
    if config is not None:
        config_candidates.append(config)
        for attr_name in ("text_config", "llm_config", "language_config", "decoder", "model_config"):
            try:
                nested_config = getattr(config, attr_name)
            except AttributeError:
                continue
            if nested_config is not None:
                config_candidates.append(nested_config)
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            for kwargs in ({}, {"decoder": True}):
                try:
                    text_config = get_text_config(**kwargs)
                except TypeError:
                    continue
                if text_config is not None:
                    config_candidates.append(text_config)

    for candidate in config_candidates:
        for attr_name in ("hidden_size", "d_model", "n_embd", "embed_dim", "embedding_dim"):
            try:
                value = getattr(candidate, attr_name)
            except AttributeError:
                continue
            if value is not None:
                return int(value)

    get_input_embeddings = getattr(language_model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        for attr_name in ("embedding_dim", "num_features", "out_features"):
            value = getattr(embeddings, attr_name, None)
            if value is not None:
                return int(value)
        weight = getattr(embeddings, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) >= 2:
            return int(weight.shape[1])

    config_type = type(config).__name__ if config is not None else "None"
    raise ValueError(f"Could not resolve language model hidden size from config type {config_type}.")


def _resolve_per_layer_text_model(language_model: nn.Module) -> nn.Module:
    model = getattr(language_model, "model", language_model)
    return getattr(model, "language_model", model)


def build_soft_prefix_per_layer_inputs(
    language_model: nn.Module,
    *,
    input_ids: torch.Tensor,
    prefix_length: int,
    device: torch.device,
    dtype: torch.dtype,
    prefix_token_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build Gemma4-style per-layer inputs for soft-prefix + text embeddings."""
    if prefix_token_mask is None and prefix_length <= 0:
        return None

    text_model = _resolve_per_layer_text_model(language_model)
    hidden_size_per_layer_input = int(getattr(text_model, "hidden_size_per_layer_input", 0) or 0)
    if hidden_size_per_layer_input <= 0:
        config = getattr(text_model, "config", None)
        hidden_size_per_layer_input = int(getattr(config, "hidden_size_per_layer_input", 0) or 0)
    if hidden_size_per_layer_input <= 0:
        return None

    get_per_layer_inputs = getattr(text_model, "get_per_layer_inputs", None)
    if not callable(get_per_layer_inputs):
        return None

    with torch.no_grad():
        text_per_layer_inputs = get_per_layer_inputs(input_ids, None)
    text_per_layer_inputs = text_per_layer_inputs.to(device=device, dtype=dtype)
    if prefix_token_mask is not None:
        prefix_mask = prefix_token_mask.to(device=device, dtype=torch.bool)
        if prefix_mask.shape != input_ids.shape:
            raise ValueError(
                "prefix_token_mask shape must match input_ids shape for interleaved soft-prefix inputs. "
                f"Got mask={tuple(prefix_mask.shape)} input_ids={tuple(input_ids.shape)}."
            )
        return text_per_layer_inputs.masked_fill(prefix_mask[:, :, None, None], 0)

    prefix_per_layer_inputs = torch.zeros(
        (
            input_ids.shape[0],
            prefix_length,
            text_per_layer_inputs.shape[2],
            text_per_layer_inputs.shape[3],
        ),
        device=device,
        dtype=dtype,
    )
    return torch.cat([prefix_per_layer_inputs, text_per_layer_inputs], dim=1)


def _text_config(language_model: nn.Module) -> Any:
    config = getattr(language_model, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()
    return getattr(config, "text_config", config)


def _conditional_lm_inner_modules(language_model: nn.Module) -> tuple[nn.Module, nn.Module] | None:
    outer_model = getattr(language_model, "model", None)
    text_model = getattr(outer_model, "language_model", None)
    lm_head = getattr(language_model, "lm_head", None)
    if text_model is None or lm_head is None:
        return None
    return text_model, lm_head


def forward_language_model_with_soft_prefix(
    language_model: nn.Module,
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    labels: torch.Tensor | None,
    prefix_length: int,
    prefix_token_mask: torch.Tensor | None = None,
    logits_to_keep: int | None = None,
) -> Any:
    per_layer_inputs = build_soft_prefix_per_layer_inputs(
        language_model,
        input_ids=input_ids,
        prefix_length=prefix_length,
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
        prefix_token_mask=prefix_token_mask,
    )
    conditional_modules = _conditional_lm_inner_modules(language_model)
    if per_layer_inputs is None or conditional_modules is None:
        model_kwargs = {"per_layer_inputs": per_layer_inputs} if per_layer_inputs is not None else {}
        if logits_to_keep is not None:
            model_kwargs["logits_to_keep"] = int(logits_to_keep)
        return language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **model_kwargs,
        )

    text_model, lm_head = conditional_modules
    outputs = text_model(
        per_layer_inputs=per_layer_inputs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    if logits_to_keep is not None:
        hidden_states = hidden_states[:, -int(logits_to_keep) :, :]
    logits = lm_head(hidden_states)
    config = _text_config(language_model)
    final_logit_softcapping = getattr(config, "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        logits = logits / final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * final_logit_softcapping

    loss = None
    if labels is not None:
        logits_for_loss = logits.float()
        shift_logits = logits_for_loss[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits_for_loss.device)
        shift_logits = shift_logits[shift_attention_mask != 0].contiguous()
        shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
        vocab_size = int(getattr(config, "vocab_size", logits.shape[-1]))
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, vocab_size), shift_labels.view(-1).to(shift_logits.device))

    return SimpleNamespace(
        loss=loss,
        logits=logits,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    )


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
