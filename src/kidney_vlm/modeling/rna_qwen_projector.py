from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kidney_vlm.modeling.path_projectors import (
    ModalityProjector,
    forward_language_model_with_soft_prefix,
    masked_mean_pool,
    resolve_language_model_hidden_size,
)


class RnaPrefixExpander(nn.Module):
    """Turn a compact RNA representation into token-conditioned soft prompts."""

    def __init__(
        self,
        hidden_size: int,
        output_tokens: int,
        *,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.output_tokens = int(output_tokens)
        if self.output_tokens < 1:
            raise ValueError("RnaPrefixExpander.output_tokens must be >= 1.")
        hidden_dim = max(self.hidden_size, int(round(self.hidden_size * float(mlp_ratio))))
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.token_embeddings = nn.Parameter(torch.empty(self.output_tokens, self.hidden_size))
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_size),
        )
        nn.init.normal_(self.token_embeddings, mean=0.0, std=0.02)

    def forward(self, projected_tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = masked_mean_pool(projected_tokens, mask=mask)
        base = self.input_norm(pooled).unsqueeze(1)
        token_conditioned = base + self.token_embeddings.unsqueeze(0)
        return self.net(token_conditioned)


class RnaQwenProjectorLM(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        rna_in_dim: int,
        *,
        projector_type: str = "mlp",
        projector_num_latents: int = 64,
        projector_depth: int = 2,
        projector_num_heads: int = 8,
        projector_mlp_ratio: float = 4.0,
        projector_dropout: float = 0.0,
        rna_prefix_tokens: int = 0,
        rna_prefix_expander_mlp_ratio: float = 1.0,
        language_model_is_quantized: bool = False,
    ):
        super().__init__()
        self.language_model = language_model
        self.language_model_is_quantized = bool(language_model_is_quantized)
        self.rna_in_dim = int(rna_in_dim)
        self.hidden_size = resolve_language_model_hidden_size(language_model)
        self.projector_config = {
            "projector_type": str(projector_type).strip().lower() or "mlp",
            "projector_num_latents": int(projector_num_latents),
            "projector_depth": int(projector_depth),
            "projector_num_heads": int(projector_num_heads),
            "projector_mlp_ratio": float(projector_mlp_ratio),
            "projector_dropout": float(projector_dropout),
            "rna_prefix_tokens": int(rna_prefix_tokens),
            "rna_prefix_expander_mlp_ratio": float(rna_prefix_expander_mlp_ratio),
        }
        self.rna_prefix_tokens = max(0, int(rna_prefix_tokens))
        self.projectors = nn.ModuleDict(
            {
                "rna": ModalityProjector(
                    in_dim=self.rna_in_dim,
                    out_dim=self.hidden_size,
                    projector_type=self.projector_config["projector_type"],
                    num_latents=self.projector_config["projector_num_latents"],
                    depth=self.projector_config["projector_depth"],
                    num_heads=self.projector_config["projector_num_heads"],
                    mlp_ratio=self.projector_config["projector_mlp_ratio"],
                    dropout=self.projector_config["projector_dropout"],
                )
            }
        )
        if self.projector_config["projector_type"] == "mlp" and self.rna_prefix_tokens > 0:
            self.projectors["rna_prefix_expander"] = RnaPrefixExpander(
                hidden_size=self.hidden_size,
                output_tokens=self.rna_prefix_tokens,
                mlp_ratio=self.projector_config["rna_prefix_expander_mlp_ratio"],
                dropout=self.projector_config["projector_dropout"],
            )
        if hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = False
        self.freeze_language_model()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        rna_in_dim: int,
        projector_type: str = "mlp",
        projector_num_latents: int = 64,
        projector_depth: int = 2,
        projector_num_heads: int = 8,
        projector_mlp_ratio: float = 4.0,
        projector_dropout: float = 0.0,
        rna_prefix_tokens: int = 0,
        rna_prefix_expander_mlp_ratio: float = 1.0,
        trust_remote_code: bool = True,
        torch_dtype: str | torch.dtype | None = None,
        attn_implementation: str | None = None,
        load_in_8bit: bool = False,
        device_map: Any | None = None,
        **kwargs: Any,
    ) -> "RnaQwenProjectorLM":
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("transformers is not installed. Install project dependencies first.") from exc

        resolved_dtype = _resolve_torch_dtype(torch_dtype)
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["low_cpu_mem_usage"] = True
            if device_map is not None:
                model_kwargs["device_map"] = device_map
        elif resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        language_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        return cls(
            language_model=language_model,
            rna_in_dim=rna_in_dim,
            projector_type=projector_type,
            projector_num_latents=projector_num_latents,
            projector_depth=projector_depth,
            projector_num_heads=projector_num_heads,
            projector_mlp_ratio=projector_mlp_ratio,
            projector_dropout=projector_dropout,
            rna_prefix_tokens=rna_prefix_tokens,
            rna_prefix_expander_mlp_ratio=rna_prefix_expander_mlp_ratio,
            language_model_is_quantized=load_in_8bit,
        )

    def freeze_language_model(self) -> None:
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> "RnaQwenProjectorLM":
        super().train(mode)
        self.language_model.eval()
        return self

    def move_trainable_modules_to(self, device: torch.device, *, dtype: torch.dtype | None = None) -> None:
        self.projectors.to(device=device, dtype=dtype)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        rna_features: torch.Tensor | None = None,
        rna_feature_mask: torch.Tensor | None = None,
    ) -> Any:
        if rna_features is None:
            raise ValueError("rna_features are required for RNA projector training.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if rna_feature_mask is None:
            rna_feature_mask = torch.ones(
                rna_features.shape[:2],
                device=rna_features.device,
                dtype=attention_mask.dtype,
            )

        rna_projected, _ = self.projectors["rna"](rna_features, rna_feature_mask)
        projected_mask = self.projectors["rna"].build_output_mask(
            rna_feature_mask,
            batch_size=rna_projected.shape[0],
            output_length=rna_projected.shape[1],
            device=rna_projected.device,
            dtype=rna_projected.dtype,
        )
        if "rna_prefix_expander" in self.projectors:
            rna_projected = self.projectors["rna_prefix_expander"](rna_projected, mask=projected_mask)
            prefix_attention = torch.ones(
                (rna_projected.shape[0], rna_projected.shape[1]),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
        else:
            prefix_attention = self.projectors["rna"].build_output_mask(
                rna_feature_mask,
                batch_size=rna_projected.shape[0],
                output_length=rna_projected.shape[1],
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        rna_projected = rna_projected.to(dtype=text_embeddings.dtype)
        combined_embeddings = torch.cat([rna_projected, text_embeddings], dim=1)
        combined_attention = torch.cat([prefix_attention, attention_mask], dim=1)

        combined_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (labels.shape[0], rna_projected.shape[1]),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)

        position_ids = combined_attention.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp_min(0)
        position_ids = position_ids.masked_fill(combined_attention == 0, 0)
        return forward_language_model_with_soft_prefix(
            self.language_model,
            input_ids=input_ids,
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention,
            position_ids=position_ids,
            labels=combined_labels,
            prefix_length=rna_projected.shape[1],
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _resolve_torch_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[normalized]
