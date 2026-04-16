from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kidney_vlm.modeling.path_projectors import ModalityProjector


class DnamQwenProjectorLM(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        dnam_in_dim: int,
        *,
        projector_type: str = "mlp",
        projector_num_latents: int = 64,
        projector_depth: int = 2,
        projector_num_heads: int = 8,
        projector_mlp_ratio: float = 4.0,
        projector_dropout: float = 0.0,
    ):
        super().__init__()
        self.language_model = language_model
        self.dnam_in_dim = int(dnam_in_dim)
        self.hidden_size = int(getattr(language_model.config, "hidden_size"))
        self.projector_config = {
            "projector_type": str(projector_type).strip().lower() or "mlp",
            "projector_num_latents": int(projector_num_latents),
            "projector_depth": int(projector_depth),
            "projector_num_heads": int(projector_num_heads),
            "projector_mlp_ratio": float(projector_mlp_ratio),
            "projector_dropout": float(projector_dropout),
        }
        self.projectors = nn.ModuleDict(
            {
                "dnam": ModalityProjector(
                    in_dim=self.dnam_in_dim,
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
        if hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = False
        self.freeze_language_model()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        dnam_in_dim: int,
        projector_type: str = "mlp",
        projector_num_latents: int = 64,
        projector_depth: int = 2,
        projector_num_heads: int = 8,
        projector_mlp_ratio: float = 4.0,
        projector_dropout: float = 0.0,
        trust_remote_code: bool = True,
        torch_dtype: str | torch.dtype | None = None,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> "DnamQwenProjectorLM":
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("transformers is not installed. Install project dependencies first.") from exc

        resolved_dtype = _resolve_torch_dtype(torch_dtype)
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        language_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        return cls(
            language_model=language_model,
            dnam_in_dim=dnam_in_dim,
            projector_type=projector_type,
            projector_num_latents=projector_num_latents,
            projector_depth=projector_depth,
            projector_num_heads=projector_num_heads,
            projector_mlp_ratio=projector_mlp_ratio,
            projector_dropout=projector_dropout,
        )

    def freeze_language_model(self) -> None:
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> "DnamQwenProjectorLM":
        super().train(mode)
        self.language_model.eval()
        return self

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        dnam_features: torch.Tensor | None = None,
        dnam_feature_mask: torch.Tensor | None = None,
    ) -> Any:
        if dnam_features is None:
            raise ValueError("dnam_features are required for DNAm projector training.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if dnam_feature_mask is None:
            dnam_feature_mask = torch.ones(
                dnam_features.shape[:2],
                device=dnam_features.device,
                dtype=attention_mask.dtype,
            )

        dnam_projected, _ = self.projectors["dnam"](dnam_features, dnam_feature_mask)
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        dnam_projected = dnam_projected.to(dtype=text_embeddings.dtype)

        prefix_attention = self.projectors["dnam"].build_output_mask(
            dnam_feature_mask,
            batch_size=dnam_projected.shape[0],
            output_length=dnam_projected.shape[1],
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        combined_embeddings = torch.cat([dnam_projected, text_embeddings], dim=1)
        combined_attention = torch.cat([prefix_attention, attention_mask], dim=1)

        combined_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (labels.shape[0], dnam_projected.shape[1]),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)

        position_ids = combined_attention.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp_min(0)
        position_ids = position_ids.masked_fill(combined_attention == 0, 0)

        return self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention,
            position_ids=position_ids,
            labels=combined_labels,
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
