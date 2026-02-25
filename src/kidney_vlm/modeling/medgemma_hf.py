from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MedGemmaHFModel(nn.Module):
    """Generic HF wrapper for MedGemma-family checkpoints.

    Notes:
    - This scaffold intentionally avoids guessing task-specific processor/model APIs.
    - Update this wrapper once the exact checkpoint and multimodal I/O contract are fixed.
    """

    def __init__(self, backbone: nn.Module, processor: Any):
        super().__init__()
        self.backbone = backbone
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> "MedGemmaHFModel":
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is not installed. Install project dependencies first.") from exc

        backbone = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        return cls(backbone=backbone, processor=processor)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone(*args, **kwargs)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
