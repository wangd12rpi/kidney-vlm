from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _dtype_from_name(name: str) -> torch.dtype | str:
    normalized = str(name).strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return dtype_map[normalized]


@dataclass
class PMCCaptionProjectorConfig:
    input_dim: int
    output_dim: int
    has_absence_token: bool = True


class PMCCaptionProjector(nn.Module):
    """Project a single pooled MedSigLIP embedding into the LLM token space."""

    def __init__(self, config: PMCCaptionProjectorConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.input_dim)
        self.fc1 = nn.Linear(config.input_dim, config.output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.output_dim, config.output_dim)
        # Learned absence token — inactive during Stage 1 but saved in checkpoint
        # for architectural compatibility with Stage 2a/2b.
        if config.has_absence_token:
            self.absence_token = nn.Parameter(torch.zeros(1, 1, config.output_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        if hasattr(self, "absence_token"):
            nn.init.normal_(self.absence_token, std=0.02)

    def forward(
        self,
        visual_features: torch.Tensor,
        *,
        feature_mask: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modality_mask is not None and not torch.all(modality_mask.to(dtype=torch.bool)):
            raise ValueError(
                "PMC-OA projector training expects a pooled radiology embedding for every sample."
            )

        if visual_features.ndim == 3:
            if visual_features.size(1) != 1:
                raise ValueError(
                    "PMC-OA extractor saves one pooled embedding per image, so the projector "
                    f"expects a single visual token. Got shape {tuple(visual_features.shape)}."
                )
            if feature_mask is not None and feature_mask.ndim != 2:
                raise ValueError(
                    f"Expected feature_mask to have shape [batch, 1], got {tuple(feature_mask.shape)}"
                )
            visual_features = visual_features[:, 0, :]
        elif visual_features.ndim != 2:
            raise ValueError(
                f"Expected visual_features to have shape [batch, dim] or [batch, 1, dim], got {tuple(visual_features.shape)}"
            )

        projected = self.norm(visual_features)
        projected = self.fc1(projected)
        projected = self.act(projected)
        projected = self.fc2(projected).unsqueeze(1)
        projected_mask = torch.ones(
            (projected.size(0), 1),
            dtype=torch.long,
            device=projected.device,
        )
        return projected, projected_mask


class PMCOACaptionProjectorModel(nn.Module):
    """Frozen causal LM plus a trainable projector for PMC-OA caption generation."""

    def __init__(
        self,
        *,
        llm: nn.Module,
        tokenizer: Any,
        projector: PMCCaptionProjector,
        llm_model_name_or_path: str,
        freeze_llm: bool = True,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.projector = projector
        self.llm_model_name_or_path = str(llm_model_name_or_path)
        self.text_embeddings = self.llm.get_input_embeddings()
        if freeze_llm:
            _freeze_module(self.llm)

    @classmethod
    def from_pretrained(
        cls,
        *,
        llm_model_name_or_path: str,
        visual_input_dim: int,
        trust_remote_code: bool = True,
        llm_dtype: str = "auto",
        freeze_llm: bool = True,
        gradient_checkpointing: bool = False,
        use_cache: bool | None = None,
    ) -> "PMCOACaptionProjectorModel":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is not installed. Install project dependencies first.") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "unk_token_id", None) is not None:
                tokenizer.pad_token = tokenizer.unk_token

        dtype = _dtype_from_name(llm_dtype)
        llm_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            llm_kwargs["torch_dtype"] = dtype
        llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name_or_path,
            **llm_kwargs,
        )
        if use_cache is not None and hasattr(llm, "config"):
            llm.config.use_cache = bool(use_cache)
        if gradient_checkpointing:
            if hasattr(llm, "gradient_checkpointing_enable"):
                llm.gradient_checkpointing_enable()
            if hasattr(llm, "enable_input_require_grads"):
                llm.enable_input_require_grads()
        elif hasattr(llm, "gradient_checkpointing_disable"):
            llm.gradient_checkpointing_disable()
        embedding_layer = llm.get_input_embeddings()
        output_dim = int(embedding_layer.weight.shape[-1])
        projector = PMCCaptionProjector(
            PMCCaptionProjectorConfig(
                input_dim=int(visual_input_dim),
                output_dim=output_dim,
            )
        )
        return cls(
            llm=llm,
            tokenizer=tokenizer,
            projector=projector,
            llm_model_name_or_path=llm_model_name_or_path,
            freeze_llm=freeze_llm,
        )

    def count_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def count_total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save_projector(self, output_dir: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / "projector.pt"
        payload = {
            "llm_model_name_or_path": self.llm_model_name_or_path,
            "projector_config": asdict(self.projector.config),
            "projector_state_dict": self.projector.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path
    
    def _build_inputs(
        self,
        *,
        visual_tokens: torch.Tensor,
        visual_token_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_embed_dtype = self.text_embeddings.weight.dtype
        prompt_embeds = self.text_embeddings(prompt_input_ids).to(dtype=text_embed_dtype)
        target_embeds = self.text_embeddings(target_input_ids).to(dtype=text_embed_dtype)
        visual_tokens = visual_tokens.to(dtype=text_embed_dtype)

        batch_size = visual_tokens.size(0)
        hidden_dim = visual_tokens.size(-1)

        # Compute per-sample lengths from masks (no Python loop needed).
        visual_lengths = visual_token_mask.sum(dim=-1)     # (B,)
        prompt_lengths = prompt_attention_mask.sum(dim=-1)  # (B,)
        target_lengths = target_attention_mask.sum(dim=-1)  # (B,)
        total_lengths = visual_lengths + prompt_lengths + target_lengths  # (B,)
        max_length = int(total_lengths.max().item())

        # Pre-allocate output tensors.
        device = visual_tokens.device
        padded_embeddings = visual_tokens.new_zeros((batch_size, max_length, hidden_dim))
        padded_masks = visual_token_mask.new_zeros((batch_size, max_length))
        padded_labels = target_input_ids.new_full((batch_size, max_length), -100)

        for i in range(batch_size):
            vl = int(visual_lengths[i].item())
            pl = int(prompt_lengths[i].item())
            tl = int(target_lengths[i].item())

            # Place visual → prompt → target contiguously.
            offset = 0
            padded_embeddings[i, offset : offset + vl] = visual_tokens[i, :vl]
            offset += vl
            padded_embeddings[i, offset : offset + pl] = prompt_embeds[i, :pl]
            offset += pl
            padded_embeddings[i, offset : offset + tl] = target_embeds[i, :tl]

            total = vl + pl + tl
            padded_masks[i, :total] = 1
            # Labels: -100 for visual+prompt positions, actual token IDs for target.
            if tl > 0:
                padded_labels[i, vl + pl : vl + pl + tl] = target_input_ids[i, :tl]

        return padded_embeddings, padded_masks, padded_labels

    # def _build_inputs(
    #     self,
    #     *,
    #     visual_tokens: torch.Tensor,
    #     visual_token_mask: torch.Tensor,
    #     prompt_input_ids: torch.Tensor,
    #     prompt_attention_mask: torch.Tensor,
    #     target_input_ids: torch.Tensor,
    #     target_attention_mask: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     text_embed_dtype = self.text_embeddings.weight.dtype
    #     prompt_embeds = self.text_embeddings(prompt_input_ids).to(dtype=text_embed_dtype)
    #     target_embeds = self.text_embeddings(target_input_ids).to(dtype=text_embed_dtype)
    #     visual_tokens = visual_tokens.to(dtype=text_embed_dtype)

    #     batch_size = visual_tokens.size(0)
    #     hidden_dim = visual_tokens.size(-1)
    #     combined_embeddings: list[torch.Tensor] = []
    #     combined_masks: list[torch.Tensor] = []
    #     combined_labels: list[torch.Tensor] = []

    #     for index in range(batch_size):
    #         visual_length = int(visual_token_mask[index].sum().item())
    #         prompt_length = int(prompt_attention_mask[index].sum().item())
    #         target_length = int(target_attention_mask[index].sum().item())

    #         sample_embeddings = torch.cat(
    #             [
    #                 visual_tokens[index, :visual_length],
    #                 prompt_embeds[index, :prompt_length],
    #                 target_embeds[index, :target_length],
    #             ],
    #             dim=0,
    #         )
    #         sample_mask = torch.ones(
    #             sample_embeddings.size(0),
    #             dtype=torch.long,
    #             device=sample_embeddings.device,
    #         )
    #         sample_labels = torch.full(
    #             (sample_embeddings.size(0),),
    #             -100,
    #             dtype=torch.long,
    #             device=sample_embeddings.device,
    #         )
    #         if target_length > 0:
    #             start = visual_length + prompt_length
    #             sample_labels[start : start + target_length] = target_input_ids[index, :target_length]

    #         combined_embeddings.append(sample_embeddings)
    #         combined_masks.append(sample_mask)
    #         combined_labels.append(sample_labels)

    #     max_length = max(embeddings.size(0) for embeddings in combined_embeddings)
    #     padded_embeddings = visual_tokens.new_zeros((batch_size, max_length, hidden_dim))
    #     padded_masks = visual_token_mask.new_zeros((batch_size, max_length))
    #     padded_labels = target_input_ids.new_full((batch_size, max_length), -100)

    #     for index, sample_embeddings in enumerate(combined_embeddings):
    #         length = sample_embeddings.size(0)
    #         padded_embeddings[index, :length] = sample_embeddings
    #         padded_masks[index, :length] = combined_masks[index]
    #         padded_labels[index, :length] = combined_labels[index]

    #     return padded_embeddings, padded_masks, padded_labels

    def forward(
        self,
        *,
        visual_features: torch.Tensor,
        visual_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_tokens, visual_token_mask = self.projector(
            visual_features,
            feature_mask=visual_attention_mask,
            modality_mask=modality_mask,
        )
        inputs_embeds, attention_mask, labels = self._build_inputs(
            visual_tokens=visual_tokens,
            visual_token_mask=visual_token_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    @torch.inference_mode()
    def caption_token_loss(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
