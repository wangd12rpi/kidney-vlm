from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kidney_vlm.modeling.dnam_qwen_projector import DnamPrefixExpander
from kidney_vlm.modeling.path_projectors import (
    ModalityProjector,
    forward_language_model_with_soft_prefix,
    resolve_language_model_hidden_size,
)
from kidney_vlm.modeling.rna_qwen_projector import RnaPrefixExpander
from kidney_vlm.vqa.constants import MODALITIES
from kidney_vlm.vqa.stage_config import cfg_get, cfg_list, clean_text, resolve_repo_path, resolve_torch_dtype

PROJECTOR_STATE_KEYS = {
    "pathology": "path_projector_state_dict",
    "radiology": "radiology_projector_state_dict",
    "dnam": "dnam_projector_state_dict",
    "rna": "rna_projector_state_dict",
}
PROJECTOR_EMBEDDING_DIM_KEYS = {
    "pathology": "pathology_embedding_dim",
    "radiology": "radiology_embedding_dim",
    "dnam": "dnam_embedding_dim",
    "rna": "rna_embedding_dim",
}


def set_module_trainable(module: nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = bool(trainable)


def checkpoint_int(checkpoint: Mapping[str, Any], block_cfg: Any, key: str, *, default: int | None = None) -> int:
    value = checkpoint.get(key)
    if value is None:
        value = cfg_get(block_cfg, key, default)
    if value is None:
        raise ValueError(f"Projector checkpoint/config is missing required integer key: {key}")
    return int(value)


def checkpoint_float(checkpoint: Mapping[str, Any], block_cfg: Any, key: str, *, default: float) -> float:
    value = checkpoint.get(key)
    if value is None:
        value = cfg_get(block_cfg, key, default)
    return float(value)


def build_projector_module(
    *,
    repo_root: Path,
    modality: str,
    block_cfg: Any,
    hidden_size: int,
) -> tuple[nn.ModuleDict, dict[str, Any]]:
    raw_checkpoint_path = clean_text(cfg_get(block_cfg, "checkpoint_path", ""))
    if not raw_checkpoint_path:
        raise ValueError(f"vqa_train.projectors.{modality}.checkpoint_path must point to a stage-1 projector checkpoint.")
    checkpoint_path = resolve_repo_path(repo_root, raw_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{modality} projector checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{modality} projector checkpoint must be a mapping: {checkpoint_path}")
    state_key = PROJECTOR_STATE_KEYS[modality]
    if state_key not in checkpoint:
        raise KeyError(f"{modality} projector checkpoint is missing state key '{state_key}': {checkpoint_path}")

    checkpoint_hidden_size = checkpoint.get("hidden_size")
    if checkpoint_hidden_size is not None and int(checkpoint_hidden_size) != int(hidden_size):
        raise ValueError(
            f"{modality} projector hidden_size={int(checkpoint_hidden_size)} does not match "
            f"language model hidden_size={int(hidden_size)}."
        )

    embedding_dim = checkpoint_int(checkpoint, block_cfg, PROJECTOR_EMBEDDING_DIM_KEYS[modality])
    projector_type = str(checkpoint.get("projector_type") or cfg_get(block_cfg, "projector_type", "mlp")).strip() or "mlp"
    projector_num_latents = checkpoint_int(checkpoint, block_cfg, "projector_num_latents", default=64)
    projector_depth = checkpoint_int(checkpoint, block_cfg, "projector_depth", default=2)
    projector_num_heads = checkpoint_int(checkpoint, block_cfg, "projector_num_heads", default=8)
    projector_mlp_ratio = checkpoint_float(checkpoint, block_cfg, "projector_mlp_ratio", default=4.0)
    projector_dropout = checkpoint_float(checkpoint, block_cfg, "projector_dropout", default=0.0)

    module = nn.ModuleDict(
        {
            modality: ModalityProjector(
                in_dim=embedding_dim,
                out_dim=hidden_size,
                projector_type=projector_type,
                num_latents=projector_num_latents,
                depth=projector_depth,
                num_heads=projector_num_heads,
                mlp_ratio=projector_mlp_ratio,
                dropout=projector_dropout,
            )
        }
    )
    prefix_tokens = 0
    prefix_expander_mlp_ratio = 1.0
    if modality == "dnam":
        prefix_tokens = checkpoint_int(checkpoint, block_cfg, "dnam_prefix_tokens", default=0)
        prefix_expander_mlp_ratio = checkpoint_float(
            checkpoint,
            block_cfg,
            "dnam_prefix_expander_mlp_ratio",
            default=1.0,
        )
        if projector_type == "mlp" and prefix_tokens > 0:
            module["dnam_prefix_expander"] = DnamPrefixExpander(
                hidden_size=hidden_size,
                output_tokens=prefix_tokens,
                mlp_ratio=prefix_expander_mlp_ratio,
                dropout=projector_dropout,
            )
    if modality == "rna":
        prefix_tokens = checkpoint_int(checkpoint, block_cfg, "rna_prefix_tokens", default=0)
        prefix_expander_mlp_ratio = checkpoint_float(
            checkpoint,
            block_cfg,
            "rna_prefix_expander_mlp_ratio",
            default=1.0,
        )
        if projector_type == "mlp" and prefix_tokens > 0:
            module["rna_prefix_expander"] = RnaPrefixExpander(
                hidden_size=hidden_size,
                output_tokens=prefix_tokens,
                mlp_ratio=prefix_expander_mlp_ratio,
                dropout=projector_dropout,
            )

    module.load_state_dict(checkpoint[state_key], strict=True)
    trainable = bool(cfg_get(block_cfg, "trainable", False))
    set_module_trainable(module, trainable)
    metadata = {
        "modality": modality,
        "checkpoint_path": str(checkpoint_path),
        "state_key": state_key,
        "trainable": trainable,
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
        "projector_type": projector_type,
        "projector_num_latents": projector_num_latents,
        "projector_depth": projector_depth,
        "projector_num_heads": projector_num_heads,
        "projector_mlp_ratio": projector_mlp_ratio,
        "projector_dropout": projector_dropout,
        "prefix_tokens": prefix_tokens,
        "prefix_expander_mlp_ratio": prefix_expander_mlp_ratio,
    }
    return module, metadata


def load_projectors(
    stage_cfg: Any,
    *,
    repo_root: Path,
    hidden_size: int,
) -> tuple[dict[str, nn.ModuleDict], dict[str, dict[str, Any]]]:
    projectors_cfg = cfg_get(stage_cfg, "projectors", {})
    projectors: dict[str, nn.ModuleDict] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        block = cfg_get(projectors_cfg, modality, {})
        if not bool(cfg_get(block, "enabled", False)):
            continue
        module, modality_metadata = build_projector_module(
            repo_root=repo_root,
            modality=modality,
            block_cfg=block,
            hidden_size=hidden_size,
        )
        projectors[modality] = module
        metadata[modality] = modality_metadata
    if not projectors:
        raise RuntimeError("No VQA projectors are enabled. Enable at least one vqa_train.projectors.<modality> block.")
    return projectors, metadata


class OncoVLMVQASFTModel(nn.Module):
    def __init__(
        self,
        *,
        language_model: nn.Module,
        projectors: dict[str, nn.ModuleDict] | None = None,
        projector_metadata: dict[str, dict[str, Any]] | None = None,
    ):
        super().__init__()
        projectors = projectors or {}
        self.language_model = language_model
        self.path_projectors = projectors.get("pathology")
        self.radiology_projectors = projectors.get("radiology")
        self.dnam_projectors = projectors.get("dnam")
        self.rna_projectors = projectors.get("rna")
        self.projector_metadata = projector_metadata or {}
        self.hidden_size = resolve_language_model_hidden_size(language_model)
        config = getattr(self.language_model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False

    def projector_modules(self) -> list[tuple[str, nn.ModuleDict]]:
        modules: list[tuple[str, nn.ModuleDict]] = []
        for modality, module in [
            ("pathology", self.path_projectors),
            ("radiology", self.radiology_projectors),
            ("dnam", self.dnam_projectors),
            ("rna", self.rna_projectors),
        ]:
            if module is not None:
                modules.append((modality, module))
        return modules

    def move_projectors_to(self, device: torch.device, *, dtype: torch.dtype | None = None) -> None:
        for _, module in self.projector_modules():
            module.to(device=device, dtype=dtype)

    def set_frozen_projectors_eval(self) -> None:
        for modality, module in self.projector_modules():
            if not bool(self.projector_metadata[modality]["trainable"]):
                module.eval()

    def _project_prefix(
        self,
        *,
        modality: str,
        module: nn.ModuleDict | None,
        features: torch.Tensor | None,
        feature_mask: torch.Tensor | None,
        attention_device: torch.device,
        attention_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if features is None:
            return None
        if module is None:
            raise RuntimeError(f"Batch contains {modality} features, but no {modality} projector is loaded.")
        if feature_mask is None:
            feature_mask = torch.ones(features.shape[:2], device=features.device, dtype=torch.long)

        projected, _ = module[modality](features, feature_mask)
        projected_mask = module[modality].build_output_mask(
            feature_mask,
            batch_size=projected.shape[0],
            output_length=projected.shape[1],
            device=projected.device,
            dtype=projected.dtype,
        )
        expander_key = f"{modality}_prefix_expander"
        if expander_key in module:
            projected = module[expander_key](projected, mask=projected_mask)
            active_rows = feature_mask.to(device=attention_device).sum(dim=1) > 0
            prefix_attention = active_rows.to(dtype=attention_dtype).unsqueeze(1).expand(projected.shape[0], projected.shape[1])
        else:
            prefix_attention = projected_mask.to(device=attention_device, dtype=attention_dtype)
        return projected, prefix_attention

    def _project_available_prefixes(
        self,
        *,
        attention_mask: torch.Tensor,
        pathology_features: torch.Tensor | None,
        pathology_feature_mask: torch.Tensor | None,
        radiology_features: torch.Tensor | None,
        radiology_feature_mask: torch.Tensor | None,
        dnam_features: torch.Tensor | None,
        dnam_feature_mask: torch.Tensor | None,
        rna_features: torch.Tensor | None,
        rna_feature_mask: torch.Tensor | None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        prefix_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for modality, module, features, feature_mask in [
            ("pathology", self.path_projectors, pathology_features, pathology_feature_mask),
            ("radiology", self.radiology_projectors, radiology_features, radiology_feature_mask),
            ("dnam", self.dnam_projectors, dnam_features, dnam_feature_mask),
            ("rna", self.rna_projectors, rna_features, rna_feature_mask),
        ]:
            projected = self._project_prefix(
                modality=modality,
                module=module,
                features=features,
                feature_mask=feature_mask,
                attention_device=attention_mask.device,
                attention_dtype=attention_mask.dtype,
            )
            if projected is not None:
                prefix_outputs[modality] = projected
        if not prefix_outputs:
            raise RuntimeError("VQA batch has no projector prefix features.")
        return prefix_outputs

    def _cached_available_prefixes(
        self,
        *,
        attention_mask: torch.Tensor,
        pathology_prefix_embeddings: torch.Tensor | None,
        pathology_prefix_mask: torch.Tensor | None,
        radiology_prefix_embeddings: torch.Tensor | None,
        radiology_prefix_mask: torch.Tensor | None,
        dnam_prefix_embeddings: torch.Tensor | None,
        dnam_prefix_mask: torch.Tensor | None,
        rna_prefix_embeddings: torch.Tensor | None,
        rna_prefix_mask: torch.Tensor | None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        prefix_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for modality, prefix_embeddings, prefix_mask in [
            ("pathology", pathology_prefix_embeddings, pathology_prefix_mask),
            ("radiology", radiology_prefix_embeddings, radiology_prefix_mask),
            ("dnam", dnam_prefix_embeddings, dnam_prefix_mask),
            ("rna", rna_prefix_embeddings, rna_prefix_mask),
        ]:
            if prefix_embeddings is None:
                continue
            if prefix_embeddings.ndim != 3:
                raise ValueError(f"Cached {modality} prefixes must be 3D [batch, tokens, hidden], got {tuple(prefix_embeddings.shape)}")
            if prefix_mask is None:
                prefix_mask = torch.ones(prefix_embeddings.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype)
            else:
                prefix_mask = prefix_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
            prefix_outputs[modality] = (prefix_embeddings, prefix_mask)
        if not prefix_outputs:
            raise RuntimeError("VQA batch has no cached prefix embeddings.")
        return prefix_outputs

    def _assemble_interleaved_prefix_sequence(
        self,
        *,
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        prefix_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]],
        prefix_spans: list[list[dict[str, Any]]],
    ) -> dict[str, torch.Tensor | None]:
        row_embeddings: list[torch.Tensor] = []
        row_attention: list[torch.Tensor] = []
        row_labels: list[torch.Tensor] = []
        row_per_layer_input_ids: list[torch.Tensor] = []
        row_prefix_token_mask: list[torch.Tensor] = []

        for row_idx, spans in enumerate(prefix_spans):
            real_text_len = int(attention_mask[row_idx].sum().item())
            cursor = 0
            parts: list[torch.Tensor] = []
            attention_parts: list[torch.Tensor] = []
            label_parts: list[torch.Tensor] = []
            per_layer_id_parts: list[torch.Tensor] = []
            prefix_mask_parts: list[torch.Tensor] = []

            for span in sorted(spans, key=lambda item: int(item["start"])):
                start = int(span["start"])
                end = int(span["end"])
                modality = str(span["modality"])
                if start < cursor or end > real_text_len:
                    raise ValueError(
                        f"Invalid interleaved prefix span for row {row_idx}: {span}; "
                        f"cursor={cursor}, real_text_len={real_text_len}."
                    )
                if modality not in prefix_outputs:
                    raise RuntimeError(f"Missing projected prefix output for modality '{modality}' in row {row_idx}.")

                if start > cursor:
                    parts.append(text_embeddings[row_idx, cursor:start])
                    attention_parts.append(attention_mask[row_idx, cursor:start])
                    if labels is not None:
                        label_parts.append(labels[row_idx, cursor:start])
                    per_layer_id_parts.append(input_ids[row_idx, cursor:start])
                    prefix_mask_parts.append(torch.zeros((start - cursor,), device=input_ids.device, dtype=torch.bool))

                projected_tokens, projected_attention = prefix_outputs[modality]
                active_mask = projected_attention[row_idx].to(device=projected_tokens.device).bool()
                if not active_mask.any():
                    raise RuntimeError(f"Prefix span requests {modality}, but row {row_idx} has no active projected tokens.")
                row_prefix = projected_tokens[row_idx, active_mask].to(device=text_embeddings.device, dtype=text_embeddings.dtype)
                prefix_len = int(row_prefix.shape[0])
                parts.append(row_prefix)
                attention_parts.append(torch.ones((prefix_len,), device=attention_mask.device, dtype=attention_mask.dtype))
                if labels is not None:
                    label_parts.append(torch.full((prefix_len,), -100, device=labels.device, dtype=labels.dtype))
                per_layer_id_parts.append(torch.zeros((prefix_len,), device=input_ids.device, dtype=input_ids.dtype))
                prefix_mask_parts.append(torch.ones((prefix_len,), device=input_ids.device, dtype=torch.bool))
                cursor = end

            if cursor < real_text_len:
                parts.append(text_embeddings[row_idx, cursor:real_text_len])
                attention_parts.append(attention_mask[row_idx, cursor:real_text_len])
                if labels is not None:
                    label_parts.append(labels[row_idx, cursor:real_text_len])
                per_layer_id_parts.append(input_ids[row_idx, cursor:real_text_len])
                prefix_mask_parts.append(torch.zeros((real_text_len - cursor,), device=input_ids.device, dtype=torch.bool))

            if not parts:
                raise RuntimeError(f"Interleaved VQA row {row_idx} produced an empty embedding sequence.")
            row_embeddings.append(torch.cat(parts, dim=0))
            row_attention.append(torch.cat(attention_parts, dim=0))
            if labels is not None:
                row_labels.append(torch.cat(label_parts, dim=0))
            row_per_layer_input_ids.append(torch.cat(per_layer_id_parts, dim=0))
            row_prefix_token_mask.append(torch.cat(prefix_mask_parts, dim=0))

        batch_size = len(row_embeddings)
        max_len = max(int(item.shape[0]) for item in row_embeddings)
        combined_embeddings = torch.zeros(
            (batch_size, max_len, text_embeddings.shape[-1]),
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )
        combined_attention = torch.zeros((batch_size, max_len), device=attention_mask.device, dtype=attention_mask.dtype)
        combined_per_layer_input_ids = torch.zeros((batch_size, max_len), device=input_ids.device, dtype=input_ids.dtype)
        combined_prefix_token_mask = torch.zeros((batch_size, max_len), device=input_ids.device, dtype=torch.bool)
        combined_labels = None
        if labels is not None:
            combined_labels = torch.full((batch_size, max_len), -100, device=labels.device, dtype=labels.dtype)

        for row_idx in range(batch_size):
            row_len = int(row_embeddings[row_idx].shape[0])
            combined_embeddings[row_idx, :row_len] = row_embeddings[row_idx]
            combined_attention[row_idx, :row_len] = row_attention[row_idx]
            combined_per_layer_input_ids[row_idx, :row_len] = row_per_layer_input_ids[row_idx]
            combined_prefix_token_mask[row_idx, :row_len] = row_prefix_token_mask[row_idx]
            if combined_labels is not None:
                combined_labels[row_idx, :row_len] = row_labels[row_idx]

        position_ids = combined_attention.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp_min(0)
        position_ids = position_ids.masked_fill(combined_attention == 0, 0)
        return {
            "input_ids": combined_per_layer_input_ids,
            "inputs_embeds": combined_embeddings,
            "attention_mask": combined_attention,
            "position_ids": position_ids,
            "labels": combined_labels,
            "prefix_token_mask": combined_prefix_token_mask,
        }

    def _forward_with_interleaved_prefixes(
        self,
        *,
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        prefix_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]],
        prefix_spans: list[list[dict[str, Any]]],
    ) -> Any:
        assembled = self._assemble_interleaved_prefix_sequence(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            prefix_outputs=prefix_outputs,
            prefix_spans=prefix_spans,
        )
        return forward_language_model_with_soft_prefix(
            self.language_model,
            input_ids=assembled["input_ids"],
            inputs_embeds=assembled["inputs_embeds"],
            attention_mask=assembled["attention_mask"],
            position_ids=assembled["position_ids"],
            labels=assembled["labels"],
            prefix_length=0,
            prefix_token_mask=assembled["prefix_token_mask"],
        )

    def _forward_with_prepended_prefixes(
        self,
        *,
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        prefix_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> Any:
        prefix_parts: list[torch.Tensor] = []
        prefix_attention_parts: list[torch.Tensor] = []
        for modality in MODALITIES:
            if modality not in prefix_outputs:
                continue
            projected_tokens, projected_attention = prefix_outputs[modality]
            prefix_parts.append(projected_tokens.to(device=text_embeddings.device, dtype=text_embeddings.dtype))
            prefix_attention_parts.append(projected_attention)

        prefix_embeddings = torch.cat(prefix_parts, dim=1)
        prefix_attention = torch.cat(prefix_attention_parts, dim=1)
        combined_embeddings = torch.cat([prefix_embeddings, text_embeddings], dim=1)
        combined_attention = torch.cat([prefix_attention, attention_mask], dim=1)

        combined_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (labels.shape[0], prefix_embeddings.shape[1]),
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
            prefix_length=prefix_embeddings.shape[1],
        )

    def _prefix_outputs_from_inputs(
        self,
        *,
        attention_mask: torch.Tensor,
        pathology_features: torch.Tensor | None = None,
        pathology_feature_mask: torch.Tensor | None = None,
        radiology_features: torch.Tensor | None = None,
        radiology_feature_mask: torch.Tensor | None = None,
        dnam_features: torch.Tensor | None = None,
        dnam_feature_mask: torch.Tensor | None = None,
        rna_features: torch.Tensor | None = None,
        rna_feature_mask: torch.Tensor | None = None,
        pathology_prefix_embeddings: torch.Tensor | None = None,
        pathology_prefix_mask: torch.Tensor | None = None,
        radiology_prefix_embeddings: torch.Tensor | None = None,
        radiology_prefix_mask: torch.Tensor | None = None,
        dnam_prefix_embeddings: torch.Tensor | None = None,
        dnam_prefix_mask: torch.Tensor | None = None,
        rna_prefix_embeddings: torch.Tensor | None = None,
        rna_prefix_mask: torch.Tensor | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        has_cached_prefixes = any(
            tensor is not None
            for tensor in [
                pathology_prefix_embeddings,
                radiology_prefix_embeddings,
                dnam_prefix_embeddings,
                rna_prefix_embeddings,
            ]
        )
        has_raw_features = any(
            tensor is not None
            for tensor in [
                pathology_features,
                radiology_features,
                dnam_features,
                rna_features,
            ]
        )
        if has_cached_prefixes and has_raw_features:
            raise RuntimeError("VQA forward received both cached prefixes and raw features. Use exactly one prefix source.")
        if has_cached_prefixes:
            return self._cached_available_prefixes(
                attention_mask=attention_mask,
                pathology_prefix_embeddings=pathology_prefix_embeddings,
                pathology_prefix_mask=pathology_prefix_mask,
                radiology_prefix_embeddings=radiology_prefix_embeddings,
                radiology_prefix_mask=radiology_prefix_mask,
                dnam_prefix_embeddings=dnam_prefix_embeddings,
                dnam_prefix_mask=dnam_prefix_mask,
                rna_prefix_embeddings=rna_prefix_embeddings,
                rna_prefix_mask=rna_prefix_mask,
            )
        return self._project_available_prefixes(
            attention_mask=attention_mask,
            pathology_features=pathology_features,
            pathology_feature_mask=pathology_feature_mask,
            radiology_features=radiology_features,
            radiology_feature_mask=radiology_feature_mask,
            dnam_features=dnam_features,
            dnam_feature_mask=dnam_feature_mask,
            rna_features=rna_features,
            rna_feature_mask=rna_feature_mask,
        )

    def prepare_interleaved_generation_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pathology_features: torch.Tensor | None = None,
        pathology_feature_mask: torch.Tensor | None = None,
        radiology_features: torch.Tensor | None = None,
        radiology_feature_mask: torch.Tensor | None = None,
        dnam_features: torch.Tensor | None = None,
        dnam_feature_mask: torch.Tensor | None = None,
        rna_features: torch.Tensor | None = None,
        rna_feature_mask: torch.Tensor | None = None,
        pathology_prefix_embeddings: torch.Tensor | None = None,
        pathology_prefix_mask: torch.Tensor | None = None,
        radiology_prefix_embeddings: torch.Tensor | None = None,
        radiology_prefix_mask: torch.Tensor | None = None,
        dnam_prefix_embeddings: torch.Tensor | None = None,
        dnam_prefix_mask: torch.Tensor | None = None,
        rna_prefix_embeddings: torch.Tensor | None = None,
        rna_prefix_mask: torch.Tensor | None = None,
        prefix_spans: list[list[dict[str, Any]]],
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        prefix_outputs = self._prefix_outputs_from_inputs(
            attention_mask=attention_mask,
            pathology_features=pathology_features,
            pathology_feature_mask=pathology_feature_mask,
            radiology_features=radiology_features,
            radiology_feature_mask=radiology_feature_mask,
            dnam_features=dnam_features,
            dnam_feature_mask=dnam_feature_mask,
            rna_features=rna_features,
            rna_feature_mask=rna_feature_mask,
            pathology_prefix_embeddings=pathology_prefix_embeddings,
            pathology_prefix_mask=pathology_prefix_mask,
            radiology_prefix_embeddings=radiology_prefix_embeddings,
            radiology_prefix_mask=radiology_prefix_mask,
            dnam_prefix_embeddings=dnam_prefix_embeddings,
            dnam_prefix_mask=dnam_prefix_mask,
            rna_prefix_embeddings=rna_prefix_embeddings,
            rna_prefix_mask=rna_prefix_mask,
        )
        assembled = self._assemble_interleaved_prefix_sequence(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            attention_mask=attention_mask,
            labels=None,
            prefix_outputs=prefix_outputs,
            prefix_spans=prefix_spans,
        )
        input_ids = assembled["input_ids"]
        inputs_embeds = assembled["inputs_embeds"]
        attention_mask = assembled["attention_mask"]
        if input_ids.shape[0] > 1:
            row_lengths = attention_mask.sum(dim=1).long()
            max_len = int(attention_mask.shape[1])
            left_input_ids = torch.zeros_like(input_ids)
            left_inputs_embeds = torch.zeros_like(inputs_embeds)
            left_attention_mask = torch.zeros_like(attention_mask)
            for row_idx, row_len_tensor in enumerate(row_lengths):
                row_len = int(row_len_tensor.item())
                if row_len == 0:
                    continue
                dst_start = max_len - row_len
                left_input_ids[row_idx, dst_start:] = input_ids[row_idx, :row_len]
                left_inputs_embeds[row_idx, dst_start:] = inputs_embeds[row_idx, :row_len]
                left_attention_mask[row_idx, dst_start:] = attention_mask[row_idx, :row_len]
            input_ids = left_input_ids
            inputs_embeds = left_inputs_embeds
            attention_mask = left_attention_mask
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp_min(0)
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pathology_features: torch.Tensor | None = None,
        pathology_feature_mask: torch.Tensor | None = None,
        radiology_features: torch.Tensor | None = None,
        radiology_feature_mask: torch.Tensor | None = None,
        dnam_features: torch.Tensor | None = None,
        dnam_feature_mask: torch.Tensor | None = None,
        rna_features: torch.Tensor | None = None,
        rna_feature_mask: torch.Tensor | None = None,
        pathology_prefix_embeddings: torch.Tensor | None = None,
        pathology_prefix_mask: torch.Tensor | None = None,
        radiology_prefix_embeddings: torch.Tensor | None = None,
        radiology_prefix_mask: torch.Tensor | None = None,
        dnam_prefix_embeddings: torch.Tensor | None = None,
        dnam_prefix_mask: torch.Tensor | None = None,
        rna_prefix_embeddings: torch.Tensor | None = None,
        rna_prefix_mask: torch.Tensor | None = None,
        prefix_spans: list[list[dict[str, Any]]] | None = None,
    ) -> Any:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        prefix_outputs = self._prefix_outputs_from_inputs(
            attention_mask=attention_mask,
            pathology_features=pathology_features,
            pathology_feature_mask=pathology_feature_mask,
            radiology_features=radiology_features,
            radiology_feature_mask=radiology_feature_mask,
            dnam_features=dnam_features,
            dnam_feature_mask=dnam_feature_mask,
            rna_features=rna_features,
            rna_feature_mask=rna_feature_mask,
            pathology_prefix_embeddings=pathology_prefix_embeddings,
            pathology_prefix_mask=pathology_prefix_mask,
            radiology_prefix_embeddings=radiology_prefix_embeddings,
            radiology_prefix_mask=radiology_prefix_mask,
            dnam_prefix_embeddings=dnam_prefix_embeddings,
            dnam_prefix_mask=dnam_prefix_mask,
            rna_prefix_embeddings=rna_prefix_embeddings,
            rna_prefix_mask=rna_prefix_mask,
        )
        if prefix_spans is None:
            return self._forward_with_prepended_prefixes(
                input_ids=input_ids,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                prefix_outputs=prefix_outputs,
            )
        return self._forward_with_interleaved_prefixes(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            prefix_outputs=prefix_outputs,
            prefix_spans=prefix_spans,
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_tokenizer(model_name_or_path: str, trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for VQA LoRA training.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad_token, eos_token, or unk_token.")
    tokenizer.padding_side = "right"
    return tokenizer


def build_language_model(stage_cfg: Any, *, device: torch.device) -> nn.Module:
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required for VQA LoRA training.") from exc

    load_in_8bit = bool(cfg_get(stage_cfg, "load_in_8bit", False))
    resolved_dtype = resolve_torch_dtype(cfg_get(stage_cfg, "torch_dtype", None))
    model_kwargs: dict[str, Any] = {"trust_remote_code": bool(cfg_get(stage_cfg, "trust_remote_code", True))}
    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["low_cpu_mem_usage"] = True
        model_kwargs["device_map"] = {"": str(device)}
    elif resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    attn_implementation = cfg_get(stage_cfg, "attn_implementation", None)
    if attn_implementation:
        model_kwargs["attn_implementation"] = str(attn_implementation)
    return AutoModelForCausalLM.from_pretrained(str(cfg_get(stage_cfg, "model_name_or_path")), **model_kwargs)


def apply_lora(language_model: nn.Module, stage_cfg: Any) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("peft is required for VQA LoRA training. Install project dependencies first.") from exc

    load_in_8bit = bool(cfg_get(stage_cfg, "load_in_8bit", False))
    gradient_checkpointing = bool(cfg_get(stage_cfg, "gradient_checkpointing", False))
    gradient_checkpointing_kwargs = {
        "use_reentrant": bool(cfg_get(stage_cfg, "gradient_checkpointing_use_reentrant", False))
    }
    if load_in_8bit:
        language_model = prepare_model_for_kbit_training(
            language_model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs if gradient_checkpointing else None,
        )
    elif gradient_checkpointing and hasattr(language_model, "gradient_checkpointing_enable"):
        language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(language_model, "enable_input_require_grads"):
            language_model.enable_input_require_grads()

    lora_cfg = cfg_get(stage_cfg, "lora", {})
    target_modules = cfg_list(cfg_get(lora_cfg, "target_modules", []))
    if not target_modules:
        raise ValueError("vqa_train.lora.target_modules must list at least one module name.")
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(cfg_get(lora_cfg, "r", 16)),
        lora_alpha=int(cfg_get(lora_cfg, "alpha", 32)),
        lora_dropout=float(cfg_get(lora_cfg, "dropout", 0.05)),
        target_modules=target_modules,
        bias=str(cfg_get(lora_cfg, "bias", "none")),
    )
    return get_peft_model(language_model, config)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    *,
    floating_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.is_floating_point() and floating_dtype is not None:
                output[key] = value.to(device=device, dtype=floating_dtype)
            else:
                output[key] = value.to(device)
        else:
            output[key] = value
    return output


def save_vqa_model_artifacts(
    *,
    artifact_dir: Path,
    stage_cfg: Any,
    model: OncoVLMVQASFTModel,
    tokenizer: Any,
    global_step: int,
    epoch: int | None,
    validation_loss: float | None,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = artifact_dir / "lora_adapter"
    model.language_model.save_pretrained(adapter_dir)
    if bool(cfg_get(stage_cfg, "save_tokenizer_snapshot", True)):
        tokenizer.save_pretrained(artifact_dir / "tokenizer")

    payload: dict[str, Any] = {
        "model_name_or_path": str(cfg_get(stage_cfg, "model_name_or_path")),
        "global_step": int(global_step),
        "epoch": int(epoch) if epoch is not None else None,
        "validation_loss": float(validation_loss) if validation_loss is not None and math.isfinite(validation_loss) else None,
        "projector_metadata": model.projector_metadata,
    }
    if model.path_projectors is not None:
        payload["path_projector_state_dict"] = model.path_projectors.state_dict()
    if model.radiology_projectors is not None:
        payload["radiology_projector_state_dict"] = model.radiology_projectors.state_dict()
    if model.dnam_projectors is not None:
        payload["dnam_projector_state_dict"] = model.dnam_projectors.state_dict()
    if model.rna_projectors is not None:
        payload["rna_projector_state_dict"] = model.rna_projectors.state_dict()

    projectors_path = artifact_dir / "projectors.ckpt"
    torch.save(payload, projectors_path)
    return {
        "artifact_dir": str(artifact_dir),
        "lora_adapter_dir": str(adapter_dir),
        "projectors_checkpoint": str(projectors_path),
    }
