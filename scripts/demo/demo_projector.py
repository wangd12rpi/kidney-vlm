#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.modeling.dnam_qwen_projector import DnamQwenProjectorLM
from kidney_vlm.modeling.path_projectors import build_soft_prefix_per_layer_inputs
from kidney_vlm.modeling.pathology_qwen_projector import PathologyQwenProjectorLM
from kidney_vlm.modeling.rna_qwen_projector import RnaQwenProjectorLM
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.training.collator import (
    _coerce_token_ids,
    _load_h5_patch_features as _load_h5_patch_features_shared,
    _load_pt_feature_tensor as _load_pt_feature_tensor_shared,
)
from kidney_vlm.vqa.data import _find_placeholder_span, load_modality_feature_tensor
from kidney_vlm.vqa.modeling import (
    OncoVLMVQASFTModel,
    build_language_model,
    build_projector_module,
    build_tokenizer,
    generate_language_model_with_soft_prefix,
)
from kidney_vlm.vqa.prompts import prefix_placeholder_for_modality
from kidney_vlm.vqa.stage_config import clean_text, resolve_torch_dtype

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

DEMO_PROMPT = "what is this?"
SUPPORTED_MODALITIES = {"pathology", "radiology", "dnam", "rna"}
SUPPORTED_INFERENCE_MODES = {"projector_training_mode", "vqa_mode"}


def load_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    OmegaConf.set_struct(cfg, False)
    return cfg


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _resolve_device(device_value: str | None) -> torch.device:
    requested = str(device_value or "").strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested}' but CUDA is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def _projector_cfg(cfg: Any, modality: str) -> Any:
    if modality == "pathology":
        return cfg.pathology_proj
    if modality == "radiology":
        return cfg.radiology_proj
    if modality == "dnam":
        return cfg.dnam_proj
    if modality == "rna":
        return cfg.rna_proj
    raise ValueError(f"Unsupported demo modality: {modality}")


def _selected_modality_cfg(cfg: Any) -> tuple[str, str, Any]:
    demo_cfg = cfg.demo
    model_key = str(demo_cfg.get("model", "")).strip().lower()
    if not model_key:
        raise RuntimeError("demo.model must be configured, for example 'gemma' or 'qwen'.")
    modality = str(demo_cfg.get("modality", "")).strip().lower()
    if modality not in SUPPORTED_MODALITIES:
        raise RuntimeError(f"demo.modality must be one of {sorted(SUPPORTED_MODALITIES)}, got: {modality!r}")
    model_cfg = demo_cfg.get("modalities", {}).get(model_key)
    if model_cfg is None:
        raise RuntimeError(f"demo.modalities.{model_key} must be configured.")
    modality_cfg = model_cfg.get(modality)
    if modality_cfg is None:
        raise RuntimeError(f"demo.modalities.{model_key}.{modality} must be configured.")
    return model_key, modality, modality_cfg


def _build_tokenizer(model_name_or_path: str, trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for the projector demo.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("The projector demo requires an instruct tokenizer with apply_chat_template.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def _build_chat_prompt_input_ids(tokenizer: Any, *, prompt_text: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt_text}]
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        token_ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                chat_template_kwargs={"enable_thinking": False},
                **kwargs,
            )
        except TypeError:
            token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    return torch.tensor([_coerce_token_ids(token_ids)], dtype=torch.long, device=device)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _read_projector_frame(modality_cfg: Any, columns: list[str]) -> pd.DataFrame:
    qa_parquet_path = _resolve_path(modality_cfg.qa_parquet_path)
    if not qa_parquet_path.exists():
        raise FileNotFoundError(f"Projector QA parquet not found: {qa_parquet_path}")
    return pd.read_parquet(qa_parquet_path, columns=columns)


def _resolve_feature_path(
    cfg: Any,
    modality: str,
    modality_cfg: Any,
    sample_key: str,
    *,
    feature_paths_by_sample_key: dict[str, str] | None = None,
) -> Path:
    mapped_feature_path = str((feature_paths_by_sample_key or {}).get(sample_key, "")).strip()
    if mapped_feature_path:
        resolved_mapped = _resolve_path(mapped_feature_path)
        if resolved_mapped.exists():
            return resolved_mapped
        raise FileNotFoundError(f"Feature file from projector parquet not found for '{sample_key}': {resolved_mapped}")

    if modality == "pathology":
        patch_encoder_name = str(cfg.pathology_features.patch_encoder)
        save_format = str(cfg.pathology_features.get("save_format", "h5")).lower()
        features_root = _resolve_path(cfg.pathology_features.features_root)
        feature_path = features_root / f"features_{patch_encoder_name}" / f"{sample_key}.{save_format}"
        if feature_path.exists():
            return feature_path

    raise FileNotFoundError(
        f"No feature path found for {modality} sample '{sample_key}'. "
        "Use a QA parquet row with the configured feature_path_field."
    )


def _load_h5_patch_features(
    path: Path,
    max_patch_tokens: int,
    *,
    compression_method: str = "none",
    compression_kernel_size: int = 1,
) -> torch.Tensor:
    return _load_h5_patch_features_shared(
        path,
        max_patch_tokens=max_patch_tokens,
        compression_method=compression_method,
        compression_kernel_size=compression_kernel_size,
    )


def _load_feature_tensor(modality: str, modality_cfg: Any, feature_path: Path) -> torch.Tensor:
    max_feature_tokens = int(modality_cfg.get("max_feature_tokens", 0))
    if modality == "pathology":
        return _load_h5_patch_features(
            feature_path,
            max_patch_tokens=max_feature_tokens,
            compression_method=str(modality_cfg.get("patch_compression_method", "none")),
            compression_kernel_size=int(modality_cfg.get("patch_compression_kernel_size", 1)),
        )
    if modality in {"dnam", "rna"}:
        return _load_pt_feature_tensor_shared(feature_path, max_tokens=max_feature_tokens)
    raise ValueError(f"Unsupported demo modality: {modality}")


def _load_projector_state(checkpoint_path: Path) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def _resolve_checkpoint_path(path_value: str | Path, checkpoint_name: str | None = None) -> Path:
    resolved = _resolve_path(path_value)
    desired_name = str(checkpoint_name or "").strip()
    if resolved.is_file():
        return resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Projector checkpoint path not found: {resolved}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Projector checkpoint path is neither a file nor a directory: {resolved}")

    if desired_name:
        direct_named = resolved / desired_name
        if direct_named.exists():
            return direct_named
    else:
        direct_best = resolved / "best.ckpt"
        if direct_best.exists():
            return direct_best

    def _sorted_checkpoint_candidates(pattern: str) -> list[Path]:
        return sorted(
            [path for path in resolved.rglob(pattern) if path.is_file()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    if desired_name:
        named_candidates = _sorted_checkpoint_candidates(desired_name)
        if named_candidates:
            return named_candidates[0]
        raise FileNotFoundError(f"Requested checkpoint '{desired_name}' was not found under: {resolved}")

    best_candidates = _sorted_checkpoint_candidates("best.ckpt")
    if best_candidates:
        return best_candidates[0]

    epoch_candidates = _sorted_checkpoint_candidates("epoch_*.ckpt")
    if epoch_candidates:
        return epoch_candidates[0]

    raise FileNotFoundError(f"No best.ckpt or epoch_*.ckpt found under: {resolved}")


def _load_projector_model(
    cfg: Any,
    modality: str,
    state: dict[str, Any],
    device: torch.device,
) -> PathologyQwenProjectorLM | DnamQwenProjectorLM | RnaQwenProjectorLM:
    stage_cfg = _projector_cfg(cfg, modality)
    model_name_or_path = str(state.get("model_name_or_path") or stage_cfg.model_name_or_path)
    load_in_8bit = bool(stage_cfg.get("load_in_8bit", False))
    common_kwargs = {
        "projector_type": str(state.get("projector_type") or stage_cfg.get("projector_type", "mlp")),
        "projector_num_latents": int(state.get("projector_num_latents") or stage_cfg.get("projector_num_latents", 64)),
        "projector_depth": int(state.get("projector_depth") or stage_cfg.get("projector_depth", 2)),
        "projector_num_heads": int(state.get("projector_num_heads") or stage_cfg.get("projector_num_heads", 8)),
        "projector_mlp_ratio": float(state.get("projector_mlp_ratio") or stage_cfg.get("projector_mlp_ratio", 4.0)),
        "projector_dropout": float(state.get("projector_dropout") or stage_cfg.get("projector_dropout", 0.0)),
        "trust_remote_code": bool(stage_cfg.trust_remote_code),
        "torch_dtype": stage_cfg.get("torch_dtype"),
        "attn_implementation": stage_cfg.get("attn_implementation"),
        "load_in_8bit": load_in_8bit,
        "device_map": {"": str(device)} if load_in_8bit else None,
    }

    if modality == "pathology":
        embedding_dim = int(state.get("pathology_embedding_dim") or stage_cfg.pathology_embedding_dim)
        model = PathologyQwenProjectorLM.from_pretrained(
            model_name_or_path,
            pathology_in_dim=embedding_dim,
            **common_kwargs,
        )
        state_dict = state.get("path_projector_state_dict")
        if state_dict is None:
            raise KeyError("Pathology checkpoint is missing 'path_projector_state_dict'.")
        model.path_projectors.load_state_dict(state_dict)
    elif modality == "dnam":
        embedding_dim = int(state.get("dnam_embedding_dim") or stage_cfg.dnam_embedding_dim)
        model = DnamQwenProjectorLM.from_pretrained(
            model_name_or_path,
            dnam_in_dim=embedding_dim,
            dnam_prefix_tokens=int(state.get("dnam_prefix_tokens") or stage_cfg.get("dnam_prefix_tokens", 0)),
            dnam_prefix_expander_mlp_ratio=float(
                state.get("dnam_prefix_expander_mlp_ratio") or stage_cfg.get("dnam_prefix_expander_mlp_ratio", 1.0)
            ),
            **common_kwargs,
        )
        state_dict = state.get("dnam_projector_state_dict")
        if state_dict is None:
            raise KeyError("DNAm checkpoint is missing 'dnam_projector_state_dict'.")
        model.projectors.load_state_dict(state_dict)
    elif modality == "rna":
        embedding_dim = int(state.get("rna_embedding_dim") or stage_cfg.rna_embedding_dim)
        model = RnaQwenProjectorLM.from_pretrained(
            model_name_or_path,
            rna_in_dim=embedding_dim,
            rna_prefix_tokens=int(state.get("rna_prefix_tokens") or stage_cfg.get("rna_prefix_tokens", 0)),
            rna_prefix_expander_mlp_ratio=float(
                state.get("rna_prefix_expander_mlp_ratio") or stage_cfg.get("rna_prefix_expander_mlp_ratio", 1.0)
            ),
            **common_kwargs,
        )
        state_dict = state.get("rna_projector_state_dict")
        if state_dict is None:
            raise KeyError("RNA checkpoint is missing 'rna_projector_state_dict'.")
        model.projectors.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported demo modality: {modality}")

    model.eval()
    if load_in_8bit:
        model.move_trainable_modules_to(device)
    else:
        model.to(device)
    if hasattr(model.language_model.config, "use_cache"):
        model.language_model.config.use_cache = True
    return model


def _resolve_demo_sample_keys(demo_cfg: Any, modality_cfg: Any) -> list[str]:
    max_samples = int(demo_cfg.get("max_samples", 1))
    if max_samples <= 0:
        raise RuntimeError("demo.max_samples must be a positive integer.")

    explicit_sample_key = str(demo_cfg.get("sample_key", "")).strip()
    configured_sample_keys = demo_cfg.get("sample_keys")
    explicit_sample_keys: list[str] = []
    if configured_sample_keys not in (None, "", "null"):
        for value in list(configured_sample_keys):
            sample_key = str(value).strip()
            if sample_key:
                explicit_sample_keys.append(sample_key)

    if explicit_sample_key:
        sample_keys = [explicit_sample_key]
    elif explicit_sample_keys:
        sample_keys = explicit_sample_keys
    else:
        key_field = str(modality_cfg.sample_key_field)
        split_filter = str(demo_cfg.get("split_filter", "")).strip().lower()
        columns = [key_field]
        feature_path_field = str(modality_cfg.get("feature_path_field", "")).strip()
        if feature_path_field and feature_path_field not in columns:
            columns.append(feature_path_field)
        if split_filter:
            columns.append("split")
        frame = _read_projector_frame(modality_cfg, columns=columns)
        if split_filter:
            if "split" not in frame.columns:
                raise RuntimeError(
                    f"demo.split_filter='{split_filter}' was requested, but the parquet is missing 'split'."
                )
            frame = frame.loc[frame["split"].fillna("train").astype(str).str.lower() == split_filter].copy()
        if feature_path_field:
            frame = frame.loc[frame[feature_path_field].map(lambda value: bool(_as_list(value)))].copy()
        sample_keys = (
            frame[key_field]
            .fillna("")
            .astype(str)
            .str.strip()
            .loc[lambda series: series != ""]
            .drop_duplicates()
            .tolist()
        )
    if not sample_keys:
        raise RuntimeError("No demo samples found. Set demo.sample_key, demo.sample_keys, or populate the QA parquet.")
    if explicit_sample_key or explicit_sample_keys or len(sample_keys) <= max_samples:
        return sample_keys[:max_samples]
    sample_seed = int(demo_cfg.get("sample_seed", 42))
    rng = random.Random(sample_seed)
    return rng.sample(sample_keys, k=max_samples)


def _load_labels_by_sample_key(modality_cfg: Any) -> dict[str, str]:
    key_field = str(modality_cfg.sample_key_field)
    label_field = str(modality_cfg.get("label_field", "")).strip()
    if not label_field:
        return {}
    frame = _read_projector_frame(modality_cfg, columns=[key_field, label_field])
    mapping: dict[str, str] = {}
    for row in frame.drop_duplicates(key_field).itertuples(index=False):
        sample_key = str(getattr(row, key_field)).strip()
        label = str(getattr(row, label_field)).strip()
        if sample_key and label:
            mapping[sample_key] = label
    return mapping


def _load_feature_paths_by_sample_key(modality_cfg: Any) -> dict[str, str]:
    key_field = str(modality_cfg.sample_key_field)
    feature_path_field = str(modality_cfg.feature_path_field)
    frame = _read_projector_frame(modality_cfg, columns=[key_field, feature_path_field])
    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        sample_key = str(getattr(row, key_field)).strip()
        feature_paths = _as_list(getattr(row, feature_path_field))
        if sample_key and feature_paths and sample_key not in mapping:
            mapping[sample_key] = feature_paths[0]
    return mapping


def _load_feature_refs_by_sample_key(modality_cfg: Any) -> dict[str, list[str]]:
    key_field = str(modality_cfg.sample_key_field)
    feature_path_field = str(modality_cfg.feature_path_field)
    frame = _read_projector_frame(modality_cfg, columns=[key_field, feature_path_field])
    mapping: dict[str, list[str]] = {}
    for row in frame.itertuples(index=False):
        sample_key = str(getattr(row, key_field)).strip()
        feature_refs = _as_list(getattr(row, feature_path_field))
        if sample_key and feature_refs and sample_key not in mapping:
            mapping[sample_key] = feature_refs
    return mapping


def _checkpoint_path_for_block(checkpoint_path: Path) -> str:
    try:
        return checkpoint_path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(checkpoint_path)


def _demo_stage_cfg(demo_cfg: Any, state: dict[str, Any]) -> dict[str, Any]:
    model_name_or_path = clean_text(state.get("model_name_or_path")) or clean_text(demo_cfg.get("model_name_or_path"))
    if not model_name_or_path:
        raise ValueError("Projector checkpoint/demo config must define model_name_or_path.")
    return {
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": bool(demo_cfg.get("trust_remote_code", True)),
        "load_in_8bit": bool(demo_cfg.get("load_in_8bit", False)),
        "torch_dtype": demo_cfg.get("torch_dtype", None),
        "attn_implementation": demo_cfg.get("attn_implementation", None),
    }


def _load_single_modality_vqa_model(
    *,
    modality: str,
    demo_cfg: Any,
    modality_cfg: Any,
    state: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[OncoVLMVQASFTModel, dict[str, Any]]:
    stage_cfg = _demo_stage_cfg(demo_cfg, state)
    language_model = build_language_model(stage_cfg, device=device)
    hidden_size = int(state.get("hidden_size") or 0)
    if hidden_size <= 0:
        raise ValueError(f"{modality} checkpoint is missing hidden_size: {checkpoint_path}")

    block_cfg = {
        "enabled": True,
        "trainable": False,
        "checkpoint_path": _checkpoint_path_for_block(checkpoint_path),
        "max_tokens": int(modality_cfg.get("max_feature_tokens", state.get("max_slice_tokens") or 32)),
        "patch_compression_method": str(modality_cfg.get("patch_compression_method", "none")),
        "patch_compression_kernel_size": int(modality_cfg.get("patch_compression_kernel_size", 1)),
        "token_dropout_prob": 0.0,
    }
    projector, metadata = build_projector_module(
        repo_root=ROOT,
        modality=modality,
        block_cfg=block_cfg,
        hidden_size=hidden_size,
    )
    model = OncoVLMVQASFTModel(
        language_model=language_model,
        projectors={modality: projector},
        projector_metadata={modality: metadata},
    )
    projector_dtype = resolve_torch_dtype(demo_cfg.get("projector_dtype", "float32"))
    model.move_projectors_to(device, dtype=projector_dtype)
    if not bool(stage_cfg.get("load_in_8bit", False)):
        model.language_model.to(device)
    model.eval()
    return model, block_cfg


def _demo_prompt_with_prefix(modality: str) -> str:
    return f"{prefix_placeholder_for_modality(modality)}\n\n{DEMO_PROMPT}"


def _load_demo_feature_tensor(modality: str, modality_cfg: Any, feature_refs: list[str]) -> torch.Tensor:
    block_cfg = {
        "max_tokens": int(modality_cfg.get("max_feature_tokens", 32)),
        "patch_compression_method": str(modality_cfg.get("patch_compression_method", "none")),
        "patch_compression_kernel_size": int(modality_cfg.get("patch_compression_kernel_size", 1)),
        "token_dropout_prob": 0.0,
    }
    row = {
        "question_id": f"demo_{modality}",
        "pathology_feature_paths": feature_refs if modality == "pathology" else [],
        "radiology_feature_paths": feature_refs if modality == "radiology" else [],
        "dnam_feature_path": feature_refs[0] if modality == "dnam" and feature_refs else "",
        "rna_feature_path": feature_refs[0] if modality == "rna" and feature_refs else "",
    }
    return load_modality_feature_tensor(ROOT, row, modality, block_cfg)


def _feature_kwargs_for_modality(
    modality: str,
    features: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    modality_features = features.unsqueeze(0).to(device=device)
    modality_feature_mask = torch.ones(modality_features.shape[:2], dtype=torch.long, device=device)
    return {
        f"{modality}_features": modality_features,
        f"{modality}_feature_mask": modality_feature_mask,
    }


def _prepare_projector_training_mode_inputs(
    *,
    modality: str,
    model: OncoVLMVQASFTModel,
    tokenizer: Any,
    features: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = _build_chat_prompt_input_ids(tokenizer, prompt_text=DEMO_PROMPT, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    text_embeddings = model.language_model.get_input_embeddings()(input_ids)
    feature_kwargs = _feature_kwargs_for_modality(modality, features, device)
    prefix_outputs = model._prefix_outputs_from_inputs(
        attention_mask=attention_mask,
        **feature_kwargs,
    )
    projected_tokens, projected_attention = prefix_outputs[modality]
    active_mask = projected_attention[0].to(device=projected_tokens.device).bool()
    if not active_mask.any():
        raise RuntimeError(f"{modality} projector produced no active prefix tokens.")
    prefix_embeddings = projected_tokens[:, active_mask].to(device=text_embeddings.device, dtype=text_embeddings.dtype)
    prefix_attention = torch.ones(
        prefix_embeddings.shape[:2],
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )

    inputs_embeds = torch.cat([prefix_embeddings, text_embeddings], dim=1)
    combined_attention = torch.cat([prefix_attention, attention_mask], dim=1)
    prefix_input_ids = torch.zeros(
        (input_ids.shape[0], prefix_embeddings.shape[1]),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    combined_input_ids = torch.cat([prefix_input_ids, input_ids], dim=1)
    position_ids = combined_attention.long().cumsum(dim=1) - 1
    position_ids = position_ids.clamp_min(0)
    position_ids = position_ids.masked_fill(combined_attention == 0, 0)

    inputs = {
        "input_ids": combined_input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": combined_attention,
        "position_ids": position_ids,
    }
    per_layer_inputs = build_soft_prefix_per_layer_inputs(
        model.language_model,
        input_ids=input_ids,
        prefix_length=prefix_embeddings.shape[1],
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    if per_layer_inputs is not None:
        inputs["per_layer_inputs"] = per_layer_inputs
    return inputs


def _prepare_vqa_mode_inputs(
    *,
    modality: str,
    model: OncoVLMVQASFTModel,
    tokenizer: Any,
    features: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prompt_text = _demo_prompt_with_prefix(modality)
    input_ids = _build_chat_prompt_input_ids(tokenizer, prompt_text=prompt_text, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    token_ids = input_ids[0].detach().cpu().tolist()
    start, end = _find_placeholder_span(tokenizer, token_ids, prefix_placeholder_for_modality(modality))
    prefix_spans = [[{"modality": modality, "start": start, "end": end}]]
    feature_kwargs = _feature_kwargs_for_modality(modality, features, device)
    return model.prepare_interleaved_generation_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **feature_kwargs,
        prefix_spans=prefix_spans,
    )


def _generate_single_modality_response(
    *,
    modality: str,
    inference_mode: str,
    model: OncoVLMVQASFTModel,
    tokenizer: Any,
    features: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    autocast_dtype: torch.dtype,
) -> str:
    inference_mode = clean_text(inference_mode) or "vqa_mode"
    if inference_mode not in SUPPORTED_INFERENCE_MODES:
        raise ValueError(f"demo.inference_mode must be one of {sorted(SUPPORTED_INFERENCE_MODES)}, got {inference_mode!r}.")
    use_autocast = device.type == "cuda" and autocast_dtype != torch.float32
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
        if inference_mode == "projector_training_mode":
            inputs = _prepare_projector_training_mode_inputs(
                modality=modality,
                model=model,
                tokenizer=tokenizer,
                features=features,
                device=device,
            )
        else:
            inputs = _prepare_vqa_mode_inputs(
                modality=modality,
                model=model,
                tokenizer=tokenizer,
                features=features,
                device=device,
            )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        generated_ids = generate_language_model_with_soft_prefix(
            model.language_model,
            inputs=inputs,
            generation_kwargs=generation_kwargs,
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def _project_modality_tokens(
    *,
    model: PathologyQwenProjectorLM | DnamQwenProjectorLM | RnaQwenProjectorLM,
    modality: str,
    features: torch.Tensor,
    text_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.unsqueeze(0).to(device=text_embeddings.device)
    feature_mask = torch.ones(features.shape[:2], dtype=torch.long, device=text_embeddings.device)

    if modality == "pathology":
        projected, _ = model.path_projectors["pathology"](features, feature_mask)
        prefix_attention = model.path_projectors["pathology"].build_output_mask(
            feature_mask,
            batch_size=projected.shape[0],
            output_length=projected.shape[1],
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
    elif modality in {"dnam", "rna"}:
        projected, _ = model.projectors[modality](features, feature_mask)
        projected_mask = model.projectors[modality].build_output_mask(
            feature_mask,
            batch_size=projected.shape[0],
            output_length=projected.shape[1],
            device=projected.device,
            dtype=projected.dtype,
        )
        prefix_expander_key = f"{modality}_prefix_expander"
        if prefix_expander_key in model.projectors:
            projected = model.projectors[prefix_expander_key](projected, mask=projected_mask)
            prefix_attention = torch.ones(
                (projected.shape[0], projected.shape[1]),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
        else:
            prefix_attention = model.projectors[modality].build_output_mask(
                feature_mask,
                batch_size=projected.shape[0],
                output_length=projected.shape[1],
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
    else:
        raise ValueError(f"Unsupported demo modality: {modality}")

    return projected.to(dtype=text_embeddings.dtype), prefix_attention


def _generate_response(
    *,
    model: PathologyQwenProjectorLM | DnamQwenProjectorLM | RnaQwenProjectorLM,
    modality: str,
    tokenizer: Any,
    features: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    input_ids = _build_chat_prompt_input_ids(tokenizer, prompt_text=DEMO_PROMPT, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    text_embeddings = model.language_model.get_input_embeddings()(input_ids)
    projected, prefix_attention = _project_modality_tokens(
        model=model,
        modality=modality,
        features=features,
        text_embeddings=text_embeddings,
        attention_mask=attention_mask,
    )

    inputs_embeds = torch.cat([projected, text_embeddings], dim=1)
    combined_attention = torch.cat([prefix_attention, attention_mask], dim=1)
    position_ids = combined_attention.long().cumsum(dim=1) - 1
    position_ids = position_ids.clamp_min(0)
    position_ids = position_ids.masked_fill(combined_attention == 0, 0)

    generate_kwargs: dict[str, Any] = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": combined_attention,
        "position_ids": position_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated_ids = model.language_model.generate(**generate_kwargs)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    if generated_text.startswith(DEMO_PROMPT):
        generated_text = generated_text[len(DEMO_PROMPT):].strip()
    return generated_text


def main() -> None:
    cfg = load_cfg()
    demo_cfg = cfg.demo
    model_key, modality, modality_cfg = _selected_modality_cfg(cfg)
    inference_mode = clean_text(demo_cfg.get("inference_mode")) or "vqa_mode"
    if inference_mode not in SUPPORTED_INFERENCE_MODES:
        raise ValueError(f"demo.inference_mode must be one of {sorted(SUPPORTED_INFERENCE_MODES)}, got {inference_mode!r}.")

    sample_keys = _resolve_demo_sample_keys(demo_cfg, modality_cfg)
    checkpoint_path = _resolve_checkpoint_path(
        modality_cfg.projector_ckpt_path,
        checkpoint_name=demo_cfg.get("checkpoint_name"),
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Projector checkpoint not found: {checkpoint_path}")

    state = _load_projector_state(checkpoint_path)
    if modality in SUPPORTED_MODALITIES:
        device = _resolve_device(demo_cfg.get("device"))
        model_name_or_path = clean_text(state.get("model_name_or_path")) or clean_text(demo_cfg.get("model_name_or_path"))
        tokenizer = build_tokenizer(
            model_name_or_path=model_name_or_path,
            trust_remote_code=bool(demo_cfg.get("trust_remote_code", True)),
        )
        model, block_cfg = _load_single_modality_vqa_model(
            modality=modality,
            demo_cfg=demo_cfg,
            modality_cfg=modality_cfg,
            state=state,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        labels_by_sample_key = _load_labels_by_sample_key(modality_cfg)
        feature_refs_by_sample_key = _load_feature_refs_by_sample_key(modality_cfg)
        autocast_dtype = resolve_torch_dtype(demo_cfg.get("autocast_dtype", "bfloat16")) or torch.bfloat16

        print(f"Demo model: {model_key}")
        print(f"Modality: {modality}")
        print(f"Inference mode: {inference_mode}")
        print(f"Model: {model_name_or_path}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Prompt: {DEMO_PROMPT}")
        print(f"Max feature tokens per ref: {block_cfg['max_tokens']}")
        if str(demo_cfg.get("split_filter", "")).strip():
            print(f"Split filter: {str(demo_cfg.get('split_filter')).strip()}")
        print(f"Samples: {len(sample_keys)}")
        for index, sample_key in enumerate(sample_keys, start=1):
            feature_refs = feature_refs_by_sample_key.get(sample_key, [])
            if not feature_refs:
                raise FileNotFoundError(f"No {modality} feature refs found for sample '{sample_key}'.")
            feature_tensor = _load_demo_feature_tensor(modality, modality_cfg, feature_refs)
            response = _generate_single_modality_response(
                modality=modality,
                inference_mode=inference_mode,
                model=model,
                tokenizer=tokenizer,
                features=feature_tensor,
                device=device,
                max_new_tokens=int(demo_cfg.max_new_tokens),
                do_sample=bool(demo_cfg.do_sample),
                temperature=float(demo_cfg.temperature),
                top_p=float(demo_cfg.top_p),
                autocast_dtype=autocast_dtype,
            )
            print(f"Sample {index}: {sample_key}")
            project_label = labels_by_sample_key.get(sample_key)
            if project_label:
                print(f"TCGA project: {project_label}")
            print("Response:")
            print(response)
            if index != len(sample_keys):
                print()
        return

    stage_cfg = _projector_cfg(cfg, modality)
    model_name_or_path = str(state.get("model_name_or_path") or stage_cfg.model_name_or_path)
    device = _resolve_device(stage_cfg.get("device"))
    tokenizer = _build_tokenizer(
        model_name_or_path=model_name_or_path,
        trust_remote_code=bool(stage_cfg.trust_remote_code),
    )
    model = _load_projector_model(cfg, modality=modality, state=state, device=device)
    labels_by_sample_key = _load_labels_by_sample_key(modality_cfg)
    feature_paths_by_sample_key = _load_feature_paths_by_sample_key(modality_cfg)
    print(f"Modality: {modality}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompt: {DEMO_PROMPT}")
    if str(demo_cfg.get("split_filter", "")).strip():
        print(f"Split filter: {str(demo_cfg.get('split_filter')).strip()}")
    print(f"Samples: {len(sample_keys)}")
    if modality == "pathology":
        print(
            "Patch compression: "
            f"{str(modality_cfg.get('patch_compression_method', 'none'))} "
            f"(kernel={int(modality_cfg.get('patch_compression_kernel_size', 1))}); "
            "max_feature_tokens is applied after compression."
        )
    for index, sample_key in enumerate(sample_keys, start=1):
        feature_path = _resolve_feature_path(
            cfg,
            modality,
            modality_cfg,
            sample_key,
            feature_paths_by_sample_key=feature_paths_by_sample_key,
        )
        feature_tensor = _load_feature_tensor(modality, modality_cfg, feature_path)
        response = _generate_response(
            model=model,
            tokenizer=tokenizer,
            modality=modality,
            features=feature_tensor,
            device=device,
            max_new_tokens=int(demo_cfg.max_new_tokens),
            do_sample=bool(demo_cfg.do_sample),
            temperature=float(demo_cfg.temperature),
            top_p=float(demo_cfg.top_p),
        )
        print(f"Sample {index}: {sample_key}")
        project_label = labels_by_sample_key.get(sample_key)
        if project_label:
            print(f"TCGA project: {project_label}")
        print(f"Feature path: {feature_path}")
        print("Response:")
        print(response)
        if index != len(sample_keys):
            print()


if __name__ == "__main__":
    main()
