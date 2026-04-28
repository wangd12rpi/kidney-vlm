#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.training.collator import _normalize_list
from kidney_vlm.vqa.constants import MODALITIES
from kidney_vlm.vqa.data import load_modality_feature_tensor
from kidney_vlm.vqa.modeling import build_projector_module
from kidney_vlm.vqa.prefix_cache import prefix_cache_path, repo_relative_path
from kidney_vlm.vqa.stage_config import cfg_get, cfg_list, clean_text, resolve_repo_path, resolve_torch_dtype

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)
CONFIG_RELATIVE_PATH = "06_vqa_train/cache_vqa_prefixes.yaml"


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path=CONFIG_RELATIVE_PATH,
        overrides=sys.argv[1:],
    )


def _resolve_device(device_value: str | None) -> torch.device:
    requested = str(device_value or "").strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{requested}', but CUDA is unavailable.")
    return torch.device(requested)


def _filter_unified_frame(frame: pd.DataFrame, stage_cfg: Any) -> pd.DataFrame:
    dataset_cfg = cfg_get(stage_cfg, "dataset", {})
    out = frame.copy()
    for column, key in [("split", "splits"), ("source", "sources"), ("project_id", "project_ids")]:
        values = cfg_list(cfg_get(dataset_cfg, key, []))
        if not values:
            continue
        if column not in out.columns:
            raise ValueError(f"Unified parquet filter requested {key}, but column {column!r} is missing.")
        out = out[out[column].astype(str).isin(values)]
    return out.reset_index(drop=True)


def _feature_refs_for_modality(frame: pd.DataFrame, stage_cfg: Any, modality: str) -> list[str]:
    feature_columns = cfg_get(stage_cfg, "feature_columns", {})
    column = clean_text(cfg_get(feature_columns, modality, ""))
    if not column:
        raise ValueError(f"vqa_train.feature_columns.{modality} must name a unified parquet column.")
    if column not in frame.columns:
        raise ValueError(f"Unified parquet is missing feature column for {modality}: {column}")

    refs: list[str] = []
    for value in frame[column].tolist():
        for raw_ref in _normalize_list(value):
            ref = clean_text(raw_ref)
            if ref:
                refs.append(ref)

    unique_refs = sorted(set(refs))
    max_items = cfg_get(cfg_get(stage_cfg, "dataset", {}), "max_items_per_modality", None)
    if max_items not in (None, "", "null"):
        unique_refs = unique_refs[: int(max_items)]
    return unique_refs


def _feature_row_for_ref(modality: str, feature_ref: str) -> dict[str, Any]:
    if Path(feature_ref).expanduser().is_absolute():
        raise ValueError(f"Feature references must be project-relative, got absolute path: {feature_ref}")
    if modality == "pathology":
        return {"question_id": feature_ref, "pathology_feature_paths": [feature_ref]}
    if modality == "radiology":
        return {"question_id": feature_ref, "radiology_feature_paths": [feature_ref]}
    if modality == "dnam":
        return {"question_id": feature_ref, "dnam_feature_path": feature_ref}
    if modality == "rna":
        return {"question_id": feature_ref, "rna_feature_path": feature_ref}
    raise ValueError(f"Unsupported modality: {modality}")


def _load_feature_batch(*, modality: str, refs: list[str], root_dir: Path, block_cfg: Any) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = [load_modality_feature_tensor(root_dir, _feature_row_for_ref(modality, ref), modality, block_cfg) for ref in refs]
    if not tensors:
        raise ValueError("Cannot load an empty feature batch.")
    feature_dim = int(tensors[0].shape[1])
    for ref, tensor in zip(refs, tensors, strict=True):
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D feature tensor for {modality} {ref}, got {tuple(tensor.shape)}")
        if int(tensor.shape[1]) != feature_dim:
            raise ValueError(f"{modality} feature dimension mismatch for {ref}: expected {feature_dim}, got {int(tensor.shape[1])}")

    max_tokens = max(int(tensor.shape[0]) for tensor in tensors)
    features = torch.zeros((len(tensors), max_tokens, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(tensors), max_tokens), dtype=torch.long)
    for index, tensor in enumerate(tensors):
        token_count = int(tensor.shape[0])
        features[index, :token_count] = tensor
        mask[index, :token_count] = 1
    return features, mask


def _project_prefix_batch(
    *,
    modality: str,
    module: torch.nn.ModuleDict,
    features: torch.Tensor,
    feature_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected, _ = module[modality](features, feature_mask)
    output_mask = module[modality].build_output_mask(
        feature_mask,
        batch_size=projected.shape[0],
        output_length=projected.shape[1],
        device=projected.device,
        dtype=projected.dtype,
    )
    expander_key = f"{modality}_prefix_expander"
    if expander_key in module:
        projected = module[expander_key](projected, mask=output_mask)
        active_rows = feature_mask.sum(dim=1) > 0
        output_mask = active_rows.to(device=projected.device, dtype=projected.dtype).unsqueeze(1).expand(
            projected.shape[0],
            projected.shape[1],
        )
    return projected, output_mask


def _save_prefix_tensor(path: Path, tensor: torch.Tensor, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().cpu(), tmp_path)
    tmp_path.replace(path)


def _cache_modality(
    *,
    stage_cfg: Any,
    frame: pd.DataFrame,
    modality: str,
    device: torch.device,
    projector_dtype: torch.dtype | None,
    cache_dtype: torch.dtype | None,
) -> tuple[int, int]:
    projectors_cfg = cfg_get(stage_cfg, "projectors", {})
    block_cfg = cfg_get(projectors_cfg, modality, {})
    if not bool(cfg_get(block_cfg, "enabled", False)):
        return 0, 0
    if bool(cfg_get(block_cfg, "trainable", False)):
        raise ValueError(f"Projected-prefix caching requires frozen projectors; {modality}.trainable is true.")

    checkpoint_path = resolve_repo_path(ROOT, cfg_get(block_cfg, "checkpoint_path"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hidden_size = int(checkpoint.get("hidden_size") or checkpoint.get("language_hidden_size") or 0)
    if hidden_size <= 0:
        raise ValueError(f"{modality} projector checkpoint is missing hidden_size: {checkpoint_path}")

    module, _ = build_projector_module(
        repo_root=ROOT,
        modality=modality,
        block_cfg=block_cfg,
        hidden_size=hidden_size,
    )
    if projector_dtype is None:
        module.to(device=device)
    else:
        module.to(device=device, dtype=projector_dtype)
    module.eval()

    feature_refs = _feature_refs_for_modality(frame, stage_cfg, modality)
    batch_size = int(cfg_get(cfg_get(stage_cfg, "batch_size", {}), modality, 1))
    if batch_size <= 0:
        raise ValueError(f"vqa_train.batch_size.{modality} must be positive.")

    overwrite = bool(cfg_get(stage_cfg, "overwrite", False))
    cache_root = cfg_get(stage_cfg, "cache_root")
    model_name_or_path = str(cfg_get(stage_cfg, "model_name_or_path"))
    output_paths = [
        prefix_cache_path(
            repo_root=ROOT,
            cache_root=cache_root,
            model_name_or_path=model_name_or_path,
            modality=modality,
            checkpoint_path=checkpoint_path,
            feature_ref=ref,
        )
        for ref in feature_refs
    ]
    missing_pairs = [(ref, path) for ref, path in zip(feature_refs, output_paths, strict=True) if overwrite or not path.exists()]
    if not missing_pairs:
        return len(feature_refs), 0

    created = 0
    description = f"cache {modality}"
    with torch.no_grad():
        for start in tqdm(range(0, len(missing_pairs), batch_size), desc=description):
            pairs = missing_pairs[start : start + batch_size]
            refs = [ref for ref, _ in pairs]
            features, feature_mask = _load_feature_batch(
                modality=modality,
                refs=refs,
                root_dir=ROOT,
                block_cfg=block_cfg,
            )
            if projector_dtype is None:
                features = features.to(device=device)
            else:
                features = features.to(device=device, dtype=projector_dtype)
            feature_mask = feature_mask.to(device=device)
            projected, output_mask = _project_prefix_batch(
                modality=modality,
                module=module,
                features=features,
                feature_mask=feature_mask,
            )
            for row_index, (_, path) in enumerate(pairs):
                active = output_mask[row_index].to(device=projected.device).bool()
                if not active.any():
                    raise RuntimeError(f"{modality} projector produced no active prefix tokens for {refs[row_index]}")
                prefix_tensor = projected[row_index, active]
                if cache_dtype is not None:
                    prefix_tensor = prefix_tensor.to(dtype=cache_dtype)
                _save_prefix_tensor(path, prefix_tensor, overwrite=overwrite)
                created += 1
    return len(feature_refs), created


def main() -> None:
    cfg = load_cfg()
    stage_cfg = cfg.vqa_train
    device = _resolve_device(cfg_get(stage_cfg, "device", None))
    projector_dtype = resolve_torch_dtype(cfg_get(stage_cfg, "projector_dtype", "float32"))
    cache_dtype = resolve_torch_dtype(cfg_get(stage_cfg, "cache_dtype", "float16"))

    dataset_cfg = cfg_get(stage_cfg, "dataset", {})
    unified_path = resolve_repo_path(ROOT, cfg_get(dataset_cfg, "unified_parquet_path"))
    if not unified_path.exists():
        raise FileNotFoundError(f"Unified parquet not found: {unified_path}")

    frame = _filter_unified_frame(pd.read_parquet(unified_path), stage_cfg)
    if frame.empty:
        raise RuntimeError(f"No unified rows selected from {unified_path}")

    cache_root = ROOT / repo_relative_path(ROOT, cfg_get(stage_cfg, "cache_root"))
    print("VQA projected-prefix cache")
    print(f"Unified parquet: {unified_path.relative_to(ROOT)} ({len(frame):,} selected rows)")
    print(f"Cache root: {cache_root.relative_to(ROOT)}")
    print(f"Model namespace: {cfg_get(stage_cfg, 'model_name_or_path')}")
    print(f"Device: {device}")

    requested_modalities = cfg_list(cfg_get(stage_cfg, "modalities", MODALITIES))
    invalid_modalities = sorted(set(requested_modalities).difference(MODALITIES))
    if invalid_modalities:
        raise ValueError(f"Unsupported VQA cache modalities: {invalid_modalities}")

    total_refs = 0
    total_created = 0
    for modality in requested_modalities:
        refs, created = _cache_modality(
            stage_cfg=stage_cfg,
            frame=frame,
            modality=modality,
            device=device,
            projector_dtype=projector_dtype,
            cache_dtype=cache_dtype,
        )
        total_refs += refs
        total_created += created
        print(f"{modality}: {refs:,} feature refs, {created:,} cached this run")

    print(f"Done. Feature refs: {total_refs:,}; newly cached: {total_created:,}")


if __name__ == "__main__":
    main()
