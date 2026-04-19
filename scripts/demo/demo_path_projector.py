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

from kidney_vlm.modeling.pathology_qwen_projector import PathologyQwenProjectorLM
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.training.collator import _load_h5_patch_features as _load_h5_patch_features_shared

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config")
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


def _build_tokenizer(model_name_or_path: str, trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for the pathology projector demo.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


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


def _resolve_feature_path(
    cfg: Any,
    file_stem: str,
    *,
    feature_paths_by_slide_stem: dict[str, str] | None = None,
) -> Path:
    mapped_feature_path = str((feature_paths_by_slide_stem or {}).get(file_stem, "")).strip()
    if mapped_feature_path:
        resolved_mapped = _resolve_path(mapped_feature_path)
        if resolved_mapped.exists():
            return resolved_mapped
        raise FileNotFoundError(
            f"Feature file from pathology projector parquet not found for stem '{file_stem}': {resolved_mapped}"
        )

    patch_encoder_name = str(cfg.pathology_features.patch_encoder)
    save_format = str(cfg.pathology_features.get("save_format", "h5")).lower()
    features_root = _resolve_path(cfg.pathology_features.features_root)
    feature_path = features_root / f"features_{patch_encoder_name}" / f"{file_stem}.{save_format}"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found for stem '{file_stem}': {feature_path}")
    return feature_path


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


def _load_path_projector_state(checkpoint_path: Path) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu")


def _resolve_checkpoint_path(path_value: str | Path, checkpoint_name: str | None = None) -> Path:
    resolved = _resolve_path(path_value)
    desired_name = str(checkpoint_name or "").strip()
    if resolved.is_file():
        return resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Path projector checkpoint path not found: {resolved}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Path projector checkpoint path is neither a file nor a directory: {resolved}")

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


def _load_path_projector_model(cfg: Any, state: dict[str, Any], device: torch.device) -> PathologyQwenProjectorLM:
    model_name_or_path = str(state.get("model_name_or_path") or cfg.pathology_proj.model_name_or_path)
    pathology_embedding_dim = int(state.get("pathology_embedding_dim") or cfg.pathology_proj.pathology_embedding_dim)

    model = PathologyQwenProjectorLM.from_pretrained(
        model_name_or_path,
        pathology_in_dim=pathology_embedding_dim,
        projector_type=str(state.get("projector_type") or cfg.pathology_proj.get("projector_type", "mlp")),
        projector_num_latents=int(state.get("projector_num_latents") or cfg.pathology_proj.get("projector_num_latents", 64)),
        projector_depth=int(state.get("projector_depth") or cfg.pathology_proj.get("projector_depth", 2)),
        projector_num_heads=int(state.get("projector_num_heads") or cfg.pathology_proj.get("projector_num_heads", 8)),
        projector_mlp_ratio=float(state.get("projector_mlp_ratio") or cfg.pathology_proj.get("projector_mlp_ratio", 4.0)),
        projector_dropout=float(state.get("projector_dropout") or cfg.pathology_proj.get("projector_dropout", 0.0)),
        trust_remote_code=bool(cfg.pathology_proj.trust_remote_code),
        torch_dtype=cfg.pathology_proj.get("torch_dtype"),
        attn_implementation=cfg.pathology_proj.get("attn_implementation"),
    )
    state_dict = state.get("path_projector_state_dict") or state.get("projector_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint is missing both 'path_projector_state_dict' and legacy 'projector_state_dict'.")
    model.path_projectors.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if hasattr(model.language_model.config, "use_cache"):
        model.language_model.config.use_cache = True
    return model


def _resolve_demo_file_stems(cfg: Any, demo_cfg: Any) -> list[str]:
    max_samples = int(demo_cfg.get("max_samples", 1))
    if max_samples <= 0:
        raise RuntimeError("demo.max_samples must be a positive integer.")
    split_filter = str(demo_cfg.get("split_filter", "")).strip().lower()

    explicit_file_stem = str(demo_cfg.get("file_stem", "")).strip()
    configured_file_stems = demo_cfg.get("file_stems")
    explicit_file_stems: list[str] = []
    if configured_file_stems not in (None, "", "null"):
        for value in list(configured_file_stems):
            stem = str(value).strip()
            if stem:
                explicit_file_stems.append(stem)

    if explicit_file_stem:
        stems = [explicit_file_stem]
    elif explicit_file_stems:
        stems = explicit_file_stems
    else:
        qa_parquet_path = _resolve_path(cfg.pathology_proj.qa_parquet_path)
        if not qa_parquet_path.exists():
            raise FileNotFoundError(f"Pathology projector training parquet not found: {qa_parquet_path}")
        columns = ["slide_stem"]
        if split_filter:
            columns.append("split")
        frame = pd.read_parquet(qa_parquet_path, columns=columns)
        if "slide_stem" not in frame.columns:
            raise RuntimeError(f"Pathology projector training parquet is missing 'slide_stem': {qa_parquet_path}")
        if split_filter:
            if "split" not in frame.columns:
                raise RuntimeError(
                    f"demo.split_filter='{split_filter}' was requested, but the parquet is missing the 'split' column: {qa_parquet_path}"
                )
            frame = frame.loc[frame["split"].fillna("train").astype(str).str.lower() == split_filter].copy()
        stems = (
            frame["slide_stem"]
            .fillna("")
            .astype(str)
            .str.strip()
            .loc[lambda series: series != ""]
            .drop_duplicates()
            .tolist()
        )
    if not stems:
        raise RuntimeError("No demo samples found. Set demo.file_stem, demo.file_stems, or populate the training parquet.")
    if explicit_file_stem or explicit_file_stems:
        return stems[:max_samples]
    if len(stems) <= max_samples:
        return stems
    sample_seed = int(demo_cfg.get("sample_seed", 42))
    rng = random.Random(sample_seed)
    return rng.sample(stems, k=max_samples)


def _load_project_labels_by_slide_stem(cfg: Any) -> dict[str, str]:
    qa_parquet_path = _resolve_path(cfg.pathology_proj.qa_parquet_path)
    if not qa_parquet_path.exists():
        return {}
    frame = pd.read_parquet(qa_parquet_path, columns=["slide_stem", "project_id"])
    if "slide_stem" not in frame.columns or "project_id" not in frame.columns:
        return {}
    mapping: dict[str, str] = {}
    for row in frame.drop_duplicates("slide_stem").itertuples(index=False):
        slide_stem = str(row.slide_stem).strip()
        project_id = str(row.project_id).strip()
        if slide_stem and project_id:
            mapping[slide_stem] = project_id
    return mapping


def _load_feature_paths_by_slide_stem(cfg: Any) -> dict[str, str]:
    qa_parquet_path = _resolve_path(cfg.pathology_proj.qa_parquet_path)
    if not qa_parquet_path.exists():
        return {}
    frame = pd.read_parquet(qa_parquet_path, columns=["slide_stem", "pathology_tile_embedding_paths"])
    if "slide_stem" not in frame.columns or "pathology_tile_embedding_paths" not in frame.columns:
        return {}
    mapping: dict[str, str] = {}
    for row in frame.drop_duplicates("slide_stem").itertuples(index=False):
        slide_stem = str(row.slide_stem).strip()
        tile_paths = _as_list(row.pathology_tile_embedding_paths)
        if slide_stem and tile_paths:
            mapping[slide_stem] = tile_paths[0]
    return mapping


def _generate_response(
    *,
    model: PathologyQwenProjectorLM,
    tokenizer: Any,
    prompt: str,
    pathology_features: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    pathology_features = pathology_features.unsqueeze(0).to(device)
    pathology_feature_mask = torch.ones(
        pathology_features.shape[:2],
        dtype=torch.long,
        device=device,
    )

    text_embeddings = model.language_model.get_input_embeddings()(input_ids)
    pathology_projected, _ = model.path_projectors["pathology"](pathology_features, pathology_feature_mask)
    pathology_projected = pathology_projected.to(dtype=text_embeddings.dtype)

    prefix_attention = model.path_projectors["pathology"].build_output_mask(
        pathology_feature_mask,
        batch_size=pathology_projected.shape[0],
        output_length=pathology_projected.shape[1],
        device=device,
        dtype=attention_mask.dtype,
    )
    inputs_embeds = torch.cat([pathology_projected, text_embeddings], dim=1)
    combined_attention = torch.cat([prefix_attention, attention_mask], dim=1)

    generate_kwargs: dict[str, Any] = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": combined_attention,
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
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    return generated_text


def main() -> None:
    cfg = load_cfg()
    demo_cfg = cfg.demo

    prompt = str(demo_cfg.prompt).strip()
    if not prompt:
        raise RuntimeError("demo.prompt must be set in the config.")
    file_stems = _resolve_demo_file_stems(cfg, demo_cfg)

    checkpoint_path = _resolve_checkpoint_path(
        demo_cfg.path_projector_ckpt_path,
        checkpoint_name=demo_cfg.get("checkpoint_name"),
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Path projector checkpoint not found: {checkpoint_path}")

    state = _load_path_projector_state(checkpoint_path)
    model_name_or_path = str(state.get("model_name_or_path") or cfg.pathology_proj.model_name_or_path)
    device = _resolve_device(cfg.pathology_proj.device)
    tokenizer = _build_tokenizer(
        model_name_or_path=model_name_or_path,
        trust_remote_code=bool(cfg.pathology_proj.trust_remote_code),
    )
    model = _load_path_projector_model(cfg, state=state, device=device)
    project_labels_by_slide_stem = _load_project_labels_by_slide_stem(cfg)
    feature_paths_by_slide_stem = _load_feature_paths_by_slide_stem(cfg)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompt: {prompt}")
    if str(demo_cfg.get("split_filter", "")).strip():
        print(f"Split filter: {str(demo_cfg.get('split_filter')).strip()}")
    print(f"Samples: {len(file_stems)}")
    print(
        "Patch compression: "
        f"{str(demo_cfg.get('patch_compression_method', 'none'))} "
        f"(kernel={int(demo_cfg.get('patch_compression_kernel_size', 1))}); "
        "max_patch_tokens is applied after compression."
    )
    for index, file_stem in enumerate(file_stems, start=1):
        feature_path = _resolve_feature_path(
            cfg,
            file_stem,
            feature_paths_by_slide_stem=feature_paths_by_slide_stem,
        )
        pathology_features = _load_h5_patch_features(
            feature_path,
            max_patch_tokens=int(demo_cfg.max_patch_tokens),
            compression_method=str(demo_cfg.get("patch_compression_method", "none")),
            compression_kernel_size=int(demo_cfg.get("patch_compression_kernel_size", 1)),
        )
        response = _generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            pathology_features=pathology_features,
            device=device,
            max_new_tokens=int(demo_cfg.max_new_tokens),
            do_sample=bool(demo_cfg.do_sample),
            temperature=float(demo_cfg.temperature),
            top_p=float(demo_cfg.top_p),
        )
        print(f"Sample {index}: {file_stem}")
        project_label = project_labels_by_slide_stem.get(file_stem)
        if project_label:
            print(f"TCGA project: {project_label}")
        print(f"Feature path: {feature_path}")
        print("Response:")
        print(response)
        if index != len(file_stems):
            print()


if __name__ == "__main__":
    main()
