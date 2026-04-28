#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.modeling.path_projectors import resolve_language_model_hidden_size
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.vqa.data import (
    VQADataset,
    VQATrainingCollator,
    assert_vqa_rows_are_trainable,
    row_missing_prefix_cache_entries,
    prefix_cache_enabled,
    select_vqa_rows,
)
from kidney_vlm.vqa.modeling import (
    OncoVLMVQASFTModel,
    apply_lora,
    build_language_model,
    build_tokenizer,
    load_projectors,
    move_batch_to_device,
    save_vqa_model_artifacts,
)
from kidney_vlm.vqa.prompts import build_vqa_prompt_preview
from kidney_vlm.vqa.schema import VQA_COLUMNS, normalize_vqa_df
from kidney_vlm.vqa.stage_config import (
    cfg_get,
    clean_text,
    generate_run_name,
    projector_trainable_summary,
    resolve_repo_path,
    resolve_torch_dtype,
    slugify_label,
)

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="06_vqa_train/vqa_lora_sft.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_device(device_value: str | None) -> torch.device:
    requested = str(device_value or "").strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{requested}', but CUDA is unavailable.")
    return torch.device(requested)


def _init_ddp(stage_cfg: Any) -> dict[str, Any]:
    requested = bool(cfg_get(stage_cfg, "ddp", False))
    state = {
        "requested": requested,
        "initialized": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "is_main": True,
    }
    if not requested:
        return state
    if not torch.cuda.is_available():
        raise RuntimeError("vqa_train.ddp=true requires CUDA.")
    missing_env = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ]
    if missing_env:
        raise RuntimeError(
            "vqa_train.ddp=true requires launching with torchrun, for example: "
            "torchrun --standalone --nproc_per_node=gpu scripts/06_vqa_train/train_vqa_lora.py"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    state.update(
        {
            "initialized": True,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "is_main": rank == 0,
        }
    )
    return state


def _cleanup_ddp(ddp_state: dict[str, Any]) -> None:
    if bool(ddp_state.get("initialized")) and dist.is_initialized():
        dist.destroy_process_group()


def _ddp_barrier(ddp_state: dict[str, Any]) -> None:
    if bool(ddp_state.get("initialized")):
        dist.barrier()


def _ddp_broadcast_object(value: Any, ddp_state: dict[str, Any]) -> Any:
    if not bool(ddp_state.get("initialized")):
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _ddp_reduce_loss_sum(loss_sum: float, count: int, *, device: torch.device, ddp_state: dict[str, Any]) -> tuple[float, int]:
    if not bool(ddp_state.get("initialized")):
        return loss_sum, count
    tensor = torch.tensor([float(loss_sum), float(count)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item()), int(tensor[1].item())


def _resolve_training_device(stage_cfg: Any, ddp_state: dict[str, Any]) -> torch.device:
    if bool(ddp_state.get("requested")):
        return torch.device(f"cuda:{int(ddp_state['local_rank'])}")
    return _resolve_device(cfg_get(stage_cfg, "device", None))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_run_output_dir(stage_cfg: Any, *, train_rows: int) -> Path:
    output_root = resolve_repo_path(ROOT, cfg_get(stage_cfg, "output_dir"))
    output_root.mkdir(parents=True, exist_ok=True)
    configured_name = clean_text(cfg_get(stage_cfg, "run_name", ""))
    run_name = configured_name or generate_run_name(stage_cfg, train_rows=train_rows)
    run_name = slugify_label(run_name, default="vqa_lora_sft")
    run_output_dir = output_root / run_name
    suffix = 1
    while run_output_dir.exists():
        run_output_dir = output_root / f"{run_name}_{suffix:02d}"
        suffix += 1
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir


def _compute_total_optimizer_steps(*, num_batches_per_epoch: int, num_epochs: int, gradient_accumulation_steps: int) -> int:
    if num_batches_per_epoch <= 0 or num_epochs <= 0:
        return 0
    updates_per_epoch = math.ceil(num_batches_per_epoch / max(1, gradient_accumulation_steps))
    return updates_per_epoch * num_epochs


def _resolve_warmup_steps(*, total_optimizer_steps: int, warmup_steps_cfg: Any, warmup_ratio: float) -> int:
    if total_optimizer_steps <= 0:
        return 0
    if warmup_steps_cfg not in (None, "", "null"):
        return max(0, min(int(warmup_steps_cfg), total_optimizer_steps))
    return max(0, min(int(round(total_optimizer_steps * max(0.0, warmup_ratio))), total_optimizer_steps))


def _build_lr_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    stage_cfg: Any,
    total_optimizer_steps: int,
) -> tuple[Any | None, str, int]:
    scheduler_type = str(cfg_get(stage_cfg, "lr_scheduler_type", "cosine")).strip().lower() or "cosine"
    warmup_ratio = float(cfg_get(stage_cfg, "warmup_ratio", 0.0) or 0.0)
    warmup_steps = _resolve_warmup_steps(
        total_optimizer_steps=total_optimizer_steps,
        warmup_steps_cfg=cfg_get(stage_cfg, "warmup_steps", None),
        warmup_ratio=warmup_ratio,
    )
    if total_optimizer_steps <= 0:
        return None, scheduler_type, warmup_steps

    try:
        from transformers import get_scheduler
    except ImportError as exc:
        raise RuntimeError("transformers is required for VQA scheduler setup.") from exc

    scheduler_name = scheduler_type
    if scheduler_type == "constant" and warmup_steps > 0:
        scheduler_name = "constant_with_warmup"
    elif scheduler_type == "none":
        return None, scheduler_type, warmup_steps
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    return scheduler, scheduler_type, warmup_steps


def _maybe_init_wandb(stage_cfg: Any, *, run_name: str, train_rows: int, val_rows: int):
    wandb_cfg = cfg_get(stage_cfg, "wandb", {})
    if not bool(cfg_get(wandb_cfg, "enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is enabled but wandb is not installed. Install project dependencies first.") from exc

    tags = [str(tag).strip() for tag in cfg_get(wandb_cfg, "tags", []) if str(tag).strip()]
    config_payload = OmegaConf.to_container(stage_cfg, resolve=True)
    if isinstance(config_payload, dict):
        config_payload["train_rows"] = int(train_rows)
        config_payload["val_rows"] = int(val_rows)
    return wandb.init(
        project=str(cfg_get(wandb_cfg, "project", "oncovlm")),
        entity=clean_text(cfg_get(wandb_cfg, "entity", "")) or None,
        name=run_name,
        mode=str(cfg_get(wandb_cfg, "mode", "online")),
        tags=tags,
        config=config_payload,
    )


def _forward_vqa_batch(model: OncoVLMVQASFTModel, batch: dict[str, Any]):
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        pathology_features=batch.get("pathology_features"),
        pathology_feature_mask=batch.get("pathology_feature_mask"),
        radiology_features=batch.get("radiology_features"),
        radiology_feature_mask=batch.get("radiology_feature_mask"),
        dnam_features=batch.get("dnam_features"),
        dnam_feature_mask=batch.get("dnam_feature_mask"),
        rna_features=batch.get("rna_features"),
        rna_feature_mask=batch.get("rna_feature_mask"),
        pathology_prefix_embeddings=batch.get("pathology_prefix_embeddings"),
        pathology_prefix_mask=batch.get("pathology_prefix_mask"),
        radiology_prefix_embeddings=batch.get("radiology_prefix_embeddings"),
        radiology_prefix_mask=batch.get("radiology_prefix_mask"),
        dnam_prefix_embeddings=batch.get("dnam_prefix_embeddings"),
        dnam_prefix_mask=batch.get("dnam_prefix_mask"),
        rna_prefix_embeddings=batch.get("rna_prefix_embeddings"),
        rna_prefix_mask=batch.get("rna_prefix_mask"),
        prefix_spans=batch.get("prefix_spans"),
    )


def _print_prompt_previews(*, stage_cfg: Any, train_frame: pd.DataFrame) -> bool:
    preview_cfg = cfg_get(stage_cfg, "preview", {})
    sample_count = int(cfg_get(preview_cfg, "num_train_samples", 0) or 0)
    if sample_count <= 0:
        return False
    prompt_cfg = cfg_get(stage_cfg, "prompt", {})
    print(f"VQA prompt preview: first {min(sample_count, len(train_frame))} train samples")
    for preview_index, (_, row) in enumerate(train_frame.head(sample_count).iterrows(), start=1):
        print(f"\n--- VQA prompt preview {preview_index} | question_id={row.get('question_id')} ---")
        print(build_vqa_prompt_preview(row, prompt_cfg))
        print(f"\n<assistant_answer>\n{clean_text(row.get('answer', ''))}\n</assistant_answer>")
    return bool(cfg_get(preview_cfg, "exit_after_preview", False))


def _cached_projector_metadata(stage_cfg: Any) -> dict[str, dict[str, Any]]:
    projectors_cfg = cfg_get(stage_cfg, "projectors", {})
    metadata: dict[str, dict[str, Any]] = {}
    for modality in ("pathology", "radiology", "dnam", "rna"):
        block_cfg = cfg_get(projectors_cfg, modality, {})
        if not bool(cfg_get(block_cfg, "enabled", False)):
            continue
        if bool(cfg_get(block_cfg, "trainable", False)):
            raise ValueError(f"vqa_train.prefix_cache.enabled=true requires frozen projectors; {modality}.trainable=true.")
        metadata[modality] = {
            "modality": modality,
            "checkpoint_path": str(resolve_repo_path(ROOT, cfg_get(block_cfg, "checkpoint_path"))),
            "trainable": False,
            "prefix_cache_enabled": True,
            "prefix_cache_root": str(resolve_repo_path(ROOT, cfg_get(cfg_get(stage_cfg, "prefix_cache", {}), "cache_root"))),
        }
    if not metadata:
        raise RuntimeError("Prefix-cache training needs at least one enabled projector block.")
    return metadata


def _filter_missing_prefix_cache_rows(
    *,
    stage_cfg: Any,
    frame: pd.DataFrame,
    split_label: str,
    log: bool = True,
) -> pd.DataFrame:
    if not prefix_cache_enabled(stage_cfg):
        return frame
    prefix_cfg = cfg_get(stage_cfg, "prefix_cache", {})
    if not bool(cfg_get(prefix_cfg, "scan_before_training", True)):
        if log:
            print(f"Prefix cache: scan_before_training=false; not pre-scanning {split_label} rows.")
        return frame

    keep_mask: list[bool] = []
    skipped: list[dict[str, Any]] = []
    for row_index, row in tqdm(
        frame.iterrows(),
        total=len(frame),
        desc=f"scan {split_label} prefix cache",
        leave=False,
        disable=not log,
    ):
        missing = row_missing_prefix_cache_entries(ROOT, stage_cfg, row)
        keep = not missing
        keep_mask.append(keep)
        if not keep:
            skipped.append(
                {
                    "row_index": int(row_index),
                    "question_id": row.get("question_id"),
                    "case_id": row.get("case_id"),
                    "missing": missing,
                }
            )

    filtered_frame = frame.loc[keep_mask].reset_index(drop=True)
    if not skipped:
        if log:
            print(f"Prefix cache: no missing {split_label} rows.")
        return filtered_frame

    if not bool(cfg_get(prefix_cfg, "skip_missing_rows", True)):
        first = skipped[0]
        raise FileNotFoundError(
            f"Missing cached VQA prefixes for {split_label} row question_id={first.get('question_id')}: "
            f"{first.get('missing')}"
        )

    if not log:
        return filtered_frame

    print(f"Prefix cache: skipped {len(skipped):,} {split_label} rows with missing cached prefixes.")
    max_examples = int(cfg_get(prefix_cfg, "max_missing_examples", 10) or 0)
    for item in skipped[:max_examples]:
        missing = item["missing"]
        first_missing = missing[0] if missing else "<unknown>"
        print(
            "  skipped "
            f"question_id={item.get('question_id')} case_id={item.get('case_id')} missing={first_missing}"
        )
    if len(skipped) > max_examples:
        print(f"  ... {len(skipped) - max_examples:,} more skipped rows")
    return filtered_frame


def _run_validation(
    *,
    model: OncoVLMVQASFTModel,
    val_loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    use_autocast: bool,
    floating_input_dtype: torch.dtype | None,
) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        loop = tqdm(val_loader, total=len(val_loader), desc="Validation", leave=False)
        for step, batch in enumerate(loop, start=1):
            batch = move_batch_to_device(batch, device, floating_dtype=floating_input_dtype)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                outputs = _forward_vqa_batch(model, batch)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model did not return a loss during validation.")
            running_loss += float(loss.detach().cpu())
            loop.set_postfix(loss=f"{running_loss / step:.4f}")
    model.train()
    model.set_frozen_projectors_eval()
    return running_loss / max(1, len(val_loader))


def _portable_path(path_value: str | Path) -> str:
    resolved = Path(path_value).expanduser().resolve()
    return Path(os.path.relpath(resolved, start=ROOT)).as_posix()


def _write_run_metadata(
    *,
    run_output_dir: Path,
    stage_cfg: Any,
    model: OncoVLMVQASFTModel,
    train_rows: int,
    val_rows: int,
    global_step: int,
    best_validation_loss: float | None,
    best_epoch: int | None,
    final_artifacts: dict[str, str] | None,
    best_artifacts: dict[str, str] | None,
) -> Path:
    metadata = {
        "run_output_dir": _portable_path(run_output_dir),
        "model_name_or_path": str(cfg_get(stage_cfg, "model_name_or_path")),
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "global_step": int(global_step),
        "trainable_parameters": int(model.trainable_parameter_count()),
        "total_parameters": int(model.total_parameter_count()),
        "projectors": model.projector_metadata,
        "config_path": _portable_path(run_output_dir / "config.yaml"),
    }
    if final_artifacts is not None:
        metadata["final_artifacts"] = {key: _portable_path(value) for key, value in final_artifacts.items()}
    if best_artifacts is not None:
        metadata["best_artifacts"] = {key: _portable_path(value) for key, value in best_artifacts.items()}
    if best_validation_loss is not None and math.isfinite(best_validation_loss):
        metadata["best_validation_loss"] = float(best_validation_loss)
    if best_epoch is not None:
        metadata["best_epoch"] = int(best_epoch)
    metadata_path = run_output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _is_improved_validation_loss(validation_loss: float | None, best_validation_loss: float | None) -> bool:
    if validation_loss is None or not math.isfinite(validation_loss):
        return False
    if best_validation_loss is None:
        return True
    return validation_loss < best_validation_loss


def main() -> None:
    cfg = load_cfg()
    stage_cfg = cfg.vqa_train
    if not bool(cfg_get(stage_cfg, "instantiate_model", True)):
        print("vqa_train.instantiate_model=false; nothing to do.")
        return

    ddp_state = _init_ddp(stage_cfg)
    try:
        is_main = bool(ddp_state["is_main"])
        seed = int(cfg_get(stage_cfg, "seed", 42))
        _set_seed(seed)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        dataset_cfg = cfg_get(stage_cfg, "dataset", {})
        vqa_parquet_path = resolve_repo_path(ROOT, cfg_get(dataset_cfg, "vqa_parquet_path"))
        if not vqa_parquet_path.exists():
            raise FileNotFoundError(f"VQA training parquet not found: {vqa_parquet_path}")

        frame = normalize_vqa_df(pd.read_parquet(vqa_parquet_path))
        missing_columns = [column for column in VQA_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"VQA parquet is missing required columns: {missing_columns}")
        if frame.empty:
            raise RuntimeError(f"VQA parquet is empty: {vqa_parquet_path}")

        train_split = str(cfg_get(dataset_cfg, "train_split", "train"))
        validation_split = str(cfg_get(dataset_cfg, "validation_split", "val"))
        train_frame = select_vqa_rows(
            frame,
            stage_cfg,
            split=train_split,
            max_samples_key="max_train_samples",
            sample_key="sample_train",
        )
        train_frame = _filter_missing_prefix_cache_rows(
            stage_cfg=stage_cfg,
            frame=train_frame,
            split_label=train_split,
            log=is_main,
        )
        assert_vqa_rows_are_trainable(train_frame)
        exit_after_preview = False
        if is_main:
            exit_after_preview = _print_prompt_previews(stage_cfg=stage_cfg, train_frame=train_frame)
        exit_after_preview = bool(_ddp_broadcast_object(exit_after_preview, ddp_state))
        if exit_after_preview:
            if is_main:
                print("preview.exit_after_preview=true; stopping before tokenizer/model/projector loading.")
            _ddp_barrier(ddp_state)
            return

        validation_cfg = cfg_get(stage_cfg, "validation", {})
        val_frame = pd.DataFrame(columns=frame.columns)
        if bool(cfg_get(validation_cfg, "enabled", True)):
            val_frame = select_vqa_rows(
                frame,
                stage_cfg,
                split=validation_split,
                max_samples_key="max_val_samples",
                sample_key="sample_val",
            )
            val_frame = _filter_missing_prefix_cache_rows(
                stage_cfg=stage_cfg,
                frame=val_frame,
                split_label=validation_split,
                log=is_main,
            )
            if val_frame.empty and bool(cfg_get(validation_cfg, "required", False)):
                raise RuntimeError(f"Validation is required but no rows were selected for split={validation_split!r}.")
            if not val_frame.empty:
                assert_vqa_rows_are_trainable(val_frame)

        device = _resolve_training_device(stage_cfg, ddp_state)
        load_in_8bit = bool(cfg_get(stage_cfg, "load_in_8bit", False))
        if load_in_8bit and device.type != "cuda":
            raise RuntimeError("vqa_train.load_in_8bit=true requires a CUDA device.")

        tokenizer = build_tokenizer(
            str(cfg_get(stage_cfg, "model_name_or_path")),
            trust_remote_code=bool(cfg_get(stage_cfg, "trust_remote_code", True)),
        )
        base_language_model = build_language_model(stage_cfg, device=device)
        hidden_size = resolve_language_model_hidden_size(base_language_model)
        language_model = apply_lora(base_language_model, stage_cfg)
        if is_main and hasattr(language_model, "print_trainable_parameters"):
            language_model.print_trainable_parameters()

        use_prefix_cache = prefix_cache_enabled(stage_cfg)
        if use_prefix_cache:
            projectors = {}
            projector_metadata = _cached_projector_metadata(stage_cfg)
        else:
            projectors, projector_metadata = load_projectors(stage_cfg, repo_root=ROOT, hidden_size=hidden_size)
        model = OncoVLMVQASFTModel(
            language_model=language_model,
            projectors=projectors,
            projector_metadata=projector_metadata,
        )

        autocast_dtype = resolve_torch_dtype(cfg_get(stage_cfg, "autocast_dtype", "bfloat16")) or torch.bfloat16
        projector_dtype = resolve_torch_dtype(cfg_get(stage_cfg, "projector_dtype", "float32"))
        if load_in_8bit:
            if not use_prefix_cache:
                model.move_projectors_to(device, dtype=projector_dtype)
        else:
            model.to(device=device)
            if not use_prefix_cache:
                model.move_projectors_to(device, dtype=projector_dtype)
        model.train()
        model.set_frozen_projectors_eval()

        train_model: torch.nn.Module = model
        if bool(ddp_state["initialized"]) and int(ddp_state["world_size"]) > 1:
            find_unused_parameters = any(bool(metadata.get("trainable", False)) for metadata in projector_metadata.values())
            train_model = DistributedDataParallel(
                model,
                device_ids=[int(ddp_state["local_rank"])],
                output_device=int(ddp_state["local_rank"]),
                find_unused_parameters=find_unused_parameters,
            )

        collator = VQATrainingCollator(tokenizer=tokenizer, root_dir=ROOT, stage_cfg=stage_cfg)
        train_dataset = VQADataset(train_frame)
        train_sampler = None
        if bool(ddp_state["initialized"]) and int(ddp_state["world_size"]) > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=int(ddp_state["world_size"]),
                rank=int(ddp_state["rank"]),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg_get(stage_cfg, "batch_size", 1)),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=int(cfg_get(stage_cfg, "dataloader_num_workers", 0)),
            collate_fn=collator,
        )
        if len(train_loader) == 0:
            raise RuntimeError("Training loader is empty after batching.")
        validation_loader = None
        if not val_frame.empty and is_main:
            validation_loader = DataLoader(
                VQADataset(val_frame),
                batch_size=int(cfg_get(stage_cfg, "batch_size", 1)),
                shuffle=False,
                num_workers=int(cfg_get(stage_cfg, "dataloader_num_workers", 0)),
                collate_fn=VQATrainingCollator(tokenizer=tokenizer, root_dir=ROOT, stage_cfg=stage_cfg),
            )

        trainable_parameters = [parameter for parameter in train_model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters found for VQA LoRA training.")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(cfg_get(stage_cfg, "learning_rate", 2e-5)),
            weight_decay=float(cfg_get(stage_cfg, "weight_decay", 0.0)),
        )

        if is_main:
            run_output_dir = _build_run_output_dir(stage_cfg, train_rows=len(train_frame))
            OmegaConf.save(config=stage_cfg, f=str(run_output_dir / "config.yaml"))
            run_output_dir_value = str(run_output_dir)
        else:
            run_output_dir_value = None
        run_output_dir = Path(_ddp_broadcast_object(run_output_dir_value, ddp_state))
        _ddp_barrier(ddp_state)

        num_epochs = int(cfg_get(stage_cfg, "num_epochs", 1))
        grad_accum = max(1, int(cfg_get(stage_cfg, "gradient_accumulation_steps", 1)))
        total_optimizer_steps = _compute_total_optimizer_steps(
            num_batches_per_epoch=len(train_loader),
            num_epochs=num_epochs,
            gradient_accumulation_steps=grad_accum,
        )
        lr_scheduler, scheduler_type, warmup_steps = _build_lr_scheduler(
            optimizer=optimizer,
            stage_cfg=stage_cfg,
            total_optimizer_steps=total_optimizer_steps,
        )
        grad_clip_norm = float(cfg_get(stage_cfg, "grad_clip_norm", 1.0))
        use_autocast = device.type == "cuda" and autocast_dtype != torch.float32
        use_grad_scaler = use_autocast and autocast_dtype == torch.float16
        grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

        if is_main:
            print("Stage 2 VQA LoRA SFT")
            print(f"VQA parquet: {vqa_parquet_path}")
            print(f"Train split: {train_split} ({len(train_frame):,} rows)")
            print(f"Validation split: {validation_split} ({len(val_frame):,} rows)")
            print(f"Model: {cfg_get(stage_cfg, 'model_name_or_path')}")
            print(f"Hidden size: {hidden_size}")
            print(f"Device: {device}")
            if bool(ddp_state["requested"]):
                print(f"DDP: world_size={int(ddp_state['world_size'])}")
            print(f"LoRA r: {int(cfg_get(cfg_get(stage_cfg, 'lora', {}), 'r', 16))}")
            if use_prefix_cache:
                print(f"Prefix source: cached embeddings from {cfg_get(cfg_get(stage_cfg, 'prefix_cache', {}), 'cache_root')}")
                print("Loaded projectors: none (prefix cache enabled)")
            else:
                print(f"Loaded projectors: {', '.join(projector_metadata)}")
            print(f"Trainable projector mode: {projector_trainable_summary(stage_cfg)}")
            print(f"Run output dir: {run_output_dir}")
            print(f"Trainable parameters: {model.trainable_parameter_count():,}")
            print(f"Total parameters: {model.total_parameter_count():,}")
            print(f"Scheduler: {scheduler_type}")
            print(f"Warmup steps: {warmup_steps}")
            print(f"Total optimizer steps: {total_optimizer_steps}")
            if validation_loader is None:
                print("Validation loader is unavailable; only final artifacts will be saved.")

        wandb_run = None
        if is_main:
            wandb_run = _maybe_init_wandb(
                stage_cfg,
                run_name=run_output_dir.name,
                train_rows=len(train_frame),
                val_rows=len(val_frame),
            )
        global_step = 0
        best_validation_loss = None
        best_epoch = None
        best_artifacts = None
        final_artifacts = None
        wandb_cfg = cfg_get(stage_cfg, "wandb", {})
        wandb_log_every_n_steps = int(cfg_get(wandb_cfg, "log_every_n_steps", 0) or 0)
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_model.train()
            model.set_frozen_projectors_eval()
            running_loss = 0.0
            accum_loss = 0.0
            accum_count = 0
            if is_main:
                loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            else:
                loop = train_loader
            for step, batch in enumerate(loop, start=1):
                batch = move_batch_to_device(batch, device, floating_dtype=None if use_prefix_cache else projector_dtype)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                    outputs = _forward_vqa_batch(train_model, batch)
                    loss = outputs.loss
                    if loss is None:
                        raise RuntimeError("Model did not return a loss during VQA LoRA training.")
                    scaled_loss = loss / grad_accum

                if use_grad_scaler:
                    grad_scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                batch_loss = float(loss.detach().cpu())
                accum_loss += batch_loss
                accum_count += 1
                if step % grad_accum == 0 or step == len(train_loader):
                    if grad_clip_norm > 0:
                        if use_grad_scaler:
                            grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
                    optimizer_stepped = True
                    if use_grad_scaler:
                        scale_before_step = grad_scaler.get_scale()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        optimizer_stepped = grad_scaler.get_scale() >= scale_before_step
                    else:
                        optimizer.step()
                    if optimizer_stepped and lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if optimizer_stepped:
                        global_step += 1
                        if (
                            wandb_run is not None
                            and wandb_log_every_n_steps > 0
                            and global_step % wandb_log_every_n_steps == 0
                        ):
                            wandb_run.log(
                                {
                                    "train/loss": accum_loss / max(1, accum_count),
                                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                                    "train/epoch": epoch + 1,
                                    "train/epoch_step": step,
                                    "train/optimizer_step": global_step,
                                },
                                step=global_step,
                            )
                    accum_loss = 0.0
                    accum_count = 0
                running_loss += batch_loss
                if is_main:
                    loop.set_postfix(loss=f"{running_loss / step:.4f}")

            reduced_loss_sum, reduced_loss_count = _ddp_reduce_loss_sum(
                running_loss,
                len(train_loader),
                device=device,
                ddp_state=ddp_state,
            )
            epoch_loss = reduced_loss_sum / max(1, reduced_loss_count)
            if is_main:
                print(f"Epoch {epoch + 1} mean loss: {epoch_loss:.4f}")
                if wandb_run is not None:
                    wandb_run.log({"train/epoch_mean_loss": epoch_loss, "train/epoch": epoch + 1}, step=global_step)

            validation_loss = None
            if validation_loader is not None and len(validation_loader) > 0:
                validation_loss = _run_validation(
                    model=model,
                    val_loader=validation_loader,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    use_autocast=use_autocast,
                    floating_input_dtype=None if use_prefix_cache else projector_dtype,
                )
                print(f"Epoch {epoch + 1} validation loss: {validation_loss:.4f}")
                if wandb_run is not None:
                    wandb_run.log({"val/loss": validation_loss, "val/epoch": epoch + 1}, step=global_step)
                if _is_improved_validation_loss(validation_loss, best_validation_loss):
                    best_validation_loss = validation_loss
                    best_epoch = epoch + 1
                    best_artifacts = save_vqa_model_artifacts(
                        artifact_dir=run_output_dir / "best",
                        stage_cfg=stage_cfg,
                        model=model,
                        tokenizer=tokenizer,
                        global_step=global_step,
                        epoch=best_epoch,
                        validation_loss=best_validation_loss,
                    )
                    print(f"Saved best VQA LoRA artifacts to: {run_output_dir / 'best'}")
            _ddp_barrier(ddp_state)

            if is_main and bool(cfg_get(stage_cfg, "save_every_epoch", False)):
                save_vqa_model_artifacts(
                    artifact_dir=run_output_dir / f"epoch_{epoch + 1:03d}",
                    stage_cfg=stage_cfg,
                    model=model,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    epoch=epoch + 1,
                    validation_loss=validation_loss,
                )

            if is_main:
                _write_run_metadata(
                    run_output_dir=run_output_dir,
                    stage_cfg=stage_cfg,
                    model=model,
                    train_rows=len(train_frame),
                    val_rows=len(val_frame),
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    best_epoch=best_epoch,
                    final_artifacts=final_artifacts,
                    best_artifacts=best_artifacts,
                )
            _ddp_barrier(ddp_state)

        if is_main:
            final_artifacts = save_vqa_model_artifacts(
                artifact_dir=run_output_dir,
                stage_cfg=stage_cfg,
                model=model,
                tokenizer=tokenizer,
                global_step=global_step,
                epoch=num_epochs,
                validation_loss=best_validation_loss,
            )
            metadata_path = _write_run_metadata(
                run_output_dir=run_output_dir,
                stage_cfg=stage_cfg,
                model=model,
                train_rows=len(train_frame),
                val_rows=len(val_frame),
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                best_epoch=best_epoch,
                final_artifacts=final_artifacts,
                best_artifacts=best_artifacts,
            )
            if wandb_run is not None:
                payload = {
                    "artifacts/output_dir": str(run_output_dir),
                    "artifacts/global_step": global_step,
                }
                if best_validation_loss is not None:
                    payload["val/best_loss"] = best_validation_loss
                if best_epoch is not None:
                    payload["val/best_epoch"] = best_epoch
                wandb_run.log(payload, step=global_step)
                wandb_run.finish()

            print(f"Saved final LoRA adapter to: {final_artifacts['lora_adapter_dir']}")
            print(f"Saved final projector checkpoint to: {final_artifacts['projectors_checkpoint']}")
            print(f"Saved run metadata to: {metadata_path}")
            if best_validation_loss is not None and best_epoch is not None:
                print(f"Best validation loss: {best_validation_loss:.4f} at epoch {best_epoch}")
        _ddp_barrier(ddp_state)
    finally:
        _cleanup_ddp(ddp_state)


if __name__ == "__main__":
    main()
