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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
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
    assert_vqa_rows_are_trainable(train_frame)
    if _print_prompt_previews(stage_cfg=stage_cfg, train_frame=train_frame):
        print("preview.exit_after_preview=true; stopping before tokenizer/model/projector loading.")
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
        if val_frame.empty and bool(cfg_get(validation_cfg, "required", False)):
            raise RuntimeError(f"Validation is required but no rows were selected for split={validation_split!r}.")
        if not val_frame.empty:
            assert_vqa_rows_are_trainable(val_frame)

    device = _resolve_device(cfg_get(stage_cfg, "device", None))
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
    if hasattr(language_model, "print_trainable_parameters"):
        language_model.print_trainable_parameters()

    projectors, projector_metadata = load_projectors(stage_cfg, repo_root=ROOT, hidden_size=hidden_size)
    model = OncoVLMVQASFTModel(
        language_model=language_model,
        projectors=projectors,
        projector_metadata=projector_metadata,
    )

    autocast_dtype = resolve_torch_dtype(cfg_get(stage_cfg, "autocast_dtype", "bfloat16")) or torch.bfloat16
    projector_dtype = resolve_torch_dtype(cfg_get(stage_cfg, "projector_dtype", "float32"))
    if load_in_8bit:
        model.move_projectors_to(device, dtype=projector_dtype)
    else:
        model.to(device=device)
        model.move_projectors_to(device, dtype=projector_dtype)
    model.train()
    model.set_frozen_projectors_eval()

    collator = VQATrainingCollator(tokenizer=tokenizer, root_dir=ROOT, stage_cfg=stage_cfg)
    train_loader = DataLoader(
        VQADataset(train_frame),
        batch_size=int(cfg_get(stage_cfg, "batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg_get(stage_cfg, "dataloader_num_workers", 0)),
        collate_fn=collator,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty after batching.")
    validation_loader = None
    if not val_frame.empty:
        validation_loader = DataLoader(
            VQADataset(val_frame),
            batch_size=int(cfg_get(stage_cfg, "batch_size", 1)),
            shuffle=False,
            num_workers=int(cfg_get(stage_cfg, "dataloader_num_workers", 0)),
            collate_fn=VQATrainingCollator(tokenizer=tokenizer, root_dir=ROOT, stage_cfg=stage_cfg),
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found for VQA LoRA training.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg_get(stage_cfg, "learning_rate", 2e-5)),
        weight_decay=float(cfg_get(stage_cfg, "weight_decay", 0.0)),
    )

    run_output_dir = _build_run_output_dir(stage_cfg, train_rows=len(train_frame))
    OmegaConf.save(config=stage_cfg, f=str(run_output_dir / "config.yaml"))
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

    print("Stage 2 VQA LoRA SFT")
    print(f"VQA parquet: {vqa_parquet_path}")
    print(f"Train split: {train_split} ({len(train_frame):,} rows)")
    print(f"Validation split: {validation_split} ({len(val_frame):,} rows)")
    print(f"Model: {cfg_get(stage_cfg, 'model_name_or_path')}")
    print(f"Hidden size: {hidden_size}")
    print(f"Device: {device}")
    print(f"LoRA r: {int(cfg_get(cfg_get(stage_cfg, 'lora', {}), 'r', 16))}")
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
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(num_epochs):
        model.train()
        model.set_frozen_projectors_eval()
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(loop, start=1):
            batch = move_batch_to_device(batch, device, floating_dtype=projector_dtype)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                outputs = _forward_vqa_batch(model, batch)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model did not return a loss during VQA LoRA training.")
                scaled_loss = loss / grad_accum

            if use_grad_scaler:
                grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if step % grad_accum == 0 or step == len(train_loader):
                if grad_clip_norm > 0:
                    if use_grad_scaler:
                        grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
                if use_grad_scaler:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": float(loss.detach().cpu()),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/epoch": epoch + 1,
                            "train/optimizer_step": global_step,
                        },
                        step=global_step,
                    )

            running_loss += float(loss.detach().cpu())
            loop.set_postfix(loss=f"{running_loss / step:.4f}")

        epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1} mean loss: {epoch_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log({"train/epoch_mean_loss": epoch_loss, "train/epoch": epoch + 1}, step=max(global_step, 1))

        validation_loss = None
        if validation_loader is not None and len(validation_loader) > 0:
            validation_loss = _run_validation(
                model=model,
                val_loader=validation_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                use_autocast=use_autocast,
                floating_input_dtype=projector_dtype,
            )
            print(f"Epoch {epoch + 1} validation loss: {validation_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"val/loss": validation_loss, "val/epoch": epoch + 1}, step=max(global_step, 1))
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

        if bool(cfg_get(stage_cfg, "save_every_epoch", False)):
            save_vqa_model_artifacts(
                artifact_dir=run_output_dir / f"epoch_{epoch + 1:03d}",
                stage_cfg=stage_cfg,
                model=model,
                tokenizer=tokenizer,
                global_step=global_step,
                epoch=epoch + 1,
                validation_loss=validation_loss,
            )

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
        wandb_run.log(payload, step=max(global_step, 1))
        wandb_run.finish()

    print(f"Saved final LoRA adapter to: {final_artifacts['lora_adapter_dir']}")
    print(f"Saved final projector checkpoint to: {final_artifacts['projectors_checkpoint']}")
    print(f"Saved run metadata to: {metadata_path}")
    if best_validation_loss is not None and best_epoch is not None:
        print(f"Best validation loss: {best_validation_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
