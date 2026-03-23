from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

MODEL_INPUT_KEYS = {
    "visual_features",
    "visual_attention_mask",
    "prompt_input_ids",
    "prompt_attention_mask",
    "target_input_ids",
    "target_attention_mask",
    "modality_mask",
}


def resolve_device(device: str | None = None) -> torch.device:
    requested = str(device or "auto").strip().lower()
    if requested in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device, precision: str):
    normalized = str(precision).strip().lower()
    if device.type != "cuda":
        return nullcontext()
    if normalized in {"bfloat16", "bf16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if normalized in {"float16", "fp16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _grad_scaler(device: torch.device, precision: str) -> torch.cuda.amp.GradScaler:
    normalized = str(precision).strip().lower()
    enabled = device.type == "cuda" and normalized in {"float16", "fp16"}
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key not in MODEL_INPUT_KEYS:
            continue
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def _estimate_max_sequence_length(model_inputs: dict[str, Any]) -> int | None:
    component_lengths: list[torch.Tensor] = []
    for key in ("visual_attention_mask", "prompt_attention_mask", "target_attention_mask"):
        value = model_inputs.get(key)
        if torch.is_tensor(value) and value.ndim >= 2:
            component_lengths.append(value.sum(dim=-1))
    if not component_lengths:
        return None
    return int(torch.stack(component_lengths, dim=0).sum(dim=0).max().item())


def _scheduler_lambda(current_step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    # Cosine annealing after warmup.
    remaining_steps = max(1, total_steps - warmup_steps)
    progress = float(current_step - warmup_steps) / float(remaining_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate_fn: Any,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def _wandb_is_enabled(wandb_config: dict[str, Any] | None) -> bool:
    if not wandb_config:
        return False
    return bool(wandb_config.get("enabled", False))


def _init_wandb_run(
    *,
    wandb_config: dict[str, Any] | None,
    run_config: dict[str, Any] | None,
) -> Any | None:
    if not _wandb_is_enabled(wandb_config):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging is enabled for projector training, but the 'wandb' package is not installed."
        ) from exc

    config = dict(wandb_config or {})
    tags = config.get("tags")
    if tags is not None and not isinstance(tags, list):
        tags = [str(tags)]

    # return wandb.init(
    #     project=str(config.get("project", "kidney-vlm-projectors")).strip() or "kidney-vlm-projectors",
    #     entity=str(config.get("entity", "")).strip() or None,
    #     name=str(config.get("name", "")).strip() or None,
    #     group=str(config.get("group", "")).strip() or None,
    #     job_type=str(config.get("job_type", "train_projector")).strip() or None,
    #     mode=str(config.get("mode", "online")).strip() or "online",
    #     tags=tags,
    #     config=run_config or {},
    # )
    
    raw_entity = config.get("entity", None)
    raw_name   = config.get("name", None)
    raw_group  = config.get("group", None)
    raw_job    = config.get("job_type", None)

    return wandb.init(
        project=str(config.get("project", "kidney-vlm-projectors")).strip() or "kidney-vlm-projectors",
        entity=str(raw_entity).strip() if raw_entity is not None else None,
        name=str(raw_name).strip() if raw_name is not None else None,
        group=str(raw_group).strip() if raw_group is not None else None,
        job_type=str(raw_job).strip() if raw_job is not None else None,
        mode=str(config.get("mode", "online")).strip() or "online",
        tags=tags,
        config=run_config or {},
    )


@torch.inference_mode()
def evaluate_projector_caption_model(
    model: Any,
    dataloader: DataLoader,
    *,
    device: torch.device,
    precision: str = "float32",
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(dataloader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        model_inputs = _batch_to_device(batch, device)
        with _autocast_context(device, precision):
            outputs = model(**model_inputs)
        losses.append(float(outputs["loss"].detach().float().cpu().item()))

    average_loss = float(sum(losses) / len(losses)) if losses else math.nan
    return {
        "loss": average_loss,
        "num_batches": float(len(losses)),
    }


def train_projector_caption_model(
    model: Any,
    *,
    train_dataset: Any,
    collator: Any,
    output_dir: str | Path,
    batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.0,
    warmup_steps: int = 0,
    num_workers: int = 0,
    seed: int = 42,
    device: str | None = None,
    precision: str = "float32",
    log_every_steps: int = 10,
    max_train_batches_per_epoch: int | None = None,
    eval_dataset: Any = None,
    eval_batch_size: int | None = None,
    max_eval_batches: int | None = None,
    wandb_config: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be >= 1")
    if grad_accumulation_steps <= 0:
        raise ValueError("grad_accumulation_steps must be >= 1")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)
    wandb_run = _init_wandb_run(wandb_config=wandb_config, run_config=run_config)

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        device=resolved_device,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = build_dataloader(
            eval_dataset,
            batch_size=eval_batch_size or batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            device=resolved_device,
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    effective_batches_per_epoch = len(train_loader)
    if max_train_batches_per_epoch is not None:
        effective_batches_per_epoch = min(effective_batches_per_epoch, max_train_batches_per_epoch)
    optimizer_steps_per_epoch = max(1, math.ceil(effective_batches_per_epoch / grad_accumulation_steps))
    total_optimizer_steps = optimizer_steps_per_epoch * num_epochs
    resolved_warmup_steps = int(warmup_steps)
    if resolved_warmup_steps <= 0 and warmup_ratio > 0:
        resolved_warmup_steps = int(total_optimizer_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _scheduler_lambda(
            step,
            warmup_steps=resolved_warmup_steps,
            total_steps=total_optimizer_steps,
        ),
    )
    scaler = _grad_scaler(resolved_device, precision)

    history: list[dict[str, Any]] = []
    best_val_loss = math.inf
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_losses: list[float] = []
            batches_seen = 0
            optimizer_steps = 0

            for batch_index, batch in enumerate(train_loader, start=1):
                if max_train_batches_per_epoch is not None and batch_index > max_train_batches_per_epoch:
                    break

                model_inputs = _batch_to_device(batch, resolved_device)
                try:
                    with _autocast_context(resolved_device, precision):
                        outputs = model(**model_inputs)
                        loss = outputs["loss"]
                        scaled_loss = loss / grad_accumulation_steps
                except torch.OutOfMemoryError as exc:
                    if resolved_device.type == "cuda":
                        torch.cuda.empty_cache()
                    approx_seq_len = _estimate_max_sequence_length(model_inputs)
                    seq_len_message = (
                        f", approx_max_seq_len={approx_seq_len}"
                        if approx_seq_len is not None
                        else ""
                    )
                    raise RuntimeError(
                        "CUDA OOM during projector training "
                        f"(epoch={epoch}, batch={batch_index}, batch_size={batch_size}, "
                        f"grad_accumulation_steps={grad_accumulation_steps}, precision={precision}"
                        f"{seq_len_message}). Lower projector_train.batch_size or "
                        "projector_train.eval_batch_size, increase "
                        "projector_train.gradient_accumulation_steps, or reduce "
                        "projector_train.prompt_max_length/projector_train.target_max_length."
                    ) from exc

                epoch_losses.append(float(loss.detach().float().cpu().item()))
                batches_seen += 1

                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (
                    batch_index % grad_accumulation_steps == 0
                    or batch_index == effective_batches_per_epoch
                )
                if not should_step:
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                optimizer_steps += 1
                current_lr = float(optimizer.param_groups[0]["lr"])
                current_loss = epoch_losses[-1] if epoch_losses else math.nan

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": current_loss,
                            "train/lr": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )

                if log_every_steps > 0 and global_step % log_every_steps == 0:
                    print(
                        f"[train] epoch={epoch} step={global_step} "
                        f"loss={current_loss:.4f} lr={current_lr:.6g}"
                    )

            train_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else math.nan
            metrics: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_batches": batches_seen,
                "optimizer_steps": optimizer_steps,
            }

            if eval_loader is not None:
                eval_metrics = evaluate_projector_caption_model(
                    model,
                    eval_loader,
                    device=resolved_device,
                    precision=precision,
                    max_batches=max_eval_batches,
                )
                metrics["val_loss"] = float(eval_metrics["loss"])
                metrics["val_batches"] = int(eval_metrics["num_batches"])
                if metrics["val_loss"] < best_val_loss:
                    best_val_loss = metrics["val_loss"]
                    best_dir = output_path / "best"
                    checkpoint_path = model.save_projector(
                        best_dir,
                        metadata={
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss": train_loss,
                            "val_loss": metrics["val_loss"],
                        },
                    )
                    metrics["best_checkpoint"] = str(checkpoint_path)

            last_dir = output_path / "last"
            last_checkpoint = model.save_projector(
                last_dir,
                metadata={
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": metrics.get("val_loss"),
                },
            )
            metrics["last_checkpoint"] = str(last_checkpoint)
            history.append(metrics)

            if wandb_run is not None:
                wandb_metrics = {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "train/batches": batches_seen,
                    "train/optimizer_steps": optimizer_steps,
                }
                if "val_loss" in metrics:
                    wandb_metrics["val/loss"] = metrics["val_loss"]
                    wandb_metrics["val/batches"] = metrics["val_batches"]
                wandb_run.log(wandb_metrics, step=global_step)

            print(
                f"[epoch {epoch}] train_loss={train_loss:.4f}"
                + (
                    f" val_loss={metrics['val_loss']:.4f}"
                    if "val_loss" in metrics and not math.isnan(metrics["val_loss"])
                    else ""
                )
            )

        summary = {
            "device": str(resolved_device),
            "seed": seed,
            "num_epochs": num_epochs,
            "global_step": global_step,
            "history": history,
            "best_val_loss": best_val_loss if best_val_loss < math.inf else None,
            "trainable_parameters": int(sum(parameter.numel() for parameter in trainable_parameters)),
        }
        summary_path = output_path / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = summary["best_val_loss"]
            wandb_run.summary["global_step"] = summary["global_step"]
            wandb_run.summary["trainable_parameters"] = summary["trainable_parameters"]

        return summary
    finally:
        if wandb_run is not None:
            wandb_run.finish()
