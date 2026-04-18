#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.modeling.radiology_qwen_projector import RadiologyQwenProjectorLM
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.training.collator import RadiologyProjectorQACollator

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)
EST = timezone(timedelta(hours=-5), name="EST")


class RadiologyProjectorQADataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.records = frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def load_cfg():
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:
        raise RuntimeError("hydra is required for radiology projector training.") from exc

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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_tokenizer(model_name_or_path: str, trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for radiology projector training.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def _maybe_init_wandb(cfg: Any):
    wandb_cfg = cfg.rad_proj_train.get("wandb")
    if wandb_cfg is None or not bool(wandb_cfg.get("enabled", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is enabled but wandb is not installed. Install it with: uv add wandb") from exc

    tags = [str(tag).strip() for tag in wandb_cfg.get("tags", []) if str(tag).strip()]
    run = wandb.init(
        project=str(wandb_cfg.get("project", "kidney-vlm")),
        tags=tags,
        config=OmegaConf.to_container(cfg.rad_proj_train, resolve=True),
    )
    return run


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
    scheduler_type = str(stage_cfg.get("lr_scheduler_type", "cosine")).strip().lower() or "cosine"
    warmup_ratio = float(stage_cfg.get("warmup_ratio", 0.0) or 0.0)
    warmup_steps = _resolve_warmup_steps(
        total_optimizer_steps=total_optimizer_steps,
        warmup_steps_cfg=stage_cfg.get("warmup_steps"),
        warmup_ratio=warmup_ratio,
    )
    if total_optimizer_steps <= 0:
        return None, scheduler_type, warmup_steps

    try:
        from transformers import get_scheduler
    except ImportError as exc:
        raise RuntimeError("transformers is required for radiology projector scheduler setup.") from exc

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


def _split_train_validation(frame: pd.DataFrame, seed: int, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty or len(frame) < 2 or validation_fraction <= 0:
        return frame.reset_index(drop=True), frame.iloc[0:0].copy().reset_index(drop=True)

    if "series_stem" not in frame.columns:
        raise RuntimeError("Radiology projector training parquet must include 'series_stem' for validation holdout.")

    series_stems = frame["series_stem"].fillna("").astype(str).str.strip()
    if (series_stems == "").any():
        raise RuntimeError("Radiology projector training parquet contains empty 'series_stem' values.")

    unique_series = series_stems.drop_duplicates().tolist()
    if len(unique_series) < 2:
        return frame.reset_index(drop=True), frame.iloc[0:0].copy().reset_index(drop=True)

    rng = random.Random(seed)
    rng.shuffle(unique_series)
    validation_count = max(1, int(round(len(unique_series) * validation_fraction)))
    validation_count = min(validation_count, len(unique_series) - 1)
    validation_series = set(unique_series[:validation_count])

    validation_mask = series_stems.isin(validation_series)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    train_frame = frame.loc[~validation_mask].reset_index(drop=True)
    return train_frame, validation_frame


def _partition_by_split_or_holdout(
    frame: pd.DataFrame,
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if "split" not in frame.columns:
        train_frame, validation_frame = _split_train_validation(frame, seed=seed, validation_fraction=validation_fraction)
        return train_frame, validation_frame, frame.iloc[0:0].copy().reset_index(drop=True), "holdout"

    split_series = frame["split"].fillna("train").astype(str).str.strip().str.lower()
    train_mask = split_series.isin({"", "train", "training"})
    validation_mask = split_series.isin({"validation", "valid", "val", "dev"})
    test_mask = split_series.eq("test")

    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    test_frame = frame.loc[test_mask].reset_index(drop=True)

    if train_frame.empty:
        raise RuntimeError("No training rows found in radiology projector training parquet.")
    if not validation_frame.empty:
        return train_frame, validation_frame, test_frame, "explicit"

    train_frame, validation_frame = _split_train_validation(train_frame, seed=seed, validation_fraction=validation_fraction)
    return train_frame, validation_frame, test_frame, "holdout"


def _top_validation_series_stems(frame: pd.DataFrame, limit: int = 5) -> list[str]:
    if frame.empty:
        return []
    return (
        frame["series_stem"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .head(limit)
        .tolist()
    )


def _run_validation(
    *,
    model: RadiologyQwenProjectorLM,
    val_loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    use_autocast: bool,
) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        loop = tqdm(val_loader, total=len(val_loader), desc="Validation", leave=False)
        for step, batch in enumerate(loop, start=1):
            batch = _move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    radiology_features=batch["radiology_features"],
                    radiology_feature_mask=batch["radiology_feature_mask"],
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model did not return a loss during radiology projector validation.")
            running_loss += float(loss.detach().cpu())
            loop.set_postfix(loss=f"{running_loss / step:.4f}")
    model.train()
    return running_loss / max(1, len(val_loader))


def _build_run_output_dir(
    *,
    output_root: Path,
    modality_tag: str,
    projector_type: str,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(EST).strftime("%Y%m%d_%H%M%S_EST")
    base_name = f"{modality_tag}_{projector_type}_{timestamp}"
    run_output_dir = output_root / base_name
    suffix = 1
    while run_output_dir.exists():
        run_output_dir = output_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir


def _resolve_modality_tag(stage_cfg: Any) -> str:
    raw_value = str(stage_cfg.get("modality_tag", "rad")).strip().lower() or "rad"
    return "".join(character for character in raw_value if character.isalnum() or character in {"-", "_"}).strip("_-") or "rad"


def _save_artifacts(
    *,
    run_output_dir: Path,
    checkpoint_name: str,
    cfg: Any,
    model: RadiologyQwenProjectorLM,
    tokenizer: Any,
    global_step: int,
    epoch: int | None = None,
    validation_loss: float | None = None,
) -> Path:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_output_dir / checkpoint_name
    projector_type = str(cfg.rad_proj_train.get("projector_type", "mlp")).strip() or "mlp"
    torch.save(
        {
            "rad_projector_state_dict": model.rad_projectors.state_dict(),
            "model_name_or_path": str(cfg.rad_proj_train.model_name_or_path),
            "radiology_embedding_dim": int(cfg.rad_proj_train.radiology_embedding_dim),
            "projector_type": projector_type,
            "projector_num_latents": int(cfg.rad_proj_train.get("projector_num_latents", 64)),
            "projector_depth": int(cfg.rad_proj_train.get("projector_depth", 2)),
            "projector_num_heads": int(cfg.rad_proj_train.get("projector_num_heads", 8)),
            "projector_mlp_ratio": float(cfg.rad_proj_train.get("projector_mlp_ratio", 4.0)),
            "projector_dropout": float(cfg.rad_proj_train.get("projector_dropout", 0.0)),
            "hidden_size": int(model.hidden_size),
            "max_slice_tokens": int(cfg.rad_proj_train.max_slice_tokens),
            "global_step": int(global_step),
            "epoch": int(epoch) if epoch is not None else None,
            "validation_loss": float(validation_loss) if validation_loss is not None and math.isfinite(validation_loss) else None,
        },
        state_path,
    )
    tokenizer_dir = run_output_dir / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer.save_pretrained(tokenizer_dir)
    config_path = run_output_dir / "config.yaml"
    if not config_path.exists():
        OmegaConf.save(config=cfg.rad_proj_train, f=str(config_path))
    return state_path


def _write_run_metadata(
    *,
    run_output_dir: Path,
    cfg: Any,
    model: RadiologyQwenProjectorLM,
    global_step: int,
    epoch_checkpoint_paths: list[str],
    best_checkpoint_path: Path | None,
    best_epoch: int | None,
    best_validation_loss: float | None,
) -> Path:
    metadata = {
        "global_step": int(global_step),
        "trainable_parameters": int(model.trainable_parameter_count()),
        "total_parameters": int(model.total_parameter_count()),
        "model_name_or_path": str(cfg.rad_proj_train.model_name_or_path),
        "run_output_dir": str(run_output_dir),
        "config_path": str(run_output_dir / "config.yaml"),
        "tokenizer_dir": str(run_output_dir / "tokenizer"),
        "epoch_checkpoint_paths": list(epoch_checkpoint_paths),
    }
    if best_checkpoint_path is not None:
        metadata["best_checkpoint_path"] = str(best_checkpoint_path)
    if best_epoch is not None:
        metadata["best_epoch"] = int(best_epoch)
    if best_validation_loss is not None and math.isfinite(best_validation_loss):
        metadata["best_validation_loss"] = float(best_validation_loss)
    metadata_path = run_output_dir / "rad_projector_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def _is_improved_validation_loss(validation_loss: float | None, best_validation_loss: float | None) -> bool:
    if validation_loss is None or not math.isfinite(validation_loss):
        return False
    if best_validation_loss is None:
        return True
    return validation_loss < best_validation_loss


def main() -> None:
    cfg = load_cfg()
    stage_cfg = cfg.rad_proj_train

    qa_parquet_path = _resolve_path(stage_cfg.qa_parquet_path)
    if not qa_parquet_path.exists():
        raise FileNotFoundError(f"Radiology projector training parquet not found: {qa_parquet_path}")

    if not bool(stage_cfg.instantiate_model):
        print("rad_proj_train.instantiate_model=false; nothing to do.")
        return

    seed = int(stage_cfg.seed)
    _set_seed(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    frame = pd.read_parquet(qa_parquet_path)
    if frame.empty:
        raise RuntimeError(f"Radiology projector training parquet is empty: {qa_parquet_path}")

    max_train_samples = stage_cfg.get("max_train_samples")
    if max_train_samples not in (None, "", "null"):
        frame = frame.head(int(max_train_samples)).reset_index(drop=True)

    validation_fraction = float(stage_cfg.get("validation_fraction", 0.05))
    train_frame, validation_frame, test_frame, validation_mode = _partition_by_split_or_holdout(
        frame,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    validation_holdout_stems = _top_validation_series_stems(validation_frame, limit=5)

    train_dataset = RadiologyProjectorQADataset(train_frame)
    validation_dataset = RadiologyProjectorQADataset(validation_frame)

    device = _resolve_device(stage_cfg.device)
    tokenizer = _build_tokenizer(
        model_name_or_path=str(stage_cfg.model_name_or_path),
        trust_remote_code=bool(stage_cfg.trust_remote_code),
    )
    collator = RadiologyProjectorQACollator(
        tokenizer=tokenizer,
        root_dir=ROOT,
        max_text_length=int(stage_cfg.max_text_length),
        max_slice_tokens=int(stage_cfg.max_slice_tokens),
        slice_token_dropout_prob=float(stage_cfg.get("slice_token_dropout_prob", 0.0)),
    )
    validation_collator = RadiologyProjectorQACollator(
        tokenizer=tokenizer,
        root_dir=ROOT,
        max_text_length=int(stage_cfg.max_text_length),
        max_slice_tokens=int(stage_cfg.max_slice_tokens),
        slice_token_dropout_prob=0.0,
    )

    projector_type = str(stage_cfg.get("projector_type", "mlp")).strip() or "mlp"
    model = RadiologyQwenProjectorLM.from_pretrained(
        str(stage_cfg.model_name_or_path),
        radiology_in_dim=int(stage_cfg.radiology_embedding_dim),
        projector_type=projector_type,
        projector_num_latents=int(stage_cfg.get("projector_num_latents", 64)),
        projector_depth=int(stage_cfg.get("projector_depth", 2)),
        projector_num_heads=int(stage_cfg.get("projector_num_heads", 8)),
        projector_mlp_ratio=float(stage_cfg.get("projector_mlp_ratio", 4.0)),
        projector_dropout=float(stage_cfg.get("projector_dropout", 0.0)),
        trust_remote_code=bool(stage_cfg.trust_remote_code),
        torch_dtype=stage_cfg.get("torch_dtype"),
        attn_implementation=stage_cfg.get("attn_implementation"),
    )
    if bool(stage_cfg.get("gradient_checkpointing", False)) and hasattr(model.language_model, "gradient_checkpointing_enable"):
        model.language_model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(stage_cfg.batch_size),
        shuffle=True,
        num_workers=int(stage_cfg.dataloader_num_workers),
        collate_fn=collator,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty after batching.")

    validation_loader = None
    if len(validation_dataset) > 0:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(stage_cfg.batch_size),
            shuffle=False,
            num_workers=int(stage_cfg.dataloader_num_workers),
            collate_fn=validation_collator,
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found for radiology projector stage.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(stage_cfg.learning_rate),
        weight_decay=float(stage_cfg.weight_decay),
    )

    output_root = _resolve_path(stage_cfg.output_dir)
    modality_tag = _resolve_modality_tag(stage_cfg)
    run_output_dir = _build_run_output_dir(
        output_root=output_root,
        modality_tag=modality_tag,
        projector_type=projector_type,
    )
    num_epochs = int(stage_cfg.num_epochs)
    grad_accum = max(1, int(stage_cfg.gradient_accumulation_steps))
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
    grad_clip_norm = float(stage_cfg.grad_clip_norm)
    autocast_dtype_name = str(stage_cfg.get("autocast_dtype", "bfloat16")).strip().lower()
    autocast_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }.get(autocast_dtype_name, torch.bfloat16)
    use_autocast = device.type == "cuda"

    print("Stage 1 radiology projector training")
    print(f"Radiology projector parquet: {qa_parquet_path}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples: {len(test_frame)}")
    print(f"Validation source: {validation_mode}")
    print(f"Model: {stage_cfg.model_name_or_path}")
    print(f"Projector type: {projector_type}")
    print(f"Device: {device}")
    print(f"Run output dir: {run_output_dir}")
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")
    print(f"Total parameters: {model.total_parameter_count():,}")
    print(f"Max slice tokens: {int(stage_cfg.max_slice_tokens)}")
    print(f"Train slice token dropout: {float(stage_cfg.get('slice_token_dropout_prob', 0.0)):.2f}")
    print(f"Max text length: {int(stage_cfg.max_text_length)}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Total optimizer steps: {total_optimizer_steps}")
    print("Checkpointing: save every epoch checkpoint plus best.ckpt when validation improves")
    if validation_loader is None or len(validation_loader) == 0:
        print("Validation is unavailable; epoch checkpoints will still be saved, but best.ckpt will not be created.")

    wandb_run = _maybe_init_wandb(cfg)
    global_step = 0
    best_validation_loss = None
    best_epoch = None
    best_checkpoint_path = None
    epoch_checkpoint_paths: list[str] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(loop, start=1):
            batch = _move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    radiology_features=batch["radiology_features"],
                    radiology_feature_mask=batch["radiology_feature_mask"],
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model did not return a loss during radiology projector training.")
                scaled_loss = loss / grad_accum

            scaled_loss.backward()
            if step % grad_accum == 0 or step == len(train_loader):
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
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
                            "train/batch_step": step,
                        },
                        step=global_step,
                    )

            running_loss += float(loss.detach().cpu())
            loop.set_postfix(loss=f"{running_loss / step:.4f}")

        epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1} mean loss: {epoch_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_mean_loss": epoch_loss,
                    "train/epoch": epoch + 1,
                },
                step=max(global_step, 1),
            )

        validation_loss = None
        if validation_loader is not None and len(validation_loader) > 0:
            validation_loss = _run_validation(
                model=model,
                val_loader=validation_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                use_autocast=use_autocast,
            )
            print(f"Epoch {epoch + 1} validation loss: {validation_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/loss": validation_loss,
                        "val/epoch": epoch + 1,
                    },
                    step=max(global_step, 1),
                )
            if _is_improved_validation_loss(validation_loss, best_validation_loss):
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                best_checkpoint_path = _save_artifacts(
                    run_output_dir=run_output_dir,
                    checkpoint_name="best.ckpt",
                    cfg=cfg,
                    model=model,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    epoch=best_epoch,
                    validation_loss=best_validation_loss,
                )
                _write_run_metadata(
                    run_output_dir=run_output_dir,
                    cfg=cfg,
                    model=model,
                    global_step=global_step,
                    epoch_checkpoint_paths=epoch_checkpoint_paths,
                    best_checkpoint_path=best_checkpoint_path,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                )
                print(
                    f"Saved new best checkpoint to: {best_checkpoint_path} "
                    f"(epoch {best_epoch}, val_loss={best_validation_loss:.4f})"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "val/best_loss": best_validation_loss,
                            "val/best_epoch": best_epoch,
                        },
                        step=max(global_step, 1),
                    )

        epoch_checkpoint_path = _save_artifacts(
            run_output_dir=run_output_dir,
            checkpoint_name=f"epoch_{epoch + 1:03d}.ckpt",
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            global_step=global_step,
            epoch=epoch + 1,
            validation_loss=validation_loss,
        )
        epoch_checkpoint_paths.append(str(epoch_checkpoint_path))
        _write_run_metadata(
            run_output_dir=run_output_dir,
            cfg=cfg,
            model=model,
            global_step=global_step,
            epoch_checkpoint_paths=epoch_checkpoint_paths,
            best_checkpoint_path=best_checkpoint_path,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
        )
        print(f"Saved epoch checkpoint to: {epoch_checkpoint_path}")

    if wandb_run is not None:
        log_payload = {
            "artifacts/output_dir": str(run_output_dir),
            "artifacts/global_step": global_step,
        }
        if best_validation_loss is not None:
            log_payload["val/best_loss"] = best_validation_loss
        if best_epoch is not None:
            log_payload["val/best_epoch"] = best_epoch
        wandb_run.log(log_payload, step=max(global_step, 1))
        wandb_run.finish()
    _write_run_metadata(
        run_output_dir=run_output_dir,
        cfg=cfg,
        model=model,
        global_step=global_step,
        epoch_checkpoint_paths=epoch_checkpoint_paths,
        best_checkpoint_path=best_checkpoint_path,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
    print(f"Saved rad projector run artifacts to: {run_output_dir}")
    print(f"Epoch checkpoints saved: {len(epoch_checkpoint_paths)}")
    if best_validation_loss is not None and best_epoch is not None and best_checkpoint_path is not None:
        print(f"Best checkpoint: {best_checkpoint_path}")
        print(f"Best validation loss: {best_validation_loss:.4f} at epoch {best_epoch}")
    else:
        print("Best checkpoint: not created because validation loss was unavailable.")
    if validation_holdout_stems:
        print("Validation series stems (first 5):")
        for series_stem in validation_holdout_stems:
            print(f"  {series_stem}")
    else:
        print("Validation series stems: none")


if __name__ == "__main__":
    main()
