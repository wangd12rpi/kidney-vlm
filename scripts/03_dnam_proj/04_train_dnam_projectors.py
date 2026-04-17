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
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.modeling.dnam_qwen_projector import DnamQwenProjectorLM
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.training.collator import DNAMProjectorQACollator

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)
EST = timezone(timedelta(hours=-5), name="EST")


class DNAMProjectorQADataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.records = frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="03_dnam_proj/04_train_dnam_projectors.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _slugify_label(raw_value: Any, *, default: str) -> str:
    text = str(raw_value or "").strip().lower()
    if not text:
        return default
    pieces: list[str] = []
    last_was_sep = False
    for character in text:
        if character.isalnum():
            pieces.append(character)
            last_was_sep = False
        elif not last_was_sep:
            pieces.append("_")
            last_was_sep = True
    normalized = "".join(pieces).strip("_")
    return normalized or default


def _resolve_llm_tag(model_name_or_path: str) -> str:
    segments = [segment for segment in str(model_name_or_path).strip().split("/") if segment]
    candidate = segments[-1] if segments else model_name_or_path
    return _slugify_label(candidate, default="llm")


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
        raise RuntimeError("transformers is required for DNAm projector training.") from exc

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
        raise RuntimeError("transformers is required for DNAm projector scheduler setup.") from exc

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


def _run_validation(
    *,
    model: DnamQwenProjectorLM,
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
                    dnam_features=batch["dnam_features"],
                    dnam_feature_mask=batch["dnam_feature_mask"],
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model did not return a loss during validation.")
            running_loss += float(loss.detach().cpu())
            loop.set_postfix(loss=f"{running_loss / step:.4f}")
    model.train()
    return running_loss / max(1, len(val_loader))


def _build_run_output_dir(
    *,
    output_root: Path,
    llm_tag: str,
    modality_dir_name: str,
    modality_tag: str,
    projector_type: str,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    modality_root = output_root / llm_tag / modality_dir_name
    modality_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(EST).strftime("%Y%m%d_%H%M%S_EST")
    base_name = f"{modality_tag}_{projector_type}_{timestamp}"
    run_output_dir = modality_root / base_name
    suffix = 1
    while run_output_dir.exists():
        run_output_dir = modality_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir


def _resolve_modality_tag(stage_cfg: Any) -> str:
    raw_value = str(stage_cfg.get("modality_tag", "dnam")).strip().lower() or "dnam"
    return "".join(character for character in raw_value if character.isalnum() or character in {"-", "_"}).strip("_-") or "dnam"


def _resolve_modality_dir_name(stage_cfg: Any) -> str:
    raw_value = str(stage_cfg.get("modality_dir_name", "")).strip().lower()
    if raw_value:
        return _slugify_label(raw_value, default="dnam")
    modality_tag = _resolve_modality_tag(stage_cfg)
    return _slugify_label(modality_tag, default="dnam")


def _save_artifacts(
    *,
    run_output_dir: Path,
    checkpoint_name: str,
    cfg: Any,
    model: DnamQwenProjectorLM,
    tokenizer: Any,
    global_step: int,
    epoch: int | None = None,
    validation_loss: float | None = None,
) -> Path:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_output_dir / checkpoint_name
    projector_type = str(cfg.dnam_proj.get("projector_type", "mlp")).strip() or "mlp"
    torch.save(
        {
            "dnam_projector_state_dict": model.projectors.state_dict(),
            "model_name_or_path": str(cfg.dnam_proj.model_name_or_path),
            "dnam_embedding_dim": int(cfg.dnam_proj.dnam_embedding_dim),
            "projector_type": projector_type,
            "projector_num_latents": int(cfg.dnam_proj.get("projector_num_latents", 64)),
            "projector_depth": int(cfg.dnam_proj.get("projector_depth", 2)),
            "projector_num_heads": int(cfg.dnam_proj.get("projector_num_heads", 8)),
            "projector_mlp_ratio": float(cfg.dnam_proj.get("projector_mlp_ratio", 4.0)),
            "projector_dropout": float(cfg.dnam_proj.get("projector_dropout", 0.0)),
            "hidden_size": int(model.hidden_size),
            "max_dnam_tokens": int(cfg.dnam_proj.max_dnam_tokens),
            "global_step": int(global_step),
            "epoch": int(epoch) if epoch is not None else None,
            "validation_loss": float(validation_loss) if validation_loss is not None and math.isfinite(validation_loss) else None,
        },
        state_path,
    )
    save_tokenizer_snapshot = bool(cfg.dnam_proj.get("save_tokenizer_snapshot", False))
    if save_tokenizer_snapshot:
        tokenizer_dir = run_output_dir / "tokenizer"
        if not tokenizer_dir.exists():
            tokenizer.save_pretrained(tokenizer_dir)
    config_path = run_output_dir / "config.yaml"
    if not config_path.exists():
        OmegaConf.save(config=cfg.dnam_proj, f=str(config_path))
    return state_path


def _write_run_metadata(
    *,
    run_output_dir: Path,
    cfg: Any,
    model: DnamQwenProjectorLM,
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
        "model_name_or_path": str(cfg.dnam_proj.model_name_or_path),
        "run_output_dir": str(run_output_dir),
        "config_path": str(run_output_dir / "config.yaml"),
        "tokenizer_model_name_or_path": str(cfg.dnam_proj.model_name_or_path),
        "epoch_checkpoint_paths": list(epoch_checkpoint_paths),
    }
    if bool(cfg.dnam_proj.get("save_tokenizer_snapshot", False)):
        metadata["tokenizer_dir"] = str(run_output_dir / "tokenizer")
    if best_checkpoint_path is not None:
        metadata["best_checkpoint_path"] = str(best_checkpoint_path)
    if best_epoch is not None:
        metadata["best_epoch"] = int(best_epoch)
    if best_validation_loss is not None and math.isfinite(best_validation_loss):
        metadata["best_validation_loss"] = float(best_validation_loss)
    metadata_path = run_output_dir / "dnam_projector_metadata.json"
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
    stage_cfg = cfg.dnam_proj

    qa_parquet_path = _resolve_path(stage_cfg.qa_parquet_path)
    if not qa_parquet_path.exists():
        raise FileNotFoundError(f"DNAm projector training parquet not found: {qa_parquet_path}")

    if not bool(stage_cfg.instantiate_model):
        print("dnam_proj.instantiate_model=false; nothing to do.")
        return

    seed = int(stage_cfg.seed)
    _set_seed(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    frame = pd.read_parquet(qa_parquet_path)
    if frame.empty:
        raise RuntimeError(f"DNAm projector training parquet is empty: {qa_parquet_path}")
    if "split" not in frame.columns:
        raise RuntimeError("DNAm projector training parquet must include the unified registry split column.")

    split_series = frame["split"].fillna("").astype(str).str.strip().str.lower()
    train_frame = frame.loc[split_series == "train"].reset_index(drop=True)
    validation_frame = frame.loc[split_series == "val"].reset_index(drop=True)
    test_frame = frame.loc[split_series == "test"].reset_index(drop=True)

    if train_frame.empty:
        raise RuntimeError("No train rows found in DNAm projector training parquet.")
    if validation_frame.empty:
        raise RuntimeError("No val rows found in DNAm projector training parquet. Rebuild unified/projector data first.")

    max_train_samples = stage_cfg.get("max_train_samples")
    if max_train_samples not in (None, "", "null"):
        train_frame = train_frame.head(int(max_train_samples)).reset_index(drop=True)

    train_dataset = DNAMProjectorQADataset(train_frame)
    validation_dataset = DNAMProjectorQADataset(validation_frame)

    device = _resolve_device(stage_cfg.device)
    tokenizer = _build_tokenizer(
        model_name_or_path=str(stage_cfg.model_name_or_path),
        trust_remote_code=bool(stage_cfg.trust_remote_code),
    )
    collator = DNAMProjectorQACollator(
        tokenizer=tokenizer,
        root_dir=ROOT,
        max_text_length=int(stage_cfg.max_text_length),
        max_dnam_tokens=int(stage_cfg.max_dnam_tokens),
    )
    validation_collator = DNAMProjectorQACollator(
        tokenizer=tokenizer,
        root_dir=ROOT,
        max_text_length=int(stage_cfg.max_text_length),
        max_dnam_tokens=int(stage_cfg.max_dnam_tokens),
    )

    model = DnamQwenProjectorLM.from_pretrained(
        str(stage_cfg.model_name_or_path),
        dnam_in_dim=int(stage_cfg.dnam_embedding_dim),
        projector_type=str(stage_cfg.get("projector_type", "mlp")),
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
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(stage_cfg.batch_size),
        shuffle=False,
        num_workers=int(stage_cfg.dataloader_num_workers),
        collate_fn=validation_collator,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty after batching.")
    if len(validation_loader) == 0:
        raise RuntimeError("Validation loader is empty after batching.")

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found for DNAm projector stage.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(stage_cfg.learning_rate),
        weight_decay=float(stage_cfg.weight_decay),
    )

    output_root = _resolve_path(stage_cfg.output_dir)
    llm_tag = _resolve_llm_tag(stage_cfg.model_name_or_path)
    modality_dir_name = _resolve_modality_dir_name(stage_cfg)
    projector_type = str(stage_cfg.get("projector_type", "mlp")).strip() or "mlp"
    modality_tag = _resolve_modality_tag(stage_cfg)
    run_output_dir = _build_run_output_dir(
        output_root=output_root,
        llm_tag=llm_tag,
        modality_dir_name=modality_dir_name,
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

    print("Stage 1 DNAm projector training")
    print(f"DNAm projector parquet: {qa_parquet_path}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Held-out test samples present but unused during training: {len(test_frame)}")
    print(f"Model: {stage_cfg.model_name_or_path}")
    print(f"Projector type: {projector_type}")
    print(f"Device: {device}")
    print(f"Run output dir: {run_output_dir}")
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")
    print(f"Total parameters: {model.total_parameter_count():,}")
    print(f"Max DNAm tokens: {int(stage_cfg.max_dnam_tokens)}")
    print(f"Max text length: {int(stage_cfg.max_text_length)}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Total optimizer steps: {total_optimizer_steps}")
    print("Split policy: train from unified train rows, validate on unified val rows, never fit on unified test rows.")

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
                    dnam_features=batch["dnam_features"],
                    dnam_feature_mask=batch["dnam_feature_mask"],
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model did not return a loss during training.")
                loss = loss / grad_accum

            loss.backward()
            running_loss += float(loss.detach().cpu()) * grad_accum

            should_step = step % grad_accum == 0 or step == len(train_loader)
            if should_step:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            avg_loss = running_loss / step
            current_lr = optimizer.param_groups[0]["lr"]
            loop.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

        validation_loss = _run_validation(
            model=model,
            val_loader=validation_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            use_autocast=use_autocast,
        )
        print(f"Epoch {epoch + 1}: validation loss = {validation_loss:.4f}")

        epoch_checkpoint_path = _save_artifacts(
            run_output_dir=run_output_dir,
            checkpoint_name=f"epoch_{epoch + 1:02d}.ckpt",
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            global_step=global_step,
            epoch=epoch + 1,
            validation_loss=validation_loss,
        )
        epoch_checkpoint_paths.append(str(epoch_checkpoint_path))

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
                epoch=epoch + 1,
                validation_loss=validation_loss,
            )
            print(f"Saved new best checkpoint: {best_checkpoint_path}")

    metadata_path = _write_run_metadata(
        run_output_dir=run_output_dir,
        cfg=cfg,
        model=model,
        global_step=global_step,
        epoch_checkpoint_paths=epoch_checkpoint_paths,
        best_checkpoint_path=best_checkpoint_path,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )

    print("DNAm projector training complete.")
    print(f"Metadata: {metadata_path}")
    if best_checkpoint_path is not None:
        print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
