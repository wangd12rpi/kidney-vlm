#!/usr/bin/env python3
from __future__ import annotations

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
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.modeling.qwen_projector import PathologyQwenProjectorLM
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.training.collator import ProjectorQACollator

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


class ProjectorQADataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.records = frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


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
        raise RuntimeError("transformers is required for projector training.") from exc

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
    wandb_cfg = cfg.projector_train.get("wandb")
    if wandb_cfg is None or not bool(wandb_cfg.get("enabled", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is enabled but wandb is not installed. Install it with: uv add wandb") from exc

    tags = [str(tag).strip() for tag in wandb_cfg.get("tags", []) if str(tag).strip()]
    run = wandb.init(
        project=str(wandb_cfg.get("project", "kidney-vlm")),
        # entity=str(wandb_cfg.get("entity", "")).strip() or None,
        # name=str(wandb_cfg.get("run_name", "")).strip() or None,
        # mode=str(wandb_cfg.get("mode", "online")),
        tags=tags,
        config=OmegaConf.to_container(cfg.projector_train, resolve=True),
    )
    return run


def _split_train_validation(frame: pd.DataFrame, seed: int, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty or len(frame) < 2 or validation_fraction <= 0:
        return frame.reset_index(drop=True), frame.iloc[0:0].copy().reset_index(drop=True)

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation_frame = shuffled.iloc[:validation_count].reset_index(drop=True)
    train_frame = shuffled.iloc[validation_count:].reset_index(drop=True)
    return train_frame, validation_frame


def _run_validation(
    *,
    model: PathologyQwenProjectorLM,
    val_loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    use_autocast: bool,
) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        loop = tqdm(val_loader, total=len(val_loader), desc='Validation', leave=False)
        for step, batch in enumerate(loop, start=1):
            batch = _move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    pathology_features=batch['pathology_features'],
                    pathology_feature_mask=batch['pathology_feature_mask'],
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError('Model did not return a loss during validation.')
            running_loss += float(loss.detach().cpu())
            loop.set_postfix(loss=f'{running_loss / step:.4f}')
    model.train()
    return running_loss / max(1, len(val_loader))


def _save_artifacts(
    *,
    output_dir: Path,
    cfg: Any,
    model: PathologyQwenProjectorLM,
    tokenizer: Any,
    global_step: int,
    epoch: int | None = None,
    validation_loss: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_name = str(cfg.projector_train.get("save_name", "pathology_projector.pt")).strip() or "pathology_projector.pt"
    state_path = output_dir / save_name
    torch.save(
        {
            "projector_state_dict": model.projectors.state_dict(),
            "model_name_or_path": str(cfg.projector_train.model_name_or_path),
            "pathology_embedding_dim": int(cfg.projector_train.pathology_embedding_dim),
            "hidden_size": int(model.hidden_size),
            "max_patch_tokens": int(cfg.projector_train.max_patch_tokens),
            "global_step": int(global_step),
        },
        state_path,
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    OmegaConf.save(config=cfg.projector_train, f=str(output_dir / "projector_train_config.yaml"))
    metadata = {
        "global_step": int(global_step),
        "trainable_parameters": int(model.trainable_parameter_count()),
        "total_parameters": int(model.total_parameter_count()),
        "model_name_or_path": str(cfg.projector_train.model_name_or_path),
    }
    if epoch is not None:
        metadata["best_epoch"] = int(epoch)
    if validation_loss is not None and math.isfinite(validation_loss):
        metadata["best_validation_loss"] = float(validation_loss)
    (output_dir / "projector_metadata.json").write_text(json.dumps(metadata, indent=2))


def _is_improved_validation_loss(validation_loss: float | None, best_validation_loss: float | None) -> bool:
    if validation_loss is None or not math.isfinite(validation_loss):
        return False
    if best_validation_loss is None:
        return True
    return validation_loss < best_validation_loss


def main() -> None:
    cfg = load_cfg()
    stage_cfg = cfg.projector_train

    qa_parquet_path = _resolve_path(stage_cfg.qa_parquet_path)
    if not qa_parquet_path.exists():
        raise FileNotFoundError(f"Projector QA parquet not found: {qa_parquet_path}")

    if not bool(stage_cfg.instantiate_model):
        print("projector_train.instantiate_model=false; nothing to do.")
        return

    seed = int(stage_cfg.seed)
    _set_seed(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    frame = pd.read_parquet(qa_parquet_path)
    if frame.empty:
        raise RuntimeError(f"Projector QA parquet is empty: {qa_parquet_path}")

    if "split" in frame.columns:
        train_pool = frame[frame["split"].fillna("train").astype(str).str.lower() == "train"].copy()
    else:
        train_pool = frame.copy()
    if train_pool.empty:
        raise RuntimeError("No training rows found in projector QA parquet.")

    max_train_samples = stage_cfg.get("max_train_samples")
    if max_train_samples not in (None, "", "null"):
        train_pool = train_pool.head(int(max_train_samples)).reset_index(drop=True)

    validation_fraction = float(stage_cfg.get("validation_fraction", 0.05))
    train_frame, validation_frame = _split_train_validation(train_pool, seed=seed, validation_fraction=validation_fraction)

    train_dataset = ProjectorQADataset(train_frame)
    validation_dataset = ProjectorQADataset(validation_frame)

    device = _resolve_device(stage_cfg.device)
    tokenizer = _build_tokenizer(
        model_name_or_path=str(stage_cfg.model_name_or_path),
        trust_remote_code=bool(stage_cfg.trust_remote_code),
    )
    collator = ProjectorQACollator(
        tokenizer=tokenizer,
        root_dir=ROOT,
        max_text_length=int(stage_cfg.max_text_length),
        max_patch_tokens=int(stage_cfg.max_patch_tokens),
    )

    model = PathologyQwenProjectorLM.from_pretrained(
        str(stage_cfg.model_name_or_path),
        pathology_in_dim=int(stage_cfg.pathology_embedding_dim),
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
            collate_fn=collator,
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found for projector stage.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(stage_cfg.learning_rate),
        weight_decay=float(stage_cfg.weight_decay),
    )

    output_dir = _resolve_path(stage_cfg.output_dir)
    num_epochs = int(stage_cfg.num_epochs)
    grad_accum = max(1, int(stage_cfg.gradient_accumulation_steps))
    grad_clip_norm = float(stage_cfg.grad_clip_norm)
    autocast_dtype_name = str(stage_cfg.get("autocast_dtype", "bfloat16")).strip().lower()
    autocast_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }.get(autocast_dtype_name, torch.bfloat16)
    use_autocast = device.type == "cuda"

    print("Stage 1 projector training")
    print(f"QA parquet: {qa_parquet_path}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Model: {stage_cfg.model_name_or_path}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")
    print(f"Total parameters: {model.total_parameter_count():,}")
    print(f"Max patch tokens: {int(stage_cfg.max_patch_tokens)}")
    print(f"Max text length: {int(stage_cfg.max_text_length)}")
    print("Checkpointing: save best validation-loss checkpoint only")

    if bool(stage_cfg.get("save_every_epoch", False)):
        print("projector_train.save_every_epoch is ignored; best-only checkpointing is enforced.")
    if validation_loader is None or len(validation_loader) == 0:
        print("Validation is unavailable; no checkpoint will be saved because checkpointing now depends on validation loss.")

    wandb_run = _maybe_init_wandb(cfg)
    global_step = 0
    best_validation_loss = None
    best_epoch = None
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
                    pathology_features=batch["pathology_features"],
                    pathology_feature_mask=batch["pathology_feature_mask"],
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model did not return a loss during projector training.")
                scaled_loss = loss / grad_accum

            scaled_loss.backward()
            if step % grad_accum == 0 or step == len(train_loader):
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": float(loss.detach().cpu()),
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
                _save_artifacts(
                    output_dir=output_dir,
                    cfg=cfg,
                    model=model,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    epoch=best_epoch,
                    validation_loss=best_validation_loss,
                )
                print(
                    f"Saved new best checkpoint to: {output_dir} "
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
    if wandb_run is not None:
        log_payload = {
            "artifacts/output_dir": str(output_dir),
            "artifacts/global_step": global_step,
        }
        if best_validation_loss is not None:
            log_payload["val/best_loss"] = best_validation_loss
        if best_epoch is not None:
            log_payload["val/best_epoch"] = best_epoch
        wandb_run.log(log_payload, step=max(global_step, 1))
        wandb_run.finish()
    if best_validation_loss is not None and best_epoch is not None:
        print(f"Saved best projector artifacts to: {output_dir}")
        print(f"Best validation loss: {best_validation_loss:.4f} at epoch {best_epoch}")
    else:
        print("No checkpoint saved because validation loss was unavailable.")


if __name__ == "__main__":
    main()
