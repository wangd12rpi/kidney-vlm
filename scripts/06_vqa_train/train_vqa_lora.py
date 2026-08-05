#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import math
import os
import random
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
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
from transformers import StoppingCriteria, StoppingCriteriaList

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.modeling.path_projectors import resolve_language_model_hidden_size
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.vqa.data import (
    VQAGRPOCollator,
    VQADataset,
    VQATrainingCollator,
    assert_vqa_rows_are_trainable,
    build_vqa_training_target,
    row_missing_prefix_cache_entries,
    prefix_cache_enabled,
    select_vqa_rows,
)
from kidney_vlm.vqa.grpo import (
    append_completions_to_prompts,
    centered_group_rewards,
    completion_span_mask,
    completion_logprobs,
    grpo_advantages,
    grpo_loss,
    repeat_batch_for_generations,
    score_grpo_completion,
)
from kidney_vlm.vqa.modeling import (
    OncoVLMVQASFTModel,
    apply_lora,
    build_language_model,
    build_tokenizer,
    generate_language_model_with_soft_prefix,
    load_projectors,
    move_batch_to_device,
    save_vqa_model_artifacts,
)
from kidney_vlm.vqa.pathology_step1_judge import (
    PathologyJudgeResult,
    PathologyStep1Judge,
)
from kidney_vlm.vqa.prompts import build_vqa_prompt_preview
from kidney_vlm.vqa.schema import VQA_COLUMNS, normalize_vqa_df
from kidney_vlm.vqa.stage_config import (
    as_bool,
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


class StopAfterGeneratedSubsequence(StoppingCriteria):
    def __init__(self, *, stop_ids: list[int], prompt_lengths: torch.Tensor) -> None:
        super().__init__()
        self.stop_ids = stop_ids
        self.prompt_lengths = prompt_lengths.detach().cpu().tolist()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> torch.BoolTensor:
        stop_len = len(self.stop_ids)
        stopped = torch.zeros(
            (int(input_ids.shape[0]),), device=input_ids.device, dtype=torch.bool
        )
        if stop_len == 0:
            return stopped
        stop_ids = torch.tensor(
            self.stop_ids, device=input_ids.device, dtype=input_ids.dtype
        )
        for row_idx in range(int(input_ids.shape[0])):
            start = int(self.prompt_lengths[row_idx])
            generated = input_ids[row_idx, start:]
            if len(generated) < stop_len:
                continue
            windows = generated.unfold(dimension=0, size=stop_len, step=1)
            stopped[row_idx] = windows.eq(stop_ids).all(dim=1).any()
        return stopped


def _extract_method(overrides: list[str]) -> tuple[str, list[str]]:
    method = "sft"
    kept: list[str] = []
    for override in overrides:
        if override.startswith("method="):
            method = override.split("=", 1)[1].strip()
        elif override.startswith("vqa_train.post_train_method="):
            method = override.split("=", 1)[1].strip()
            kept.append(override)
        else:
            kept.append(override)
    return method or "sft", kept


def _extract_profile(overrides: list[str]) -> tuple[str, list[str]]:
    profile = ""
    kept: list[str] = []
    for override in overrides:
        if override.startswith("profile="):
            profile = override.split("=", 1)[1].strip()
        else:
            kept.append(override)
    if profile and not all(
        character.isalnum() or character in {"_", "-"} for character in profile
    ):
        raise ValueError(
            "Training profile names may contain only letters, numbers, underscores, and hyphens."
        )
    return profile, kept


def _wrap_vqa_train_cfg(raw_cfg: Any) -> Any:
    if "defaults" in raw_cfg:
        del raw_cfg["defaults"]
    if "vqa_train" in raw_cfg:
        return raw_cfg
    return OmegaConf.create(
        {"vqa_train": OmegaConf.to_container(raw_cfg, resolve=False)}
    )


def load_cfg_from_overrides(overrides: list[str] | None = None):
    overrides = list(overrides or [])
    method, kept_overrides = _extract_method(overrides)
    profile, kept_overrides = _extract_profile(kept_overrides)
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="06_vqa_train/vqa_common.yaml",
        overrides=[],
    )
    config_stem = f"vqa_{method}_{profile}" if profile else f"vqa_{method}"
    method_path = ROOT / "conf" / "06_vqa_train" / f"{config_stem}.yaml"
    if not method_path.exists():
        raise FileNotFoundError(f"VQA training profile not found: {method_path}")
    method_cfg = _wrap_vqa_train_cfg(OmegaConf.load(method_path))
    cfg = OmegaConf.merge(cfg, method_cfg)
    if kept_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(kept_overrides))
    cfg.vqa_train.post_train_method = method
    cfg.project.root_dir = str(ROOT)
    return cfg


def load_cfg():
    return load_cfg_from_overrides(sys.argv[1:])


def _resolve_device(device_value: str | None) -> torch.device:
    requested = str(device_value or "").strip() or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
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
    missing_env = [
        name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ
    ]
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


def _ddp_reduce_loss_sum(
    loss_sum: float, count: int, *, device: torch.device, ddp_state: dict[str, Any]
) -> tuple[float, int]:
    if not bool(ddp_state.get("initialized")):
        return loss_sum, count
    tensor = torch.tensor(
        [float(loss_sum), float(count)], device=device, dtype=torch.float64
    )
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
    method = slugify_label(
        cfg_get(stage_cfg, "post_train_method", "sft"), default="sft"
    )
    output_root = resolve_repo_path(ROOT, cfg_get(stage_cfg, "output_dir")) / method
    output_root.mkdir(parents=True, exist_ok=True)
    configured_name = clean_text(cfg_get(stage_cfg, "run_name", ""))
    run_name = configured_name or generate_run_name(stage_cfg, train_rows=train_rows)
    run_name = slugify_label(run_name, default=f"vqa_lora_{method}")
    run_output_dir = output_root / run_name
    suffix = 1
    while run_output_dir.exists():
        run_output_dir = output_root / f"{run_name}_{suffix:02d}"
        suffix += 1
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir


def _compute_total_optimizer_steps(
    *, num_batches_per_epoch: int, num_epochs: int, gradient_accumulation_steps: int
) -> int:
    if num_batches_per_epoch <= 0 or num_epochs <= 0:
        return 0
    updates_per_epoch = math.ceil(
        num_batches_per_epoch / max(1, gradient_accumulation_steps)
    )
    return updates_per_epoch * num_epochs


def _build_sft_optimizer(
    *,
    model: OncoVLMVQASFTModel,
    train_model: torch.nn.Module,
    stage_cfg: Any,
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
    trainable_parameters = [
        parameter for parameter in train_model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found for VQA LoRA training.")

    projector_parameter_ids = {
        id(parameter)
        for _, module in model.projector_modules()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    projector_parameters = [
        parameter
        for parameter in trainable_parameters
        if id(parameter) in projector_parameter_ids
    ]
    language_parameters = [
        parameter
        for parameter in trainable_parameters
        if id(parameter) not in projector_parameter_ids
    ]

    learning_rate = float(cfg_get(stage_cfg, "learning_rate", 2e-5))
    weight_decay = float(cfg_get(stage_cfg, "weight_decay", 0.0))
    raw_projector_lr = cfg_get(stage_cfg, "projector_learning_rate", None)
    projector_learning_rate = (
        learning_rate
        if raw_projector_lr in (None, "", "null")
        else float(raw_projector_lr)
    )
    raw_projector_weight_decay = cfg_get(stage_cfg, "projector_weight_decay", None)
    projector_weight_decay = (
        weight_decay
        if raw_projector_weight_decay in (None, "", "null")
        else float(raw_projector_weight_decay)
    )

    parameter_groups: list[dict[str, Any]] = []
    if language_parameters:
        parameter_groups.append(
            {
                "params": language_parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": "language_lora",
            }
        )
    if projector_parameters:
        parameter_groups.append(
            {
                "params": projector_parameters,
                "lr": projector_learning_rate,
                "weight_decay": projector_weight_decay,
                "group_name": "projectors",
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    return optimizer, trainable_parameters


def _resolve_warmup_steps(
    *, total_optimizer_steps: int, warmup_steps_cfg: Any, warmup_ratio: float
) -> int:
    if total_optimizer_steps <= 0:
        return 0
    if warmup_steps_cfg not in (None, "", "null"):
        return max(0, min(int(warmup_steps_cfg), total_optimizer_steps))
    return max(
        0,
        min(
            int(round(total_optimizer_steps * max(0.0, warmup_ratio))),
            total_optimizer_steps,
        ),
    )


def _build_lr_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    stage_cfg: Any,
    total_optimizer_steps: int,
) -> tuple[Any | None, str, int]:
    scheduler_type = (
        str(cfg_get(stage_cfg, "lr_scheduler_type", "cosine")).strip().lower()
        or "cosine"
    )
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
        raise RuntimeError(
            "wandb is enabled but wandb is not installed. Install project dependencies first."
        ) from exc

    tags = [
        str(tag).strip() for tag in cfg_get(wandb_cfg, "tags", []) if str(tag).strip()
    ]
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
    print(
        f"VQA prompt preview: first {min(sample_count, len(train_frame))} train samples"
    )
    for preview_index, (_, row) in enumerate(
        train_frame.head(sample_count).iterrows(), start=1
    ):
        print(
            f"\n--- VQA prompt preview {preview_index} | question_id={row.get('question_id')} ---"
        )
        prompt_cfg = dict(prompt_cfg or {})
        prompt_cfg["use_cot"] = bool(
            cfg_get(cfg_get(stage_cfg, "cot", {}), "enabled", False)
        )
        print(build_vqa_prompt_preview(row, prompt_cfg))
        if str(cfg_get(stage_cfg, "post_train_method", "sft")).lower() == "grpo":
            print(
                "\n<assistant_answer>\n[GRPO sampled completion]\n</assistant_answer>"
            )
        else:
            print(
                f"\n<assistant_answer>\n{build_vqa_training_target(row, stage_cfg)}\n</assistant_answer>"
            )
    return bool(cfg_get(preview_cfg, "exit_after_preview", False))


def _cached_projector_metadata(stage_cfg: Any) -> dict[str, dict[str, Any]]:
    projectors_cfg = cfg_get(stage_cfg, "projectors", {})
    metadata: dict[str, dict[str, Any]] = {}
    for modality in ("pathology", "radiology", "dnam", "rna"):
        block_cfg = cfg_get(projectors_cfg, modality, {})
        if not bool(cfg_get(block_cfg, "enabled", False)):
            continue
        if bool(cfg_get(block_cfg, "trainable", False)):
            raise ValueError(
                f"vqa_train.prefix_cache.enabled=true requires frozen projectors; {modality}.trainable=true."
            )
        metadata[modality] = {
            "modality": modality,
            "checkpoint_path": str(
                resolve_repo_path(ROOT, cfg_get(block_cfg, "checkpoint_path"))
            ),
            "trainable": False,
            "prefix_cache_enabled": True,
            "prefix_cache_root": str(
                resolve_repo_path(
                    ROOT, cfg_get(cfg_get(stage_cfg, "prefix_cache", {}), "cache_root")
                )
            ),
        }
    if not metadata:
        raise RuntimeError(
            "Prefix-cache training needs at least one enabled projector block."
        )
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
            print(
                f"Prefix cache: scan_before_training=false; not pre-scanning {split_label} rows."
            )
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

    print(
        f"Prefix cache: skipped {len(skipped):,} {split_label} rows with missing cached prefixes."
    )
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
            batch = move_batch_to_device(
                batch, device, floating_dtype=floating_input_dtype
            )
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
            ):
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
        metadata["final_artifacts"] = {
            key: _portable_path(value) for key, value in final_artifacts.items()
        }
    if best_artifacts is not None:
        metadata["best_artifacts"] = {
            key: _portable_path(value) for key, value in best_artifacts.items()
        }
    if best_validation_loss is not None and math.isfinite(best_validation_loss):
        metadata["best_validation_loss"] = float(best_validation_loss)
    if best_epoch is not None:
        metadata["best_epoch"] = int(best_epoch)
    metadata_path = run_output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _is_improved_validation_loss(
    validation_loss: float | None, best_validation_loss: float | None
) -> bool:
    if validation_loss is None or not math.isfinite(validation_loss):
        return False
    if best_validation_loss is None:
        return True
    return validation_loss < best_validation_loss


def _new_grpo_metric_accumulator() -> dict[str, Any]:
    return {"count": 0, "sums": {}, "mins": {}, "maxs": {}}


def _float_mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _score_float_values(score_rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in score_rows]


def _disable_dropout_for_grpo(model: torch.nn.Module) -> int:
    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0:
            module.p = 0.0
            changed += 1
    return changed


def _pathology_judge_weights(
    grpo_cfg: Any,
) -> tuple[Any, float, float, float, float]:
    judge_cfg = cfg_get(grpo_cfg, "visual_judge", {}) or {}
    observation_weight = float(cfg_get(judge_cfg, "observation_weight", 0.0) or 0.0)
    reasoning_weight = float(cfg_get(judge_cfg, "reasoning_weight", 0.0) or 0.0)
    observation_min_score = float(
        cfg_get(judge_cfg, "observation_min_score", 0.75) or 0.0
    )
    reasoning_min_score = float(cfg_get(judge_cfg, "reasoning_min_score", 0.75) or 0.0)
    if observation_weight < 0.0 or reasoning_weight < 0.0:
        raise ValueError("Pathology judge reward weights must be non-negative.")
    if not 0.0 <= observation_min_score <= 1.0 or not 0.0 <= reasoning_min_score <= 1.0:
        raise ValueError("Pathology judge minimum scores must be between 0 and 1.")
    if as_bool(cfg_get(judge_cfg, "enabled", False)):
        correctness_weight = float(
            cfg_get(cfg_get(grpo_cfg, "reward_weights", {}), "correctness", 1.0) or 0.0
        )
        if observation_weight + reasoning_weight >= correctness_weight:
            raise ValueError(
                "The maximum pathology judge bonus must be smaller than the correctness reward."
            )
    return (
        judge_cfg,
        observation_weight,
        reasoning_weight,
        observation_min_score,
        reasoning_min_score,
    )


def _apply_pathology_judge_result(
    *,
    score_rows: list[dict[str, Any]],
    result: PathologyJudgeResult,
    observation_weight: float,
    reasoning_weight: float,
    observation_min_score: float,
    reasoning_min_score: float,
    latency_seconds: float,
) -> None:
    if len(result.scores) != len(score_rows):
        raise ValueError(
            f"Pathology judge returned {len(result.scores)} scores for {len(score_rows)} completions."
        )
    for index, row in enumerate(score_rows):
        observation_score = float(result.observation_scores[index])
        reasoning_score = float(result.reasoning_scores[index])
        reward_eligible = float(
            bool(row.get("format", 0.0)) and bool(row.get("two_step", 0.0))
        )
        observation_structure_eligible = reward_eligible * float(
            bool(row.get("observation", 0.0))
        )
        observation_reward_eligible = observation_structure_eligible * float(
            bool(row.get("correct", 0.0))
            and observation_score >= observation_min_score
        )
        judge_observation_reward = (
            observation_reward_eligible * observation_weight * observation_score
        )
        reasoning_structure_eligible = reward_eligible * float(
            bool(row.get("reasoning", 0.0))
        )
        reasoning_reward_eligible = reasoning_structure_eligible * float(
            bool(row.get("correct", 0.0)) and reasoning_score >= reasoning_min_score
        )
        judge_reasoning_reward = (
            reasoning_reward_eligible * reasoning_weight * reasoning_score
        )
        judge_reward = judge_observation_reward + judge_reasoning_reward
        row["reward"] = float(row["reward"]) + judge_reward
        row.update(
            {
                "judge_score": float(result.scores[index]),
                "judge_observation_score": observation_score,
                "judge_reasoning_score": reasoning_score,
                "judge_observation_support": float(result.observation_support[index])
                / 4.0,
                "judge_observation_salience": float(result.observation_salience[index])
                / 4.0,
                "judge_reasoning_validity": float(result.reasoning_validity[index])
                / 4.0,
                "judge_reasoning_answer_alignment": float(
                    result.reasoning_answer_alignment[index]
                )
                / 4.0,
                "judge_reward": judge_reward,
                "judge_observation_reward": judge_observation_reward,
                "judge_reasoning_reward": judge_reasoning_reward,
                "judge_reward_eligible": reward_eligible,
                "judge_observation_reward_eligible": observation_reward_eligible,
                "judge_reasoning_reward_eligible": reasoning_reward_eligible,
                "judge_enabled": 1.0,
                "judge_success": 1.0,
                "judge_failure": 0.0,
                "judge_cache_hit": float(result.cache_hit),
                "judge_latency_seconds": float(latency_seconds),
                "judge_issue": result.issues[index],
                "judge_image_inventory": list(result.image_inventory),
            }
        )


def _record_pathology_judge_failure(
    *,
    score_rows: list[dict[str, Any]],
    error: Exception,
    latency_seconds: float,
) -> None:
    print(
        f"Pathology visual judge failed for this group; using a constant zero bonus: {error}"
    )
    for row in score_rows:
        row.update(
            {
                "judge_score": 0.0,
                "judge_observation_score": 0.0,
                "judge_reasoning_score": 0.0,
                "judge_observation_support": 0.0,
                "judge_observation_salience": 0.0,
                "judge_reasoning_validity": 0.0,
                "judge_reasoning_answer_alignment": 0.0,
                "judge_reward": 0.0,
                "judge_observation_reward": 0.0,
                "judge_reasoning_reward": 0.0,
                "judge_reward_eligible": float(
                    bool(row.get("format", 0.0)) and bool(row.get("two_step", 0.0))
                ),
                "judge_observation_reward_eligible": 0.0,
                "judge_reasoning_reward_eligible": 0.0,
                "judge_enabled": 1.0,
                "judge_success": 0.0,
                "judge_failure": 1.0,
                "judge_cache_hit": 0.0,
                "judge_latency_seconds": float(latency_seconds),
                "judge_issue": f"judge_error: {error}",
                "judge_image_inventory": [],
            }
        )


def _grpo_batch_metrics(
    *,
    loss: torch.Tensor,
    rewards: torch.Tensor,
    policy_rewards: torch.Tensor,
    advantages: torch.Tensor,
    judge_observation_advantages: torch.Tensor,
    judge_reasoning_advantages: torch.Tensor,
    judge_observation_token_mask: torch.Tensor,
    judge_reasoning_token_mask: torch.Tensor,
    score_rows: list[dict[str, Any]],
    completions: list[str],
    completion_attention_mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    current_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    clip_range: float,
    max_completion_tokens: int,
) -> dict[str, float]:
    reward_values = rewards.detach().float().cpu()
    policy_reward_values = policy_rewards.detach().float().cpu()
    advantage_values = advantages.detach().float().cpu()
    judge_observation_advantage_values = (
        judge_observation_advantages.detach().float().cpu()
    )
    judge_reasoning_advantage_values = judge_reasoning_advantages.detach().float().cpu()
    completion_tokens = completion_attention_mask.sum(dim=1).detach().float().cpu()
    completion_words = [float(len(completion.split())) for completion in completions]
    lower_completions = [completion.lower() for completion in completions]
    parsed_answers = [clean_text(row.get("parsed_answer", "")) for row in score_rows]

    log_ratios = (current_logprobs.detach().float() - old_logprobs.detach().float())[
        token_mask
    ]
    valid_ratios = torch.exp(log_ratios)
    if int(valid_ratios.numel()) > 0:
        approx_kl = float(((valid_ratios - 1.0) - log_ratios).mean().cpu())
        clip_fraction = float(
            ((valid_ratios < (1.0 - clip_range)) | (valid_ratios > (1.0 + clip_range)))
            .float()
            .mean()
            .cpu()
        )
    else:
        approx_kl = 0.0
        clip_fraction = 0.0

    correct_values = _score_float_values(score_rows, "correct")
    unique_answers = len({answer for answer in parsed_answers if answer})
    reward_is_varied = bool(
        float(reward_values.max().item()) - float(reward_values.min().item()) > 1e-8
    )
    policy_reward_is_varied = bool(
        float(policy_reward_values.max().item())
        - float(policy_reward_values.min().item())
        > 1e-8
    )
    judge_values = _score_float_values(score_rows, "judge_score")
    judge_is_varied = bool(
        judge_values and max(judge_values) - min(judge_values) > 1e-8
    )
    return {
        "loss": float(loss.detach().cpu()),
        "reward_mean": float(reward_values.mean().item()),
        "reward_std": float(reward_values.std(unbiased=False).item()),
        "reward_min": float(reward_values.min().item()),
        "reward_max": float(reward_values.max().item()),
        "reward_nonzero_frac": float((reward_values != 0).float().mean().item()),
        "policy_reward_mean": float(policy_reward_values.mean().item()),
        "policy_reward_std": float(policy_reward_values.std(unbiased=False).item()),
        "correct_frac": _float_mean(_score_float_values(score_rows, "correct")),
        "valid_choice_frac": _float_mean(
            _score_float_values(score_rows, "valid_choice")
        ),
        "format_frac": _float_mean(_score_float_values(score_rows, "format")),
        "two_step_frac": _float_mean(_score_float_values(score_rows, "two_step")),
        "observation_frac": _float_mean(_score_float_values(score_rows, "observation")),
        "observation_presence_only_frac": _float_mean(
            _score_float_values(score_rows, "observation_presence_only")
        ),
        "reasoning_frac": _float_mean(_score_float_values(score_rows, "reasoning")),
        "walkthrough_frac": _float_mean(_score_float_values(score_rows, "walkthrough")),
        "choice_copy_frac": _float_mean(_score_float_values(score_rows, "choice_copy")),
        "correct_and_format_frac": _float_mean(
            [
                float(row.get("correct", 0.0) or 0.0)
                * float(row.get("format", 0.0) or 0.0)
                for row in score_rows
            ]
        ),
        "think_words_mean": _float_mean(_score_float_values(score_rows, "think_words")),
        "think_words_max": max(
            _score_float_values(score_rows, "think_words"), default=0.0
        ),
        "observation_words_mean": _float_mean(
            _score_float_values(score_rows, "observation_words")
        ),
        "reasoning_words_mean": _float_mean(
            _score_float_values(score_rows, "reasoning_words")
        ),
        "answer_tag_frac": _float_mean(
            [
                float("<answer>" in completion and "</answer>" in completion)
                for completion in lower_completions
            ]
        ),
        "close_think_frac": _float_mean(
            [float("</think>" in completion) for completion in lower_completions]
        ),
        "placeholder_answer_frac": _float_mean(
            [
                float(
                    answer.lower()
                    in {"one displayed choice copied exactly", "exact full choice text"}
                )
                for answer in parsed_answers
            ]
        ),
        "letter_answer_frac": _float_mean(
            [float(answer.lower() in {"a", "b", "c", "d"}) for answer in parsed_answers]
        ),
        "completion_tokens_mean": float(completion_tokens.mean().item()),
        "completion_tokens_max": float(completion_tokens.max().item()),
        "completion_words_mean": _float_mean(completion_words),
        "completion_words_max": max(completion_words, default=0.0),
        "completion_at_cap_frac": float(
            (completion_tokens >= max_completion_tokens).float().mean().item()
        ),
        "all_wrong_group_frac": float(not any(correct_values)),
        "all_correct_group_frac": float(bool(correct_values) and all(correct_values)),
        "mixed_correctness_group_frac": float(
            any(correct_values) and not all(correct_values)
        ),
        "reward_varied_group_frac": float(reward_is_varied),
        "policy_reward_varied_group_frac": float(policy_reward_is_varied),
        "judge_score_varied_group_frac": float(judge_is_varied),
        "judge_score_mean": _float_mean(judge_values),
        "judge_observation_mean": _float_mean(
            _score_float_values(score_rows, "judge_observation_score")
        ),
        "judge_reasoning_mean": _float_mean(
            _score_float_values(score_rows, "judge_reasoning_score")
        ),
        "judge_observation_support_mean": _float_mean(
            _score_float_values(score_rows, "judge_observation_support")
        ),
        "judge_observation_salience_mean": _float_mean(
            _score_float_values(score_rows, "judge_observation_salience")
        ),
        "judge_reasoning_validity_mean": _float_mean(
            _score_float_values(score_rows, "judge_reasoning_validity")
        ),
        "judge_reasoning_answer_alignment_mean": _float_mean(
            _score_float_values(score_rows, "judge_reasoning_answer_alignment")
        ),
        "judge_reward_mean": _float_mean(
            _score_float_values(score_rows, "judge_reward")
        ),
        "judge_observation_reward_mean": _float_mean(
            _score_float_values(score_rows, "judge_observation_reward")
        ),
        "judge_reasoning_reward_mean": _float_mean(
            _score_float_values(score_rows, "judge_reasoning_reward")
        ),
        "judge_reward_eligible_frac": _float_mean(
            _score_float_values(score_rows, "judge_reward_eligible")
        ),
        "judge_observation_reward_eligible_frac": _float_mean(
            _score_float_values(score_rows, "judge_observation_reward_eligible")
        ),
        "judge_reasoning_reward_eligible_frac": _float_mean(
            _score_float_values(score_rows, "judge_reasoning_reward_eligible")
        ),
        "judge_enabled_frac": _float_mean(
            _score_float_values(score_rows, "judge_enabled")
        ),
        "judge_success_frac": _float_mean(
            _score_float_values(score_rows, "judge_success")
        ),
        "judge_failure_frac": _float_mean(
            _score_float_values(score_rows, "judge_failure")
        ),
        "judge_cache_hit_frac": _float_mean(
            _score_float_values(score_rows, "judge_cache_hit")
        ),
        "judge_latency_seconds": _float_mean(
            _score_float_values(score_rows, "judge_latency_seconds")
        ),
        "unique_answers_mean": float(unique_answers),
        "advantage_mean": float(advantage_values.mean().item()),
        "advantage_std": float(advantage_values.std(unbiased=False).item()),
        "advantage_abs_mean": float(advantage_values.abs().mean().item()),
        "advantage_nonzero_frac": float((advantage_values != 0).float().mean().item()),
        "judge_observation_advantage_abs_mean": float(
            judge_observation_advantage_values.abs().mean().item()
        ),
        "judge_observation_advantage_nonzero_frac": float(
            (judge_observation_advantage_values != 0).float().mean().item()
        ),
        "judge_reasoning_advantage_abs_mean": float(
            judge_reasoning_advantage_values.abs().mean().item()
        ),
        "judge_reasoning_advantage_nonzero_frac": float(
            (judge_reasoning_advantage_values != 0).float().mean().item()
        ),
        "judge_observation_token_frac": float(
            (judge_observation_token_mask & token_mask)
            .sum()
            .detach()
            .float()
            .cpu()
            .item()
            / token_mask.sum().clamp_min(1).detach().float().cpu().item()
        ),
        "judge_reasoning_token_frac": float(
            (judge_reasoning_token_mask & token_mask)
            .sum()
            .detach()
            .float()
            .cpu()
            .item()
            / token_mask.sum().clamp_min(1).detach().float().cpu().item()
        ),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "logprob_tokens_mean": float(
            token_mask.sum(dim=1).detach().float().cpu().mean().item()
        ),
    }


def _accumulate_grpo_metrics(
    accumulator: dict[str, Any], metrics: dict[str, float]
) -> None:
    accumulator["count"] += 1
    for key, value in metrics.items():
        value = float(value)
        if key.endswith("_min"):
            accumulator["mins"][key] = min(value, accumulator["mins"].get(key, value))
        elif key.endswith("_max"):
            accumulator["maxs"][key] = max(value, accumulator["maxs"].get(key, value))
        else:
            accumulator["sums"][key] = accumulator["sums"].get(key, 0.0) + value


def _finalize_grpo_metrics(accumulator: dict[str, Any]) -> dict[str, float]:
    count = max(1, int(accumulator["count"]))
    metrics = {key: float(value) / count for key, value in accumulator["sums"].items()}
    metrics.update({key: float(value) for key, value in accumulator["mins"].items()})
    metrics.update({key: float(value) for key, value in accumulator["maxs"].items()})
    return metrics


def _slice_grpo_rows(batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    row_count = int(batch["input_ids"].shape[0])
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (row_count,):
            sliced[key] = value[start:end]
        elif isinstance(value, list) and len(value) == row_count:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _pad_2d(
    tensor: torch.Tensor, width: int, *, fill_value: float | bool = 0
) -> torch.Tensor:
    if int(tensor.shape[1]) == width:
        return tensor
    padded = torch.full(
        (int(tensor.shape[0]), width),
        fill_value,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[:, : int(tensor.shape[1])] = tensor
    return padded


def _concat_padded_2d(
    chunks: list[torch.Tensor], *, fill_value: float | bool = 0
) -> torch.Tensor:
    width = max(int(chunk.shape[1]) for chunk in chunks)
    return torch.cat(
        [_pad_2d(chunk, width, fill_value=fill_value) for chunk in chunks], dim=0
    )


def _completion_logprobs_microbatched(
    *,
    model: OncoVLMVQASFTModel,
    batch: dict[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    micro_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_count = int(input_ids.shape[0])
    if micro_batch_size <= 0 or micro_batch_size >= row_count:
        return completion_logprobs(
            model=model,
            batch=batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    logprob_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    for start in range(0, row_count, micro_batch_size):
        end = min(start + micro_batch_size, row_count)
        logprobs, token_mask = completion_logprobs(
            model=model,
            batch=_slice_grpo_rows(batch, start, end),
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
            labels=labels[start:end],
        )
        logprob_chunks.append(logprobs)
        mask_chunks.append(token_mask)
    return _concat_padded_2d(logprob_chunks), _concat_padded_2d(
        mask_chunks, fill_value=False
    )


def _grpo_loss_sum(
    *,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    auxiliary_terms: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
) -> torch.Tensor:
    ratio = torch.exp(current_logprobs - old_logprobs)
    unclipped = ratio * advantages.unsqueeze(1)
    clipped = torch.clamp(
        ratio, 1.0 - clip_range, 1.0 + clip_range
    ) * advantages.unsqueeze(1)
    per_token_loss = -torch.minimum(unclipped, clipped)
    completion_token_counts = token_mask.sum(dim=1)
    completion_losses = (per_token_loss * token_mask).sum(
        dim=1
    ) / completion_token_counts.clamp_min(1)
    for auxiliary_advantages, auxiliary_token_mask in auxiliary_terms:
        auxiliary_mask = token_mask & auxiliary_token_mask
        auxiliary_unclipped = ratio * auxiliary_advantages.unsqueeze(1)
        auxiliary_clipped = torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        ) * auxiliary_advantages.unsqueeze(1)
        auxiliary_token_loss = -torch.minimum(auxiliary_unclipped, auxiliary_clipped)
        auxiliary_token_counts = auxiliary_mask.sum(dim=1)
        completion_losses = completion_losses + (
            (auxiliary_token_loss * auxiliary_mask).sum(dim=1)
            / auxiliary_token_counts.clamp_min(1)
        )
    return completion_losses[completion_token_counts > 0].sum()


def _run_grpo_training(
    *,
    stage_cfg: Any,
    model: OncoVLMVQASFTModel,
    tokenizer: Any,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    device: torch.device,
    autocast_dtype: torch.dtype,
    use_autocast: bool,
    use_prefix_cache: bool,
    projector_dtype: torch.dtype | None,
    hidden_size: int,
) -> None:
    grpo_cfg = cfg_get(stage_cfg, "grpo", {})
    if float(cfg_get(grpo_cfg, "beta", 0.0) or 0.0) != 0.0:
        raise RuntimeError(
            "GRPO beta/KL reference is not implemented in this prefix-cache trainer; keep grpo.beta=0.0."
        )
    (
        judge_cfg,
        judge_observation_weight,
        judge_reasoning_weight,
        judge_observation_min_score,
        judge_reasoning_min_score,
    ) = _pathology_judge_weights(grpo_cfg)
    judge_enabled = as_bool(cfg_get(judge_cfg, "enabled", False))
    max_consecutive_judge_failures = int(
        cfg_get(judge_cfg, "max_consecutive_failures", 3) or 3
    )
    if judge_enabled and max_consecutive_judge_failures < 1:
        raise ValueError("visual_judge.max_consecutive_failures must be at least 1.")
    num_generations = int(cfg_get(grpo_cfg, "num_generations", 4))
    if num_generations < 2:
        raise ValueError(
            "GRPO needs grpo.num_generations >= 2 for per-prompt reward normalization."
        )
    grpo_batch_size = int(cfg_get(stage_cfg, "batch_size", 1))
    if grpo_batch_size != 1:
        raise ValueError(
            "GRPO currently requires vqa_train.batch_size=1 so completion logprobs align by prompt group."
        )

    collator = VQAGRPOCollator(tokenizer=tokenizer, root_dir=ROOT, stage_cfg=stage_cfg)
    train_loader = DataLoader(
        VQADataset(train_frame),
        batch_size=grpo_batch_size,
        shuffle=True,
        num_workers=int(cfg_get(stage_cfg, "dataloader_num_workers", 0)),
        collate_fn=collator,
    )
    if len(train_loader) == 0:
        raise RuntimeError("GRPO training loader is empty after batching.")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found for VQA GRPO training.")
    disabled_dropout_modules = _disable_dropout_for_grpo(model)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg_get(stage_cfg, "learning_rate", 1e-6)),
        weight_decay=float(cfg_get(stage_cfg, "weight_decay", 0.0)),
    )

    run_output_dir = _build_run_output_dir(stage_cfg, train_rows=len(train_frame))
    OmegaConf.save(config=stage_cfg, f=str(run_output_dir / "config.yaml"))
    pathology_judge = (
        PathologyStep1Judge(cfg=judge_cfg, repo_root=ROOT) if judge_enabled else None
    )
    judge_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="pathology-judge")
        if pathology_judge is not None
        else None
    )

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
    clip_range = float(cfg_get(grpo_cfg, "clip_range", 0.2))
    use_grad_scaler = use_autocast and autocast_dtype == torch.float16
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    print("Stage 2 VQA LoRA GRPO")
    print(f"Train rows: {len(train_frame):,}")
    print(f"Validation rows selected: {len(val_frame):,}")
    print(f"Model: {cfg_get(stage_cfg, 'model_name_or_path')}")
    print(
        f"CoT prompt/reward: {bool(cfg_get(cfg_get(stage_cfg, 'cot', {}), 'enabled', False))}"
    )
    init_adapter = clean_text(cfg_get(grpo_cfg, "init_lora_adapter_path", ""))
    print(f"GRPO init adapter: {init_adapter or '<fresh_lora>'}")
    print(f"Hidden size: {hidden_size}")
    print(f"Device: {device}")
    print(f"LoRA r: {int(cfg_get(cfg_get(stage_cfg, 'lora', {}), 'r', 16))}")
    if use_prefix_cache:
        print(
            f"Prefix source: cached embeddings from {cfg_get(cfg_get(stage_cfg, 'prefix_cache', {}), 'cache_root')}"
        )
        print("Loaded projectors: none (prefix cache enabled)")
    else:
        print(f"Trainable projector mode: {projector_trainable_summary(stage_cfg)}")
    print(f"Run output dir: {run_output_dir}")
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")
    print(f"Total parameters: {model.total_parameter_count():,}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Total optimizer steps: {total_optimizer_steps}")
    print(f"Dropout modules disabled for on-policy GRPO: {disabled_dropout_modules}")
    if pathology_judge is not None:
        print(
            "Pathology visual judge: "
            f"deployment={pathology_judge.deployment} "
            f"observation_weight={judge_observation_weight} "
            f"observation_min_score={judge_observation_min_score} "
            f"reasoning_weight={judge_reasoning_weight} "
            f"reasoning_min_score={judge_reasoning_min_score} "
            f"max_consecutive_failures={max_consecutive_judge_failures} "
            f"cache={pathology_judge.cache_path}"
        )
        print(
            "Judge optimization: group-centered without variance normalization; "
            "observation/reasoning rewards target their exact Step 1/Step 2 spans; "
            "both rewards require a correct final answer."
        )

    wandb_run = _maybe_init_wandb(
        stage_cfg,
        run_name=run_output_dir.name,
        train_rows=len(train_frame),
        val_rows=len(val_frame),
    )
    wandb_cfg = cfg_get(stage_cfg, "wandb", {})
    wandb_log_every_n_steps = int(cfg_get(wandb_cfg, "log_every_n_steps", 0) or 0)
    sample_log_every = int(cfg_get(grpo_cfg, "sample_log_every_n_steps", 0) or 0)
    logprob_micro_batch_size = int(
        cfg_get(grpo_cfg, "logprob_micro_batch_size", 0) or 0
    )
    snapshot_every = int(cfg_get(grpo_cfg, "save_adapter_every_n_steps", 0) or 0)
    empty_cache_every = int(cfg_get(grpo_cfg, "empty_cache_every_n_steps", 0) or 0)
    samples_path = run_output_dir / "grpo_samples.jsonl"
    metrics_path = run_output_dir / "grpo_metrics.jsonl"

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for GRPO generation.")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = pad_token_id
    max_completion_tokens = int(cfg_get(grpo_cfg, "max_completion_tokens", 192))
    generation_kwargs = {
        "max_new_tokens": max_completion_tokens,
        "do_sample": True,
        "temperature": float(cfg_get(grpo_cfg, "temperature", 0.7)),
        "top_p": float(cfg_get(grpo_cfg, "top_p", 0.95)),
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(pad_token_id),
    }
    step_1_ids = [
        int(token_id)
        for token_id in tokenizer("Step 1 — Observation:", add_special_tokens=False)[
            "input_ids"
        ]
    ]
    step_2_ids = [
        int(token_id)
        for token_id in tokenizer("Step 2 — Reasoning:", add_special_tokens=False)[
            "input_ids"
        ]
    ]
    close_think_ids = [
        int(token_id)
        for token_id in tokenizer("</think>", add_special_tokens=False)["input_ids"]
    ]

    global_step = 0
    consecutive_judge_failures = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_reward = 0.0
        running_correct = 0.0
        running_count = 0
        accum_loss = 0.0
        accum_count = 0
        accum_grpo_metrics = _new_grpo_metric_accumulator()
        loop = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"GRPO epoch {epoch + 1}/{num_epochs}",
        )
        for step, batch in enumerate(loop, start=1):
            batch = move_batch_to_device(
                batch,
                device,
                floating_dtype=None if use_prefix_cache else projector_dtype,
            )
            repeated = repeat_batch_for_generations(batch, num_generations)

            model.eval()
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
                ),
            ):
                generation_inputs = model.prepare_interleaved_generation_inputs(
                    input_ids=repeated["input_ids"],
                    attention_mask=repeated["attention_mask"],
                    pathology_features=repeated.get("pathology_features"),
                    pathology_feature_mask=repeated.get("pathology_feature_mask"),
                    radiology_features=repeated.get("radiology_features"),
                    radiology_feature_mask=repeated.get("radiology_feature_mask"),
                    dnam_features=repeated.get("dnam_features"),
                    dnam_feature_mask=repeated.get("dnam_feature_mask"),
                    rna_features=repeated.get("rna_features"),
                    rna_feature_mask=repeated.get("rna_feature_mask"),
                    pathology_prefix_embeddings=repeated.get(
                        "pathology_prefix_embeddings"
                    ),
                    pathology_prefix_mask=repeated.get("pathology_prefix_mask"),
                    radiology_prefix_embeddings=repeated.get(
                        "radiology_prefix_embeddings"
                    ),
                    radiology_prefix_mask=repeated.get("radiology_prefix_mask"),
                    dnam_prefix_embeddings=repeated.get("dnam_prefix_embeddings"),
                    dnam_prefix_mask=repeated.get("dnam_prefix_mask"),
                    rna_prefix_embeddings=repeated.get("rna_prefix_embeddings"),
                    rna_prefix_mask=repeated.get("rna_prefix_mask"),
                    prefix_spans=repeated["prefix_spans"],
                )
                batch_generation_kwargs = dict(generation_kwargs)
                if bool(cfg_get(grpo_cfg, "stop_after_answer_tag", True)):
                    stop_ids = tokenizer("</answer>", add_special_tokens=False)[
                        "input_ids"
                    ]
                    batch_generation_kwargs["stop_token_sequences"] = [
                        [int(token_id) for token_id in stop_ids]
                    ]
                    prompt_length = int(generation_inputs["input_ids"].shape[1])
                    batch_generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                        [
                            StopAfterGeneratedSubsequence(
                                stop_ids=[int(token_id) for token_id in stop_ids],
                                prompt_lengths=torch.full(
                                    (int(repeated["input_ids"].shape[0]),),
                                    prompt_length,
                                    device=repeated["input_ids"].device,
                                    dtype=torch.long,
                                ),
                            )
                        ]
                    )
                generated_ids = generate_language_model_with_soft_prefix(
                    model.language_model,
                    inputs=generation_inputs,
                    generation_kwargs=batch_generation_kwargs,
                )

            completion_attention_mask = generated_ids.ne(int(pad_token_id)).long()
            full_input_ids, full_attention_mask, labels = append_completions_to_prompts(
                prompt_input_ids=repeated["input_ids"],
                prompt_attention_mask=repeated["attention_mask"],
                completion_ids=generated_ids,
                completion_attention_mask=completion_attention_mask,
                pad_token_id=int(pad_token_id),
            )
            completions = tokenizer.batch_decode(
                generated_ids.detach().cpu(), skip_special_tokens=True
            )
            completion_score_prefixes = repeated.get(
                "completion_score_prefix", [""] * len(completions)
            )
            scored_completions = [
                f"{prefix}{completion}"
                for prefix, completion in zip(
                    completion_score_prefixes, completions, strict=True
                )
            ]
            score_rows = [
                score_grpo_completion(
                    completion=completion,
                    answer=str(answer),
                    choices=[str(choice) for choice in choices],
                    reward_cfg=grpo_cfg,
                )
                for completion, answer, choices in zip(
                    scored_completions,
                    repeated["answer"],
                    repeated["choices"],
                    strict=True,
                )
            ]
            judge_future: Future[PathologyJudgeResult] | None = None
            judge_started_at = 0.0
            if pathology_judge is not None and judge_executor is not None:
                judge_started_at = time.perf_counter()
                judge_future = judge_executor.submit(
                    pathology_judge.score_group,
                    str(batch["case_id"][0]),
                    str(batch["question"][0]),
                    [str(row["observation_text"]) for row in score_rows],
                    [str(row["reasoning_text"]) for row in score_rows],
                    [str(row["parsed_answer"]) for row in score_rows],
                )

            model.train()
            model.set_frozen_projectors_eval()
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
                ),
            ):
                old_logprobs, token_mask = _completion_logprobs_microbatched(
                    model=model,
                    batch=repeated,
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    labels=labels,
                    micro_batch_size=logprob_micro_batch_size,
                )
            if judge_future is not None:
                try:
                    judge_result = judge_future.result()
                    _apply_pathology_judge_result(
                        score_rows=score_rows,
                        result=judge_result,
                        observation_weight=judge_observation_weight,
                        reasoning_weight=judge_reasoning_weight,
                        observation_min_score=judge_observation_min_score,
                        reasoning_min_score=judge_reasoning_min_score,
                        latency_seconds=time.perf_counter() - judge_started_at,
                    )
                    consecutive_judge_failures = 0
                except Exception as exc:
                    consecutive_judge_failures += 1
                    _record_pathology_judge_failure(
                        score_rows=score_rows,
                        error=exc,
                        latency_seconds=time.perf_counter() - judge_started_at,
                    )
                    if consecutive_judge_failures >= max_consecutive_judge_failures:
                        raise RuntimeError(
                            "Pathology visual judge failed for "
                            f"{consecutive_judge_failures} consecutive groups; aborting instead "
                            "of silently continuing without the judge."
                        ) from exc
            rewards = torch.tensor(
                [float(row["reward"]) for row in score_rows],
                device=device,
                dtype=torch.float32,
            )
            judge_rewards = torch.tensor(
                [float(row.get("judge_reward", 0.0) or 0.0) for row in score_rows],
                device=device,
                dtype=torch.float32,
            )
            judge_observation_rewards = torch.tensor(
                [
                    float(row.get("judge_observation_reward", 0.0) or 0.0)
                    for row in score_rows
                ],
                device=device,
                dtype=torch.float32,
            )
            judge_reasoning_rewards = torch.tensor(
                [
                    float(row.get("judge_reasoning_reward", 0.0) or 0.0)
                    for row in score_rows
                ],
                device=device,
                dtype=torch.float32,
            )
            policy_rewards = rewards - judge_rewards
            advantages = grpo_advantages(
                policy_rewards, num_generations=num_generations
            )
            judge_observation_advantages = centered_group_rewards(
                judge_observation_rewards,
                num_generations=num_generations,
            )
            judge_reasoning_advantages = centered_group_rewards(
                judge_reasoning_rewards,
                num_generations=num_generations,
            )
            if pathology_judge is not None:
                judge_observation_token_mask = completion_span_mask(
                    completion_ids=generated_ids,
                    completion_attention_mask=completion_attention_mask,
                    start_ids=step_1_ids,
                    end_ids=step_2_ids,
                )
                judge_reasoning_token_mask = completion_span_mask(
                    completion_ids=generated_ids,
                    completion_attention_mask=completion_attention_mask,
                    start_ids=step_2_ids,
                    end_ids=close_think_ids,
                )
                if not (
                    judge_observation_token_mask.shape
                    == judge_reasoning_token_mask.shape
                    == token_mask.shape
                ):
                    raise RuntimeError(
                        "Judge span masks do not align with compact completion log-probabilities."
                    )
                judge_observation_token_mask &= token_mask
                judge_reasoning_token_mask &= token_mask
            else:
                judge_observation_token_mask = torch.zeros_like(token_mask)
                judge_reasoning_token_mask = torch.zeros_like(token_mask)
            scaled_loss = None
            if 0 < logprob_micro_batch_size < int(full_input_ids.shape[0]):
                current_chunks: list[torch.Tensor] = []
                current_mask_chunks: list[torch.Tensor] = []
                loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                completion_count = (token_mask.sum(dim=1) > 0).sum().clamp_min(1)
                for start in range(
                    0, int(full_input_ids.shape[0]), logprob_micro_batch_size
                ):
                    end = min(
                        start + logprob_micro_batch_size, int(full_input_ids.shape[0])
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=autocast_dtype,
                        enabled=use_autocast,
                    ):
                        chunk_logprobs, chunk_token_mask = completion_logprobs(
                            model=model,
                            batch=_slice_grpo_rows(repeated, start, end),
                            input_ids=full_input_ids[start:end],
                            attention_mask=full_attention_mask[start:end],
                            labels=labels[start:end],
                        )
                        chunk_logprobs = _pad_2d(
                            chunk_logprobs, int(old_logprobs.shape[1])
                        )
                        chunk_token_mask = _pad_2d(
                            chunk_token_mask,
                            int(old_logprobs.shape[1]),
                            fill_value=False,
                        )
                        chunk_mask = token_mask[start:end] & chunk_token_mask
                        chunk_loss_sum = _grpo_loss_sum(
                            current_logprobs=chunk_logprobs,
                            old_logprobs=old_logprobs[start:end].detach(),
                            token_mask=chunk_mask,
                            advantages=advantages[start:end],
                            clip_range=clip_range,
                            auxiliary_terms=(
                                (
                                    judge_observation_advantages[start:end],
                                    judge_observation_token_mask[start:end],
                                ),
                                (
                                    judge_reasoning_advantages[start:end],
                                    judge_reasoning_token_mask[start:end],
                                ),
                            ),
                        )
                        scaled_chunk_loss = (
                            chunk_loss_sum / completion_count / grad_accum
                        )
                    if use_grad_scaler:
                        grad_scaler.scale(scaled_chunk_loss).backward()
                    else:
                        scaled_chunk_loss.backward()
                    loss_sum = loss_sum + chunk_loss_sum.detach()
                    current_chunks.append(chunk_logprobs.detach())
                    current_mask_chunks.append(chunk_token_mask.detach())
                current_logprobs = torch.cat(current_chunks, dim=0)
                current_token_mask = torch.cat(current_mask_chunks, dim=0)
                token_mask = token_mask & current_token_mask
                loss = loss_sum / completion_count
            else:
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
                ):
                    current_logprobs, current_token_mask = completion_logprobs(
                        model=model,
                        batch=repeated,
                        input_ids=full_input_ids,
                        attention_mask=full_attention_mask,
                        labels=labels,
                    )
                    token_mask = token_mask & current_token_mask
                    loss = grpo_loss(
                        current_logprobs=current_logprobs,
                        old_logprobs=old_logprobs.detach(),
                        token_mask=token_mask,
                        advantages=advantages,
                        clip_range=clip_range,
                        auxiliary_terms=(
                            (
                                judge_observation_advantages,
                                judge_observation_token_mask,
                            ),
                            (
                                judge_reasoning_advantages,
                                judge_reasoning_token_mask,
                            ),
                        ),
                    )
                    scaled_loss = loss / grad_accum
                if use_grad_scaler:
                    grad_scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            batch_metrics = _grpo_batch_metrics(
                loss=loss,
                rewards=rewards,
                policy_rewards=policy_rewards,
                advantages=advantages,
                judge_observation_advantages=judge_observation_advantages,
                judge_reasoning_advantages=judge_reasoning_advantages,
                judge_observation_token_mask=judge_observation_token_mask,
                judge_reasoning_token_mask=judge_reasoning_token_mask,
                score_rows=score_rows,
                completions=scored_completions,
                completion_attention_mask=completion_attention_mask,
                old_logprobs=old_logprobs,
                current_logprobs=current_logprobs,
                token_mask=token_mask,
                clip_range=clip_range,
                max_completion_tokens=max_completion_tokens,
            )
            _accumulate_grpo_metrics(accum_grpo_metrics, batch_metrics)

            batch_loss = float(loss.detach().cpu())
            batch_reward = float(rewards.mean().detach().cpu())
            batch_correct = float(
                np.mean([float(row["correct"]) for row in score_rows])
            )
            running_loss += batch_loss
            running_reward += batch_reward
            running_correct += batch_correct
            running_count += 1
            accum_loss += batch_loss
            accum_count += 1

            if sample_log_every > 0 and (
                global_step == 0 or global_step % sample_log_every == 0
            ):
                with samples_path.open("a", encoding="utf-8") as handle:
                    for completion, score_row, question_id, answer in zip(
                        scored_completions[:num_generations],
                        score_rows[:num_generations],
                        repeated["question_id"][:num_generations],
                        repeated["answer"][:num_generations],
                        strict=True,
                    ):
                        handle.write(
                            json.dumps(
                                {
                                    "optimizer_step": int(global_step),
                                    "epoch": int(epoch + 1),
                                    "question_id": str(question_id),
                                    "answer": str(answer),
                                    "completion": completion,
                                    "score": score_row,
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )

            if step % grad_accum == 0 or step == len(train_loader):
                grad_norm_value = None
                if grad_clip_norm > 0:
                    if use_grad_scaler:
                        grad_scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, grad_clip_norm
                    )
                    grad_norm_value = float(grad_norm.detach().cpu())
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
                    grpo_metrics = _finalize_grpo_metrics(accum_grpo_metrics)
                    metric_record = {
                        "optimizer_step": int(global_step),
                        "epoch": int(epoch + 1),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "grad_norm": grad_norm_value,
                        **grpo_metrics,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(metric_record, ensure_ascii=True) + "\n"
                        )
                    if (
                        wandb_run is not None
                        and wandb_log_every_n_steps > 0
                        and global_step % wandb_log_every_n_steps == 0
                    ):
                        log_payload = {
                            f"train/grpo/{key}": value
                            for key, value in grpo_metrics.items()
                        }
                        log_payload.update(
                            {
                                "train/grpo_loss": grpo_metrics.get("loss", 0.0),
                                "train/reward": grpo_metrics.get("reward_mean", 0.0),
                                "train/correct": grpo_metrics.get("correct_frac", 0.0),
                                "train/valid_choice": grpo_metrics.get(
                                    "valid_choice_frac", 0.0
                                ),
                                "train/format": grpo_metrics.get("format_frac", 0.0),
                                "train/reasoning": grpo_metrics.get(
                                    "reasoning_frac", 0.0
                                ),
                                "train/lr": float(optimizer.param_groups[0]["lr"]),
                                "train/epoch": epoch + 1,
                                "train/optimizer_step": global_step,
                            }
                        )
                        if grad_norm_value is not None:
                            log_payload["train/grad_norm"] = grad_norm_value
                        wandb_run.log(
                            log_payload,
                            step=global_step,
                        )
                    if snapshot_every > 0 and global_step % snapshot_every == 0:
                        snapshot_dir = (
                            run_output_dir
                            / "adapter_snapshots"
                            / f"step_{global_step:06d}"
                        )
                        model.language_model.save_pretrained(
                            snapshot_dir / "lora_adapter"
                        )
                        (snapshot_dir / "snapshot.json").write_text(
                            json.dumps(
                                {
                                    "optimizer_step": int(global_step),
                                    "epoch": int(epoch + 1),
                                    "next_training_row": int(step),
                                    "parent_adapter": init_adapter,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        print(f"Saved GRPO adapter snapshot: {snapshot_dir}")
                    accum_loss = 0.0
                    accum_count = 0
                    accum_grpo_metrics = _new_grpo_metric_accumulator()

            loop.set_postfix(
                loss=f"{running_loss / max(1, running_count):.4f}",
                reward=f"{running_reward / max(1, running_count):.3f}",
                acc=f"{running_correct / max(1, running_count):.3f}",
            )
            del (
                generation_inputs,
                generated_ids,
                completion_attention_mask,
                full_input_ids,
                full_attention_mask,
                labels,
                old_logprobs,
                current_logprobs,
                token_mask,
                current_token_mask,
                loss,
                scaled_loss,
                batch_metrics,
                completions,
                rewards,
                policy_rewards,
                judge_rewards,
                judge_observation_rewards,
                judge_reasoning_rewards,
                advantages,
                judge_observation_advantages,
                judge_reasoning_advantages,
                judge_observation_token_mask,
                judge_reasoning_token_mask,
                score_rows,
                judge_future,
            )
            if (
                device.type == "cuda"
                and empty_cache_every > 0
                and step % empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

        epoch_loss = running_loss / max(1, running_count)
        epoch_reward = running_reward / max(1, running_count)
        epoch_correct = running_correct / max(1, running_count)
        print(
            f"GRPO epoch {epoch + 1}: "
            f"loss={epoch_loss:.4f} reward={epoch_reward:.4f} correct={epoch_correct:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_mean_loss": epoch_loss,
                    "train/epoch_mean_reward": epoch_reward,
                    "train/epoch_mean_correct": epoch_correct,
                    "train/epoch": epoch + 1,
                },
                step=global_step,
            )

    if judge_executor is not None:
        judge_executor.shutdown(wait=True)

    final_artifacts = save_vqa_model_artifacts(
        artifact_dir=run_output_dir,
        stage_cfg=stage_cfg,
        model=model,
        tokenizer=tokenizer,
        global_step=global_step,
        epoch=num_epochs,
        validation_loss=None,
    )
    metadata_path = _write_run_metadata(
        run_output_dir=run_output_dir,
        stage_cfg=stage_cfg,
        model=model,
        train_rows=len(train_frame),
        val_rows=len(val_frame),
        global_step=global_step,
        best_validation_loss=None,
        best_epoch=None,
        final_artifacts=final_artifacts,
        best_artifacts=None,
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "artifacts/output_dir": str(run_output_dir),
                "artifacts/global_step": global_step,
            },
            step=global_step,
        )
        wandb_run.finish()

    print(f"Saved final GRPO LoRA adapter to: {final_artifacts['lora_adapter_dir']}")
    print(f"Saved run metadata to: {metadata_path}")


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
        post_train_method = (
            str(cfg_get(stage_cfg, "post_train_method", "sft")).strip().lower()
        )

        dataset_cfg = cfg_get(stage_cfg, "dataset", {})
        vqa_parquet_path = resolve_repo_path(
            ROOT, cfg_get(dataset_cfg, "vqa_parquet_path")
        )
        if not vqa_parquet_path.exists():
            raise FileNotFoundError(
                f"VQA training parquet not found: {vqa_parquet_path}"
            )

        raw_frame = pd.read_parquet(vqa_parquet_path)
        missing_columns = [
            column for column in VQA_COLUMNS if column not in raw_frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"VQA parquet is missing required columns: {missing_columns}"
            )
        frame = normalize_vqa_df(raw_frame[VQA_COLUMNS].copy())
        for column in raw_frame.columns:
            if column not in frame.columns:
                frame[column] = raw_frame[column].values
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
            exit_after_preview = _print_prompt_previews(
                stage_cfg=stage_cfg, train_frame=train_frame
            )
        exit_after_preview = bool(_ddp_broadcast_object(exit_after_preview, ddp_state))
        if exit_after_preview:
            if is_main:
                print(
                    "preview.exit_after_preview=true; stopping before tokenizer/model/projector loading."
                )
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
                raise RuntimeError(
                    f"Validation is required but no rows were selected for split={validation_split!r}."
                )
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
        method_cfg = cfg_get(stage_cfg, post_train_method, {})
        init_adapter = clean_text(cfg_get(method_cfg, "init_lora_adapter_path", ""))
        adapter_path = resolve_repo_path(ROOT, init_adapter) if init_adapter else None
        language_model = apply_lora(
            base_language_model, stage_cfg, adapter_path=adapter_path
        )
        if is_main and hasattr(language_model, "print_trainable_parameters"):
            language_model.print_trainable_parameters()

        use_prefix_cache = prefix_cache_enabled(stage_cfg)
        if use_prefix_cache:
            projectors = {}
            projector_metadata = _cached_projector_metadata(stage_cfg)
        else:
            projectors, projector_metadata = load_projectors(
                stage_cfg, repo_root=ROOT, hidden_size=hidden_size
            )
        model = OncoVLMVQASFTModel(
            language_model=language_model,
            projectors=projectors,
            projector_metadata=projector_metadata,
        )

        autocast_dtype = (
            resolve_torch_dtype(cfg_get(stage_cfg, "autocast_dtype", "bfloat16"))
            or torch.bfloat16
        )
        projector_dtype = resolve_torch_dtype(
            cfg_get(stage_cfg, "projector_dtype", "float32")
        )
        if load_in_8bit:
            if not use_prefix_cache:
                model.move_projectors_to(device, dtype=projector_dtype)
        else:
            model.to(device=device)
            if not use_prefix_cache:
                model.move_projectors_to(device, dtype=projector_dtype)
        model.train()
        model.set_frozen_projectors_eval()

        if post_train_method == "grpo":
            if bool(ddp_state["initialized"]):
                raise RuntimeError(
                    "GRPO training currently runs single-process; launch without vqa_train.ddp=true."
                )
            _run_grpo_training(
                stage_cfg=stage_cfg,
                model=model,
                tokenizer=tokenizer,
                train_frame=train_frame,
                val_frame=val_frame,
                device=device,
                autocast_dtype=autocast_dtype,
                use_autocast=device.type == "cuda" and autocast_dtype != torch.float32,
                use_prefix_cache=use_prefix_cache,
                projector_dtype=projector_dtype,
                hidden_size=hidden_size,
            )
            return
        if post_train_method != "sft":
            raise ValueError(
                f"Unsupported vqa_train.post_train_method={post_train_method!r}."
            )

        train_model: torch.nn.Module = model
        if bool(ddp_state["initialized"]) and int(ddp_state["world_size"]) > 1:
            find_unused_parameters = any(
                bool(metadata.get("trainable", False))
                for metadata in projector_metadata.values()
            )
            train_model = DistributedDataParallel(
                model,
                device_ids=[int(ddp_state["local_rank"])],
                output_device=int(ddp_state["local_rank"]),
                find_unused_parameters=find_unused_parameters,
            )

        collator = VQATrainingCollator(
            tokenizer=tokenizer, root_dir=ROOT, stage_cfg=stage_cfg
        )
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
                collate_fn=VQATrainingCollator(
                    tokenizer=tokenizer, root_dir=ROOT, stage_cfg=stage_cfg
                ),
            )

        optimizer, trainable_parameters = _build_sft_optimizer(
            model=model,
            train_model=train_model,
            stage_cfg=stage_cfg,
        )

        if is_main:
            run_output_dir = _build_run_output_dir(
                stage_cfg, train_rows=len(train_frame)
            )
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
            print(f"Initial LoRA adapter: {adapter_path or 'fresh adapter'}")
            cot_cfg = cfg_get(stage_cfg, "cot", {})
            print(
                "CoT training: "
                f"{bool(cfg_get(cot_cfg, 'enabled', False))} "
                f"(rationale_column={cfg_get(cot_cfg, 'rationale_column', 'rationale')})"
            )
            print(f"Hidden size: {hidden_size}")
            print(f"Device: {device}")
            if bool(ddp_state["requested"]):
                print(f"DDP: world_size={int(ddp_state['world_size'])}")
            print(f"LoRA r: {int(cfg_get(cfg_get(stage_cfg, 'lora', {}), 'r', 16))}")
            if use_prefix_cache:
                print(
                    f"Prefix source: cached embeddings from {cfg_get(cfg_get(stage_cfg, 'prefix_cache', {}), 'cache_root')}"
                )
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
            for parameter_group in optimizer.param_groups:
                print(
                    f"Optimizer group {parameter_group.get('group_name', '<unnamed>')}: "
                    f"lr={float(parameter_group['lr']):.3g}, "
                    f"weight_decay={float(parameter_group['weight_decay']):.3g}"
                )
            if validation_loader is None:
                print(
                    "Validation loader is unavailable; only final artifacts will be saved."
                )

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
                loop = tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{num_epochs}",
                )
            else:
                loop = train_loader
            for step, batch in enumerate(loop, start=1):
                batch = move_batch_to_device(
                    batch,
                    device,
                    floating_dtype=None if use_prefix_cache else projector_dtype,
                )
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
                ):
                    outputs = _forward_vqa_batch(train_model, batch)
                    loss = outputs.loss
                    if loss is None:
                        raise RuntimeError(
                            "Model did not return a loss during VQA LoRA training."
                        )
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
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters, grad_clip_norm
                        )
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
                    wandb_run.log(
                        {"train/epoch_mean_loss": epoch_loss, "train/epoch": epoch + 1},
                        step=global_step,
                    )

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
                    wandb_run.log(
                        {"val/loss": validation_loss, "val/epoch": epoch + 1},
                        step=global_step,
                    )
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
                    print(
                        f"Saved best VQA LoRA artifacts to: {run_output_dir / 'best'}"
                    )
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
            if "projectors_checkpoint" in final_artifacts:
                print(
                    f"Saved final projector checkpoint to: {final_artifacts['projectors_checkpoint']}"
                )
            print(f"Saved run metadata to: {metadata_path}")
            if best_validation_loss is not None and best_epoch is not None:
                print(
                    f"Best validation loss: {best_validation_loss:.4f} at epoch {best_epoch}"
                )
        _ddp_barrier(ddp_state)
    finally:
        _cleanup_ddp(ddp_state)


if __name__ == "__main__":
    main()
