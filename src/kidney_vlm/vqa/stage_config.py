from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from kidney_vlm.vqa.constants import MODALITIES

EST = timezone(timedelta(hours=-5), name="EST")


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def cfg_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    text = clean_text(value).lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"", "0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot coerce value to bool: {value!r}")


def resolve_repo_path(repo_root: Path, path_value: str | Path) -> Path:
    text = str(path_value or "").strip()
    if not text:
        raise ValueError("Received an empty path value.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def slugify_label(raw_value: Any, *, default: str) -> str:
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


def resolve_llm_tag(model_name_or_path: str) -> str:
    segments = [segment for segment in str(model_name_or_path).strip().split("/") if segment]
    candidate = segments[-1] if segments else model_name_or_path
    return slugify_label(candidate, default="llm")


def resolve_torch_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[normalized]


def enabled_modality_names(stage_cfg: Any) -> list[str]:
    projectors_cfg = cfg_get(stage_cfg, "projectors", {})
    enabled: list[str] = []
    for modality in MODALITIES:
        block = cfg_get(projectors_cfg, modality, {})
        if bool(cfg_get(block, "enabled", False)):
            enabled.append(modality)
    return enabled


def projector_trainable_summary(stage_cfg: Any) -> str:
    projectors_cfg = cfg_get(stage_cfg, "projectors", {})
    trainable = [
        modality
        for modality in MODALITIES
        if bool(cfg_get(cfg_get(projectors_cfg, modality, {}), "enabled", False))
        and bool(cfg_get(cfg_get(projectors_cfg, modality, {}), "trainable", False))
    ]
    return "projft_" + "_".join(trainable) if trainable else "projfrozen"


def generate_run_name(stage_cfg: Any, *, train_rows: int, now: datetime | None = None) -> str:
    llm_tag = resolve_llm_tag(str(cfg_get(stage_cfg, "model_name_or_path", "llm")))
    method = slugify_label(cfg_get(stage_cfg, "post_train_method", "sft"), default="sft")
    dataset_cfg = cfg_get(stage_cfg, "dataset", {})
    vqa_stem = slugify_label(Path(str(cfg_get(dataset_cfg, "vqa_parquet_path", "vqa"))).stem, default="vqa")
    lora_cfg = cfg_get(stage_cfg, "lora", {})
    lora_r = int(cfg_get(lora_cfg, "r", 0))
    timestamp = (now or datetime.now(EST)).strftime("%Y%m%d_%H%M%S_EST")
    return f"{llm_tag}_{method}_{vqa_stem}_n{int(train_rows)}_r{lora_r}_{projector_trainable_summary(stage_cfg)}_{timestamp}"
