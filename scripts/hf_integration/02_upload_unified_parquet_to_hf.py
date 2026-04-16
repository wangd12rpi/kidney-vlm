#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.hf_integration import (
    build_dataset_for_push,
    describe_dataset_payload,
    normalize_string_list,
    resolve_path,
)
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

CONFIG_PATH = ROOT / "conf" / "hf_integration" / "unified_registry_upload.yaml"


def load_cfg():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Unified-registry HF integration config not found: {CONFIG_PATH}")
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.set_struct(cfg, False)
    return cfg


def main() -> None:
    cfg = load_cfg()

    repo_id = str(cfg.repo_id).strip()
    if not repo_id:
        raise ValueError("repo_id must be set in unified_registry_upload.yaml.")

    local_parquet_path = resolve_path(ROOT, cfg.local_parquet_path)
    if not local_parquet_path.exists():
        raise FileNotFoundError(f"Local unified parquet not found: {local_parquet_path}")

    dry_run = bool(cfg.get("dry_run", True))
    private = bool(cfg.get("private", False))
    create_repo_if_missing = bool(cfg.get("create_repo_if_missing", True))
    default_split_name = str(cfg.get("default_split_name", "train")).strip().lower() or "train"
    allowed_split_names = normalize_string_list(cfg.get("allowed_split_names", []))
    split_column = str(cfg.get("split_column", "split")).strip() or None
    hf_config_name = str(cfg.get("hf_config_name", "default")).strip() or "default"
    set_default = bool(cfg.get("set_default", True))
    max_shard_size = cfg.get("max_shard_size")
    max_shard_size = None if max_shard_size in (None, "", "null") else str(max_shard_size)
    commit_message = str(cfg.get("commit_message", "Upload unified registry dataset")).strip()
    commit_description = str(cfg.get("commit_description", "")).strip() or None

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for HF upload.") from exc

    api = HfApi()
    whoami = api.whoami()
    username = str(whoami.get("name") or "").strip()
    if not username:
        raise RuntimeError("HF login check failed. Run `huggingface_hub.login()` first.")

    if create_repo_if_missing and not dry_run:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    frame = pd.read_parquet(local_parquet_path)
    dataset_payload = build_dataset_for_push(
        frame,
        split_column=split_column,
        default_split_name=default_split_name,
        allowed_split_names=allowed_split_names,
    )
    dataset_description = describe_dataset_payload(dataset_payload)

    print(f"HF user: {username}")
    print(f"Repo: {repo_id}")
    print(f"Local parquet: {local_parquet_path}")
    print(f"HF config name: {hf_config_name}")
    print(f"Payload: {dataset_description}")
    print(f"Dry run: {dry_run}")

    if dry_run:
        return

    push_kwargs = {
        "repo_id": repo_id,
        "config_name": hf_config_name,
        "set_default": set_default,
        "private": private,
        "commit_message": commit_message,
        "commit_description": commit_description,
    }
    if max_shard_size is not None:
        push_kwargs["max_shard_size"] = max_shard_size

    commit_info = dataset_payload.push_to_hub(**push_kwargs)
    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    viewer_url = f"https://huggingface.co/datasets/{repo_id}/viewer/{hf_config_name}"
    print(f"Commit: {commit_info}")
    print(f"Dataset page: {dataset_url}")
    print(f"Viewer: {viewer_url}")


if __name__ == "__main__":
    main()
