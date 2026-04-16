#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from tqdm.auto import tqdm

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

CONFIG_PATH = ROOT / "conf" / "hf_integration" / "projector_train_upload.yaml"


def load_cfg():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Projector-train HF integration config not found: {CONFIG_PATH}")
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.set_struct(cfg, False)
    return cfg


def main() -> None:
    cfg = load_cfg()

    repo_id = str(cfg.repo_id).strip()
    if not repo_id:
        raise ValueError("repo_id must be set in projector_train_upload.yaml.")

    dry_run = bool(cfg.get("dry_run", True))
    private = bool(cfg.get("private", False))
    create_repo_if_missing = bool(cfg.get("create_repo_if_missing", True))
    default_split_name = str(cfg.get("default_split_name", "train")).strip().lower() or "train"
    allowed_split_names = normalize_string_list(cfg.get("allowed_split_names", []))
    skip_missing_local_files = bool(cfg.get("skip_missing_local_files", True))
    max_shard_size = cfg.get("max_shard_size")
    max_shard_size = None if max_shard_size in (None, "", "null") else str(max_shard_size)
    commit_message_prefix = str(cfg.get("commit_message_prefix", "Upload projector-train dataset")).strip()
    commit_description = str(cfg.get("commit_description", "")).strip() or None

    artifacts = [artifact for artifact in list(cfg.get("artifacts", []) or []) if bool(artifact.get("enabled", True))]
    if not artifacts:
        raise RuntimeError("No enabled artifacts found in projector_train_upload.yaml.")

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

    print(f"HF user: {username}")
    print(f"Repo: {repo_id}")
    print(f"Dry run: {dry_run}")
    print(f"Artifacts enabled: {len(artifacts)}")

    artifact_loop = tqdm(artifacts, total=len(artifacts), desc="Preparing HF uploads", unit="artifact")
    for artifact_cfg in artifact_loop:
        artifact_name = str(artifact_cfg.get("name", "")).strip() or "unnamed_artifact"
        artifact_loop.set_postfix_str(artifact_name)

        local_parquet_path = resolve_path(ROOT, artifact_cfg.get("local_parquet_path"))
        if not local_parquet_path.exists():
            message = f"Local parquet not found for artifact '{artifact_name}': {local_parquet_path}"
            if skip_missing_local_files:
                print(f"[warning] {message}. Skipping.")
                continue
            raise FileNotFoundError(message)

        frame = pd.read_parquet(local_parquet_path)
        if frame.empty:
            print(f"[warning] Artifact '{artifact_name}' is empty at {local_parquet_path}. Skipping.")
            continue

        split_column = str(artifact_cfg.get("split_column", "")).strip() or None
        hf_config_name = str(artifact_cfg.get("hf_config_name", artifact_name)).strip() or artifact_name
        set_default = artifact_cfg.get("set_default")
        set_default = None if set_default in (None, "", "null") else bool(set_default)
        artifact_commit_message = str(
            artifact_cfg.get("commit_message", f"{commit_message_prefix}: {artifact_name}")
        ).strip()

        dataset_payload = build_dataset_for_push(
            frame,
            split_column=split_column,
            default_split_name=default_split_name,
            allowed_split_names=allowed_split_names,
        )
        dataset_description = describe_dataset_payload(dataset_payload)

        print("-" * 80)
        print(f"Artifact: {artifact_name}")
        print(f"Local parquet: {local_parquet_path}")
        print(f"HF config name: {hf_config_name}")
        print(f"Payload: {dataset_description}")

        if dry_run:
            continue

        push_kwargs = {
            "repo_id": repo_id,
            "config_name": hf_config_name,
            "set_default": set_default,
            "private": private,
            "commit_message": artifact_commit_message,
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
