#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push local unified parquet to a Hugging Face dataset repo.")
    parser.add_argument(
        "--repo-id",
        default="wangd12/kidney_vlm",
        help="Hugging Face dataset repo id (example: wangd12/kidney_vlm).",
    )
    parser.add_argument(
        "--local-parquet",
        default=str(ROOT / "data" / "registry" / "unified.parquet"),
        help="Local parquet path to upload.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="unified.parquet",
        help="Target path in the dataset repo.",
    )
    parser.add_argument(
        "--commit-message",
        default="Update unified registry parquet",
        help="Commit message used on HF Hub.",
    )
    parser.add_argument(
        "--create-if-missing",
        action="store_true",
        help="Create the dataset repo if it does not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_path = Path(args.local_parquet).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Local parquet not found: {local_path}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install it with: uv add huggingface_hub") from exc

    api = HfApi()
    whoami = api.whoami()
    username = str(whoami.get("name") or "").strip()
    if not username:
        raise RuntimeError("HF login check failed. Run `huggingface_hub.login()` first.")

    if args.create_if_missing:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    commit_url = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=str(args.path_in_repo),
        repo_id=str(args.repo_id),
        repo_type="dataset",
        commit_message=str(args.commit_message),
    )

    dataset_url = f"https://huggingface.co/datasets/{args.repo_id}"
    viewer_url = f"https://huggingface.co/datasets/{args.repo_id}/viewer/default/train"
    file_url = f"https://huggingface.co/datasets/{args.repo_id}/blob/main/{args.path_in_repo}"

    print(f"HF user: {username}")
    print(f"Uploaded local parquet: {local_path}")
    print(f"Repo: {args.repo_id}")
    print(f"Path in repo: {args.path_in_repo}")
    print(f"Commit: {commit_url}")
    print(f"Dataset page: {dataset_url}")
    print(f"Viewer: {viewer_url}")
    print(f"File: {file_url}")


if __name__ == "__main__":
    os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)
    main()
