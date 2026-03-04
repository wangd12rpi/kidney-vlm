#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull unified parquet from HF Hub and replace local copy.")
    parser.add_argument(
        "--repo-id",
        default="wangd12/kidney_vlm",
        help="Hugging Face dataset repo id (example: wangd12/kidney_vlm).",
    )
    parser.add_argument(
        "--path-in-repo",
        default="unified.parquet",
        help="Parquet path inside the dataset repo.",
    )
    parser.add_argument(
        "--local-parquet",
        default=str(ROOT / "data" / "registry" / "unified.parquet"),
        help="Local parquet destination path.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation before writing local parquet.",
    )
    return parser.parse_args()


def _read_local_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    local_path = Path(args.local_parquet).expanduser().resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install it with: uv add huggingface_hub") from exc

    api = HfApi()
    whoami = api.whoami()
    username = str(whoami.get("name") or "").strip()
    if not username:
        raise RuntimeError("HF login check failed. Run `huggingface_hub.login()` first.")

    downloaded_path = hf_hub_download(
        repo_id=str(args.repo_id),
        repo_type="dataset",
        filename=str(args.path_in_repo),
    )
    remote_df = pd.read_parquet(downloaded_path)
    local_df = _read_local_or_empty(local_path)
    output_df = remote_df.copy()

    backup_path = None
    if local_path.exists() and not args.no_backup:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup_path = local_path.with_name(f"{local_path.name}.bak.{stamp}")
        shutil.copy2(local_path, backup_path)

    output_df.to_parquet(local_path, index=False)

    print(f"HF user: {username}")
    print(f"Repo: {args.repo_id}")
    print(f"Pulled file in repo: {args.path_in_repo}")
    print(f"Local path updated: {local_path}")
    print("Mode: replace (no merge)")
    print(f"Remote rows: {len(remote_df)}")
    print(f"Local rows before: {len(local_df)}")
    print(f"Local rows after: {len(output_df)}")
    if backup_path is not None:
        print(f"Backup: {backup_path}")
    print(f"Dataset page: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
