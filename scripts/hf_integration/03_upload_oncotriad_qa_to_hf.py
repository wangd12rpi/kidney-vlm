#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

REPO_ID = "AIM-Research-Lab/OncoTriad-QA"
PRIVATE = False
CREATE_REPO_IF_MISSING = True
STAGING_DIR = ROOT / "data" / "staging" / "hf_datasets" / "oncotriad_qa"
SPLITS = ["train", "val", "test"]
SPLIT_COLUMN = "split"
MODALITY_COMBINATION_COLUMN = "modality_combination_name"
UPLOAD_MODALITY_COMBINATION = "all_available"
COMMIT_MESSAGE = "Upload OncoTriad-QA full VQA dataset"
COMMIT_DESCRIPTION = (
    "Upload full Universal MCQ, Case-specific MCQ, and Case-specific open-ended "
    "VQA parquets with task configs and train/val/test splits."
)
REMOTE_DELETE_PATTERNS = [
    "README.md",
    "universal_mcq/**",
    "case_specific_mcq/**",
    "case_specific_open_ended/**",
    "**/*.parquet",
]

TASK_DISPLAY_NAMES = {
    "universal_mcq": "Universal MCQ",
    "case_specific_mcq": "Case-specific MCQ",
    "case_specific_open_ended": "Case-specific open-ended",
}

TASK_SOURCE_PARQUETS = {
    "universal_mcq": ROOT / "data" / "vqa" / "gt_mcq_questions_full.parquet",
    "case_specific_mcq": ROOT
    / "data"
    / "vqa"
    / "caption_condensed_mcq_questions_full.parquet",
    "case_specific_open_ended": ROOT
    / "data"
    / "vqa"
    / "caption_qa_questions_full.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the full OncoTriad-QA VQA tasks to Hugging Face Hub."
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--staging-dir", default=str(STAGING_DIR))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _quote_yaml(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _write_split_files(staging_dir: Path) -> dict[str, dict[str, int]]:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {}
    for task_key, source_path in TASK_SOURCE_PARQUETS.items():
        if task_key not in TASK_DISPLAY_NAMES:
            raise KeyError(f"Missing display name for task key: {task_key}")
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source parquet: {source_path}")

        frame = pd.read_parquet(source_path)
        if frame.empty:
            raise RuntimeError(f"Source parquet is empty: {source_path}")
        if SPLIT_COLUMN not in frame.columns:
            raise ValueError(f"{source_path} is missing required split column: {SPLIT_COLUMN}")
        if MODALITY_COMBINATION_COLUMN not in frame.columns:
            raise ValueError(
                f"{source_path} is missing required modality column: "
                f"{MODALITY_COMBINATION_COLUMN}"
            )

        modality_values = frame[MODALITY_COMBINATION_COLUMN].astype(str).str.strip()
        frame = frame.loc[modality_values.eq(UPLOAD_MODALITY_COMBINATION)].reset_index(
            drop=True
        )
        if frame.empty:
            raise RuntimeError(
                f"{source_path} has no {UPLOAD_MODALITY_COMBINATION!r} rows; "
                "refusing upload."
            )

        split_values = frame[SPLIT_COLUMN].astype(str).str.strip()
        task_dir = staging_dir / task_key
        task_dir.mkdir(parents=True, exist_ok=True)
        counts[task_key] = {}
        for split in SPLITS:
            split_frame = frame.loc[split_values.eq(split)].reset_index(drop=True)
            if split_frame.empty:
                raise RuntimeError(
                    f"{source_path} has no rows for split {split!r}; refusing upload."
                )
            out_path = task_dir / f"{split}.parquet"
            split_frame.to_parquet(out_path, index=False)
            counts[task_key][split] = int(len(split_frame))
    return counts


def _write_readme(staging_dir: Path) -> None:
    lines = [
        "---",
        "pretty_name: OncoTriad-QA",
        "configs:",
    ]
    for task_key, display_name in TASK_DISPLAY_NAMES.items():
        lines.append(f"- config_name: {_quote_yaml(display_name)}")
        lines.append("  data_files:")
        for split in SPLITS:
            lines.append(f"  - split: {split}")
            lines.append(f"    path: {task_key}/{split}.parquet")
    lines.extend(
        [
            "---",
            "",
            "# OncoTriad-QA",
            "",
            "OncoTriad-QA contains three full VQA task families:",
            "",
            "- Universal MCQ",
            "- Case-specific MCQ",
            "- Case-specific open-ended",
            "",
            "Each task is exposed as a dataset configuration.",
            "Each configuration contains train, val, and test splits.",
            "",
        ]
    )
    (staging_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _print_counts(counts: dict[str, dict[str, int]], staging_dir: Path) -> None:
    print(f"Staging dir: {staging_dir}")
    for task_key, split_counts in counts.items():
        display_name = TASK_DISPLAY_NAMES[task_key]
        total = sum(split_counts.values())
        pieces = ", ".join(f"{split}={split_counts[split]}" for split in SPLITS)
        print(f"{display_name}: total={total}, {pieces}")


def main() -> None:
    args = parse_args()
    staging_dir = Path(args.staging_dir).expanduser()
    if not staging_dir.is_absolute():
        staging_dir = ROOT / staging_dir
    staging_dir = staging_dir.resolve()

    counts = _write_split_files(staging_dir)
    _write_readme(staging_dir)
    _print_counts(counts, staging_dir)

    if args.dry_run:
        print("Dry run: staged files only. Nothing uploaded.")
        return

    _load_dotenv_if_present()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for HF upload.") from exc

    api = HfApi()
    whoami = api.whoami()
    username = str(whoami.get("name") or "").strip()
    if not username:
        raise RuntimeError("HF login check failed. Set HF_TOKEN or run huggingface-cli login.")

    if CREATE_REPO_IF_MISSING:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=PRIVATE,
            exist_ok=True,
        )

    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=staging_dir,
        commit_message=COMMIT_MESSAGE,
        commit_description=COMMIT_DESCRIPTION,
        delete_patterns=REMOTE_DELETE_PATTERNS,
    )

    print(f"HF user: {username}")
    print(f"Repo: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Viewer: https://huggingface.co/datasets/{args.repo_id}/viewer")
    print(f"Commit: {commit}")


if __name__ == "__main__":
    main()
