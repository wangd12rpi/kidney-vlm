#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.vqa.schema import VQA_COLUMNS, normalize_vqa_df, validate_vqa_df, write_vqa_parquet

ROOT = find_repo_root(Path(__file__))

PART_A_PATH = ROOT / "data/vqa/gt_mcq_questions.parquet"
PART_B_PATH = ROOT / "data/vqa/caption_condensed_mcq_questions.parquet"
PART_C_PATH = ROOT / "data/vqa/caption_qa_questions.parquet"
OUTPUT_VQA_PATH = ROOT / "data/vqa/merged_vqa.parquet"

MODALITY_COLUMNS = ("all_available", "path_only", "radiology_only")
TASK_COLUMNS = ("question_type", "generation_type", "task_category")


def _read_part(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    frame = normalize_vqa_df(pd.read_parquet(path))
    validate_vqa_df(frame)
    return frame[VQA_COLUMNS].copy()


def _task_breakdown(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    split_frame = frame[frame["split"].astype(str).str.lower().eq(split)]
    if split_frame.empty:
        return pd.DataFrame(columns=[*TASK_COLUMNS, "total", *MODALITY_COLUMNS])

    grouped = (
        split_frame.groupby([*TASK_COLUMNS, "modality_combination_name"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    for column in MODALITY_COLUMNS:
        if column not in grouped.columns:
            grouped[column] = 0
    grouped["total"] = split_frame.groupby(list(TASK_COLUMNS), dropna=False).size()
    grouped = grouped.reset_index()
    return grouped[[*TASK_COLUMNS, "total", *MODALITY_COLUMNS]].sort_values(list(TASK_COLUMNS))


def _print_stats(frame: pd.DataFrame, part_counts: dict[str, int]) -> None:
    print("Merged VQA parts")
    for name, count in part_counts.items():
        print(f"  {name}: {count:,}")
    print(f"  merged: {len(frame):,}")
    print(f"Wrote: {OUTPUT_VQA_PATH}")

    for split in ("train", "test"):
        print()
        print(f"{split} task breakdown")
        breakdown = _task_breakdown(frame, split)
        if breakdown.empty:
            print("  no rows")
        else:
            print(breakdown.to_string(index=False))


def main() -> None:
    part_a = _read_part("part_a", PART_A_PATH)
    part_b = _read_part("part_b", PART_B_PATH)
    part_c = _read_part("part_c", PART_C_PATH)

    merged = pd.concat([part_a, part_b, part_c], ignore_index=True)
    write_vqa_parquet(merged, OUTPUT_VQA_PATH)

    _print_stats(
        merged,
        {
            f"part_a {PART_A_PATH.relative_to(ROOT)}": len(part_a),
            f"part_b {PART_B_PATH.relative_to(ROOT)}": len(part_b),
            f"part_c {PART_C_PATH.relative_to(ROOT)}": len(part_c),
        },
    )


if __name__ == "__main__":
    main()
