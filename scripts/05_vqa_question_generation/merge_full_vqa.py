#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))

GT_VQA_PATH = ROOT / "data/vqa/gt_mcq_questions.parquet"
CAPTION_VQA_PATH = ROOT / "data/vqa/captions_mcq_oe_questions.parquet"
OUTPUT_VQA_PATH = ROOT / "data/vqa/merged_full_vqa.parquet"


def _string_value(value) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return str(value)


def main() -> None:
    gt_df = pd.read_parquet(GT_VQA_PATH)
    caption_df = pd.read_parquet(CAPTION_VQA_PATH)

    gt_columns = list(gt_df.columns)
    caption_columns = list(caption_df.columns)
    if caption_columns != gt_columns:
        missing = [column for column in gt_columns if column not in caption_columns]
        extra = [column for column in caption_columns if column not in gt_columns]
        raise ValueError(
            "Caption VQA parquet schema does not match GT VQA parquet. "
            f"missing={missing}; extra={extra}"
        )

    for frame in (gt_df, caption_df):
        frame["question_id"] = pd.to_numeric(frame["question_id"], errors="raise").astype("Int64")
        frame["base_question_id"] = pd.to_numeric(
            frame["base_question_id"].replace("", pd.NA),
            errors="raise",
        ).astype("Int64")
        frame["caption_id"] = frame["caption_id"].astype(str).replace({"<NA>": "", "nan": "", "None": ""})
        frame["radiology_view_png_dir"] = frame["radiology_view_png_dir"].map(_string_value)

    merged_df = pd.concat([gt_df, caption_df[gt_columns]], ignore_index=True)
    duplicate_question_ids = merged_df["question_id"].duplicated()
    if duplicate_question_ids.any():
        examples = merged_df.loc[duplicate_question_ids, "question_id"].head(10).tolist()
        raise ValueError(f"Merged VQA has duplicate question_id values. Examples: {examples}")

    OUTPUT_VQA_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(OUTPUT_VQA_PATH, index=False)

    print(f"GT rows: {len(gt_df):,}")
    print(f"Caption rows: {len(caption_df):,}")
    print(f"Merged rows: {len(merged_df):,}")
    print(f"Wrote: {OUTPUT_VQA_PATH}")


if __name__ == "__main__":
    main()
