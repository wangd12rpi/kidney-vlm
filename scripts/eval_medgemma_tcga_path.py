"""Prompt-based baseline: evaluate MedGemma on TCGA pathology WSI tiles.

Assumes you have already:
1) built index: python data/tcga_build_path_index.py
2) downloaded SVS: python data/tcga_download_path_from_index.py
3) tiled SVS: python data/tcga_tile_svs.py

Run:
  python scripts/eval_medgemma_tcga_path.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from kidney_vlm.datasets.tcga_path import load_tcga_path_index
from kidney_vlm.medgemma import MedGemma, MedGemmaConfig
from kidney_vlm.tcga_eval import evaluate_tcga_task, print_tcga_eval
from kidney_vlm.tcga_tasks import TCGA_TASKS, TCGATask


# -----------------
# Config (edit me)
# -----------------
MODEL_ID = "google/medgemma-1.5-4b-it"
DTYPE = torch.bfloat16
DEVICE_MAP = "auto"

INDEX_PATH = Path("data/tcga_path_index.jsonl")

# Pick a task from kidney_vlm/tcga_tasks.py
# Examples:
#   "tumor_grade", "tumor_stage", "kidney_subtype", "vital_status"
TASK_NAME = "tumor_grade"

# Optional override: define a custom task here (then set CUSTOM_TASK to it)
CUSTOM_TASK: TCGATask | None = None

# "case" = multi-tile -> one answer per case
# "tile" = one tile -> one answer per tile (ground truth is still case-level)
EVAL_MODE = "case"

MAX_CASES = None  # set e.g. 50 for quick tests
TILES_PER_CASE = 16
MAX_NEW_TOKENS = 32
DO_SAMPLE = False


def main() -> None:
    cases = load_tcga_path_index(INDEX_PATH)
    if MAX_CASES is not None:
        cases = cases[:MAX_CASES]
    print(f"Cases loaded: {len(cases)}")

    model = MedGemma(MedGemmaConfig(model_id=MODEL_ID, dtype=DTYPE, device_map=DEVICE_MAP))

    task = CUSTOM_TASK if CUSTOM_TASK is not None else TCGA_TASKS[TASK_NAME]
    print(f"Task: {task.name} (labels['{task.label_key}'])")
    print(f"Classes: {task.classes}")

    result = evaluate_tcga_task(
        model=model,
        cases=cases,
        task=task,
        eval_mode=EVAL_MODE,
        tiles_per_case=TILES_PER_CASE,
        max_new_tokens=MAX_NEW_TOKENS,
        max_cases=MAX_CASES,
        do_sample=DO_SAMPLE,
    )

    print_tcga_eval(result)


if __name__ == "__main__":
    main()
