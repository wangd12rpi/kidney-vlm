"""Prompt-based baseline: classify TCGA kidney subtype from pathology WSI tiles.

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
from PIL import Image
from tqdm import tqdm

from kidney_vlm.datasets.tcga_path import load_tcga_path_index, get_tile_paths
from kidney_vlm.medgemma import MedGemma, MedGemmaConfig
from kidney_vlm.eval_utils import compute_metrics, extract_label, print_report


# -----------------
# Config (edit me)
# -----------------
MODEL_ID = "google/medgemma-1.5-4b-it"
DTYPE = torch.bfloat16
DEVICE_MAP = "auto"

INDEX_PATH = Path("../data/data/tcga_path_index.jsonl")
MAX_CASES = None  # set e.g. 50 for quick tests

TILES_PER_CASE = 16
MAX_NEW_TOKENS = 32

LABELS = ["KICH"]

PROMPT = (
    "You are a pathology assistant. "
    "These are multiple patches from the same kidney tumor whole-slide image (H&E). "
    "Classify the TCGA kidney subtype. "
    "Answer with exactly one label: KICH, KIRC, or KIRP."
)


def main() -> None:
    cases = load_tcga_path_index(INDEX_PATH)
    if MAX_CASES is not None:
        cases = cases[:MAX_CASES]

    print(f"Cases in index: {len(cases)}")

    model = MedGemma(MedGemmaConfig(model_id=MODEL_ID, dtype=DTYPE, device_map=DEVICE_MAP))

    y_true = []
    y_pred = []

    skipped_no_tiles = 0

    for c in tqdm(cases, desc="infer"):
        tile_paths = get_tile_paths(c, max_tiles=TILES_PER_CASE)
        if len(tile_paths) == 0:
            skipped_no_tiles += 1
            continue

        images = [Image.open(p).convert("RGB") for p in tile_paths]
        out = model.generate(images, PROMPT, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        pred = extract_label(out, LABELS)
        if pred is None:
            pred = "Unknown"

        y_true.append(c.label)
        y_pred.append(pred)

    print(f"Skipped (no tiles): {skipped_no_tiles}")
    print(f"Evaluated: {len(y_true)}")

    kept_true = [t for t, p in zip(y_true, y_pred) if p in LABELS]
    kept_pred = [p for p in y_pred if p in LABELS]

    if not kept_true:
        print("No valid predictions to score (all Unknown).")
        return

    metrics = compute_metrics(kept_true, kept_pred, labels=LABELS)

    print("\nMetrics (on non-Unknown predictions only):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print_report(kept_true, kept_pred, LABELS)

    unk = sum(1 for p in y_pred if p not in LABELS)
    print(f"\nUnknown predictions: {unk}/{len(y_pred)}")


if __name__ == "__main__":
    main()
