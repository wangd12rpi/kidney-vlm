"""Prompt-based baseline: kidney stone vs non-stone using base MedGemma.

Run:
  python scripts/eval_medgemma_kidney_stone.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from kidney_vlm.datasets.kidney_stone import load_kidney_stone_dataset
from kidney_vlm.medgemma import MedGemma, MedGemmaConfig
from kidney_vlm.eval_utils import compute_metrics, extract_label_with_synonyms, print_report


# -----------------
# Config (edit me)
# -----------------
MODEL_ID = "google/medgemma-1.5-4b-it"
DTYPE = torch.bfloat16
DEVICE_MAP = "auto"

DATA_ROOT = Path("data/raw/kidney_stone/Original")
MAX_SAMPLES = None  # set e.g. 20 for quick tests

LABELS = ["Stone", "Non-Stone"]
LABEL_SYNONYMS = {
    "Stone": ["STONE", "KIDNEY STONE"],
    "Non-Stone": ["NONSTONE", "NON STONE", "NO STONE", "WITHOUT STONE"],
}

PROMPT = (
    "You are a medical image classifier. "
    "Does this image show a kidney stone? "
    "Answer with exactly one label: Stone or Non-Stone."
)

MAX_NEW_TOKENS = 16


def main() -> None:
    samples = load_kidney_stone_dataset(DATA_ROOT)
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]

    print(f"Samples: {len(samples)}")

    model = MedGemma(MedGemmaConfig(model_id=MODEL_ID, dtype=DTYPE, device_map=DEVICE_MAP))

    y_true = []
    y_pred = []

    for s in tqdm(samples, desc="infer"):
        img = Image.open(s.image_path).convert("RGB")
        out = model.generate([img], PROMPT, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        pred = extract_label_with_synonyms(out, LABEL_SYNONYMS)
        if pred is None:
            pred = "Unknown"

        y_true.append(s.label)
        y_pred.append(pred)

    # Drop unknowns for metric computation (or keep them as errors)
    # Here we keep them; scikit-learn will treat Unknown as another label
    # so we compute metrics only over the target labels.
    metrics = compute_metrics(
        y_true=[t for t, p in zip(y_true, y_pred) if p in LABELS],
        y_pred=[p for p in y_pred if p in LABELS],
        labels=LABELS,
    )

    print("\nMetrics (on non-Unknown predictions only):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Full report including Unknown counts
    kept_true = [t for t, p in zip(y_true, y_pred) if p in LABELS]
    kept_pred = [p for p in y_pred if p in LABELS]
    if kept_true:
        print_report(kept_true, kept_pred, LABELS)

    unk = sum(1 for p in y_pred if p not in LABELS)
    print(f"\nUnknown predictions: {unk}/{len(y_pred)}")


if __name__ == "__main__":
    main()
