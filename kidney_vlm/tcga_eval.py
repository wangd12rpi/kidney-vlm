from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from kidney_vlm.datasets.tcga_path import TCGACase, get_tile_paths
from kidney_vlm.eval_utils import compute_metrics, print_report
from kidney_vlm.medgemma import MedGemma
from kidney_vlm.tcga_tasks import TCGATask


def _gt_for_task(case: TCGACase, task: TCGATask) -> Optional[str]:
    v = case.labels.get(task.label_key)
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    for c in task.classes:
        if s.upper() == c.upper():
            return c
    return None


def evaluate_tcga_task(
    model: MedGemma,
    cases: List[TCGACase],
    task: TCGATask,
    eval_mode: str,
    tiles_per_case: int,
    max_new_tokens: int,
    max_cases: Optional[int] = None,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """Evaluate a classification task from labels{}.

    eval_mode:
      - "case": multiple tiles -> one answer per case
      - "tile": one tile -> one answer per tile

    Skips cases where:
      - task ground truth is missing or not in task.classes
      - there are no tiles on disk
    """
    if eval_mode not in ["case", "tile"]:
        raise ValueError("eval_mode must be 'case' or 'tile'")

    if max_cases is not None:
        cases = cases[:max_cases]

    prompt = task.build_prompt()

    y_true: List[str] = []
    y_pred: List[str] = []

    skipped_no_gt = 0
    skipped_no_tiles = 0
    unknown = 0

    total_units = 0

    for c in tqdm(cases, desc=f"infer ({eval_mode})"):
        gt = _gt_for_task(c, task)
        if gt is None:
            skipped_no_gt += 1
            continue

        tile_paths = get_tile_paths(c, max_tiles=tiles_per_case)
        if len(tile_paths) == 0:
            skipped_no_tiles += 1
            continue

        if eval_mode == "case":
            images = [Image.open(p).convert("RGB") for p in tile_paths]
            out = model.generate(images, prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
            pred = task.parse_prediction(out) or "Unknown"
            y_true.append(gt)
            y_pred.append(pred)
            total_units += 1
            if pred == "Unknown":
                unknown += 1

        else:
            for p in tile_paths:
                img = Image.open(p).convert("RGB")
                out = model.generate([img], prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
                pred = task.parse_prediction(out) or "Unknown"
                y_true.append(gt)
                y_pred.append(pred)
                total_units += 1
                if pred == "Unknown":
                    unknown += 1

    kept_true = [t for t, p in zip(y_true, y_pred) if p in task.classes]
    kept_pred = [p for p in y_pred if p in task.classes]

    metrics = None
    if kept_true:
        metrics = compute_metrics(kept_true, kept_pred, labels=task.classes)

    return {
        "task_name": task.name,
        "task_label_key": task.label_key,
        "task_classes": task.classes,
        "eval_mode": eval_mode,
        "tiles_per_case": tiles_per_case,
        "max_new_tokens": max_new_tokens,
        "n_units_total": total_units,
        "n_skipped_no_gt": skipped_no_gt,
        "n_skipped_no_tiles": skipped_no_tiles,
        "n_unknown": unknown,
        "n_scored": len(kept_true),
        "metrics": metrics,
        "y_true_scored": kept_true,
        "y_pred_scored": kept_pred,
    }


def print_tcga_eval(result: Dict[str, Any]) -> None:
    print("\n=== TCGA Task Eval ===")
    print(f"task: {result['task_name']} (labels['{result['task_label_key']}'])")
    print(f"mode: {result['eval_mode']}  tiles_per_case: {result['tiles_per_case']}")
    print(f"total_units: {result['n_units_total']}")
    print(f"skipped_no_gt: {result['n_skipped_no_gt']}")
    print(f"skipped_no_tiles: {result['n_skipped_no_tiles']}")
    print(f"unknown: {result['n_unknown']}")
    print(f"scored: {result['n_scored']}")

    if result["metrics"] is None:
        print("No valid predictions to score (all Unknown or no samples).")
        return

    print("\nMetrics (on non-Unknown predictions only):")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}")

    print_report(result["y_true_scored"], result["y_pred_scored"], result["task_classes"])