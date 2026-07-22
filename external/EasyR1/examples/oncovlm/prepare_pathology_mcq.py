#!/usr/bin/env python3
"""Prepare the OncoVLM pathology-finding MCQs for EasyR1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 417
MAX_IMAGES = 4
TRAIN_BATCH_SIZE = 8
EXPECTED_COUNTS = {"train": 3_873, "val": 322}
OUTPUT_COLUMNS = ["problem", "answer", "images", "question_id", "case_id"]
OPTION_COLUMNS = ["option_a", "option_b", "option_c", "option_d"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[4]
    easy_r1_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=project_root,
        help="Path to the kidney-vlm repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=easy_r1_root / "data" / "oncovlm_pathology_mcq",
        help="Directory in which to write the EasyR1 parquet files.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES,
        help="Maximum number of pathology ROI images per question.",
    )
    return parser.parse_args()


def as_path_list(value: object) -> list[str]:
    """Convert a parquet list cell to a plain list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def existing_roi_paths(value: object, repo_root: Path) -> list[str]:
    """Resolve ROI paths against the project root and retain existing files."""
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in as_path_list(value):
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        resolved = str(path.resolve())
        if resolved not in seen and Path(resolved).is_file():
            paths.append(resolved)
            seen.add(resolved)
    return paths


def select_evenly_spaced(paths: list[str], max_images: int) -> list[str]:
    """Select at most ``max_images`` paths while retaining the first and last."""
    if max_images < 1:
        raise ValueError("--max-images must be at least 1")
    if len(paths) <= max_images:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=max_images, dtype=int)
    return [paths[index] for index in indices]


def build_prompt(row: pd.Series, image_count: int) -> str:
    image_markers = "\n".join("<image>" for _ in range(image_count))
    options = "\n".join(
        f"{label}. {row[column]}" for label, column in zip("ABCD", OPTION_COLUMNS, strict=True)
    )
    return (
        f"{image_markers}\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{options}"
    )


def validate_rows(frame: pd.DataFrame, name: str) -> None:
    if frame.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError(f"{name}: unexpected columns: {frame.columns.tolist()}")
    if frame["question_id"].duplicated().any():
        raise ValueError(f"{name}: question_id values must be unique")
    for row in frame.itertuples(index=False):
        if not row.images:
            raise ValueError(f"{name}: {row.question_id} has no pathology ROI image")
        if row.problem.count("<image>") != len(row.images):
            raise ValueError(f"{name}: image placeholder mismatch for {row.question_id}")
        if not all(Path(path).is_file() for path in row.images):
            raise ValueError(f"{name}: missing ROI image for {row.question_id}")


def deterministic_sample(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    if len(frame) < size:
        raise ValueError(f"Cannot sample {size} rows from a {len(frame)}-row dataset")
    ordered = frame.sort_values("question_id", kind="stable")
    return ordered.sample(n=size, random_state=SEED).reset_index(drop=True)


def pad_to_batch_size(frame: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Append deterministic repeats so EasyR1's drop-last loader sees every source row."""
    remainder = len(frame) % batch_size
    if remainder == 0:
        return frame.copy().reset_index(drop=True)

    padding = deterministic_sample(frame, batch_size - remainder)
    if pd.api.types.is_numeric_dtype(frame["question_id"]):
        first_padding_id = int(frame["question_id"].min()) - 1
        padding["question_id"] = [first_padding_id - index for index in range(len(padding))]
    else:
        padding["question_id"] = [
            f"{question_id}__batch_pad_{index}"
            for index, question_id in enumerate(padding["question_id"])
        ]
    return pd.concat([frame, padding], ignore_index=True)


def prepare_datasets(repo_root: Path, max_images: int) -> dict[str, pd.DataFrame]:
    vqa_path = repo_root / "data" / "vqa" / "merged_vqa.parquet"
    registry_path = repo_root / "data" / "registry" / "unified.parquet"

    vqa = pd.read_parquet(vqa_path)
    registry = pd.read_parquet(registry_path, columns=["patient_id", "pathology_png_roi_paths"])
    if registry["patient_id"].duplicated().any():
        raise ValueError("Registry patient_id values must be unique for a many-to-one join")

    selected = vqa[
        (vqa["question_type"] == "mcq")
        & (vqa["generation_type"] == "from_caption")
        & (vqa["modality_combination_name"] == "all_available")
        & (vqa["task_id"] == "pathology_findings")
        & vqa["use_pathology"]
        & vqa["split"].isin(["train", "val"])
    ].copy()
    selected = selected.merge(
        registry,
        left_on="case_id",
        right_on="patient_id",
        how="left",
        validate="many_to_one",
    )
    selected["images"] = selected["pathology_png_roi_paths"].map(
        lambda paths: select_evenly_spaced(existing_roi_paths(paths, repo_root), max_images)
    )
    selected = selected[selected["images"].map(bool)].copy()

    option_membership = selected.apply(
        lambda row: row["answer"] in [row[column] for column in OPTION_COLUMNS], axis=1
    )
    if not option_membership.all():
        bad_ids = selected.loc[~option_membership, "question_id"].tolist()
        raise ValueError(f"Answers must exactly match one option; invalid question IDs: {bad_ids[:10]}")
    if selected["question_id"].duplicated().any():
        raise ValueError("Filtered question_id values must be unique")

    selected["problem"] = selected.apply(lambda row: build_prompt(row, len(row["images"])), axis=1)
    prepared = selected[OUTPUT_COLUMNS + ["split"]]
    train = prepared[prepared["split"] == "train"][OUTPUT_COLUMNS].reset_index(drop=True)
    val = prepared[prepared["split"] == "val"][OUTPUT_COLUMNS].reset_index(drop=True)

    actual_counts = {"train": len(train), "val": len(val)}
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"Expected {EXPECTED_COUNTS}, found {actual_counts}")
    overlap = set(train["case_id"]) & set(val["case_id"])
    if overlap:
        raise ValueError(f"Train/validation case leakage: {sorted(overlap)[:10]}")

    datasets = {
        "train": train,
        "train_batch8": pad_to_batch_size(train, TRAIN_BATCH_SIZE),
        "val": val,
        "smoke32": deterministic_sample(train, 32),
        "pilot128": deterministic_sample(train, 128),
        "val_smoke4": deterministic_sample(val, 4),
        "val_monitor64": deterministic_sample(val, 64),
    }
    for name, frame in datasets.items():
        validate_rows(frame, name)
    return datasets


def write_datasets(
    datasets: dict[str, pd.DataFrame], repo_root: Path, output_dir: Path, max_images: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in datasets.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)

    manifest = {
        "seed": SEED,
        "max_images": max_images,
        "counts": {name: len(frame) for name, frame in datasets.items()},
        "sources": {
            "vqa": str((repo_root / "data" / "vqa" / "merged_vqa.parquet").resolve()),
            "registry": str((repo_root / "data" / "registry" / "unified.parquet").resolve()),
        },
        "outputs": {name: str((output_dir / f"{name}.parquet").resolve()) for name in datasets},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    datasets = prepare_datasets(repo_root=repo_root, max_images=args.max_images)
    write_datasets(datasets, repo_root=repo_root, output_dir=output_dir, max_images=args.max_images)
    print(f"Wrote EasyR1 pathology MCQ data to {output_dir}")
    for name, frame in datasets.items():
        print(f"  {name}: {len(frame):,}")


if __name__ == "__main__":
    main()
