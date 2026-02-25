#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print per-source registry status and check whether referenced local binaries exist "
            "(SVS, images, PDFs, feature files, masks, etc.)."
        )
    )
    parser.add_argument(
        "--registry",
        default=str(ROOT / "data" / "registry" / "unified.parquet"),
        help="Path to unified parquet registry.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source filter (example: tcga).",
    )
    parser.add_argument(
        "--samples-per-source",
        type=int,
        default=1,
        help="How many sample rows to print per source.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling rows when source has many rows.",
    )
    parser.add_argument(
        "--path-columns",
        default=None,
        help=(
            "Comma-separated path columns to check. "
            "Default auto-detects columns ending in _path or _paths."
        ),
    )
    parser.add_argument(
        "--missing-examples",
        type=int,
        default=5,
        help="Number of missing local path examples to print per column.",
    )
    return parser.parse_args()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]

    if isinstance(value, float) and pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]

    return [text]


def detect_path_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.endswith("_paths") or column.endswith("_path")]


def summarize_path_column(
    values: pd.Series,
    *,
    exists_cache: dict[str, bool],
    missing_examples_limit: int,
) -> dict[str, Any]:
    total_refs = 0
    local_refs = 0
    uri_refs = 0
    existing_local_refs = 0
    missing_local_refs = 0

    missing_unique: set[str] = set()
    missing_examples: list[str] = []

    non_empty_rows = 0

    for raw_value in values.tolist():
        refs = as_list(raw_value)
        if refs:
            non_empty_rows += 1

        for ref in refs:
            total_refs += 1
            if "://" in ref:
                uri_refs += 1
                continue

            local_refs += 1
            is_present = exists_cache.get(ref)
            if is_present is None:
                is_present = Path(ref).exists()
                exists_cache[ref] = is_present

            if is_present:
                existing_local_refs += 1
            else:
                missing_local_refs += 1
                if ref not in missing_unique and len(missing_examples) < missing_examples_limit:
                    missing_examples.append(ref)
                missing_unique.add(ref)

    return {
        "non_empty_rows": non_empty_rows,
        "total_refs": total_refs,
        "local_refs": local_refs,
        "uri_refs": uri_refs,
        "existing_local_refs": existing_local_refs,
        "missing_local_refs": missing_local_refs,
        "missing_local_unique": len(missing_unique),
        "missing_examples": missing_examples,
    }


def split_counts(frame: pd.DataFrame) -> str:
    if "split" not in frame.columns:
        return "split column missing"
    counts = frame["split"].fillna("").map(str).value_counts().to_dict()
    parts = [f"{name}={value}" for name, value in sorted(counts.items())]
    return ", ".join(parts) if parts else "none"


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            return str(value.tolist())
        except Exception:
            return str(value)
    return str(value)


def print_samples(
    source_frame: pd.DataFrame,
    *,
    source: str,
    path_columns: list[str],
    sample_rows: int,
    seed: int,
) -> None:
    del path_columns

    if source_frame.empty:
        print("Sample rows: none")
        return

    if sample_rows <= 0:
        return

    count = min(sample_rows, len(source_frame))
    if len(source_frame) > count:
        sample = source_frame.sample(n=count, random_state=seed)
    else:
        sample = source_frame.head(count)

    for idx, (_, row) in enumerate(sample.iterrows(), start=1):
        if count == 1:
            print(f"Sample row ({source}):")
        else:
            print(f"Sample row {idx}/{count} ({source}):")
        for column in source_frame.columns:
            print(f"{column}: {_display_value(row[column])}")
        if idx < count:
            print("-" * 80)


def main() -> None:
    args = parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry parquet not found: {registry_path}")

    frame = pd.read_parquet(registry_path)
    if frame.empty:
        print(f"Registry: {registry_path}")
        print("Rows: 0")
        print("No data in registry.")
        return

    if "source" not in frame.columns:
        raise ValueError("Registry is missing required column: source")

    if args.source:
        frame = frame[frame["source"].map(str) == str(args.source)]
        if frame.empty:
            raise ValueError(f"No rows found for source='{args.source}'")

    if args.path_columns:
        path_columns = [col.strip() for col in str(args.path_columns).split(",") if col.strip()]
        missing = [col for col in path_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Requested path columns not found: {missing}")
    else:
        path_columns = detect_path_columns(frame)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 180)

    print(f"Registry: {registry_path}")
    print(f"Rows total (post-filter): {len(frame)}")
    print(f"Sources: {', '.join(sorted(frame['source'].map(str).unique().tolist()))}")
    print(f"Path columns checked: {path_columns if path_columns else 'none'}")
    print("=" * 120)

    exists_cache: dict[str, bool] = {}
    for source in sorted(frame["source"].map(str).unique().tolist()):
        source_frame = frame[frame["source"].map(str) == source]
        print(f"\nSource: {source}")
        print(f"Rows: {len(source_frame)}")
        print(f"Split counts: {split_counts(source_frame)}")

        source_missing_total = 0
        source_missing_unique_total = 0

        for column in path_columns:
            summary = summarize_path_column(
                source_frame[column],
                exists_cache=exists_cache,
                missing_examples_limit=max(0, int(args.missing_examples)),
            )
            source_missing_total += int(summary["missing_local_refs"])
            source_missing_unique_total += int(summary["missing_local_unique"])

            print(
                f"- {column}: non_empty_rows={summary['non_empty_rows']}, refs={summary['total_refs']}, "
                f"local={summary['local_refs']}, uri={summary['uri_refs']}, "
                f"existing_local={summary['existing_local_refs']}, "
                f"missing_local={summary['missing_local_refs']} (unique={summary['missing_local_unique']})"
            )
            if summary["missing_examples"]:
                print("  missing examples:")
                for ref in summary["missing_examples"]:
                    print(f"  - {ref}")

        print(
            f"Missing local refs total for source '{source}': {source_missing_total} "
            f"(sum of unique-per-column={source_missing_unique_total})"
        )

        print_samples(
            source_frame,
            source=source,
            path_columns=path_columns,
            sample_rows=max(0, int(args.samples_per_source)),
            seed=int(args.seed),
        )


if __name__ == "__main__":
    main()
