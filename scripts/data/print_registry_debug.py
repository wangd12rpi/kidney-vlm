#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a filtered preview of the registry parquet for debugging.",
    )
    parser.add_argument(
        "--path",
        default="/Users/wdn/tsa/kidney-vlm/data/registry/unified.parquet",
        help="Path to parquet registry file.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source filter (e.g. tcga).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of rows to print.",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated list of columns to print. Default prints all columns.",
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="Use head(rows). Default is sample(rows, random_state=42) when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Registry parquet not found: {path}")

    df = pd.read_parquet(path)
    original_rows = len(df)

    if args.source:
        if "source" not in df.columns:
            raise ValueError("Column 'source' not found in parquet, cannot apply --source filter.")
        df = df[df["source"] == str(args.source)]

    filtered_rows = len(df)

    if args.columns:
        selected = [col.strip() for col in args.columns.split(",") if col.strip()]
        missing = [col for col in selected if col not in df.columns]
        if missing:
            raise ValueError(f"Requested columns missing from parquet: {missing}")
        view = df[selected]
    else:
        view = df

    if args.rows <= 0:
        raise ValueError("--rows must be > 0")

    if len(view) == 0:
        preview = view
    elif args.head or len(view) <= args.rows:
        preview = view.head(args.rows)
    else:
        preview = view.sample(n=args.rows, random_state=42)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 160)

    print(f"Path: {path}")
    print(f"Rows total: {original_rows}")
    print(f"Rows after filter: {filtered_rows}")
    if args.source:
        print(f"Source filter: {args.source}")
    print(f"Columns: {list(view.columns)}")
    print("-" * 120)
    if len(preview) == 0:
        print("No rows to display.")
        return
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
