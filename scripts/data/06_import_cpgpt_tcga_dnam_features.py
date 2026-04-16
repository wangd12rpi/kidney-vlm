#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.dnam_feature_import import (
    build_cpgpt_hash_index,
    build_cpgpt_output_path,
)
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


# External source locations
HESCAPEDNA_ROOT = Path("/media/volume/patho_meth/hescapedna")
HESCAPEDNA_INDEX_DIR = HESCAPEDNA_ROOT / "data" / "wsi_dnameth"
HESCAPEDNA_RAW_DNAM_ROOT = HESCAPEDNA_INDEX_DIR / "raw"
HESCAPEDNA_CPGPT_CACHE_DIR = HESCAPEDNA_ROOT / "cache" / "cpgpt_embeddings"

# Local output locations
OUTPUT_FEATURES_DIR = ROOT / "data" / "features" / "features_cpgpt_dnam"
MANIFEST_PARQUET_PATH = ROOT / "data" / "features" / "features_cpgpt_dnam_manifest.parquet"
MANIFEST_CSV_PATH = ROOT / "data" / "features" / "features_cpgpt_dnam_manifest.csv"
UNMATCHED_HASHES_PATH = ROOT / "data" / "features" / "features_cpgpt_dnam_unmatched_hashes.txt"

# Selection
# Empty means all indexed TCGA projects.
ALLOWED_PROJECT_IDS: list[str] = []
TOP_K_FEATURES: int | None = None

# Copy behavior
OVERWRITE_EXISTING = False
WRITE_CSV_MANIFEST = True


def _selected_index_paths() -> list[Path]:
    return sorted(path for path in HESCAPEDNA_INDEX_DIR.glob("*.jsonl") if path.is_file())


def _relative_to_repo(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _filtered_records_by_hash(records_by_hash):
    allowed = {project.strip() for project in ALLOWED_PROJECT_IDS if str(project).strip()}
    selected_hashes: list[str] = []
    seen_output_paths: dict[Path, str] = {}

    for cache_hash, record in records_by_hash.items():
        if allowed and record.project_id not in allowed:
            continue
        output_path = build_cpgpt_output_path(OUTPUT_FEATURES_DIR, record)
        existing_hash = seen_output_paths.get(output_path)
        if existing_hash is not None and existing_hash != cache_hash:
            raise ValueError(
                "Output feature path collision detected for CpGPT import: "
                f"{output_path} maps to both {existing_hash} and {cache_hash}"
            )
        seen_output_paths[output_path] = cache_hash
        selected_hashes.append(cache_hash)

    selected_hashes.sort()
    if TOP_K_FEATURES is not None:
        selected_hashes = selected_hashes[: max(0, int(TOP_K_FEATURES))]
    return selected_hashes


def _write_unmatched_hashes(unmatched_hashes: list[str]) -> None:
    UNMATCHED_HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [hash_value.strip() for hash_value in unmatched_hashes if hash_value.strip()]
    UNMATCHED_HASHES_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    if not HESCAPEDNA_ROOT.exists():
        raise FileNotFoundError(f"hescapedna repo not found: {HESCAPEDNA_ROOT}")
    if not HESCAPEDNA_INDEX_DIR.exists():
        raise FileNotFoundError(f"hescapedna index dir not found: {HESCAPEDNA_INDEX_DIR}")
    if not HESCAPEDNA_CPGPT_CACHE_DIR.exists():
        raise FileNotFoundError(f"CpGPT cache dir not found: {HESCAPEDNA_CPGPT_CACHE_DIR}")

    index_paths = _selected_index_paths()
    if not index_paths:
        raise RuntimeError(f"No JSONL index files found under {HESCAPEDNA_INDEX_DIR}")

    print(f"hescapedna root: {HESCAPEDNA_ROOT}")
    print(f"Index files found: {len(index_paths)}")
    print(f"CpGPT cache dir: {HESCAPEDNA_CPGPT_CACHE_DIR}")
    print(f"Output features dir: {OUTPUT_FEATURES_DIR}")
    print(f"Allowed project ids: {ALLOWED_PROJECT_IDS if ALLOWED_PROJECT_IDS else ['ALL']}")
    print(f"Top-k features: {TOP_K_FEATURES if TOP_K_FEATURES is not None else 'ALL'}")

    print("Building CpGPT hash index from hescapedna metadata...")
    records_by_hash = build_cpgpt_hash_index(index_paths, raw_root=HESCAPEDNA_RAW_DNAM_ROOT)
    print(f"Indexed CpGPT hash entries: {len(records_by_hash)}")

    cache_paths = sorted(path for path in HESCAPEDNA_CPGPT_CACHE_DIR.glob("*.pt") if path.is_file())
    if not cache_paths:
        raise RuntimeError(f"No cached CpGPT feature files found under {HESCAPEDNA_CPGPT_CACHE_DIR}")
    print(f"Cached CpGPT feature files found: {len(cache_paths)}")

    selected_hashes = set(_filtered_records_by_hash(records_by_hash))
    cache_paths = [path for path in cache_paths if path.stem in selected_hashes]
    if not cache_paths:
        raise RuntimeError("No cached CpGPT feature files remain after applying filters.")

    unmatched_hashes = sorted(
        path.stem for path in HESCAPEDNA_CPGPT_CACHE_DIR.glob("*.pt") if path.is_file() and path.stem not in records_by_hash
    )
    _write_unmatched_hashes(unmatched_hashes)

    manifest_rows: list[dict[str, object]] = []
    copied_count = 0
    skipped_existing_count = 0

    loop = tqdm(cache_paths, total=len(cache_paths), desc="Importing CpGPT DNAm features", unit="file")
    for cache_path in loop:
        record = records_by_hash[cache_path.stem]
        output_path = build_cpgpt_output_path(OUTPUT_FEATURES_DIR, record)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existed_before = output_path.exists()
        if existed_before and not OVERWRITE_EXISTING:
            skipped_existing_count += 1
        else:
            shutil.copy2(cache_path, output_path)
            copied_count += 1

        manifest_rows.append(
            {
                "project_id": record.project_id,
                "case_submitter_id": record.case_submitter_id,
                "sample_submitter_id": record.sample_submitter_id,
                "beta_file_id": record.beta_file_id,
                "beta_file_name": record.beta_file_name,
                "beta_path": record.beta_path,
                "source_index_files": list(record.source_index_files),
                "cache_hash": record.cache_hash,
                "source_cache_path": cache_path.as_posix(),
                "feature_path": _relative_to_repo(output_path),
                "feature_filename": output_path.name,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(
        by=["project_id", "case_submitter_id", "sample_submitter_id", "beta_file_id", "feature_filename"],
        kind="stable",
    )
    MANIFEST_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_parquet(MANIFEST_PARQUET_PATH, index=False)
    if WRITE_CSV_MANIFEST:
        manifest_df.to_csv(MANIFEST_CSV_PATH, index=False)

    print("CpGPT DNAm feature import complete.")
    print(f"Manifest rows written: {len(manifest_df)}")
    print(f"Copied feature files: {copied_count}")
    print(f"Skipped existing feature files: {skipped_existing_count}")
    print(f"Unmatched cache hashes: {len(unmatched_hashes)}")
    if unmatched_hashes:
        print(f"Unmatched hash list written to: {UNMATCHED_HASHES_PATH}")
        print(f"First unmatched hash: {unmatched_hashes[0]}")
    print(f"Manifest parquet: {MANIFEST_PARQUET_PATH}")
    if WRITE_CSV_MANIFEST:
        print(f"Manifest csv: {MANIFEST_CSV_PATH}")


if __name__ == "__main__":
    main()
