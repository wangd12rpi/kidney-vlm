#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.pathology.feature_registry import register_existing_pathology_features
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


# Input/output locations
REGISTRY_PATH = ROOT / "data" / "registry" / "unified.parquet"
PATCH_FEATURES_DIR = ROOT / "data" / "features" / "features_uni"

# UNI metadata
SAVE_FORMAT = "h5"
PATCH_SIZE = 256
TARGET_MAGNIFICATION = 20

# Patch-count fallback
# UNI H5 files already contain both `features` and `coords`, so this can stay
# as a placeholder path; the registry utility will fall back to the H5 itself.
COORDS_ROOT = ROOT / "data" / "features" / "coords_uni_unused"

# Selection
# Empty means all projects in the registry.
ALLOWED_PROJECT_IDS: list[str] = []

# Registry behavior
CLEAR_EXISTING_PATHOLOGY_PATCH_EMBEDDINGS_BEFORE_REGISTER = True


def _normalized_string_list(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _count_cases_with_patch_embeddings(frame) -> int:
    def _has_paths(value) -> bool:
        if value is None:
            return False
        if hasattr(value, "tolist") and not isinstance(value, str):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return any(str(item).strip() for item in value)
        return bool(str(value).strip())

    if "pathology_tile_embedding_paths" not in frame.columns:
        return 0
    return int(frame["pathology_tile_embedding_paths"].map(_has_paths).sum())


def _clear_patch_embedding_fields(frame):
    out = frame.copy()
    out["pathology_tile_embedding_paths"] = [[] for _ in range(len(out))]
    if "pathology_tile_embedding_patch_counts" in out.columns:
        out["pathology_tile_embedding_patch_counts"] = [[] for _ in range(len(out))]
    if "pathology_embedding_patch_size" in out.columns:
        out["pathology_embedding_patch_size"] = None
    if "pathology_embedding_magnification" in out.columns:
        out["pathology_embedding_magnification"] = None
    return out


def main() -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Unified registry not found: {REGISTRY_PATH}")
    if not PATCH_FEATURES_DIR.exists():
        raise FileNotFoundError(f"UNI features dir not found: {PATCH_FEATURES_DIR}")

    registry_df = read_parquet_or_empty(REGISTRY_PATH)
    if registry_df.empty:
        raise RuntimeError(f"Unified registry is empty: {REGISTRY_PATH}")

    allowed_project_ids = _normalized_string_list(ALLOWED_PROJECT_IDS)
    if allowed_project_ids and "project_id" in registry_df.columns:
        selected_registry_df = registry_df[registry_df["project_id"].astype(str).isin(allowed_project_ids)].copy()
    else:
        selected_registry_df = registry_df.copy()

    if selected_registry_df.empty:
        raise RuntimeError("No registry rows remain after applying ALLOWED_PROJECT_IDS.")

    print(f"Registry path: {REGISTRY_PATH}")
    print(f"UNI features dir: {PATCH_FEATURES_DIR}")
    print(f"Save format: {SAVE_FORMAT}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Target magnification: {TARGET_MAGNIFICATION}")
    print(f"Allowed project ids: {allowed_project_ids if allowed_project_ids else ['ALL']}")
    print(f"Rows selected: {len(selected_registry_df)}")
    print(f"Rows with patch embeddings before: {_count_cases_with_patch_embeddings(selected_registry_df)}")

    feature_paths = sorted(PATCH_FEATURES_DIR.glob(f"*.{SAVE_FORMAT}"))
    print(f"UNI feature files found: {len(feature_paths)}")
    preview = [path.name for path in feature_paths[:5]]
    if preview:
        print("Sample UNI feature files:")
        for name in tqdm(preview, total=len(preview), desc="Preview", unit="file", leave=False):
            print(f"  - {name}")

    working_df = selected_registry_df
    if CLEAR_EXISTING_PATHOLOGY_PATCH_EMBEDDINGS_BEFORE_REGISTER:
        print("Clearing existing pathology patch embedding fields before UNI registration...")
        working_df = _clear_patch_embedding_fields(working_df)

    print("Registering existing UNI feature files into the registry...")
    updated_selected_df, stats = register_existing_pathology_features(
        working_df,
        patch_features_dir=PATCH_FEATURES_DIR,
        coords_root=COORDS_ROOT,
        save_format=SAVE_FORMAT,
        patch_size=int(PATCH_SIZE),
        target_mag=int(TARGET_MAGNIFICATION),
        root_dir=ROOT,
        progress=True,
    )

    final_registry_df = registry_df.copy()
    for column in updated_selected_df.columns:
        if column not in final_registry_df.columns:
            final_registry_df[column] = None
        try:
            final_registry_df.loc[updated_selected_df.index, column] = updated_selected_df[column]
        except (TypeError, ValueError):
            final_registry_df[column] = final_registry_df[column].astype("object")
            final_registry_df.loc[updated_selected_df.index, column] = updated_selected_df[column].astype("object")
    write_registry_parquet(final_registry_df, REGISTRY_PATH, validate=True)

    print("UNI registry insertion complete.")
    print(f"Cases scanned: {stats.cases_scanned}")
    print(f"Cases with slide paths: {stats.cases_with_slide_paths}")
    print(f"Cases with matched features: {stats.cases_with_matches}")
    print(f"Matched feature paths written: {stats.matched_feature_paths}")
    print(f"Feature files indexed: {stats.feature_files_indexed}")
    print(f"Invalid feature files skipped: {stats.invalid_feature_files}")
    print(f"Rows with patch embeddings after: {_count_cases_with_patch_embeddings(updated_selected_df)}")


if __name__ == "__main__":
    main()
