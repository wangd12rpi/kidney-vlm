#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.rna_feature_import import build_case_level_rna_assignments
from kidney_vlm.data.unified_registry import merge_case_level_rna_feature_paths
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


# Input/output locations
REGISTRY_PATH = ROOT / "data" / "registry" / "unified.parquet"
RNA_MANIFEST_PATH = ROOT / "data" / "features" / "features_bulkformer_rna_manifest.parquet"

# Selection
# Empty means all TCGA projects present in the manifest.
ALLOWED_PROJECT_IDS: list[str] = []

# Registry behavior
CLEAR_EXISTING_RNA_FEATURE_PATH_BEFORE_REGISTER = False
OVERWRITE_EXISTING_RNA_FEATURE_PATH = True
REQUIRE_EXISTING_FEATURE_FILES = True


def _normalized_string_list(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _count_cases_with_rna_feature_path(frame: pd.DataFrame) -> int:
    if "genomics_rna_bulk_feature_path" not in frame.columns:
        return 0
    return int(frame["genomics_rna_bulk_feature_path"].fillna("").astype(str).str.strip().ne("").sum())


def _backup_registry(registry_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = registry_path.with_name(f"{registry_path.name}.bak.{timestamp}")
    suffix = 1
    while backup_path.exists():
        backup_path = registry_path.with_name(f"{registry_path.name}.bak.{timestamp}.{suffix}")
        suffix += 1
    shutil.copy2(registry_path, backup_path)
    return backup_path


def _resolve_feature_path(path_value: str) -> Path:
    path = Path(str(path_value).strip()).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _filter_to_existing_features(case_assignments_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    exists_mask = case_assignments_df["genomics_rna_bulk_feature_path"].map(
        lambda value: _resolve_feature_path(str(value)).exists()
    )
    missing_count = int((~exists_mask).sum())
    return case_assignments_df[exists_mask].copy(), missing_count


def _assert_only_rna_feature_path_changed(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    if before_df.shape != after_df.shape:
        raise RuntimeError(f"Registry shape changed unexpectedly: before={before_df.shape}, after={after_df.shape}")
    before_cols = before_df.columns.tolist()
    after_cols = after_df.columns.tolist()
    if before_cols != after_cols:
        raise RuntimeError("Registry columns changed unexpectedly while registering RNA feature paths.")

    protected_columns = [column for column in before_cols if column != "genomics_rna_bulk_feature_path"]
    before_protected = before_df[protected_columns].reset_index(drop=True)
    after_protected = after_df[protected_columns].reset_index(drop=True)
    if not before_protected.equals(after_protected):
        changed = [
            column
            for column in protected_columns
            if not before_protected[column].equals(after_protected[column])
        ]
        raise RuntimeError(f"Unexpected registry columns changed during RNA registration: {changed[:20]}")


def _verify_populated_feature_paths(frame: pd.DataFrame) -> int:
    populated = frame["genomics_rna_bulk_feature_path"].fillna("").astype(str).str.strip()
    populated = populated[populated.ne("")]
    missing = [path for path in populated.tolist() if not _resolve_feature_path(path).exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"RNA feature paths missing on disk: {len(missing)}. First missing: {preview}")
    return len(populated)


def _assert_split_columns_unchanged(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    for column in ("split", "split_group_id", "split_scheme_version"):
        if column in before_df.columns or column in after_df.columns:
            if column not in before_df.columns or column not in after_df.columns:
                raise RuntimeError(f"Split column presence changed unexpectedly: {column}")
            if not before_df[column].reset_index(drop=True).equals(after_df[column].reset_index(drop=True)):
                raise RuntimeError(f"Split column changed unexpectedly during RNA registration: {column}")


def main() -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Unified registry not found: {REGISTRY_PATH}")
    if not RNA_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"RNA manifest not found: {RNA_MANIFEST_PATH}. "
            "Run scripts/04_rna_features/01_extract_bulkformer_tcga_rna_features.py first."
        )

    registry_df = read_parquet_or_empty(REGISTRY_PATH)
    if registry_df.empty:
        raise RuntimeError(f"Unified registry is empty: {REGISTRY_PATH}")

    manifest_df = pd.read_parquet(RNA_MANIFEST_PATH)
    if manifest_df.empty:
        raise RuntimeError(f"RNA manifest is empty: {RNA_MANIFEST_PATH}")

    allowed_project_ids = _normalized_string_list(ALLOWED_PROJECT_IDS)
    selected_manifest_df = manifest_df.copy()
    if allowed_project_ids:
        selected_manifest_df = selected_manifest_df[
            selected_manifest_df["project_id"].fillna("").astype(str).isin(allowed_project_ids)
        ].copy()
    if selected_manifest_df.empty:
        raise RuntimeError("No RNA manifest rows remain after applying ALLOWED_PROJECT_IDS.")

    case_assignments_df = build_case_level_rna_assignments(selected_manifest_df)
    if case_assignments_df.empty:
        raise RuntimeError("No case-level RNA assignments could be built from the manifest.")

    missing_feature_count = 0
    if REQUIRE_EXISTING_FEATURE_FILES:
        case_assignments_df, missing_feature_count = _filter_to_existing_features(case_assignments_df)
        if case_assignments_df.empty:
            raise RuntimeError("No case-level RNA assignments have feature files that exist on disk.")

    print(f"Registry path: {REGISTRY_PATH}")
    print(f"RNA manifest path: {RNA_MANIFEST_PATH}")
    print(f"Allowed project ids: {allowed_project_ids if allowed_project_ids else ['ALL']}")
    print(f"Manifest rows selected: {len(selected_manifest_df)}")
    print(f"Case-level assignments: {len(case_assignments_df)}")
    print(f"Missing feature files skipped: {missing_feature_count}")
    print(f"Rows with RNA feature path before: {_count_cases_with_rna_feature_path(registry_df)}")

    working_registry_df, report = merge_case_level_rna_feature_paths(
        registry_df,
        case_assignments_df,
        allowed_project_ids=allowed_project_ids,
        overwrite_existing=OVERWRITE_EXISTING_RNA_FEATURE_PATH,
        clear_existing=CLEAR_EXISTING_RNA_FEATURE_PATH_BEFORE_REGISTER,
    )

    _assert_only_rna_feature_path_changed(registry_df, working_registry_df)
    _assert_split_columns_unchanged(registry_df, working_registry_df)
    populated_count = _verify_populated_feature_paths(working_registry_df)

    backup_path = _backup_registry(REGISTRY_PATH)
    write_registry_parquet(working_registry_df, REGISTRY_PATH, validate=True)

    print("BulkFormer RNA registry insertion complete.")
    print(f"Backup written: {backup_path}")
    print(f"Matched registry rows: {report.matched_registry_rows}")
    print(f"Matched TCGA cases: {report.matched_cases}")
    print(f"Feature rows updated: {report.updated_feature_rows}")
    print(f"Existing feature rows skipped: {report.skipped_existing_feature_rows}")
    print(f"Unmatched manifest cases: {report.unmatched_assignment_count}")
    print(f"Rows with RNA feature path after: {populated_count}")


if __name__ == "__main__":
    main()
