#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.dnam_feature_import import build_case_level_dnam_assignments
from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


# Input/output locations
REGISTRY_PATH = ROOT / "data" / "registry" / "unified.parquet"
DNAM_MANIFEST_PATH = ROOT / "data" / "features" / "features_cpgpt_dnam_manifest.parquet"

# Selection
# Empty means all TCGA projects present in the manifest.
ALLOWED_PROJECT_IDS: list[str] = []

# Registry behavior
CLEAR_EXISTING_DNAM_FIELDS_BEFORE_REGISTER = False
OVERWRITE_EXISTING_DNAM_FEATURE_PATH = True


def _normalized_string_list(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _count_cases_with_dnam_feature_path(frame) -> int:
    if "genomics_dna_methylation_feature_path" not in frame.columns:
        return 0
    return int(frame["genomics_dna_methylation_feature_path"].fillna("").astype(str).str.strip().ne("").sum())


def _clear_dnam_fields(frame):
    out = frame.copy()
    out["genomics_dna_methylation_paths"] = [[] for _ in range(len(out))]
    out["genomics_dna_methylation_feature_path"] = ""
    return out


def main() -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Unified registry not found: {REGISTRY_PATH}")
    if not DNAM_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Dnam manifest not found: {DNAM_MANIFEST_PATH}")

    registry_df = read_parquet_or_empty(REGISTRY_PATH)
    if registry_df.empty:
        raise RuntimeError(f"Unified registry is empty: {REGISTRY_PATH}")

    manifest_df = pd.read_parquet(DNAM_MANIFEST_PATH)
    if manifest_df.empty:
        raise RuntimeError(f"Dnam manifest is empty: {DNAM_MANIFEST_PATH}")

    allowed_project_ids = _normalized_string_list(ALLOWED_PROJECT_IDS)
    selected_manifest_df = manifest_df.copy()
    if allowed_project_ids:
        selected_manifest_df = selected_manifest_df[
            selected_manifest_df["project_id"].fillna("").astype(str).isin(allowed_project_ids)
        ].copy()
    if selected_manifest_df.empty:
        raise RuntimeError("No DNAm manifest rows remain after applying ALLOWED_PROJECT_IDS.")

    case_assignments_df = build_case_level_dnam_assignments(selected_manifest_df)
    if case_assignments_df.empty:
        raise RuntimeError("No case-level DNAm assignments could be built from the manifest.")

    print(f"Registry path: {REGISTRY_PATH}")
    print(f"DNAm manifest path: {DNAM_MANIFEST_PATH}")
    print(f"Allowed project ids: {allowed_project_ids if allowed_project_ids else ['ALL']}")
    print(f"Manifest rows selected: {len(selected_manifest_df)}")
    print(f"Case-level assignments: {len(case_assignments_df)}")
    print(f"Rows with DNAm feature path before: {_count_cases_with_dnam_feature_path(registry_df)}")

    selected_registry_mask = registry_df["source"].fillna("").astype(str).eq("tcga")
    if allowed_project_ids and "project_id" in registry_df.columns:
        selected_registry_mask = selected_registry_mask & registry_df["project_id"].fillna("").astype(str).isin(allowed_project_ids)

    working_registry_df = registry_df.copy()
    if CLEAR_EXISTING_DNAM_FIELDS_BEFORE_REGISTER:
        print("Clearing existing DNAm fields on selected TCGA registry rows before registration...")
        working_registry_df.loc[selected_registry_mask, "genomics_dna_methylation_paths"] = [
            [] for _ in range(int(selected_registry_mask.sum()))
        ]
        working_registry_df.loc[selected_registry_mask, "genomics_dna_methylation_feature_path"] = ""

    assignment_by_key = {
        (str(row.project_id).strip(), str(row.patient_id).strip()): row
        for row in case_assignments_df.itertuples(index=False)
    }

    matched_registry_rows = 0
    matched_cases = set()
    skipped_existing_feature_rows = 0
    updated_raw_path_rows = 0

    loop = tqdm(
        working_registry_df.index.tolist(),
        total=len(working_registry_df),
        desc="Registering CpGPT DNAm into registry",
        unit="row",
    )
    for row_index in loop:
        source = str(working_registry_df.at[row_index, "source"]).strip()
        if source != "tcga":
            continue
        project_id = str(working_registry_df.at[row_index, "project_id"]).strip()
        patient_id = str(working_registry_df.at[row_index, "patient_id"]).strip()
        if allowed_project_ids and project_id not in allowed_project_ids:
            continue

        assignment = assignment_by_key.get((project_id, patient_id))
        if assignment is None:
            continue

        matched_registry_rows += 1
        matched_cases.add((project_id, patient_id))

        raw_paths = list(getattr(assignment, "genomics_dna_methylation_paths"))
        current_raw_paths = working_registry_df.at[row_index, "genomics_dna_methylation_paths"]
        if current_raw_paths != raw_paths:
            working_registry_df.at[row_index, "genomics_dna_methylation_paths"] = raw_paths
            updated_raw_path_rows += 1

        current_feature_path = str(working_registry_df.at[row_index, "genomics_dna_methylation_feature_path"] or "").strip()
        new_feature_path = str(getattr(assignment, "genomics_dna_methylation_feature_path") or "").strip()
        if current_feature_path and not OVERWRITE_EXISTING_DNAM_FEATURE_PATH:
            skipped_existing_feature_rows += 1
            continue
        if current_feature_path != new_feature_path:
            working_registry_df.at[row_index, "genomics_dna_methylation_feature_path"] = new_feature_path

    unmatched_assignment_count = len(assignment_by_key) - len(matched_cases)

    write_registry_parquet(working_registry_df, REGISTRY_PATH, validate=True)

    print("DNAm registry insertion complete.")
    print(f"Matched registry rows updated: {matched_registry_rows}")
    print(f"Matched TCGA cases: {len(matched_cases)}")
    print(f"Raw-path rows updated: {updated_raw_path_rows}")
    print(f"Existing feature rows skipped: {skipped_existing_feature_rows}")
    print(f"Unmatched manifest cases: {unmatched_assignment_count}")
    print(f"Rows with DNAm feature path after: {_count_cases_with_dnam_feature_path(working_registry_df)}")


if __name__ == "__main__":
    main()
