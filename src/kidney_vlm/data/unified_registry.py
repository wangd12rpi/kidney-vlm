from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .registry_schema import empty_registry_frame


def _value_is_effectively_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _series_is_effectively_empty(series: pd.Series) -> bool:
    return bool(series.map(_value_is_effectively_empty).all())


def replace_source_slice(unified_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    source_name = str(source_name)
    source_rows = source_df.copy()
    source_rows["source"] = source_name

    if unified_df.empty:
        return source_rows

    kept = unified_df[unified_df["source"] != source_name].copy()
    stale_columns = [
        column
        for column in kept.columns
        if column not in source_rows.columns and _series_is_effectively_empty(kept[column])
    ]
    if stale_columns:
        kept = kept.drop(columns=stale_columns)
    if kept.empty:
        return source_rows
    return pd.concat([kept, source_rows], ignore_index=True, sort=False)


def source_row_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "source" not in df.columns:
        return {}
    counts = df["source"].astype(str).value_counts(dropna=False)
    return {str(source): int(count) for source, count in counts.items()}


def expected_source_row_counts_after_replace(
    unified_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    source_name: str,
) -> dict[str, int]:
    counts = source_row_counts(unified_df)
    counts[str(source_name)] = int(len(source_df))
    return counts


def initialize_if_missing(unified_df: pd.DataFrame | None) -> pd.DataFrame:
    if unified_df is None:
        return empty_registry_frame()
    return unified_df


@dataclass(frozen=True)
class RnaFeatureMergeReport:
    matched_registry_rows: int
    matched_cases: int
    skipped_existing_feature_rows: int
    updated_feature_rows: int
    unmatched_assignment_count: int


def _normalized_allowed_projects(allowed_project_ids: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    return {str(project_id).strip() for project_id in allowed_project_ids or [] if str(project_id).strip()}


def merge_case_level_rna_feature_paths(
    unified_df: pd.DataFrame,
    rna_case_assignments_df: pd.DataFrame,
    *,
    allowed_project_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    overwrite_existing: bool = True,
    clear_existing: bool = False,
) -> tuple[pd.DataFrame, RnaFeatureMergeReport]:
    """Merge case-level RNA feature paths into a unified registry copy.

    The join key mirrors the DNAm registration path: (project_id, patient_id).
    Only ``genomics_rna_bulk_feature_path`` is modified, preserving existing
    registry paths, split labels, and modality metadata.
    """
    required_assignment_columns = {"project_id", "patient_id", "genomics_rna_bulk_feature_path"}
    missing_assignment_columns = sorted(required_assignment_columns.difference(rna_case_assignments_df.columns))
    if missing_assignment_columns:
        raise ValueError(f"RNA case assignments missing required columns: {missing_assignment_columns}")

    required_registry_columns = {"source", "project_id", "patient_id", "genomics_rna_bulk_feature_path"}
    missing_registry_columns = sorted(required_registry_columns.difference(unified_df.columns))
    if missing_registry_columns:
        raise ValueError(f"Unified registry missing required columns: {missing_registry_columns}")

    allowed_projects = _normalized_allowed_projects(allowed_project_ids)
    working_df = unified_df.copy()
    working_df["genomics_rna_bulk_feature_path"] = (
        working_df["genomics_rna_bulk_feature_path"].fillna("").astype(str)
    )

    selected_registry_mask = working_df["source"].fillna("").astype(str).eq("tcga")
    if allowed_projects:
        selected_registry_mask = selected_registry_mask & working_df["project_id"].fillna("").astype(str).isin(allowed_projects)

    if clear_existing:
        working_df.loc[selected_registry_mask, "genomics_rna_bulk_feature_path"] = ""

    selected_assignments = rna_case_assignments_df.copy()
    if allowed_projects:
        selected_assignments = selected_assignments[
            selected_assignments["project_id"].fillna("").astype(str).isin(allowed_projects)
        ].copy()

    assignment_by_key = {
        (str(row.project_id).strip(), str(row.patient_id).strip()): str(
            getattr(row, "genomics_rna_bulk_feature_path", "") or ""
        ).strip()
        for row in selected_assignments.itertuples(index=False)
        if str(getattr(row, "genomics_rna_bulk_feature_path", "") or "").strip()
    }

    matched_registry_rows = 0
    matched_cases: set[tuple[str, str]] = set()
    skipped_existing_feature_rows = 0
    updated_feature_rows = 0

    for row_index in working_df.index.tolist():
        source = str(working_df.at[row_index, "source"]).strip()
        if source != "tcga":
            continue
        project_id = str(working_df.at[row_index, "project_id"]).strip()
        patient_id = str(working_df.at[row_index, "patient_id"]).strip()
        if allowed_projects and project_id not in allowed_projects:
            continue

        key = (project_id, patient_id)
        new_feature_path = assignment_by_key.get(key)
        if not new_feature_path:
            continue

        matched_registry_rows += 1
        matched_cases.add(key)

        current_feature_path = str(working_df.at[row_index, "genomics_rna_bulk_feature_path"] or "").strip()
        if current_feature_path and not overwrite_existing:
            skipped_existing_feature_rows += 1
            continue
        if current_feature_path != new_feature_path:
            working_df.at[row_index, "genomics_rna_bulk_feature_path"] = new_feature_path
            updated_feature_rows += 1

    report = RnaFeatureMergeReport(
        matched_registry_rows=matched_registry_rows,
        matched_cases=len(matched_cases),
        skipped_existing_feature_rows=skipped_existing_feature_rows,
        updated_feature_rows=updated_feature_rows,
        unmatched_assignment_count=len(assignment_by_key) - len(matched_cases),
    )
    return working_df, report
