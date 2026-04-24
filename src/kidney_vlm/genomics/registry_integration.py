from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PATH_COLUMNS_BY_MODALITY = {
    "dna_methylation_beta": "genomics_dna_methylation_paths",
    "copy_number_gene": "genomics_cnv_gene_paths",
    "copy_number_segment": "genomics_cnv_segment_paths",
    "mutation_maf": "genomics_mutation_paths",
    "mirna_expression": "genomics_mirna_paths",
}

METADATA_COLUMN_PREFIX_BY_MODALITY = {
    "dna_methylation_beta": "genomics_dna_methylation",
    "copy_number_gene": "genomics_cnv_gene",
    "copy_number_segment": "genomics_cnv_segment",
    "mutation_maf": "genomics_mutation",
    "mirna_expression": "genomics_mirna",
}

GENOMICS_JSON_COLUMNS = [
    "genomics_json_path",
    "genomics_teacher_text_path",
    "genomics_student_text_path",
    "genomics_json_errors",
    "genomics_clinical_text_path",
    "genomics_gdisc_text_path",
    "genomics_llm_input_text_path",
    "genomics_llm_input_json_path",
    "genomics_llm_input_errors",
    "genomics_available_modalities",
]


@dataclass(frozen=True)
class RegistryUpdateStats:
    matched_registry_rows: int
    updated_registry_rows: int
    unmatched_manifest_cases: int


def to_repo_relative_path(path_value: str | Path | None, *, repo_root: Path) -> str:
    if path_value is None:
        return ""
    try:
        if pd.isna(path_value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(path_value).strip()
    if not text:
        return ""
    if "://" in text:
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return Path(os.path.relpath(path.resolve(), start=repo_root.resolve())).as_posix()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _ensure_list_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        frame[column] = [[] for _ in range(len(frame))]
    else:
        frame[column] = frame[column].map(_as_list)


def _ensure_text_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        frame[column] = ""
    else:
        frame[column] = frame[column].fillna("").map(str)


def _metadata_columns_for_modality(modality: str) -> list[str]:
    prefix = METADATA_COLUMN_PREFIX_BY_MODALITY[modality]
    return [
        f"{prefix}_file_ids",
        f"{prefix}_file_names",
        f"{prefix}_sample_submitter_ids",
        f"{prefix}_workflow_types",
    ]


def ensure_extra_genomics_registry_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in PATH_COLUMNS_BY_MODALITY.values():
        _ensure_list_column(out, column)
    _ensure_list_column(out, "genomics_cnv_paths")
    for modality in METADATA_COLUMN_PREFIX_BY_MODALITY:
        for column in _metadata_columns_for_modality(modality):
            _ensure_list_column(out, column)
    for column in GENOMICS_JSON_COLUMNS:
        if column == "genomics_available_modalities":
            _ensure_list_column(out, column)
        else:
            _ensure_text_column(out, column)
    return out


def _values_for_rows(rows: pd.DataFrame, column: str) -> list[str]:
    if rows.empty or column not in rows.columns:
        return []
    return sorted({str(value).strip() for value in rows[column].tolist() if str(value).strip()})


def _manifest_index(manifest_df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    index: dict[tuple[str, str], pd.DataFrame] = {}
    if manifest_df.empty:
        return index
    for key, group in manifest_df.groupby(["project_id", "patient_id"], sort=True, dropna=False):
        project_id, patient_id = str(key[0]).strip(), str(key[1]).strip()
        if project_id and patient_id:
            index[(project_id, patient_id)] = group.copy()
    return index


def update_registry_with_extra_genomics_manifest(
    registry_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    *,
    repo_root: Path,
    source_name: str = "tcga",
    allowed_patient_ids: set[str] | None = None,
    clear_existing: bool = False,
) -> tuple[pd.DataFrame, RegistryUpdateStats]:
    out = ensure_extra_genomics_registry_columns(registry_df)
    manifest_by_case = _manifest_index(manifest_df)
    allowed_patient_ids = {str(value).strip() for value in allowed_patient_ids or set() if str(value).strip()}
    matched_keys: set[tuple[str, str]] = set()
    matched_rows = 0
    updated_rows = 0

    selected_source_mask = out["source"].fillna("").astype(str).eq(source_name)
    if allowed_patient_ids:
        selected_source_mask = selected_source_mask & out["patient_id"].fillna("").astype(str).isin(allowed_patient_ids)

    if clear_existing:
        list_columns = set(PATH_COLUMNS_BY_MODALITY.values())
        list_columns.add("genomics_cnv_paths")
        for modality in METADATA_COLUMN_PREFIX_BY_MODALITY:
            list_columns.update(_metadata_columns_for_modality(modality))
        for column in sorted(list_columns):
            out.loc[selected_source_mask, column] = [[] for _ in range(int(selected_source_mask.sum()))]

    for row_index in out.index[selected_source_mask]:
        project_id = str(out.at[row_index, "project_id"]).strip()
        patient_id = str(out.at[row_index, "patient_id"]).strip()
        case_manifest = manifest_by_case.get((project_id, patient_id))
        if case_manifest is None:
            continue
        matched_rows += 1
        matched_keys.add((project_id, patient_id))
        changed = False

        cnv_paths: list[str] = []
        available_modalities: list[str] = []
        for modality, path_column in PATH_COLUMNS_BY_MODALITY.items():
            modality_rows = case_manifest[case_manifest["modality"].fillna("").astype(str).eq(modality)]
            if modality_rows.empty:
                continue
            available_modalities.append(modality)
            paths = [
                to_repo_relative_path(value, repo_root=repo_root)
                for value in _values_for_rows(modality_rows, "output_path")
            ]
            paths = sorted({path for path in paths if path})
            if out.at[row_index, path_column] != paths:
                out.at[row_index, path_column] = paths
                changed = True
            if modality.startswith("copy_number_"):
                cnv_paths.extend(paths)

            metadata_values = {
                "file_ids": _values_for_rows(modality_rows, "file_id"),
                "file_names": _values_for_rows(modality_rows, "file_name"),
                "sample_submitter_ids": _values_for_rows(modality_rows, "sample_submitter_id"),
                "workflow_types": _values_for_rows(modality_rows, "workflow_type"),
            }
            prefix = METADATA_COLUMN_PREFIX_BY_MODALITY[modality]
            for suffix, values in metadata_values.items():
                column = f"{prefix}_{suffix}"
                if out.at[row_index, column] != values:
                    out.at[row_index, column] = values
                    changed = True

        cnv_paths = sorted({path for path in cnv_paths if path})
        if out.at[row_index, "genomics_cnv_paths"] != cnv_paths:
            out.at[row_index, "genomics_cnv_paths"] = cnv_paths
            changed = True
        if out.at[row_index, "genomics_available_modalities"] != sorted(available_modalities):
            out.at[row_index, "genomics_available_modalities"] = sorted(available_modalities)
            changed = True

        if changed:
            updated_rows += 1

    stats = RegistryUpdateStats(
        matched_registry_rows=matched_rows,
        updated_registry_rows=updated_rows,
        unmatched_manifest_cases=len(set(manifest_by_case) - matched_keys),
    )
    return out, stats


def update_registry_with_genomics_json_manifest(
    registry_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    *,
    repo_root: Path,
    source_name: str = "tcga",
    allowed_patient_ids: set[str] | None = None,
    overwrite_existing: bool = True,
) -> tuple[pd.DataFrame, RegistryUpdateStats]:
    out = ensure_extra_genomics_registry_columns(registry_df)
    manifest_by_case = _manifest_index(manifest_df)
    allowed_patient_ids = {str(value).strip() for value in allowed_patient_ids or set() if str(value).strip()}
    matched_keys: set[tuple[str, str]] = set()
    matched_rows = 0
    updated_rows = 0

    selected_source_mask = out["source"].fillna("").astype(str).eq(source_name)
    if allowed_patient_ids:
        selected_source_mask = selected_source_mask & out["patient_id"].fillna("").astype(str).isin(allowed_patient_ids)

    for row_index in out.index[selected_source_mask]:
        project_id = str(out.at[row_index, "project_id"]).strip()
        patient_id = str(out.at[row_index, "patient_id"]).strip()
        case_manifest = manifest_by_case.get((project_id, patient_id))
        if case_manifest is None or case_manifest.empty:
            continue
        row = case_manifest.iloc[0]
        matched_rows += 1
        matched_keys.add((project_id, patient_id))
        changed = False

        assignments = {
            "genomics_json_path": row.get("genomics_json_path", ""),
            "genomics_teacher_text_path": row.get("teacher_text_path", ""),
            "genomics_student_text_path": row.get("student_text_path", ""),
            "genomics_json_errors": row.get("errors", ""),
        }
        for column, raw_value in assignments.items():
            new_value = to_repo_relative_path(raw_value, repo_root=repo_root) if column.endswith("_path") else str(raw_value or "")
            current = str(out.at[row_index, column] or "").strip()
            if current and not overwrite_existing:
                continue
            if current != new_value:
                out.at[row_index, column] = new_value
                changed = True

        available = _as_list(row.get("available_modalities", []))
        if available and out.at[row_index, "genomics_available_modalities"] != available:
            out.at[row_index, "genomics_available_modalities"] = available
            changed = True

        if changed:
            updated_rows += 1

    stats = RegistryUpdateStats(
        matched_registry_rows=matched_rows,
        updated_registry_rows=updated_rows,
        unmatched_manifest_cases=len(set(manifest_by_case) - matched_keys),
    )
    return out, stats


def update_registry_with_llm_input_context_manifest(
    registry_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    *,
    repo_root: Path,
    source_name: str = "tcga",
    allowed_patient_ids: set[str] | None = None,
    overwrite_existing: bool = True,
) -> tuple[pd.DataFrame, RegistryUpdateStats]:
    out = ensure_extra_genomics_registry_columns(registry_df)
    manifest_by_case = _manifest_index(manifest_df)
    allowed_patient_ids = {str(value).strip() for value in allowed_patient_ids or set() if str(value).strip()}
    matched_keys: set[tuple[str, str]] = set()
    matched_rows = 0
    updated_rows = 0

    selected_source_mask = out["source"].fillna("").astype(str).eq(source_name)
    if allowed_patient_ids:
        selected_source_mask = selected_source_mask & out["patient_id"].fillna("").astype(str).isin(allowed_patient_ids)

    for row_index in out.index[selected_source_mask]:
        project_id = str(out.at[row_index, "project_id"]).strip()
        patient_id = str(out.at[row_index, "patient_id"]).strip()
        case_manifest = manifest_by_case.get((project_id, patient_id))
        if case_manifest is None or case_manifest.empty:
            continue
        row = case_manifest.iloc[0]
        matched_rows += 1
        matched_keys.add((project_id, patient_id))
        changed = False

        assignments = {
            "genomics_clinical_text_path": row.get("clinical_text_path", ""),
            "genomics_gdisc_text_path": row.get("gdisc_text_path", ""),
            "genomics_llm_input_text_path": row.get("llm_input_text_path", ""),
            "genomics_llm_input_json_path": row.get("llm_input_json_path", ""),
            "genomics_llm_input_errors": row.get("errors", ""),
        }
        for column, raw_value in assignments.items():
            if column.endswith("_path"):
                new_value = to_repo_relative_path(raw_value, repo_root=repo_root)
            else:
                new_value = str(raw_value or "")
            current = str(out.at[row_index, column] or "").strip()
            if current and not overwrite_existing:
                continue
            if current != new_value:
                out.at[row_index, column] = new_value
                changed = True

        available: list[str] = []
        if bool(row.get("mutation_available", False)):
            available.append("mutation_maf")
        if bool(row.get("copy_number_gene_available", False)):
            available.append("copy_number_gene")
        if bool(row.get("copy_number_segment_available", False)):
            available.append("copy_number_segment")
        if available and out.at[row_index, "genomics_available_modalities"] != available:
            out.at[row_index, "genomics_available_modalities"] = available
            changed = True

        if changed:
            updated_rows += 1

    stats = RegistryUpdateStats(
        matched_registry_rows=matched_rows,
        updated_registry_rows=updated_rows,
        unmatched_manifest_cases=len(set(manifest_by_case) - matched_keys),
    )
    return out, stats
