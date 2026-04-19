from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd

CORE_COLUMNS = [
    "sample_id",
    "source",
    "patient_id",
    "study_id",
    "split",
    "genomics_rna_bulk_paths",
    "genomics_rna_bulk_feature_path",
    "genomics_rna_bulk_file_ids",
    "genomics_rna_bulk_file_names",
    "genomics_rna_bulk_sample_types",
    "genomics_rna_bulk_workflow_types",
    "genomics_rna_bulk_molecular_subtype",
    "genomics_rna_bulk_subtype_mrna",
    "genomics_dna_methylation_subtype",
    "genomics_integrative_subtype",
    "genomics_msi_status",
    "genomics_rna_bulk_leukocyte_fraction",
    "genomics_rna_bulk_tumor_purity",
    "genomics_aneuploidy_score",
    "genomics_hrd_score",
    "genomics_rna_bulk_top_immune_cell_types",
    "genomics_rna_bulk_top_immune_cell_fractions",
    "genomics_dna_methylation_paths",
    "genomics_dna_methylation_feature_path",
    "genomics_cnv_paths",
    "genomics_cnv_feature_path",
    "pathology_wsi_paths",
    "radiology_image_paths",
    "radiology_image_modalities",
    "radiology_report_download_paths",
    "radiology_report_uri_paths",
    "radiology_report_series_descriptions",
    "pathology_mask_paths",
    "pathology_segmentation_slide_image_paths",
    "pathology_segmentation_overlay_paths",
    "pathology_segmentation_metadata_paths",
    "radiology_mask_paths",
    "pathology_tile_embedding_paths",
    "pathology_slide_embedding_paths",
    "radiology_embedding_paths",
    "biomarkers_text",
    "question",
    "answer",
]

LIST_COLUMNS = [
    "genomics_rna_bulk_paths",
    "genomics_rna_bulk_file_ids",
    "genomics_rna_bulk_file_names",
    "genomics_rna_bulk_sample_types",
    "genomics_rna_bulk_workflow_types",
    "genomics_rna_bulk_top_immune_cell_types",
    "genomics_rna_bulk_top_immune_cell_fractions",
    "genomics_dna_methylation_paths",
    "genomics_cnv_paths",
    "pathology_wsi_paths",
    "radiology_image_paths",
    "radiology_image_modalities",
    "radiology_report_download_paths",
    "radiology_report_uri_paths",
    "radiology_report_series_descriptions",
    "pathology_mask_paths",
    "pathology_segmentation_slide_image_paths",
    "pathology_segmentation_overlay_paths",
    "pathology_segmentation_metadata_paths",
    "radiology_mask_paths",
    "pathology_tile_embedding_paths",
    "pathology_slide_embedding_paths",
    "radiology_embedding_paths",
]

OPTIONAL_LIST_COLUMNS = [
    "radiology_mask_manifest_paths",
    "radiology_png_dirs",
    "radiology_download_paths",
]

OPTIONAL_INT_LIST_COLUMNS = [
    "radiology_series_slice_counts",
]

TEXT_COLUMNS = [
    "sample_id",
    "source",
    "patient_id",
    "study_id",
    "split",
    "genomics_rna_bulk_feature_path",
    "genomics_rna_bulk_molecular_subtype",
    "genomics_rna_bulk_subtype_mrna",
    "genomics_dna_methylation_subtype",
    "genomics_integrative_subtype",
    "genomics_msi_status",
    "genomics_rna_bulk_leukocyte_fraction",
    "genomics_rna_bulk_tumor_purity",
    "genomics_aneuploidy_score",
    "genomics_hrd_score",
    "genomics_dna_methylation_feature_path",
    "genomics_cnv_feature_path",
    "biomarkers_text",
    "question",
    "answer",
]


@dataclass(frozen=True)
class RegistrySchema:
    required_columns: tuple[str, ...] = tuple(CORE_COLUMNS)


def empty_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CORE_COLUMNS)


def _parse_serialized_list(text: str) -> list[str] | None:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None

    quoted_matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", stripped)
    recovered = [left or right for left, right in quoted_matches if (left or right).strip()]

    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        if recovered or not stripped[1:-1].strip():
            return recovered
        return None

    if isinstance(parsed, (list, tuple)):
        normalized: list[str] = []
        for item in parsed:
            normalized.extend(_normalize_list_value(item))
        if len(recovered) > len(normalized):
            return recovered
        return normalized
    return None


def _normalize_list_value(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            normalized.extend(_normalize_list_value(item))
        return normalized
    if isinstance(value, tuple):
        normalized: list[str] = []
        for item in value:
            normalized.extend(_normalize_list_value(item))
        return normalized
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            normalized: list[str] = []
            for item in converted:
                normalized.extend(_normalize_list_value(item))
            return normalized
    text = str(value).strip()
    if not text:
        return []
    parsed = _parse_serialized_list(text)
    if parsed is not None:
        return parsed
    return [text]


def _normalize_int_list_value(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        normalized: list[int] = []
        for item in value:
            normalized.extend(_normalize_int_list_value(item))
        return normalized
    if isinstance(value, tuple):
        normalized: list[int] = []
        for item in value:
            normalized.extend(_normalize_int_list_value(item))
        return normalized
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            normalized: list[int] = []
            for item in converted:
                normalized.extend(_normalize_int_list_value(item))
            return normalized
    text = str(value).strip()
    if not text:
        return []
    parsed = _parse_serialized_list(text)
    if parsed is not None:
        normalized: list[int] = []
        for item in parsed:
            try:
                normalized.append(int(float(str(item).strip())))
            except ValueError:
                continue
        return normalized
    try:
        return [int(float(text))]
    except ValueError:
        return []


def _default_for_column(column: str) -> object:
    if column in LIST_COLUMNS or column in OPTIONAL_LIST_COLUMNS or column in OPTIONAL_INT_LIST_COLUMNS:
        return []
    return ""


def ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in CORE_COLUMNS:
        if column not in out.columns:
            default_value = _default_for_column(column)
            if isinstance(default_value, list):
                out[column] = [list(default_value) for _ in range(len(out))]
            else:
                out[column] = default_value
    ordered = CORE_COLUMNS + [c for c in out.columns if c not in CORE_COLUMNS]
    return out[ordered]


def normalize_registry_df(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_core_columns(df)
    for column in LIST_COLUMNS:
        out[column] = out[column].map(_normalize_list_value)
    for column in OPTIONAL_LIST_COLUMNS:
        if column in out.columns:
            out[column] = out[column].map(_normalize_list_value)
    for column in OPTIONAL_INT_LIST_COLUMNS:
        if column in out.columns:
            out[column] = out[column].map(_normalize_int_list_value)
    for column in TEXT_COLUMNS:
        out[column] = out[column].fillna("").map(str)
    return out


def validate_registry_df(df: pd.DataFrame, required_columns: Iterable[str] = CORE_COLUMNS) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Registry is missing required columns: {missing}")

    for column in LIST_COLUMNS:
        if column not in df.columns:
            continue
        invalid = [idx for idx, value in enumerate(df[column].tolist()) if not isinstance(value, list)]
        if invalid:
            raise ValueError(f"Column '{column}' must contain lists. Invalid row indices: {invalid[:10]}")
    for column in OPTIONAL_LIST_COLUMNS:
        if column not in df.columns:
            continue
        invalid = [idx for idx, value in enumerate(df[column].tolist()) if not isinstance(value, list)]
        if invalid:
            raise ValueError(f"Column '{column}' must contain lists. Invalid row indices: {invalid[:10]}")
    for column in OPTIONAL_INT_LIST_COLUMNS:
        if column not in df.columns:
            continue
        invalid = [idx for idx, value in enumerate(df[column].tolist()) if not isinstance(value, list)]
        if invalid:
            raise ValueError(f"Column '{column}' must contain lists. Invalid row indices: {invalid[:10]}")
