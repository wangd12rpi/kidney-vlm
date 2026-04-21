from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from kidney_vlm.data.id_factory import make_sample_id
from kidney_vlm.data.registry_schema import CORE_COLUMNS, empty_registry_frame, normalize_registry_df
from kidney_vlm.data.sources.tcga import (
    APIQueryError,
    GDCClient,
    TCIAClient,
    _coalesce_text,
    _extract_text_values,
    _first_diagnosis,
    _first_non_empty,
    _linked_case_sample_metadata,
    _to_float_or_none,
    _to_project_relative_path,
    _unique_sorted_non_empty,
    normalize_tcia_modality,
    normalize_tcia_modality_list,
)


DEFAULT_CPTAC_CASE_FIELDS = [
    "case_id",
    "submitter_id",
    "project.project_id",
    "primary_site",
    "disease_type",
    "diagnoses.primary_diagnosis",
    "diagnoses.tumor_grade",
    "diagnoses.tumor_stage",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.age_at_diagnosis",
    "diagnoses.morphology",
    "diagnoses.last_known_disease_status",
    "diagnoses.days_to_last_known_disease_status",
    "diagnoses.days_to_recurrence",
    "diagnoses.vital_status",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_death",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.year_of_birth",
]

DEFAULT_CPTAC_FILE_FIELDS = [
    "file_id",
    "file_name",
    "data_category",
    "data_type",
    "data_format",
    "experimental_strategy",
    "analysis.workflow_type",
    "access",
    "file_size",
    "md5sum",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.primary_site",
    "cases.disease_type",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
]

DEFAULT_CPTAC_SPLIT_SCHEME_VERSION = "cptac_external_test_v1"


def _normalized_list(values: Any) -> list[str]:
    if values is None:
        return []
    out: list[str] = []
    for value in list(values or []):
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def _primary_sites_by_group(cancer_groups: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        str(group.get("name", "")).strip(): _normalized_list(group.get("primary_sites", []))
        for group in cancer_groups
        if str(group.get("name", "")).strip()
    }


def _collections_by_group(cancer_groups: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        str(group.get("name", "")).strip(): _normalized_list(group.get("tcia_collections", []))
        for group in cancer_groups
        if str(group.get("name", "")).strip()
    }


def cptac_case_filter(primary_sites: list[str]) -> dict[str, Any]:
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "project.program.name",
                    "value": ["CPTAC"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "primary_site",
                    "value": primary_sites,
                },
            },
        ],
    }


def cptac_file_filter(
    *,
    primary_sites: list[str],
    data_categories: list[str] | None = None,
    data_types: list[str] | None = None,
    data_formats: list[str] | None = None,
    experimental_strategies: list[str] | None = None,
    workflow_types: list[str] | None = None,
    access: list[str] | None = None,
    sample_types: list[str] | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [
        {
            "op": "in",
            "content": {
                "field": "cases.project.program.name",
                "value": ["CPTAC"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "cases.primary_site",
                "value": primary_sites,
            },
        },
    ]
    optional_filters = [
        ("data_category", data_categories),
        ("data_type", data_types),
        ("data_format", data_formats),
        ("experimental_strategy", experimental_strategies),
        ("analysis.workflow_type", workflow_types),
        ("access", access),
        ("cases.samples.sample_type", sample_types),
    ]
    for field_name, values in optional_filters:
        normalized = _normalized_list(values)
        if normalized:
            filters.append(
                {
                    "op": "in",
                    "content": {
                        "field": field_name,
                        "value": normalized,
                    },
                }
            )
    return {"op": "and", "content": filters}


def fetch_cptac_cases(
    gdc_client: GDCClient,
    *,
    primary_sites: list[str],
    max_cases: int | None = None,
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    payload = {
        "filters": cptac_case_filter(primary_sites),
        "fields": ",".join(fields or DEFAULT_CPTAC_CASE_FIELDS),
        "sort": "submitter_id:asc",
    }
    return gdc_client._post_hits("cases", payload, max_records=max_cases)


def fetch_cptac_files(
    gdc_client: GDCClient,
    *,
    primary_sites: list[str],
    data_categories: list[str] | None = None,
    data_types: list[str] | None = None,
    data_formats: list[str] | None = None,
    experimental_strategies: list[str] | None = None,
    workflow_types: list[str] | None = None,
    access: list[str] | None = None,
    sample_types: list[str] | None = None,
    max_files: int | None = None,
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    payload = {
        "filters": cptac_file_filter(
            primary_sites=primary_sites,
            data_categories=data_categories,
            data_types=data_types,
            data_formats=data_formats,
            experimental_strategies=experimental_strategies,
            workflow_types=workflow_types,
            access=access,
            sample_types=sample_types,
        ),
        "fields": ",".join(fields or DEFAULT_CPTAC_FILE_FIELDS),
        "sort": "file_name:asc",
    }
    return gdc_client._post_hits("files", payload, max_records=max_files)


def _file_sample_types(file_hit: dict[str, Any]) -> set[str]:
    sample_types: set[str] = set()
    for linked_case in file_hit.get("cases", []) or []:
        if not isinstance(linked_case, dict):
            continue
        _sample_ids, linked_sample_types = _linked_case_sample_metadata(file_hit, linked_case=linked_case)
        for sample_type in linked_sample_types:
            if str(sample_type).strip():
                sample_types.add(str(sample_type).strip())
    return sample_types


def _index_files_by_case_and_patient(file_hits: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for file_hit in file_hits:
        linked_cases = file_hit.get("cases", [])
        if not isinstance(linked_cases, list):
            continue
        for linked_case in linked_cases:
            if not isinstance(linked_case, dict):
                continue
            case_id = str(linked_case.get("case_id", "")).strip()
            patient_id = str(linked_case.get("submitter_id", "")).strip()
            if case_id:
                by_case.setdefault(case_id, []).append(file_hit)
            if patient_id:
                by_patient.setdefault(patient_id, []).append(file_hit)
    return by_case, by_patient


def _file_metadata(file_hit: dict[str, Any]) -> dict[str, Any]:
    workflow_type = ""
    analysis = file_hit.get("analysis")
    if isinstance(analysis, dict):
        workflow_type = str(analysis.get("workflow_type", "")).strip()

    sample_ids: list[str] = []
    sample_types: list[str] = []
    for linked_case in file_hit.get("cases", []) or []:
        if not isinstance(linked_case, dict):
            continue
        linked_sample_ids, linked_sample_types = _linked_case_sample_metadata(file_hit, linked_case=linked_case)
        sample_ids.extend(linked_sample_ids)
        sample_types.extend(linked_sample_types)

    return {
        "file_id": str(file_hit.get("file_id", "")).strip(),
        "file_name": str(file_hit.get("file_name", "")).strip(),
        "data_category": str(file_hit.get("data_category", "")).strip(),
        "data_type": str(file_hit.get("data_type", "")).strip(),
        "data_format": str(file_hit.get("data_format", "")).strip(),
        "experimental_strategy": str(file_hit.get("experimental_strategy", "")).strip(),
        "workflow_type": workflow_type,
        "sample_ids": _unique_sorted_non_empty(sample_ids),
        "sample_types": _unique_sorted_non_empty(sample_types),
    }


def _relative_download_paths(
    file_hits: list[dict[str, Any]],
    downloaded_by_file_id: dict[str, str],
    *,
    project_root: Path,
) -> list[str]:
    paths: list[str] = []
    for file_hit in file_hits:
        file_id = str(file_hit.get("file_id", "")).strip()
        local_path = str(downloaded_by_file_id.get(file_id, "")).strip()
        if local_path:
            paths.append(_to_project_relative_path(local_path, project_root))
    return sorted(set(paths))


def _metadata_lists(file_hits: list[dict[str, Any]]) -> dict[str, list[str]]:
    file_ids: list[str] = []
    file_names: list[str] = []
    sample_types: list[str] = []
    workflow_types: list[str] = []
    for file_hit in file_hits:
        metadata = _file_metadata(file_hit)
        file_ids.append(metadata["file_id"])
        file_names.append(metadata["file_name"])
        sample_types.extend(metadata["sample_types"])
        workflow_types.append(metadata["workflow_type"])
    return {
        "file_ids": _unique_sorted_non_empty(file_ids),
        "file_names": _unique_sorted_non_empty(file_names),
        "sample_types": _unique_sorted_non_empty(sample_types),
        "workflow_types": _unique_sorted_non_empty(workflow_types),
    }


def _first_case_group(case: dict[str, Any], primary_sites_by_group: dict[str, list[str]]) -> str | None:
    primary_site = str(case.get("primary_site", "")).strip()
    for group_name, primary_sites in primary_sites_by_group.items():
        if primary_site in set(primary_sites):
            return group_name
    return None


def _case_tcia_collections(case: dict[str, Any], primary_sites_by_group: dict[str, list[str]], collections_by_group: dict[str, list[str]]) -> list[str]:
    group_name = _first_case_group(case, primary_sites_by_group)
    if group_name is None:
        return []
    return list(collections_by_group.get(group_name, []))


def build_biomarkers_text(case: dict[str, Any]) -> str:
    diagnosis = _first_diagnosis(case)
    demographic = case.get("demographic", {}) if isinstance(case.get("demographic", {}), dict) else {}
    pairs = [
        ("project", (case.get("project") or {}).get("project_id")),
        ("primary_site", case.get("primary_site")),
        ("disease_type", case.get("disease_type")),
        ("primary_diagnosis", diagnosis.get("primary_diagnosis")),
        ("tumor_grade", diagnosis.get("tumor_grade")),
        ("ajcc_pathologic_stage", diagnosis.get("ajcc_pathologic_stage")),
        ("vital_status", _coalesce_text(diagnosis.get("vital_status"), demographic.get("vital_status"))),
        ("gender", demographic.get("gender")),
        ("race", demographic.get("race")),
        ("ethnicity", demographic.get("ethnicity")),
    ]
    return "; ".join(f"{key}: {value}" for key, value in pairs if str(value or "").strip())


def _stable_external_split(source_name: str) -> str:
    return f"{source_name}_external_test"


def build_cptac_registry_rows(
    *,
    cases: list[dict[str, Any]],
    cancer_groups: list[dict[str, Any]],
    rna_bulk_files: list[dict[str, Any]] | None = None,
    dnam_files: list[dict[str, Any]] | None = None,
    mutation_files: list[dict[str, Any]] | None = None,
    report_files: list[dict[str, Any]] | None = None,
    tcia_studies_by_patient: dict[str, list[dict[str, Any]]] | None = None,
    tcia_series_by_patient: dict[str, list[dict[str, str]]] | None = None,
    downloaded_rna_bulk_by_file_id: dict[str, str] | None = None,
    downloaded_dnam_by_file_id: dict[str, str] | None = None,
    downloaded_mutation_by_file_id: dict[str, str] | None = None,
    downloaded_tcia_series_by_patient: dict[str, list[dict[str, str]]] | None = None,
    raw_root: Path | None = None,
    project_root: Path | None = None,
    source_name: str = "cptac",
    split_name: str | None = None,
    show_progress: bool = True,
    progress_desc: str = "Preparing CPTAC registry rows",
) -> pd.DataFrame:
    if not cases:
        return empty_registry_frame()

    rna_bulk_files = rna_bulk_files or []
    dnam_files = dnam_files or []
    mutation_files = mutation_files or []
    report_files = report_files or []
    tcia_studies_by_patient = tcia_studies_by_patient or {}
    tcia_series_by_patient = tcia_series_by_patient or {}
    downloaded_rna_bulk_by_file_id = downloaded_rna_bulk_by_file_id or {}
    downloaded_dnam_by_file_id = downloaded_dnam_by_file_id or {}
    downloaded_mutation_by_file_id = downloaded_mutation_by_file_id or {}
    downloaded_tcia_series_by_patient = downloaded_tcia_series_by_patient or {}
    primary_sites_by_group = _primary_sites_by_group(cancer_groups)
    collections_by_group = _collections_by_group(cancer_groups)
    resolved_project_root = Path(project_root or Path.cwd()).expanduser().resolve()
    resolved_split = str(split_name or _stable_external_split(source_name)).strip()

    rna_by_case, rna_by_patient = _index_files_by_case_and_patient(rna_bulk_files)
    dnam_by_case, dnam_by_patient = _index_files_by_case_and_patient(dnam_files)
    mutation_by_case, mutation_by_patient = _index_files_by_case_and_patient(mutation_files)
    report_by_case, report_by_patient = _index_files_by_case_and_patient(report_files)

    rows: list[dict[str, Any]] = []
    case_iterable = cases
    if show_progress:
        case_iterable = tqdm(
            cases,
            total=len(cases),
            desc=progress_desc,
            unit="case",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

    for case in case_iterable:
        case_id = str(case.get("case_id", "")).strip()
        patient_id = str(case.get("submitter_id", "")).strip()
        if not patient_id:
            continue

        project_id = str((case.get("project") or {}).get("project_id", "")).strip()
        diagnosis = _first_diagnosis(case)
        demographic = case.get("demographic", {}) if isinstance(case.get("demographic", {}), dict) else {}
        cancer_group = _first_case_group(case, primary_sites_by_group)
        tcia_collections_for_case = _case_tcia_collections(case, primary_sites_by_group, collections_by_group)

        rna_entries = rna_by_case.get(case_id, rna_by_patient.get(patient_id, []))
        dnam_entries = dnam_by_case.get(case_id, dnam_by_patient.get(patient_id, []))
        mutation_entries = mutation_by_case.get(case_id, mutation_by_patient.get(patient_id, []))
        report_entries = report_by_case.get(case_id, report_by_patient.get(patient_id, []))
        rna_metadata = _metadata_lists(rna_entries)
        dnam_metadata = _metadata_lists(dnam_entries)
        mutation_metadata = _metadata_lists(mutation_entries)
        report_metadata = _metadata_lists(report_entries)

        radiology_entries = tcia_studies_by_patient.get(patient_id, [])
        series_metadata_entries = tcia_series_by_patient.get(patient_id, [])
        downloaded_series_entries = downloaded_tcia_series_by_patient.get(patient_id, [])
        tcia_study_uids: list[str] = []
        tcia_collections: list[str] = []
        tcia_study_dates: list[str] = []
        tcia_study_descriptions: list[str] = []
        tcia_series_uids: list[str] = []
        tcia_modalities: list[str] = []
        tcia_body_parts: list[str] = []
        tcia_series_descriptions: list[str] = []
        radiology_uri_paths: list[str] = []
        for study in radiology_entries:
            collection = str(study.get("collection", "")).strip()
            study_uid = str(study.get("study_instance_uid", "")).strip()
            study_date = str(study.get("study_date", "")).strip()
            study_description = str(study.get("study_description", "")).strip()
            if collection:
                tcia_collections.append(collection)
            if study_uid:
                tcia_study_uids.append(study_uid)
                radiology_uri_paths.append(f"tcia://{collection}/{patient_id}/{study_uid}")
            if study_date:
                tcia_study_dates.append(study_date)
            if study_description:
                tcia_study_descriptions.append(study_description)
            tcia_modalities.extend(normalize_tcia_modality_list(study.get("modalities_in_study", [])))

        for series in series_metadata_entries:
            series_uid = str(series.get("series_instance_uid", "")).strip()
            modality = normalize_tcia_modality(series.get("modality", ""))
            body_part = str(series.get("body_part_examined", "")).strip()
            description = str(series.get("series_description", "")).strip()
            if series_uid:
                tcia_series_uids.append(series_uid)
            if modality:
                tcia_modalities.append(modality)
            if body_part:
                tcia_body_parts.append(body_part)
            if description:
                tcia_series_descriptions.append(description)

        for series in downloaded_series_entries:
            modality = normalize_tcia_modality(series.get("modality", ""))
            if modality:
                tcia_modalities.append(modality)
            series_uid = str(series.get("series_instance_uid", "")).strip()
            if series_uid:
                tcia_series_uids.append(series_uid)

        radiology_download_paths = sorted(
            {
                _to_project_relative_path(str(entry.get("local_path", "")), resolved_project_root)
                for entry in downloaded_series_entries
                if str(entry.get("local_path", "")).strip()
            }
        )
        vital_status = _coalesce_text(diagnosis.get("vital_status"), demographic.get("vital_status"))
        days_to_death = _coalesce_text(diagnosis.get("days_to_death"), demographic.get("days_to_death"))
        days_to_last_follow_up = str(diagnosis.get("days_to_last_follow_up", "")).strip()
        overall_survival_days = _coalesce_text(days_to_death, days_to_last_follow_up)
        if str(days_to_death).strip():
            survival_event: bool | None = True
        elif str(days_to_last_follow_up).strip():
            survival_event = False
        else:
            survival_event = None

        row = {
            "sample_id": make_sample_id(source_name, patient_id, case_id or patient_id, modality_scope="patient_study"),
            "source": source_name,
            "patient_id": patient_id,
            "study_id": case_id or patient_id,
            "split": resolved_split,
            "split_group_id": f"{source_name}:{project_id}:{patient_id}",
            "split_scheme_version": DEFAULT_CPTAC_SPLIT_SCHEME_VERSION,
            "project_id": project_id,
            "cptac_cancer_group": cancer_group,
            "cptac_tcia_collections": _unique_sorted_non_empty(tcia_collections_for_case),
            "primary_site": str(case.get("primary_site", "")).strip(),
            "disease_type": str(case.get("disease_type", "")).strip(),
            "primary_diagnosis": str(diagnosis.get("primary_diagnosis", "")).strip(),
            "tumor_grade": str(diagnosis.get("tumor_grade", "")).strip(),
            "tumor_stage": str(diagnosis.get("tumor_stage", "")).strip(),
            "ajcc_pathologic_stage": str(diagnosis.get("ajcc_pathologic_stage", "")).strip(),
            "ajcc_pathologic_t": str(diagnosis.get("ajcc_pathologic_t", "")).strip(),
            "ajcc_pathologic_n": str(diagnosis.get("ajcc_pathologic_n", "")).strip(),
            "ajcc_pathologic_m": str(diagnosis.get("ajcc_pathologic_m", "")).strip(),
            "age_at_diagnosis": str(diagnosis.get("age_at_diagnosis", "")).strip(),
            "morphology": str(diagnosis.get("morphology", "")).strip(),
            "last_known_disease_status": str(diagnosis.get("last_known_disease_status", "")).strip(),
            "days_to_last_known_disease_status": str(diagnosis.get("days_to_last_known_disease_status", "")).strip(),
            "days_to_recurrence": str(diagnosis.get("days_to_recurrence", "")).strip(),
            "vital_status": str(vital_status).strip(),
            "days_to_last_follow_up": days_to_last_follow_up,
            "days_to_death": str(days_to_death).strip(),
            "gender": str(demographic.get("gender", "")).strip(),
            "race": str(demographic.get("race", "")).strip(),
            "ethnicity": str(demographic.get("ethnicity", "")).strip(),
            "year_of_birth": str(demographic.get("year_of_birth", "")).strip(),
            "task_grade_label": str(diagnosis.get("tumor_grade", "")).strip(),
            "task_stage_label": str(diagnosis.get("ajcc_pathologic_stage", "")).strip(),
            "task_survival_event": survival_event,
            "task_survival_days": _to_float_or_none(overall_survival_days),
            "genomics_rna_bulk_paths": _relative_download_paths(
                rna_entries,
                downloaded_rna_bulk_by_file_id,
                project_root=resolved_project_root,
            ),
            "genomics_rna_bulk_feature_path": "",
            "genomics_rna_bulk_file_ids": rna_metadata["file_ids"],
            "genomics_rna_bulk_file_names": rna_metadata["file_names"],
            "genomics_rna_bulk_sample_types": rna_metadata["sample_types"],
            "genomics_rna_bulk_workflow_types": rna_metadata["workflow_types"],
            "genomics_rna_bulk_molecular_subtype": "",
            "genomics_rna_bulk_subtype_mrna": "",
            "genomics_dna_methylation_subtype": "",
            "genomics_integrative_subtype": "",
            "genomics_msi_status": "",
            "genomics_rna_bulk_leukocyte_fraction": "",
            "genomics_rna_bulk_tumor_purity": "",
            "genomics_aneuploidy_score": "",
            "genomics_hrd_score": "",
            "genomics_rna_bulk_top_immune_cell_types": [],
            "genomics_rna_bulk_top_immune_cell_fractions": [],
            "genomics_dna_methylation_paths": _relative_download_paths(
                dnam_entries,
                downloaded_dnam_by_file_id,
                project_root=resolved_project_root,
            ),
            "genomics_dna_methylation_feature_path": "",
            "genomics_dna_methylation_file_ids": dnam_metadata["file_ids"],
            "genomics_dna_methylation_file_names": dnam_metadata["file_names"],
            "genomics_dna_methylation_sample_types": dnam_metadata["sample_types"],
            "genomics_dna_methylation_workflow_types": dnam_metadata["workflow_types"],
            "genomics_mutation_paths": _relative_download_paths(
                mutation_entries,
                downloaded_mutation_by_file_id,
                project_root=resolved_project_root,
            ),
            "genomics_mutation_file_ids": mutation_metadata["file_ids"],
            "genomics_mutation_file_names": mutation_metadata["file_names"],
            "genomics_mutation_sample_types": mutation_metadata["sample_types"],
            "genomics_mutation_workflow_types": mutation_metadata["workflow_types"],
            "genomics_cnv_paths": [],
            "genomics_cnv_feature_path": "",
            "pathology_wsi_paths": [],
            "pathology_mask_paths": [],
            "pathology_tile_embedding_paths": [],
            "pathology_slide_embedding_paths": [],
            "radiology_image_paths": [],
            "radiology_image_modalities": [],
            "radiology_download_paths": radiology_download_paths,
            "radiology_uri_paths": sorted(set(radiology_uri_paths)),
            "radiology_report_download_paths": [],
            "radiology_report_uri_paths": [],
            "radiology_report_series_descriptions": [],
            "radiology_mask_paths": [],
            "radiology_embedding_paths": [],
            "biomarkers_text": build_biomarkers_text(case),
            "question": "",
            "answer": "",
            "tcia_collections": _unique_sorted_non_empty(tcia_collections),
            "tcia_study_uids": _unique_sorted_non_empty(tcia_study_uids),
            "tcia_series_uids": _unique_sorted_non_empty(tcia_series_uids),
            "tcia_modalities": _unique_sorted_non_empty(tcia_modalities),
            "tcia_body_parts": _unique_sorted_non_empty(tcia_body_parts),
            "tcia_study_dates": _unique_sorted_non_empty(tcia_study_dates),
            "tcia_study_descriptions": _unique_sorted_non_empty(tcia_study_descriptions),
            "tcia_series_descriptions": _unique_sorted_non_empty(tcia_series_descriptions),
            "report_pdf_paths": [],
            "report_file_ids": report_metadata["file_ids"],
            "report_file_names": report_metadata["file_names"],
        }
        rows.append(row)

    if show_progress:
        case_iterable.close()

    frame = pd.DataFrame(rows)
    if "sample_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["sample_id"], keep="last").reset_index(drop=True)
    frame = normalize_registry_df(frame)
    return frame[CORE_COLUMNS + [col for col in frame.columns if col not in CORE_COLUMNS]]


def fetch_cptac_tcia_metadata(
    tcia_client: TCIAClient,
    *,
    collections: list[str],
    max_studies_per_collection: int | None = None,
    fetch_series_metadata: bool = True,
    max_series_per_study: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, str]]]]:
    studies_by_patient: dict[str, list[dict[str, Any]]] = {}
    for collection in collections:
        studies = tcia_client.fetch_patient_studies(
            collection=collection,
            max_studies=max_studies_per_collection,
        )
        for study in studies:
            patient_id = _first_non_empty(
                study,
                ["PatientID", "PatientId", "patientId", "SubjectID", "subject_id"],
            )
            study_uid = _first_non_empty(
                study,
                ["StudyInstanceUID", "StudyInstanceUid", "studyInstanceUid"],
            )
            if not patient_id or not study_uid:
                continue
            studies_by_patient.setdefault(str(patient_id), []).append(
                {
                    "collection": str(collection),
                    "patient_id": str(patient_id),
                    "study_instance_uid": str(study_uid),
                    "study_date": str(_first_non_empty(study, ["StudyDate", "studyDate"])),
                    "study_description": str(
                        _first_non_empty(study, ["StudyDescription", "studyDescription"])
                    ),
                    "modalities_in_study": _extract_text_values(
                        study,
                        ["ModalitiesInStudy", "modalitiesInStudy", "Modalities", "Modality"],
                    ),
                    "study_series_count": str(
                        _first_non_empty(
                            study,
                            ["NumberOfStudyRelatedSeries", "numberOfStudyRelatedSeries", "SeriesCount"],
                        )
                    ),
                    "study_instance_count": str(
                        _first_non_empty(
                            study,
                            ["NumberOfStudyRelatedInstances", "numberOfStudyRelatedInstances", "ImageCount"],
                        )
                    ),
                }
            )

    if not fetch_series_metadata:
        return studies_by_patient, {}
    series_by_patient: dict[str, list[dict[str, str]]] = {}
    for patient_id, studies in studies_by_patient.items():
        seen_study_uids: set[str] = set()
        for study in studies:
            study_uid = str(study.get("study_instance_uid", "")).strip()
            if not study_uid or study_uid in seen_study_uids:
                continue
            seen_study_uids.add(study_uid)
            collection = str(study.get("collection", "")).strip()
            series_records = tcia_client.fetch_series_for_study(
                study_uid,
                max_series=max_series_per_study,
            )
            for series in series_records:
                series_uid = _first_non_empty(
                    series,
                    ["SeriesInstanceUID", "SeriesInstanceUid", "seriesInstanceUid"],
                )
                if not series_uid:
                    continue
                series_by_patient.setdefault(patient_id, []).append(
                    {
                        "collection": collection,
                        "patient_id": patient_id,
                        "study_instance_uid": study_uid,
                        "series_instance_uid": str(series_uid),
                        "modality": str(_first_non_empty(series, ["Modality", "modality"])),
                        "body_part_examined": str(
                            _first_non_empty(series, ["BodyPartExamined", "bodyPartExamined", "BodyPart"])
                        ),
                        "series_description": str(
                            _first_non_empty(series, ["SeriesDescription", "seriesDescription"])
                        ),
                    }
                )
    return studies_by_patient, series_by_patient


def stable_file_output_path(
    *,
    raw_root: Path,
    source_name: str,
    subfolder: str,
    file_hit: dict[str, Any],
) -> Path | None:
    file_id = str(file_hit.get("file_id", "")).strip()
    file_name = str(file_hit.get("file_name", "")).strip()
    if not file_id or not file_name:
        return None
    linked_cases = file_hit.get("cases", [])
    patient_id = "unknown_patient"
    project_id = "unknown_project"
    if isinstance(linked_cases, list):
        for linked_case in linked_cases:
            if not isinstance(linked_case, dict):
                continue
            patient_id = str(linked_case.get("submitter_id", "")).strip() or patient_id
            project_id = str((linked_case.get("project") or {}).get("project_id", "")).strip() or project_id
            break
    safe_name = file_name
    if not safe_name.startswith(file_id):
        digest = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:8]
        safe_name = f"{file_id}.{digest}.{file_name}"
    return raw_root / source_name / subfolder / project_id / patient_id / safe_name


def download_cptac_files(
    gdc_client: GDCClient,
    file_hits: list[dict[str, Any]],
    *,
    raw_root: Path,
    source_name: str,
    subfolder: str,
    skip_existing: bool,
    max_downloads: int | None = None,
    progress_desc: str = "Downloading CPTAC files",
) -> tuple[dict[str, str], int]:
    downloaded: dict[str, str] = {}
    selected_hits = file_hits[:max_downloads] if max_downloads is not None else file_hits
    for file_hit in tqdm(selected_hits, desc=progress_desc, unit="file", leave=False):
        file_id = str(file_hit.get("file_id", "")).strip()
        if not file_id:
            continue
        output_path = stable_file_output_path(
            raw_root=raw_root,
            source_name=source_name,
            subfolder=subfolder,
            file_hit=file_hit,
        )
        if output_path is None:
            continue
        try:
            resolved = gdc_client.download_data_file(
                file_id=file_id,
                output_path=output_path,
                skip_existing=skip_existing,
            )
        except APIQueryError:
            raise
        downloaded[file_id] = str(resolved)
    return downloaded, len(downloaded)


def resolve_existing_cptac_files(
    file_hits: list[dict[str, Any]],
    *,
    raw_root: Path,
    source_name: str,
    subfolder: str,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for file_hit in file_hits:
        file_id = str(file_hit.get("file_id", "")).strip()
        if not file_id:
            continue
        output_path = stable_file_output_path(
            raw_root=raw_root,
            source_name=source_name,
            subfolder=subfolder,
            file_hit=file_hit,
        )
        if output_path is not None and output_path.exists():
            resolved[file_id] = str(output_path)
    return resolved
