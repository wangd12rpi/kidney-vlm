from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import requests

from kidney_vlm.data.id_factory import make_sample_id
from kidney_vlm.data.registry_schema import CORE_COLUMNS, empty_registry_frame, normalize_registry_df

from tqdm.auto import tqdm


DEFAULT_CASE_FIELDS = [
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

DEFAULT_PROJECT_FIELDS = [
    "project_id",
]

DEFAULT_PATHOLOGY_FILE_FIELDS = [
    "file_id",
    "file_name",
    "data_category",
    "data_type",
    "data_format",
    "file_size",
    "md5sum",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
]

DEFAULT_REPORT_FILE_FIELDS = [
    "file_id",
    "file_name",
    "data_category",
    "data_type",
    "data_format",
    "file_size",
    "md5sum",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
]

DEFAULT_MUTATION_PANEL_VERSION = "pancanatlas_driver_union_v1"
DEFAULT_SPLIT_SCHEME_VERSION = "tcga_project_patient_hash_v1"
PROJECT_DRIVER_GENE_JSON_PATH = Path(__file__).with_name("tcga_project_driver_genes.json")


def _load_project_driver_gene_panel_by_project(
    json_path: Path = PROJECT_DRIVER_GENE_JSON_PATH,
) -> dict[str, list[str]]:
    if not json_path.exists():
        return {}

    payload = json.loads(json_path.read_text())
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for project_id, genes in payload.items():
        if not isinstance(genes, list):
            continue
        normalized_project_id = str(project_id).strip()
        normalized_genes = sorted(
            {
                str(gene).strip().upper()
                for gene in genes
                if str(gene).strip()
            }
        )
        if normalized_project_id:
            normalized[normalized_project_id] = normalized_genes
    return normalized


DEFAULT_PROJECT_DRIVER_GENE_PANEL_BY_PROJECT = _load_project_driver_gene_panel_by_project()
DEFAULT_PANCANCER_MUTATION_GENE_PANEL = sorted(
    {
        gene
        for genes in DEFAULT_PROJECT_DRIVER_GENE_PANEL_BY_PROJECT.values()
        for gene in genes
        if str(gene).strip()
    }
)

DEFAULT_KIDNEY_MUTATION_GENE_PANEL = list(DEFAULT_PANCANCER_MUTATION_GENE_PANEL)

TCIA_MODALITY_ALIASES = {
    "MRI": "MR",
    "MAMMOGRAPHY": "MG",
    "PET": "PT",
}

TCIA_REPORT_MODALITIES = {"SR"}


class APIQueryError(RuntimeError):
    """Raised for upstream API failures with endpoint context."""


@dataclass
class GDCClient:
    base_url: str = "https://api.gdc.cancer.gov"
    timeout_seconds: int = 120
    page_size: int = 200
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    def _post_hits(self, endpoint: str, payload: dict[str, Any], max_records: int | None = None) -> list[dict[str, Any]]:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        hits: list[dict[str, Any]] = []
        offset = 0

        while True:
            remaining = None if max_records is None else max_records - len(hits)
            if remaining is not None and remaining <= 0:
                break

            request_size = self.page_size if remaining is None else min(self.page_size, max(remaining, 1))
            paged_payload = dict(payload)
            paged_payload["from"] = offset
            paged_payload["size"] = request_size

            response = None
            last_exc: Exception | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = requests.post(url, json=paged_payload, timeout=self.timeout_seconds)
                    response.raise_for_status()
                    last_exc = None
                    break
                except requests.RequestException as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff_seconds * attempt)

            if response is None or last_exc is not None:
                raise APIQueryError(f"GDC request failed for endpoint '{endpoint}': {last_exc}") from last_exc

            body = response.json()
            data = body.get("data", {})
            page_hits = data.get("hits", [])
            if not page_hits:
                break

            hits.extend(page_hits)
            offset += len(page_hits)

            pagination = data.get("pagination", {})
            total = int(pagination.get("total", len(hits)))
            if offset >= total:
                break

        if max_records is not None:
            return hits[:max_records]
        return hits

    def fetch_cases(
        self,
        project_ids: list[str],
        fields: list[str] | None = None,
        max_cases: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = {
            "filters": {
                "op": "in",
                "content": {
                    "field": "project.project_id",
                    "value": project_ids,
                },
            },
            "fields": ",".join(fields or DEFAULT_CASE_FIELDS),
            "sort": "submitter_id:asc",
        }
        return self._post_hits("cases", payload, max_records=max_cases)

    def fetch_projects(
        self,
        *,
        project_id_pattern: str | None = None,
        fields: list[str] | None = None,
        max_projects: int | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "fields": ",".join(fields or DEFAULT_PROJECT_FIELDS),
            "sort": "project_id:asc",
        }
        if project_id_pattern:
            payload["filters"] = {
                "op": "=",
                "content": {
                    "field": "project_id",
                    "value": str(project_id_pattern),
                },
            }
        return self._post_hits("projects", payload, max_records=max_projects)

    def fetch_pathology_files(
        self,
        project_ids: list[str],
        data_formats: list[str] | None = None,
        data_types: list[str] | None = None,
        fields: list[str] | None = None,
        max_files: int | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project_ids,
                },
            }
        ]

        if data_formats:
            filters.append(
                {
                    "op": "in",
                    "content": {
                        "field": "data_format",
                        "value": data_formats,
                    },
                }
            )

        if data_types:
            filters.append(
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": data_types,
                    },
                }
            )

        payload = {
            "filters": {
                "op": "and",
                "content": filters,
            },
            "fields": ",".join(fields or DEFAULT_PATHOLOGY_FILE_FIELDS),
            "sort": "file_name:asc",
        }
        return self._post_hits("files", payload, max_records=max_files)

    def fetch_files_by_ids(
        self,
        file_ids: list[str],
        *,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_file_ids = [str(file_id).strip() for file_id in file_ids if str(file_id).strip()]
        if not normalized_file_ids:
            return []

        payload = {
            "filters": {
                "op": "in",
                "content": {
                    "field": "file_id",
                    "value": normalized_file_ids,
                },
            },
            "fields": ",".join(fields or DEFAULT_PATHOLOGY_FILE_FIELDS),
            "sort": "file_name:asc",
        }
        return self._post_hits("files", payload, max_records=len(normalized_file_ids))

    def fetch_report_files(
        self,
        *,
        project_ids: list[str],
        case_ids: list[str] | None = None,
        data_formats: list[str] | None = None,
        data_types: list[str] | None = None,
        data_categories: list[str] | None = None,
        fields: list[str] | None = None,
        max_files: int | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project_ids,
                },
            }
        ]

        if case_ids:
            filters.append(
                {
                    "op": "in",
                    "content": {
                        "field": "cases.case_id",
                        "value": case_ids,
                    },
                }
            )

        filters.append(
            {
                "op": "in",
                "content": {
                    "field": "data_format",
                    "value": data_formats or ["PDF"],
                },
            }
        )

        if data_types:
            filters.append(
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": data_types,
                    },
                }
            )

        if data_categories:
            filters.append(
                {
                    "op": "in",
                    "content": {
                        "field": "data_category",
                        "value": data_categories,
                    },
                }
            )

        payload = {
            "filters": {
                "op": "and",
                "content": filters,
            },
            "fields": ",".join(fields or DEFAULT_REPORT_FILE_FIELDS),
            "sort": "file_name:asc",
        }
        return self._post_hits("files", payload, max_records=max_files)

    def fetch_ssm_hits(
        self,
        *,
        project_ids: list[str],
        gene_symbols: list[str] | None = None,
        max_hits: int | None = None,
    ) -> list[dict[str, Any]]:
        if not project_ids:
            return []

        project_field_candidates = [
            "occurrence.case.project.project_id",
            "case.project.project_id",
            "cases.project.project_id",
        ]
        gene_field_candidates = [
            "consequence.transcript.gene.symbol",
            "gene.symbol",
        ]
        normalized_genes = _unique_sorted_non_empty([str(gene).upper() for gene in (gene_symbols or [])])

        last_error: Exception | None = None
        for project_field in project_field_candidates:
            filters: list[dict[str, Any]] = [
                {
                    "op": "in",
                    "content": {
                        "field": project_field,
                        "value": project_ids,
                    },
                }
            ]
            if normalized_genes:
                filters.append(
                    {
                        "op": "in",
                        "content": {
                            "field": gene_field_candidates[0],
                            "value": normalized_genes,
                        },
                    }
                )

            payload = {
                "filters": {
                    "op": "and",
                    "content": filters,
                },
                "sort": "ssm_id:asc",
            }
            try:
                return self._post_hits("ssms", payload, max_records=max_hits)
            except APIQueryError as exc:
                last_error = exc
                if normalized_genes:
                    # Retry with an alternate gene symbol field key.
                    payload_alt = {
                        "filters": {
                            "op": "and",
                            "content": [
                                filters[0],
                                {
                                    "op": "in",
                                    "content": {
                                        "field": gene_field_candidates[1],
                                        "value": normalized_genes,
                                    },
                                },
                            ],
                        },
                        "sort": "ssm_id:asc",
                    }
                    try:
                        return self._post_hits("ssms", payload_alt, max_records=max_hits)
                    except APIQueryError as nested_exc:
                        last_error = nested_exc
                        continue
                continue

        if last_error is not None:
            raise APIQueryError(f"GDC SSM query failed across candidate filters: {last_error}") from last_error
        return []

    def download_data_file(
        self,
        *,
        file_id: str,
        output_path: Path,
        skip_existing: bool = True,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url.rstrip('/')}/data/{str(file_id).strip()}"
        response = None
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout_seconds, stream=True)
                response.raise_for_status()
                last_exc = None
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        if response is None or last_exc is not None:
            raise APIQueryError(f"GDC download failed for file_id '{file_id}': {last_exc}") from last_exc

        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
        return output_path


@dataclass
class TCIAClient:
    base_url: str = "https://services.cancerimagingarchive.net/nbia-api/services"
    api_version: str = "v1"
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url.rstrip('/')}/{self.api_version}/{endpoint}"
        response = None
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                last_exc = None
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        if response is None or last_exc is not None:
            raise APIQueryError(f"TCIA request failed for endpoint '{endpoint}': {last_exc}") from last_exc

        # TCIA sometimes returns HTTP 200 with an empty body instead of `[]`
        # for collections/studies that have no matching public records.
        if not (response.text or "").strip():
            return []

        try:
            return response.json()
        except ValueError as exc:
            response_preview = (response.text or "").strip().replace("\n", " ")
            if len(response_preview) > 200:
                response_preview = f"{response_preview[:200]}..."
            raise APIQueryError(
                f"TCIA returned a non-JSON response for endpoint '{endpoint}' with params {params}: "
                f"{response_preview or '[empty response]'}"
            ) from exc

    def fetch_patient_studies(self, collection: str, max_studies: int | None = None) -> list[dict[str, Any]]:
        payload = self._get_json(
            "getPatientStudy",
            params={"Collection": collection, "format": "json"},
        )

        records: list[dict[str, Any]]
        if isinstance(payload, list):
            records = [rec for rec in payload if isinstance(rec, dict)]
        elif isinstance(payload, dict):
            # Keep this defensive for API wrapper changes.
            nested = payload.get("result", payload.get("results", payload.get("data", [])))
            records = [rec for rec in nested if isinstance(rec, dict)] if isinstance(nested, list) else []
        else:
            records = []

        if max_studies is not None:
            return records[:max_studies]
        return records

    def fetch_series_for_study(self, study_instance_uid: str, max_series: int | None = None) -> list[dict[str, Any]]:
        payload = self._get_json(
            "getSeries",
            params={"StudyInstanceUID": study_instance_uid, "format": "json"},
        )

        records: list[dict[str, Any]]
        if isinstance(payload, list):
            records = [rec for rec in payload if isinstance(rec, dict)]
        elif isinstance(payload, dict):
            nested = payload.get("result", payload.get("results", payload.get("data", [])))
            records = [rec for rec in nested if isinstance(rec, dict)] if isinstance(nested, list) else []
        else:
            records = []

        if max_series is not None:
            return records[:max_series]
        return records

    def download_series_zip(
        self,
        *,
        series_instance_uid: str,
        output_path: Path,
        skip_existing: bool = True,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url.rstrip('/')}/{self.api_version}/getImage"
        response = None
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    params={"SeriesInstanceUID": str(series_instance_uid).strip()},
                    timeout=self.timeout_seconds,
                    stream=True,
                )
                response.raise_for_status()
                last_exc = None
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        if response is None or last_exc is not None:
            raise APIQueryError(
                f"TCIA series download failed for SeriesInstanceUID '{series_instance_uid}': {last_exc}"
            ) from last_exc

        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
        return output_path

    def fetch_studies_by_patient(
        self,
        collections: list[str],
        max_studies_per_collection: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        by_patient: dict[str, list[dict[str, Any]]] = {}

        for collection in collections:
            try:
                studies = self.fetch_patient_studies(collection=collection, max_studies=max_studies_per_collection)
            except APIQueryError as exc:
                print(f"[warning] Skipping TCIA collection '{collection}' after metadata fetch failure: {exc}")
                continue
            for study in studies:
                patient_id = _first_non_empty(
                    study,
                    ["PatientID", "PatientId", "patientId", "SubjectID", "subject_id"],
                )
                if not patient_id:
                    continue
                study_uid = _first_non_empty(
                    study,
                    ["StudyInstanceUID", "StudyInstanceUid", "studyInstanceUid"],
                )
                study_date = _first_non_empty(study, ["StudyDate", "studyDate"])
                study_description = _first_non_empty(study, ["StudyDescription", "studyDescription"])
                modalities_in_study = _extract_text_values(
                    study,
                    ["ModalitiesInStudy", "modalitiesInStudy", "Modalities", "Modality"],
                )
                study_series_count = _first_non_empty(
                    study,
                    ["NumberOfStudyRelatedSeries", "numberOfStudyRelatedSeries", "SeriesCount"],
                )
                study_instance_count = _first_non_empty(
                    study,
                    ["NumberOfStudyRelatedInstances", "numberOfStudyRelatedInstances", "ImageCount"],
                )
                entry = {
                    "collection": str(collection),
                    "patient_id": str(patient_id),
                    "study_instance_uid": str(study_uid),
                    "study_date": str(study_date),
                    "study_description": str(study_description),
                    "modalities_in_study": modalities_in_study,
                    "study_series_count": str(study_series_count),
                    "study_instance_count": str(study_instance_count),
                }
                by_patient.setdefault(str(patient_id), []).append(entry)

        return by_patient

    def fetch_series_by_patient(
        self,
        studies_by_patient: dict[str, list[dict[str, Any]]],
        max_series_per_study: int | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        by_patient: dict[str, list[dict[str, str]]] = {}

        for patient_id, studies in studies_by_patient.items():
            seen_study_uids: set[str] = set()
            for study in studies:
                study_uid = str(study.get("study_instance_uid", "")).strip()
                if not study_uid or study_uid in seen_study_uids:
                    continue
                seen_study_uids.add(study_uid)

                collection = str(study.get("collection", "")).strip()
                try:
                    series_records = self.fetch_series_for_study(study_uid, max_series=max_series_per_study)
                except APIQueryError as exc:
                    print(
                        f"[warning] Skipping TCIA series metadata for study '{study_uid}' "
                        f"(collection '{collection}') after fetch failure: {exc}"
                    )
                    continue
                for series in series_records:
                    series_uid = _first_non_empty(
                        series,
                        ["SeriesInstanceUID", "SeriesInstanceUid", "seriesInstanceUid"],
                    )
                    if not series_uid:
                        continue
                    modality = _first_non_empty(series, ["Modality", "modality"])
                    body_part = _first_non_empty(
                        series,
                        ["BodyPartExamined", "bodyPartExamined", "BodyPart"],
                    )
                    series_description = _first_non_empty(
                        series,
                        ["SeriesDescription", "seriesDescription"],
                    )
                    by_patient.setdefault(patient_id, []).append(
                        {
                            "collection": collection,
                            "patient_id": patient_id,
                            "study_instance_uid": study_uid,
                            "series_instance_uid": str(series_uid),
                            "modality": str(modality),
                            "body_part_examined": str(body_part),
                            "series_description": str(series_description),
                        }
                    )

        return by_patient


def _first_non_empty(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_text_values(record: dict[str, Any], keys: list[str]) -> list[str]:
    for key in keys:
        if key not in record:
            continue
        raw_value = record.get(key)
        values = _to_text_list(raw_value)
        if values:
            return values
    return []


def _to_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return _unique_sorted_non_empty([str(item) for item in value])

    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return _unique_sorted_non_empty([chunk.strip() for chunk in text.split(",")])
    return [text]


def normalize_tcia_modality(value: Any) -> str:
    text = str(value).strip().upper()
    if not text:
        return ""
    compact = text.replace(" ", "").replace("-", "")
    return TCIA_MODALITY_ALIASES.get(compact, compact)


def normalize_tcia_modality_list(values: Any) -> list[str]:
    normalized: list[str] = []
    raw_values = _to_text_list(values)
    for raw_value in raw_values:
        chunks = str(raw_value).replace("\\", ",").replace("/", ",").split(",")
        for chunk in chunks:
            modality = normalize_tcia_modality(chunk)
            if modality:
                normalized.append(modality)
    return _unique_sorted_non_empty(normalized)


def _walk_nested(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_nested(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_nested(nested)


def _extract_case_links_from_ssm_hit(hit: dict[str, Any]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    candidate_dicts: list[dict[str, Any]] = []
    occurrence = hit.get("occurrence", [])
    if isinstance(occurrence, list):
        for item in occurrence:
            if isinstance(item, dict):
                candidate_dicts.append(item.get("case", item))
    for direct_key in ["case", "cases"]:
        direct = hit.get(direct_key)
        if isinstance(direct, dict):
            candidate_dicts.append(direct)
        elif isinstance(direct, list):
            candidate_dicts.extend([item for item in direct if isinstance(item, dict)])

    # Fallback scan to survive slight schema variation.
    if not candidate_dicts:
        candidate_dicts.extend([node for node in _walk_nested(hit) if isinstance(node, dict)])

    for case_like in candidate_dicts:
        case_id = _first_non_empty(case_like, ["case_id", "caseId"])
        patient_id = _first_non_empty(case_like, ["submitter_id", "submitterId", "patient_id", "patientId"])
        project = case_like.get("project") if isinstance(case_like.get("project"), dict) else {}
        project_id = _first_non_empty(
            project if isinstance(project, dict) else {},
            ["project_id", "projectId"],
        )
        if not case_id and not patient_id:
            continue
        key = (case_id, patient_id, project_id)
        if key in seen:
            continue
        seen.add(key)
        links.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "project_id": project_id,
            }
        )

    return links


def _extract_gene_symbols_from_ssm_hit(hit: dict[str, Any]) -> list[str]:
    genes: set[str] = set()

    for node in _walk_nested(hit):
        if not isinstance(node, dict):
            continue
        if "gene" in node and isinstance(node.get("gene"), dict):
            symbol = _first_non_empty(node["gene"], ["symbol", "gene_symbol"])
            if symbol:
                genes.add(symbol.upper())
        for key in ["gene_symbol", "symbol"]:
            if key in node:
                value = str(node.get(key, "")).strip()
                if value:
                    genes.add(value.upper())

    return sorted(genes)


def _extract_consequence_terms_from_ssm_hit(hit: dict[str, Any]) -> list[str]:
    terms: set[str] = set()
    for node in _walk_nested(hit):
        if not isinstance(node, dict):
            continue
        for key in ["vep_consequence", "consequence_type", "consequenceType"]:
            if key in node:
                values = _to_text_list(node.get(key))
                for value in values:
                    text = str(value).strip()
                    if text:
                        terms.add(text)
    return sorted(terms)


def index_ssm_hits_by_case_and_patient(
    ssm_hits: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    by_patient: dict[str, list[dict[str, Any]]] = {}

    for hit in ssm_hits:
        if not isinstance(hit, dict):
            continue
        ssm_id = _first_non_empty(hit, ["ssm_id", "id"])
        mutation_type = _first_non_empty(hit, ["mutation_type", "mutationType", "variant_type"])
        gene_symbols = _extract_gene_symbols_from_ssm_hit(hit)
        consequence_terms = _extract_consequence_terms_from_ssm_hit(hit)
        if not gene_symbols and not ssm_id:
            continue

        entry = {
            "ssm_id": ssm_id,
            "mutation_type": mutation_type,
            "gene_symbols": gene_symbols,
            "consequence_terms": consequence_terms,
        }
        links = _extract_case_links_from_ssm_hit(hit)
        for link in links:
            case_id = str(link.get("case_id", "")).strip()
            patient_id = str(link.get("patient_id", "")).strip()
            if case_id:
                by_case.setdefault(case_id, []).append(entry)
            if patient_id:
                by_patient.setdefault(patient_id, []).append(entry)

    return by_case, by_patient


def _coalesce_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _unique_sorted_non_empty(values: list[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _dedupe_radiology_path_entries(entries: list[tuple[str, Any]]) -> tuple[list[str], list[str]]:
    path_to_modalities: dict[str, list[str]] = {}
    for raw_path, raw_modalities in entries:
        path = str(raw_path).strip()
        if not path:
            continue
        normalized_modalities = normalize_tcia_modality_list(raw_modalities)
        collected_modalities = path_to_modalities.setdefault(path, [])
        for modality in normalized_modalities:
            if modality not in collected_modalities:
                collected_modalities.append(modality)

    paths = sorted(path_to_modalities)
    modalities = ["|".join(path_to_modalities[path]) for path in paths]
    return paths, modalities


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _infer_project_root_from_raw_root(raw_root: Path) -> Path:
    resolved = raw_root.expanduser().resolve()
    if resolved.name == "raw" and resolved.parent.name == "data":
        return resolved.parent.parent
    return resolved.parent


def _to_project_relative_path(path_value: str | Path, project_root: Path) -> str:
    text = str(path_value).strip()
    if not text:
        return ""
    if "://" in text:
        return text

    path_obj = Path(text).expanduser()
    if not path_obj.is_absolute():
        return path_obj.as_posix().lstrip("/")

    resolved = path_obj.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix().lstrip("/")


def _infer_kidney_histology_subtype(project_id: str) -> str:
    normalized = str(project_id).strip().upper()
    mapping = {
        "TCGA-KIRC": "clear_cell_rcc",
        "TCGA-KIRP": "papillary_rcc",
        "TCGA-KICH": "chromophobe_rcc",
    }
    return mapping.get(normalized, "")


def _first_diagnosis(case: dict[str, Any]) -> dict[str, Any]:
    diagnoses = case.get("diagnoses", [])
    if isinstance(diagnoses, list) and diagnoses:
        first = diagnoses[0]
        if isinstance(first, dict):
            return first
    return {}


def build_biomarkers_text(case: dict[str, Any]) -> str:
    diagnosis = _first_diagnosis(case)
    demographic = case.get("demographic", {}) if isinstance(case.get("demographic", {}), dict) else {}
    project_id = str((case.get("project") or {}).get("project_id", ""))
    vital_status = _coalesce_text(diagnosis.get("vital_status"), demographic.get("vital_status"))

    pairs = [
        ("project", project_id),
        ("disease_type", case.get("disease_type")),
        ("primary_diagnosis", diagnosis.get("primary_diagnosis")),
        ("tumor_grade", diagnosis.get("tumor_grade")),
        ("tumor_stage", diagnosis.get("tumor_stage")),
        ("ajcc_pathologic_stage", diagnosis.get("ajcc_pathologic_stage")),
        ("vital_status", vital_status),
        ("gender", demographic.get("gender")),
        ("race", demographic.get("race")),
        ("ethnicity", demographic.get("ethnicity")),
    ]

    parts = [f"{key}: {value}" for key, value in pairs if value not in (None, "")]
    return "; ".join(parts)


def make_split_group_id(*, source_name: str, project_id: str, patient_id: str) -> str:
    return ":".join(
        [
            str(source_name).strip() or "unknown_source",
            str(project_id).strip() or "unknown_project",
            str(patient_id).strip() or "unknown_patient",
        ]
    )


def assign_split(split_group_id: str, split_ratios: dict[str, float]) -> str:
    train_ratio = float(split_ratios.get("train", 0.9))
    val_ratio = float(split_ratios.get("val", 0.0))
    test_ratio = float(split_ratios.get("test", 0.1))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        return "train"

    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    digest = hashlib.sha256(str(split_group_id).encode("utf-8")).hexdigest()
    stable_bucket = (int(digest[:8], 16) % 10_000) / 10_000.0
    if stable_bucket < train_ratio:
        return "train"
    if stable_bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def build_tcga_registry_rows(
    *,
    cases: list[dict[str, Any]],
    pathology_files: list[dict[str, Any]],
    tcia_studies_by_patient: dict[str, list[dict[str, Any]]],
    raw_root: Path,
    source_name: str,
    split_ratios: dict[str, float],
    report_files: list[dict[str, Any]] | None = None,
    downloaded_pathology_by_file_id: dict[str, str] | None = None,
    downloaded_reports_by_file_id: dict[str, str] | None = None,
    ssm_mutations_by_case_id: dict[str, list[dict[str, Any]]] | None = None,
    ssm_mutations_by_patient_id: dict[str, list[dict[str, Any]]] | None = None,
    mutation_gene_panel: list[str] | None = None,
    tcia_series_by_patient: dict[str, list[dict[str, str]]] | None = None,
    downloaded_tcia_series_by_patient: dict[str, list[dict[str, str]]] | None = None,
    downloaded_radiology_only: bool = False,
    project_root: Path | None = None,
    mutation_query_succeeded: bool = True,
    mutation_panel_version: str | None = None,
    split_scheme_version: str = DEFAULT_SPLIT_SCHEME_VERSION,
    project_driver_gene_panel_by_project: dict[str, list[str]] | None = None,
    show_progress: bool = True,
    progress_desc: str = "Building TCGA rows",
) -> pd.DataFrame:
    if not cases:
        return empty_registry_frame()

    report_files = report_files or []
    downloaded_pathology_by_file_id = downloaded_pathology_by_file_id or {}
    downloaded_reports_by_file_id = downloaded_reports_by_file_id or {}
    ssm_mutations_by_case_id = ssm_mutations_by_case_id or {}
    ssm_mutations_by_patient_id = ssm_mutations_by_patient_id or {}
    mutation_gene_panel = mutation_gene_panel or list(DEFAULT_PANCANCER_MUTATION_GENE_PANEL)
    mutation_panel_version = str(mutation_panel_version or DEFAULT_MUTATION_PANEL_VERSION).strip()
    project_driver_gene_panel_by_project = project_driver_gene_panel_by_project or dict(
        DEFAULT_PROJECT_DRIVER_GENE_PANEL_BY_PROJECT
    )
    tcia_series_by_patient = tcia_series_by_patient or {}
    downloaded_tcia_series_by_patient = downloaded_tcia_series_by_patient or {}
    mutation_gene_panel_upper = [str(gene).strip().upper() for gene in mutation_gene_panel if str(gene).strip()]
    resolved_project_root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else _infer_project_root_from_raw_root(raw_root)
    )

    pathology_by_case: dict[str, list[dict[str, str]]] = {}
    pathology_by_patient: dict[str, list[dict[str, str]]] = {}
    reports_by_case: dict[str, list[dict[str, str]]] = {}
    reports_by_patient: dict[str, list[dict[str, str]]] = {}

    for file_hit in pathology_files:
        file_id = str(file_hit.get("file_id", ""))
        file_name = str(file_hit.get("file_name", ""))
        linked_cases = file_hit.get("cases", [])
        if not isinstance(linked_cases, list):
            continue

        for linked_case in linked_cases:
            if not isinstance(linked_case, dict):
                continue
            case_id = str(linked_case.get("case_id", ""))
            patient_id = str(linked_case.get("submitter_id", ""))
            project_id = str((linked_case.get("project") or {}).get("project_id", ""))

            fallback_path = raw_root / source_name / "pathology" / project_id / patient_id / file_name
            local_path = Path(
                downloaded_pathology_by_file_id.get(file_id, str(fallback_path))
            )
            file_entry = {
                "file_id": file_id,
                "file_name": file_name,
                "local_path": _to_project_relative_path(local_path, resolved_project_root),
            }

            if case_id:
                pathology_by_case.setdefault(case_id, []).append(file_entry)
            if patient_id:
                pathology_by_patient.setdefault(patient_id, []).append(file_entry)

    for report_hit in report_files:
        file_id = str(report_hit.get("file_id", ""))
        file_name = str(report_hit.get("file_name", ""))
        linked_cases = report_hit.get("cases", [])
        if not isinstance(linked_cases, list):
            continue

        for linked_case in linked_cases:
            if not isinstance(linked_case, dict):
                continue

            case_id = str(linked_case.get("case_id", ""))
            patient_id = str(linked_case.get("submitter_id", ""))
            project_id = str((linked_case.get("project") or {}).get("project_id", ""))

            fallback_path = raw_root / source_name / "reports" / project_id / patient_id / file_name
            local_path = Path(
                downloaded_reports_by_file_id.get(file_id, str(fallback_path))
            )
            report_entry = {
                "file_id": file_id,
                "file_name": file_name,
                "local_path": _to_project_relative_path(local_path, resolved_project_root),
            }
            if case_id:
                reports_by_case.setdefault(case_id, []).append(report_entry)
            if patient_id:
                reports_by_patient.setdefault(patient_id, []).append(report_entry)

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
            mininterval=0.0,
            miniters=1,
            disable=False,
        )
        case_iterable.refresh()

    for case in case_iterable:
        case_id = str(case.get("case_id", "")).strip()
        patient_id = str(case.get("submitter_id", "")).strip()
        if not patient_id:
            continue

        project_id = str((case.get("project") or {}).get("project_id", "")).strip()
        diagnosis = _first_diagnosis(case)
        demographic = case.get("demographic", {}) if isinstance(case.get("demographic", {}), dict) else {}
        vital_status = _coalesce_text(diagnosis.get("vital_status"), demographic.get("vital_status"))
        days_to_death = _coalesce_text(diagnosis.get("days_to_death"), demographic.get("days_to_death"))

        pathology_entries = pathology_by_case.get(case_id, pathology_by_patient.get(patient_id, []))
        pathology_paths = sorted({entry["local_path"] for entry in pathology_entries if entry.get("local_path")})
        pathology_file_ids = sorted({entry["file_id"] for entry in pathology_entries if entry.get("file_id")})
        report_entries = reports_by_case.get(case_id, reports_by_patient.get(patient_id, []))
        report_pdf_paths = sorted({entry["local_path"] for entry in report_entries if entry.get("local_path")})
        report_file_ids = sorted({entry["file_id"] for entry in report_entries if entry.get("file_id")})
        report_file_names = sorted({entry["file_name"] for entry in report_entries if entry.get("file_name")})

        mutation_entries = ssm_mutations_by_case_id.get(case_id, ssm_mutations_by_patient_id.get(patient_id, []))
        mutation_observed = mutation_query_succeeded and len(mutation_entries) > 0
        if mutation_query_succeeded:
            mutated_gene_symbols = sorted(
                {
                    str(gene).strip().upper()
                    for entry in mutation_entries
                    for gene in list(entry.get("gene_symbols", []))
                    if str(gene).strip()
                }
            )
            mutation_ssm_ids = sorted(
                {
                    str(entry.get("ssm_id", "")).strip()
                    for entry in mutation_entries
                    if str(entry.get("ssm_id", "")).strip()
                }
            )
            mutation_types = sorted(
                {
                    str(entry.get("mutation_type", "")).strip()
                    for entry in mutation_entries
                    if str(entry.get("mutation_type", "")).strip()
                }
            )
            mutation_consequence_terms = sorted(
                {
                    str(term).strip()
                    for entry in mutation_entries
                    for term in list(entry.get("consequence_terms", []))
                    if str(term).strip()
                }
            )
        else:
            mutated_gene_symbols = []
            mutation_ssm_ids = []
            mutation_types = []
            mutation_consequence_terms = []
        mutation_gene_set = set(mutated_gene_symbols)
        project_driver_gene_panel = list(project_driver_gene_panel_by_project.get(project_id, []))
        project_driver_gene_mutations = [gene for gene in project_driver_gene_panel if gene in mutation_gene_set]
        days_to_last_follow_up = str(diagnosis.get("days_to_last_follow_up", "")).strip()
        overall_survival_days = _coalesce_text(days_to_death, days_to_last_follow_up)
        overall_survival_days_numeric = _to_float_or_none(overall_survival_days)
        if str(days_to_death).strip():
            survival_event: bool | None = True
        elif str(days_to_last_follow_up).strip():
            survival_event = False
        else:
            survival_event = None
        kidney_histology_subtype = _infer_kidney_histology_subtype(project_id)

        radiology_entries = tcia_studies_by_patient.get(patient_id, [])
        series_metadata_entries = tcia_series_by_patient.get(patient_id, [])
        radiology_study_fallback_entries: list[tuple[str, list[str]]] = []
        radiology_series_fallback_entries: list[tuple[str, str]] = []
        radiology_uri_paths: list[str] = []
        radiology_report_uri_paths: list[str] = []
        radiology_report_download_paths: list[str] = []
        tcia_study_uids: list[str] = []
        tcia_collections: list[str] = []
        tcia_study_dates: list[str] = []
        tcia_study_descriptions: list[str] = []
        tcia_modalities: list[str] = []
        for entry in radiology_entries:
            collection = str(entry.get("collection", "")).strip() or project_id
            study_uid = str(entry.get("study_instance_uid", "")).strip()
            study_date = str(entry.get("study_date", "")).strip()
            study_description = str(entry.get("study_description", "")).strip()
            modalities_in_study = normalize_tcia_modality_list(entry.get("modalities_in_study", []))
            if study_uid:
                radiology_uri_paths.append(f"tcia://{collection}/{patient_id}/{study_uid}")
                study_fallback = (
                    raw_root / source_name / "radiology" / collection / patient_id / study_uid
                )
                radiology_study_fallback_entries.append(
                    (
                        _to_project_relative_path(study_fallback, resolved_project_root),
                        modalities_in_study,
                    )
                )
                tcia_study_uids.append(study_uid)
            elif collection:
                radiology_uri_paths.append(f"tcia://{collection}/{patient_id}")
            if collection:
                tcia_collections.append(collection)
            if study_date:
                tcia_study_dates.append(study_date)
            if study_description:
                tcia_study_descriptions.append(study_description)
            tcia_modalities.extend(modalities_in_study)

        tcia_series_uids = sorted(
            {
                str(entry.get("series_instance_uid", "")).strip()
                for entry in series_metadata_entries
                if str(entry.get("series_instance_uid", "")).strip()
            }
        )
        tcia_body_parts = sorted(
            {
                str(entry.get("body_part_examined", "")).strip()
                for entry in series_metadata_entries
                if str(entry.get("body_part_examined", "")).strip()
            }
        )
        tcia_series_descriptions = sorted(
            {
                str(entry.get("series_description", "")).strip()
                for entry in series_metadata_entries
                if str(entry.get("series_description", "")).strip()
            }
        )
        radiology_report_series_descriptions = sorted(
            {
                str(entry.get("series_description", "")).strip()
                for entry in series_metadata_entries
                if normalize_tcia_modality(entry.get("modality", "")) in TCIA_REPORT_MODALITIES
                and str(entry.get("series_description", "")).strip()
            }
        )
        tcia_modalities.extend(
            [
                normalize_tcia_modality(entry.get("modality", ""))
                for entry in series_metadata_entries
                if normalize_tcia_modality(entry.get("modality", ""))
            ]
        )
        for entry in series_metadata_entries:
            collection = str(entry.get("collection", "")).strip() or project_id
            study_uid = str(entry.get("study_instance_uid", "")).strip()
            series_uid = str(entry.get("series_instance_uid", "")).strip()
            modality = normalize_tcia_modality(entry.get("modality", ""))
            if study_uid and series_uid:
                series_fallback = (
                    raw_root / source_name / "radiology" / collection / patient_id / study_uid / f"{series_uid}.zip"
                )
                radiology_series_fallback_entries.append(
                    (
                        _to_project_relative_path(series_fallback, resolved_project_root),
                        modality,
                    )
                )
                if modality in TCIA_REPORT_MODALITIES:
                    radiology_report_uri_paths.append(
                        f"tcia://{collection}/{patient_id}/{study_uid}/{series_uid}"
                    )

        downloaded_series_entries = downloaded_tcia_series_by_patient.get(patient_id, [])
        accepted_downloaded_series_entries = [
            entry
            for entry in downloaded_series_entries
            if normalize_tcia_modality(entry.get("modality", "")) not in TCIA_REPORT_MODALITIES
            if bool(entry.get("accepted", True))
            and (
                str(entry.get("png_dir", "")).strip()
                or str(entry.get("extracted_path", "")).strip()
            )
        ]
        tcia_series_uids = sorted(
            set(tcia_series_uids).union(
                {
                    str(entry.get("series_instance_uid", "")).strip()
                    for entry in downloaded_series_entries
                    if str(entry.get("series_instance_uid", "")).strip()
                }
            )
        )
        tcia_modalities.extend(
            [
                normalize_tcia_modality(entry.get("modality", ""))
                for entry in downloaded_series_entries
                if normalize_tcia_modality(entry.get("modality", ""))
            ]
        )
        radiology_download_paths = sorted(
            {
                _to_project_relative_path(str(entry.get("local_path", "")), resolved_project_root)
                for entry in downloaded_series_entries
                if normalize_tcia_modality(entry.get("modality", "")) not in TCIA_REPORT_MODALITIES
                if str(entry.get("local_path", "")).strip()
            }
        )
        radiology_report_download_paths = sorted(
            {
                _to_project_relative_path(str(entry.get("local_path", "")), resolved_project_root)
                for entry in downloaded_series_entries
                if normalize_tcia_modality(entry.get("modality", "")) in TCIA_REPORT_MODALITIES
                and str(entry.get("local_path", "")).strip()
            }
        )
        radiology_report_uri_paths = _unique_sorted_non_empty(radiology_report_uri_paths)
        radiology_download_entries = [
            (
                _to_project_relative_path(
                    str(entry.get("png_dir", "")).strip() or str(entry.get("extracted_path", "")).strip(),
                    resolved_project_root,
                ),
                str(entry.get("modality", "")).strip(),
            )
            for entry in accepted_downloaded_series_entries
            if str(entry.get("png_dir", "")).strip() or str(entry.get("extracted_path", "")).strip()
        ]
        radiology_embedding_paths: list[str] = []
        radiology_mask_paths: list[str] = []
        radiology_mask_manifest_paths: list[str] = []
        radiology_png_dirs: list[str] = []
        radiology_series_slice_counts: list[int] = []
        seen_embedding_refs: set[str] = set()
        seen_mask_paths: set[str] = set()
        seen_mask_manifest_paths: set[str] = set()
        seen_png_dirs: set[str] = set()
        for entry in accepted_downloaded_series_entries:
            embedding_ref = str(entry.get("embedding_ref", "")).strip()
            if embedding_ref and embedding_ref not in seen_embedding_refs:
                seen_embedding_refs.add(embedding_ref)
                radiology_embedding_paths.append(embedding_ref)

            for mask_path_value in list(entry.get("mask_paths", []) or []):
                mask_path = _to_project_relative_path(str(mask_path_value), resolved_project_root)
                if mask_path and mask_path not in seen_mask_paths:
                    seen_mask_paths.add(mask_path)
                    radiology_mask_paths.append(mask_path)

            mask_manifest_path = _to_project_relative_path(
                str(entry.get("mask_manifest_path", "")),
                resolved_project_root,
            )
            if mask_manifest_path and mask_manifest_path not in seen_mask_manifest_paths:
                seen_mask_manifest_paths.add(mask_manifest_path)
                radiology_mask_manifest_paths.append(mask_manifest_path)

            png_dir = _to_project_relative_path(str(entry.get("png_dir", "")), resolved_project_root)
            if png_dir and png_dir not in seen_png_dirs:
                seen_png_dirs.add(png_dir)
                radiology_png_dirs.append(png_dir)

            slice_count_text = str(entry.get("slice_count", "")).strip()
            if slice_count_text:
                try:
                    slice_count = int(float(slice_count_text))
                except ValueError:
                    slice_count = 0
                if slice_count > 0:
                    radiology_series_slice_counts.append(slice_count)

        # Keep radiology_image_paths reserved for extracted/processed image-like artifacts.
        # Raw TCIA zip downloads are tracked separately in radiology_download_paths.
        radiology_paths, radiology_modalities = _dedupe_radiology_path_entries(radiology_download_entries)
        if downloaded_radiology_only:
            has_radiology = bool(radiology_paths)
        else:
            has_radiology = bool(
                radiology_paths
                or radiology_download_paths
                or radiology_uri_paths
                or tcia_study_uids
                or tcia_series_uids
                or radiology_series_fallback_entries
                or radiology_study_fallback_entries
            )

        split_group_id = make_split_group_id(
            source_name=source_name,
            project_id=project_id,
            patient_id=patient_id,
        )
        split = assign_split(split_group_id, split_ratios)
        sample_id = make_sample_id(source_name, patient_id, case_id or patient_id, modality_scope="patient_study")
        pathology_tile_embedding_paths: list[str] = []
        pathology_slide_embedding_paths: list[str] = []

        row = {
            "sample_id": sample_id,
            "source": source_name,
            "patient_id": patient_id,
            "study_id": case_id or patient_id,
            "split": split,
            "split_group_id": split_group_id,
            "split_scheme_version": split_scheme_version,
            "genomics_rna_bulk_paths": [],
            "genomics_rna_bulk_feature_path": "",
            "genomics_dna_methylation_paths": [],
            "genomics_dna_methylation_feature_path": "",
            "genomics_cnv_paths": [],
            "genomics_cnv_feature_path": "",
            "pathology_wsi_paths": pathology_paths,
            "radiology_image_paths": radiology_paths,
            "radiology_image_modalities": radiology_modalities,
            "radiology_report_download_paths": list(radiology_report_download_paths),
            "radiology_report_uri_paths": list(radiology_report_uri_paths),
            "radiology_report_series_descriptions": list(radiology_report_series_descriptions),
            "pathology_mask_paths": [],
            "pathology_segmentation_slide_image_paths": [],
            "pathology_segmentation_overlay_paths": [],
            "pathology_segmentation_metadata_paths": [],
            "radiology_mask_paths": list(radiology_mask_paths),
            "pathology_tile_embedding_paths": list(pathology_tile_embedding_paths),
            "pathology_slide_embedding_paths": list(pathology_slide_embedding_paths),
            "radiology_embedding_paths": list(radiology_embedding_paths),
            "radiology_mask_manifest_paths": list(radiology_mask_manifest_paths),
            "radiology_png_dirs": list(radiology_png_dirs),
            "radiology_series_slice_counts": list(radiology_series_slice_counts),
            "biomarkers_text": build_biomarkers_text(case),
            "question": "",
            "answer": "",
            "project_id": project_id,
            "primary_site": str(case.get("primary_site", "")),
            "disease_type": str(case.get("disease_type", "")),
            "primary_diagnosis": str(diagnosis.get("primary_diagnosis", "")),
            "tumor_grade": str(diagnosis.get("tumor_grade", "")),
            "tumor_stage": str(diagnosis.get("tumor_stage", "")),
            "ajcc_pathologic_stage": str(diagnosis.get("ajcc_pathologic_stage", "")),
            "ajcc_pathologic_t": str(diagnosis.get("ajcc_pathologic_t", "")),
            "ajcc_pathologic_n": str(diagnosis.get("ajcc_pathologic_n", "")),
            "ajcc_pathologic_m": str(diagnosis.get("ajcc_pathologic_m", "")),
            "age_at_diagnosis": str(diagnosis.get("age_at_diagnosis", "")),
            "morphology": str(diagnosis.get("morphology", "")),
            "last_known_disease_status": str(diagnosis.get("last_known_disease_status", "")),
            "days_to_last_known_disease_status": str(diagnosis.get("days_to_last_known_disease_status", "")),
            "days_to_recurrence": str(diagnosis.get("days_to_recurrence", "")),
            "vital_status": str(vital_status),
            "days_to_last_follow_up": days_to_last_follow_up,
            "days_to_death": str(days_to_death),
            "gender": str(demographic.get("gender", "")),
            "race": str(demographic.get("race", "")),
            "ethnicity": str(demographic.get("ethnicity", "")),
            "year_of_birth": str(demographic.get("year_of_birth", "")),
            "kidney_histology_subtype": kidney_histology_subtype,
            "task_grade_label": str(diagnosis.get("tumor_grade", "")),
            "task_stage_label": str(diagnosis.get("ajcc_pathologic_stage", "")),
            "task_survival_event": survival_event,
            "task_survival_days": overall_survival_days_numeric,
            "mutation_ssm_ids": mutation_ssm_ids,
            "mutation_types": mutation_types,
            "mutation_consequence_terms": mutation_consequence_terms,
            "mutated_gene_symbols": mutated_gene_symbols,
            "mutation_query_succeeded": bool(mutation_query_succeeded),
            "mutation_panel_version": mutation_panel_version,
            "mutation_panel_observed": mutation_observed if mutation_query_succeeded else None,
            "mutation_event_count": len(mutation_entries) if mutation_query_succeeded else None,
            "mutation_unique_gene_count": len(mutated_gene_symbols) if mutation_query_succeeded else None,
            "project_driver_gene_mutations": project_driver_gene_mutations,
            "pathology_file_ids": pathology_file_ids,
            "tcia_collections": sorted(set(tcia_collections)),
            "tcia_study_uids": sorted(set(tcia_study_uids)),
            "tcia_series_uids": tcia_series_uids,
            "tcia_modalities": _unique_sorted_non_empty(tcia_modalities),
            "tcia_body_parts": tcia_body_parts,
            "tcia_study_dates": sorted(set(tcia_study_dates)),
            "tcia_study_descriptions": sorted(set(tcia_study_descriptions)),
            "tcia_series_descriptions": tcia_series_descriptions,
            "radiology_uri_paths": sorted(set(radiology_uri_paths)),
            "radiology_download_paths": radiology_download_paths,
            "report_pdf_paths": report_pdf_paths,
            "report_file_ids": report_file_ids,
            "report_file_names": report_file_names,
            "has_pathology": bool(pathology_paths),
            "has_radiology": has_radiology,
        }
        for gene in mutation_gene_panel_upper:
            row[f"mutation_{gene.lower()}"] = (gene in mutation_gene_set) if mutation_query_succeeded else None
        rows.append(row)

    if show_progress:
        case_iterable.close()

    frame = pd.DataFrame(rows)
    if "sample_id" in frame.columns:
        # Guard against accidental duplicate API hits for the same case/sample.
        frame = frame.drop_duplicates(subset=["sample_id"], keep="last").reset_index(drop=True)
    frame = normalize_registry_df(frame)
    return frame[CORE_COLUMNS + [col for col in frame.columns if col not in CORE_COLUMNS]]
