from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests

from kidney_vlm.data.id_factory import make_sample_id
from kidney_vlm.data.registry_schema import CORE_COLUMNS, empty_registry_frame, normalize_registry_df

DEFAULT_CASE_FIELDS = [
    "case_id",
    "submitter_id",
    "project.project_id",
    "primary_site",
    "disease_type",
    "diagnoses.primary_diagnosis",
    "diagnoses.tumor_grade",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.vital_status",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_death",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
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

        return response.json()

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
    ) -> dict[str, list[dict[str, str]]]:
        by_patient: dict[str, list[dict[str, str]]] = {}

        for collection in collections:
            studies = self.fetch_patient_studies(collection=collection, max_studies=max_studies_per_collection)
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
                entry = {
                    "collection": str(collection),
                    "patient_id": str(patient_id),
                    "study_instance_uid": str(study_uid),
                }
                by_patient.setdefault(str(patient_id), []).append(entry)

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

    pairs = [
        ("project", project_id),
        ("disease_type", case.get("disease_type")),
        ("primary_diagnosis", diagnosis.get("primary_diagnosis")),
        ("tumor_grade", diagnosis.get("tumor_grade")),
        ("ajcc_pathologic_stage", diagnosis.get("ajcc_pathologic_stage")),
        ("vital_status", diagnosis.get("vital_status")),
        ("gender", demographic.get("gender")),
        ("race", demographic.get("race")),
        ("ethnicity", demographic.get("ethnicity")),
    ]

    parts = [f"{key}: {value}" for key, value in pairs if value not in (None, "")]
    return "; ".join(parts)


def assign_split(submitter_id: str, split_ratios: dict[str, float]) -> str:
    train_ratio = float(split_ratios.get("train", 0.9))
    val_ratio = float(split_ratios.get("val", 0.0))
    test_ratio = float(split_ratios.get("test", 0.1))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        return "train"

    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    digest = hashlib.sha256(str(submitter_id).encode("utf-8")).hexdigest()
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
    tcia_studies_by_patient: dict[str, list[dict[str, str]]],
    raw_root: Path,
    source_name: str,
    split_ratios: dict[str, float],
    report_files: list[dict[str, Any]] | None = None,
    downloaded_pathology_by_file_id: dict[str, str] | None = None,
    downloaded_reports_by_file_id: dict[str, str] | None = None,
    downloaded_tcia_series_by_patient: dict[str, list[dict[str, str]]] | None = None,
) -> pd.DataFrame:
    if not cases:
        return empty_registry_frame()

    report_files = report_files or []
    downloaded_pathology_by_file_id = downloaded_pathology_by_file_id or {}
    downloaded_reports_by_file_id = downloaded_reports_by_file_id or {}
    downloaded_tcia_series_by_patient = downloaded_tcia_series_by_patient or {}

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
                "local_path": str(local_path),
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
                "local_path": str(local_path),
            }
            if case_id:
                reports_by_case.setdefault(case_id, []).append(report_entry)
            if patient_id:
                reports_by_patient.setdefault(patient_id, []).append(report_entry)

    rows: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case.get("case_id", "")).strip()
        patient_id = str(case.get("submitter_id", "")).strip()
        if not patient_id:
            continue

        project_id = str((case.get("project") or {}).get("project_id", "")).strip()
        diagnosis = _first_diagnosis(case)
        demographic = case.get("demographic", {}) if isinstance(case.get("demographic", {}), dict) else {}

        pathology_entries = pathology_by_case.get(case_id, pathology_by_patient.get(patient_id, []))
        pathology_paths = sorted({entry["local_path"] for entry in pathology_entries if entry.get("local_path")})
        pathology_file_ids = sorted({entry["file_id"] for entry in pathology_entries if entry.get("file_id")})
        report_entries = reports_by_case.get(case_id, reports_by_patient.get(patient_id, []))
        report_pdf_paths = sorted({entry["local_path"] for entry in report_entries if entry.get("local_path")})
        report_file_ids = sorted({entry["file_id"] for entry in report_entries if entry.get("file_id")})
        report_file_names = sorted({entry["file_name"] for entry in report_entries if entry.get("file_name")})

        radiology_entries = tcia_studies_by_patient.get(patient_id, [])
        radiology_paths: list[str] = []
        radiology_uri_paths: list[str] = []
        tcia_study_uids = []
        for entry in radiology_entries:
            collection = str(entry.get("collection", ""))
            study_uid = str(entry.get("study_instance_uid", ""))
            if study_uid:
                radiology_uri_paths.append(f"tcia://{collection}/{patient_id}/{study_uid}")
                tcia_study_uids.append(study_uid)
            elif collection:
                radiology_uri_paths.append(f"tcia://{collection}/{patient_id}")

        downloaded_series_entries = downloaded_tcia_series_by_patient.get(patient_id, [])
        tcia_series_uids = sorted(
            {
                str(entry.get("series_instance_uid", ""))
                for entry in downloaded_series_entries
                if str(entry.get("series_instance_uid", "")).strip()
            }
        )
        radiology_download_paths = sorted(
            {
                str(entry.get("local_path", ""))
                for entry in downloaded_series_entries
                if str(entry.get("local_path", "")).strip()
            }
        )
        radiology_paths = radiology_download_paths if radiology_download_paths else sorted(set(radiology_uri_paths))

        split = assign_split(patient_id, split_ratios)
        sample_id = make_sample_id(source_name, patient_id, case_id or patient_id, modality_scope="patient_study")

        row = {
            "sample_id": sample_id,
            "source": source_name,
            "patient_id": patient_id,
            "study_id": case_id or patient_id,
            "split": split,
            "pathology_wsi_paths": pathology_paths,
            "radiology_image_paths": sorted(set(radiology_paths)),
            "pathology_mask_paths": [],
            "radiology_mask_paths": [],
            "pathology_feature_paths": [],
            "radiology_feature_paths": [],
            "biomarkers_text": build_biomarkers_text(case),
            "question": "",
            "answer": "",
            "project_id": project_id,
            "primary_site": str(case.get("primary_site", "")),
            "disease_type": str(case.get("disease_type", "")),
            "primary_diagnosis": str(diagnosis.get("primary_diagnosis", "")),
            "tumor_grade": str(diagnosis.get("tumor_grade", "")),
            "ajcc_pathologic_stage": str(diagnosis.get("ajcc_pathologic_stage", "")),
            "vital_status": str(diagnosis.get("vital_status", "")),
            "days_to_last_follow_up": str(diagnosis.get("days_to_last_follow_up", "")),
            "days_to_death": str(diagnosis.get("days_to_death", "")),
            "gender": str(demographic.get("gender", "")),
            "race": str(demographic.get("race", "")),
            "ethnicity": str(demographic.get("ethnicity", "")),
            "pathology_file_ids": pathology_file_ids,
            "tcia_study_uids": sorted(set(tcia_study_uids)),
            "tcia_series_uids": tcia_series_uids,
            "radiology_uri_paths": sorted(set(radiology_uri_paths)),
            "radiology_download_paths": radiology_download_paths,
            "report_pdf_paths": report_pdf_paths,
            "report_file_ids": report_file_ids,
            "report_file_names": report_file_names,
            "has_pathology": bool(pathology_paths),
            "has_radiology": bool(radiology_paths),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    if "sample_id" in frame.columns:
        # Guard against accidental duplicate API hits for the same case/sample.
        frame = frame.drop_duplicates(subset=["sample_id"], keep="last").reset_index(drop=True)
    frame = normalize_registry_df(frame)
    return frame[CORE_COLUMNS + [col for col in frame.columns if col not in CORE_COLUMNS]]
