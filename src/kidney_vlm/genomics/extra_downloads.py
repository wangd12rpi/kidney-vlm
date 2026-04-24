"""
GDC query and download-plan helpers for the extra genomics modalities.

These mirror the patterns used in `kidney_vlm.data.sources.tcga` for pathology
and RNA-bulk downloads, but are parameterized over a single
`ExtraGenomicsDownloadSpec` so that we can stamp out DNAm, CNV, MAF and miRNA
pipelines uniformly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExtraGenomicsDownloadSpec:
    """Specification for one extra-genomics modality download."""

    key: str                       # short identifier, e.g. "dna_methylation_beta"
    subfolder: str                 # subfolder name under <raw_root>/<source_name>/
    data_category: str             # GDC data_category, e.g. "DNA Methylation"
    data_types: list[str]          # GDC data_type values to accept
    experimental_strategies: list[str]
    data_formats: list[str]
    workflow_types: list[str]
    access: str = "open"           # "open" | "controlled"
    description: str = ""          # human-readable description for logs


# ---------------------------------------------------------------------------
# GDC filter assembly
# ---------------------------------------------------------------------------


def _in_filter(field: str, values: list[str]) -> dict[str, Any]:
    return {"op": "in", "content": {"field": field, "value": list(values)}}


def _and_filter(filters: list[dict[str, Any]]) -> dict[str, Any]:
    return {"op": "and", "content": filters}


def _spec_to_gdc_filter(spec: ExtraGenomicsDownloadSpec, project_ids: list[str]) -> dict[str, Any]:
    filters = [
        _in_filter("cases.project.project_id", project_ids),
        _in_filter("data_category", [spec.data_category]),
        _in_filter("access", [spec.access]),
    ]
    if spec.data_types:
        filters.append(_in_filter("data_type", spec.data_types))
    if spec.experimental_strategies:
        filters.append(_in_filter("experimental_strategy", spec.experimental_strategies))
    if spec.data_formats:
        filters.append(_in_filter("data_format", spec.data_formats))
    if spec.workflow_types:
        filters.append(_in_filter("analysis.workflow_type", spec.workflow_types))
    return _and_filter(filters)


def spec_to_gdc_filter(
    spec: ExtraGenomicsDownloadSpec,
    *,
    project_ids: list[str],
    patient_submitter_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Public filter builder for tests and scripts."""
    filters = list(_spec_to_gdc_filter(spec, project_ids).get("content", []))
    normalized_patient_ids = [
        str(patient_id).strip()
        for patient_id in (patient_submitter_ids or [])
        if str(patient_id).strip()
    ]
    if normalized_patient_ids:
        filters.append(_in_filter("cases.submitter_id", sorted(set(normalized_patient_ids))))
    return _and_filter(filters)


# The set of file-level fields we want the GDC to return. Kept small so that
# responses stay under the GDC page-size cap and the manifest stays lean.
_GDC_FILE_FIELDS = ",".join(
    [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "data_category",
        "data_type",
        "data_format",
        "experimental_strategy",
        "analysis.workflow_type",
        "access",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.samples.submitter_id",
        "cases.samples.sample_type",
        "cases.samples.portions.analytes.aliquots.submitter_id",
    ]
)


def fetch_extra_genomics_files(
    *,
    gdc_client: Any,
    project_ids: list[str],
    spec: ExtraGenomicsDownloadSpec,
    max_files: int | None = None,
    patient_submitter_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Query the GDC `files` endpoint with the filters assembled from `spec`.

    Uses the GDC client's generic `fetch_files` helper if it exists, otherwise
    falls back to `_fetch_paginated`. Both are present in the existing
    kidney_vlm.data.sources.tcga.GDCClient; we probe for the best entry point
    at runtime so this module stays decoupled from GDCClient internals.
    """
    gdc_filter = spec_to_gdc_filter(
        spec,
        project_ids=project_ids,
        patient_submitter_ids=patient_submitter_ids,
    )
    params = {
        "filters": json.dumps(gdc_filter),
        "fields": _GDC_FILE_FIELDS,
        "format": "JSON",
    }

    # Prefer a generic `fetch_files(filters=..., fields=..., max_files=...)`
    # entry point if the client exposes one. The existing GDCClient already
    # uses this shape internally for pathology / RNA queries.
    if hasattr(gdc_client, "fetch_files"):
        return gdc_client.fetch_files(
            filters=gdc_filter,
            fields=_GDC_FILE_FIELDS,
            max_files=max_files,
        )

    # Fall back to paginated GET on /files
    return _fallback_fetch_files(gdc_client, params=params, max_files=max_files)


def _fallback_fetch_files(
    gdc_client: Any,
    *,
    params: dict[str, Any],
    max_files: int | None,
) -> list[dict[str, Any]]:
    """Minimal paginated GET for GDCClients that lack fetch_files()."""
    results: list[dict[str, Any]] = []
    page = 1
    size = int(getattr(gdc_client, "page_size", 500))
    while True:
        page_params = dict(params)
        page_params.update({"size": size, "from": (page - 1) * size})
        hits = gdc_client._get_json("/files", params=page_params)  # type: ignore[attr-defined]
        data = (hits.get("data") or {}).get("hits") or []
        if not data:
            break
        results.extend(data)
        if max_files is not None and len(results) >= max_files:
            return results[:max_files]
        if len(data) < size:
            break
        page += 1
    return results


# ---------------------------------------------------------------------------
# Download-plan assembly (mirrors 01_upsert_tcga_registry_rows._build_gdc_download_plan)
# ---------------------------------------------------------------------------


def _first_linked_case(file_hit: dict[str, Any]) -> tuple[str, str, str, str, str]:
    linked_cases = file_hit.get("cases", []) or []
    for linked in linked_cases:
        if not isinstance(linked, dict):
            continue
        case_id = str(linked.get("case_id", "")).strip()
        patient_id = str(linked.get("submitter_id", "")).strip()
        project_id = str((linked.get("project") or {}).get("project_id", "")).strip()

        sample_submitter = ""
        aliquot_submitter = ""
        samples = linked.get("samples") or []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sample_submitter = str(sample.get("submitter_id", "")).strip() or sample_submitter
            portions = sample.get("portions") or []
            for portion in portions:
                analytes = (portion or {}).get("analytes") or []
                for analyte in analytes:
                    aliquots = (analyte or {}).get("aliquots") or []
                    for aliquot in aliquots:
                        aliquot_submitter = (
                            str(aliquot.get("submitter_id", "")).strip() or aliquot_submitter
                        )
            if sample_submitter or aliquot_submitter:
                break

        if patient_id or case_id:
            return case_id, patient_id, project_id, sample_submitter, aliquot_submitter
    return "", "", "", "", ""


def build_extra_genomics_download_plan(
    file_hits: list[dict[str, Any]],
    *,
    raw_root: Path,
    source_name: str,
    subfolder: str,
) -> list[dict[str, Any]]:
    """Mirror `_build_gdc_download_plan` from 01_upsert_tcga_registry_rows, with
    extra fields (sample / aliquot submitter IDs, workflow type) that the
    downstream text-feature pipeline needs to resolve the right aliquot per case.
    """
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()

    for file_hit in file_hits:
        file_id = str(file_hit.get("file_id", "")).strip()
        file_name = str(file_hit.get("file_name", "")).strip()
        if not file_id or not file_name or file_id in seen:
            continue

        case_id, patient_id, project_id, sample_submitter, aliquot_submitter = _first_linked_case(
            file_hit
        )
        if not patient_id:
            patient_id = "unknown_patient"
        if not project_id:
            project_id = "unknown_project"

        workflow_type = str(((file_hit.get("analysis") or {}).get("workflow_type", ""))).strip()
        data_type = str(file_hit.get("data_type", "")).strip()
        data_format = str(file_hit.get("data_format", "")).strip()
        experimental_strategy = str(file_hit.get("experimental_strategy", "")).strip()
        file_size = file_hit.get("file_size")
        md5sum = str(file_hit.get("md5sum", "")).strip()

        output_path = raw_root / source_name / subfolder / project_id / patient_id / file_name
        plan.append(
            {
                "file_id": file_id,
                "file_name": file_name,
                "output_path": str(output_path),
                "project_id": project_id,
                "patient_id": patient_id,
                "case_id": case_id,
                "sample_submitter_id": sample_submitter,
                "aliquot_submitter_id": aliquot_submitter,
                "workflow_type": workflow_type,
                "data_type": data_type,
                "data_format": data_format,
                "experimental_strategy": experimental_strategy,
                "file_size": int(file_size) if file_size is not None else None,
                "md5sum": md5sum,
                "subfolder": subfolder,
            }
        )
        seen.add(file_id)

    return plan


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


def write_extra_genomics_manifest(
    *,
    manifests_by_key: dict[str, list[dict[str, Any]]],
    manifests_root: Path,
    source_name: str,
) -> Path:
    manifests_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows: list[dict[str, Any]] = []
    for key, entries in manifests_by_key.items():
        for entry in entries:
            rows.append({"modality": key, **entry})

    manifest_path = manifests_root / f"{source_name}_extra_genomics_{timestamp}.parquet"
    if rows:
        pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    else:
        pd.DataFrame(
            columns=["modality", "file_id", "file_name", "project_id", "patient_id"]
        ).to_parquet(manifest_path, index=False)
    return manifest_path
