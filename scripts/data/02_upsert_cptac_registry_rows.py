#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.manifest import write_run_manifest
from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.sources.cptac import (
    build_cptac_registry_rows,
    download_cptac_files,
    fetch_cptac_cases,
    fetch_cptac_files,
    fetch_cptac_tcia_metadata,
    resolve_existing_cptac_files,
)
from kidney_vlm.data.sources.tcga import GDCClient, TCIAClient, normalize_tcia_modality
from kidney_vlm.data.unified_registry import (
    expected_source_row_counts_after_replace,
    replace_source_slice,
    source_row_counts,
)
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _string_list(values: Any) -> list[str]:
    items: list[str] = []
    for value in list(values or []):
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _selected_primary_sites(cancer_groups: Any) -> list[str]:
    primary_sites: list[str] = []
    for group in list(cancer_groups or []):
        for primary_site in _string_list(group.get("primary_sites", [])):
            if primary_site not in primary_sites:
                primary_sites.append(primary_site)
    if not primary_sites:
        raise ValueError("CPTAC config selected no primary sites.")
    return primary_sites


def _selected_tcia_collections(cancer_groups: Any) -> list[str]:
    collections: list[str] = []
    for group in list(cancer_groups or []):
        for collection in _string_list(group.get("tcia_collections", [])):
            if collection not in collections:
                collections.append(collection)
    return collections


def _fetch_cptac_file_payload(
    *,
    gdc_client: GDCClient,
    cptac_cfg: Any,
    primary_sites: list[str],
    payload_name: str,
    max_files_key: str,
) -> list[dict[str, Any]]:
    payload_cfg = cptac_cfg.gdc[payload_name]
    return fetch_cptac_files(
        gdc_client,
        primary_sites=primary_sites,
        data_categories=_string_list(payload_cfg.data_categories),
        data_types=_string_list(payload_cfg.data_types),
        data_formats=_string_list(payload_cfg.data_formats),
        experimental_strategies=_string_list(payload_cfg.experimental_strategies),
        workflow_types=_string_list(payload_cfg.workflow_types),
        access=_string_list(cptac_cfg.gdc.access),
        sample_types=_string_list(cptac_cfg.gdc.sample_types),
        max_files=_optional_int(cptac_cfg.gdc.get(max_files_key)),
    )


def _fetch_cptac_report_files(
    *,
    gdc_client: GDCClient,
    cptac_cfg: Any,
    primary_sites: list[str],
) -> list[dict[str, Any]]:
    report_cfg = cptac_cfg.gdc.reports
    return fetch_cptac_files(
        gdc_client,
        primary_sites=primary_sites,
        data_categories=_string_list(report_cfg.data_categories),
        data_types=_string_list(report_cfg.data_types),
        data_formats=_string_list(report_cfg.data_formats),
        access=_string_list(cptac_cfg.gdc.access),
        sample_types=_string_list(cptac_cfg.gdc.sample_types),
        max_files=_optional_int(cptac_cfg.gdc.max_report_files),
    )


def _fetch_cptac_payloads(
    *,
    cptac_cfg: Any,
    primary_sites: list[str],
    tcia_collections: list[str],
    gdc_client: GDCClient,
    tcia_client: TCIAClient | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, str]]],
]:
    stage_labels = [
        "cases",
        "RNA STAR-count files",
        "DNAm SeSAMe beta files",
        "masked MAF mutation files",
        "pathology/clinical report PDFs",
        "TCIA radiology metadata" if tcia_client is not None else "TCIA radiology metadata (skipped)",
    ]
    progress = tqdm(total=len(stage_labels), desc="Pulling CPTAC metadata", unit="stage", leave=True)
    try:
        progress.set_description_str("Pulling CPTAC metadata: cases")
        cases = fetch_cptac_cases(
            gdc_client,
            primary_sites=primary_sites,
            max_cases=_optional_int(cptac_cfg.gdc.max_cases),
        )
        progress.update(1)

        progress.set_description_str("Pulling CPTAC metadata: RNA STAR-count files")
        rna_bulk_files = _fetch_cptac_file_payload(
            gdc_client=gdc_client,
            cptac_cfg=cptac_cfg,
            primary_sites=primary_sites,
            payload_name="rna_bulk",
            max_files_key="max_rna_bulk_files",
        )
        progress.update(1)

        progress.set_description_str("Pulling CPTAC metadata: DNAm SeSAMe beta files")
        dnam_files = _fetch_cptac_file_payload(
            gdc_client=gdc_client,
            cptac_cfg=cptac_cfg,
            primary_sites=primary_sites,
            payload_name="dna_methylation",
            max_files_key="max_dna_methylation_files",
        )
        progress.update(1)

        progress.set_description_str("Pulling CPTAC metadata: masked MAF mutation files")
        mutation_files = _fetch_cptac_file_payload(
            gdc_client=gdc_client,
            cptac_cfg=cptac_cfg,
            primary_sites=primary_sites,
            payload_name="mutation",
            max_files_key="max_mutation_files",
        )
        progress.update(1)

        progress.set_description_str("Pulling CPTAC metadata: pathology/clinical report PDFs")
        report_files = _fetch_cptac_report_files(
            gdc_client=gdc_client,
            cptac_cfg=cptac_cfg,
            primary_sites=primary_sites,
        )
        progress.update(1)

        progress.set_description_str("Pulling CPTAC metadata: TCIA radiology metadata")
        tcia_studies_by_patient: dict[str, list[dict[str, Any]]] = {}
        tcia_series_by_patient: dict[str, list[dict[str, str]]] = {}
        if tcia_client is not None:
            tcia_studies_by_patient, tcia_series_by_patient = fetch_cptac_tcia_metadata(
                tcia_client,
                collections=tcia_collections,
                max_studies_per_collection=_optional_int(cptac_cfg.tcia.max_studies_per_collection),
                fetch_series_metadata=bool(cptac_cfg.tcia.fetch_series_metadata),
                max_series_per_study=_optional_int(cptac_cfg.tcia.max_series_per_study_metadata),
            )
        progress.update(1)
    finally:
        progress.close()

    return (
        cases,
        rna_bulk_files,
        dnam_files,
        mutation_files,
        report_files,
        tcia_studies_by_patient,
        tcia_series_by_patient,
    )


def _download_cptac_tcia_series(
    tcia_client: TCIAClient,
    *,
    tcia_series_by_patient: dict[str, list[dict[str, str]]],
    raw_root: Path,
    source_name: str,
    skip_existing: bool,
    included_modalities: set[str],
    max_series_total: int | None,
) -> tuple[dict[str, list[dict[str, str]]], int]:
    selected_series: list[dict[str, str]] = []
    seen_series_uids: set[str] = set()
    for patient_id in sorted(tcia_series_by_patient):
        for series in tcia_series_by_patient[patient_id]:
            series_uid = str(series.get("series_instance_uid", "")).strip()
            if not series_uid or series_uid in seen_series_uids:
                continue
            modality = normalize_tcia_modality(series.get("modality", ""))
            if included_modalities and modality not in included_modalities:
                continue
            selected_series.append({**series, "modality": modality})
            seen_series_uids.add(series_uid)

    if max_series_total is not None:
        selected_series = selected_series[:max_series_total]

    downloaded_by_patient: dict[str, list[dict[str, str]]] = {}
    for series in tqdm(selected_series, desc="Downloading CPTAC radiology series", unit="series", leave=False):
        collection = str(series.get("collection", "")).strip() or "unknown_collection"
        patient_id = str(series.get("patient_id", "")).strip() or "unknown_patient"
        study_uid = str(series.get("study_instance_uid", "")).strip() or "unknown_study"
        series_uid = str(series.get("series_instance_uid", "")).strip()
        output_path = raw_root / source_name / "radiology" / collection / patient_id / study_uid / f"{series_uid}.zip"
        resolved = tcia_client.download_series_zip(
            series_instance_uid=series_uid,
            output_path=output_path,
            skip_existing=skip_existing,
        )
        downloaded_by_patient.setdefault(patient_id, []).append(
            {
                **series,
                "collection": collection,
                "patient_id": patient_id,
                "study_instance_uid": study_uid,
                "series_instance_uid": series_uid,
                "local_path": str(resolved),
            }
        )

    return downloaded_by_patient, len(selected_series)


def main() -> None:
    overrides = sys.argv[1:]
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="data/sources/cptac.yaml",
        overrides=overrides,
    )

    source_name = str(cfg.data.source.name)
    cptac_cfg = cfg.data.source.cptac
    cancer_groups = list(cptac_cfg.cancer_groups or [])
    primary_sites = _selected_primary_sites(cancer_groups)
    tcia_collections = _selected_tcia_collections(cancer_groups)

    gdc_client = GDCClient(
        base_url=str(cptac_cfg.gdc.base_url),
        timeout_seconds=int(cptac_cfg.gdc.timeout_seconds),
        page_size=int(cptac_cfg.gdc.page_size),
        max_retries=int(cptac_cfg.gdc.max_retries),
        retry_backoff_seconds=float(cptac_cfg.gdc.retry_backoff_seconds),
    )
    tcia_client = None
    if bool(cptac_cfg.tcia.enabled):
        tcia_client = TCIAClient(
            base_url=str(cptac_cfg.tcia.base_url),
            api_version=str(cptac_cfg.tcia.api_version),
            timeout_seconds=int(cptac_cfg.tcia.timeout_seconds),
            max_retries=int(cptac_cfg.tcia.max_retries),
            retry_backoff_seconds=float(cptac_cfg.tcia.retry_backoff_seconds),
        )

    print(f"Pulling CPTAC primary sites: {primary_sites}")
    if tcia_client is not None:
        print(f"Pulling TCIA collections: {tcia_collections}")

    (
        cases,
        rna_bulk_files,
        dnam_files,
        mutation_files,
        report_files,
        tcia_studies_by_patient,
        tcia_series_by_patient,
    ) = _fetch_cptac_payloads(
        cptac_cfg=cptac_cfg,
        primary_sites=primary_sites,
        tcia_collections=tcia_collections,
        gdc_client=gdc_client,
        tcia_client=tcia_client,
    )

    download_cfg = cfg.data.source.download
    download_enabled = bool(download_cfg.enabled)
    skip_existing = bool(download_cfg.skip_existing)
    raw_root = Path(str(cfg.data.raw_root))
    downloaded_rna_bulk_by_file_id = resolve_existing_cptac_files(
        rna_bulk_files,
        raw_root=raw_root,
        source_name=source_name,
        subfolder="rna_bulk",
    )
    downloaded_dnam_by_file_id = resolve_existing_cptac_files(
        dnam_files,
        raw_root=raw_root,
        source_name=source_name,
        subfolder="dna_methylation",
    )
    downloaded_mutation_by_file_id = resolve_existing_cptac_files(
        mutation_files,
        raw_root=raw_root,
        source_name=source_name,
        subfolder="mutation",
    )
    downloaded_tcia_series_by_patient: dict[str, list[dict[str, str]]] = {}
    rna_bulk_download_count = 0
    dnam_download_count = 0
    mutation_download_count = 0
    radiology_download_count = 0

    if download_enabled:
        if bool(download_cfg.include.rna_bulk):
            downloaded_paths, rna_bulk_download_count = download_cptac_files(
                gdc_client,
                rna_bulk_files,
                raw_root=raw_root,
                source_name=source_name,
                subfolder="rna_bulk",
                skip_existing=skip_existing,
                max_downloads=_optional_int(download_cfg.max_rna_bulk_downloads),
                progress_desc="Downloading CPTAC RNA STAR-count files",
            )
            downloaded_rna_bulk_by_file_id.update(downloaded_paths)
            print(f"CPTAC RNA files downloaded/resolved: {rna_bulk_download_count}")

        if bool(download_cfg.include.dna_methylation):
            downloaded_paths, dnam_download_count = download_cptac_files(
                gdc_client,
                dnam_files,
                raw_root=raw_root,
                source_name=source_name,
                subfolder="dna_methylation",
                skip_existing=skip_existing,
                max_downloads=_optional_int(download_cfg.max_dna_methylation_downloads),
                progress_desc="Downloading CPTAC DNAm beta files",
            )
            downloaded_dnam_by_file_id.update(downloaded_paths)
            print(f"CPTAC DNAm files downloaded/resolved: {dnam_download_count}")

        if bool(download_cfg.include.mutation):
            downloaded_paths, mutation_download_count = download_cptac_files(
                gdc_client,
                mutation_files,
                raw_root=raw_root,
                source_name=source_name,
                subfolder="mutation",
                skip_existing=skip_existing,
                max_downloads=_optional_int(download_cfg.max_mutation_downloads),
                progress_desc="Downloading CPTAC mutation MAF files",
            )
            downloaded_mutation_by_file_id.update(downloaded_paths)
            print(f"CPTAC mutation files downloaded/resolved: {mutation_download_count}")

        if bool(download_cfg.include.radiology):
            if tcia_client is None:
                raise ValueError("CPTAC radiology download requested, but TCIA is disabled.")
            if not bool(cptac_cfg.tcia.fetch_series_metadata):
                raise ValueError("CPTAC radiology download requires data.source.cptac.tcia.fetch_series_metadata=true.")
            included_modalities = {
                normalize_tcia_modality(value)
                for value in _string_list(cptac_cfg.tcia.download_modalities)
                if normalize_tcia_modality(value)
            }
            downloaded_tcia_series_by_patient, radiology_download_count = _download_cptac_tcia_series(
                tcia_client,
                tcia_series_by_patient=tcia_series_by_patient,
                raw_root=raw_root,
                source_name=source_name,
                skip_existing=skip_existing,
                included_modalities=included_modalities,
                max_series_total=_optional_int(download_cfg.max_radiology_series_downloads),
            )
            print(f"CPTAC radiology series downloaded/resolved: {radiology_download_count}")

    source_df = build_cptac_registry_rows(
        cases=cases,
        cancer_groups=cancer_groups,
        rna_bulk_files=rna_bulk_files,
        dnam_files=dnam_files,
        mutation_files=mutation_files,
        report_files=report_files,
        tcia_studies_by_patient=tcia_studies_by_patient,
        tcia_series_by_patient=tcia_series_by_patient,
        downloaded_rna_bulk_by_file_id=downloaded_rna_bulk_by_file_id,
        downloaded_dnam_by_file_id=downloaded_dnam_by_file_id,
        downloaded_mutation_by_file_id=downloaded_mutation_by_file_id,
        downloaded_tcia_series_by_patient=downloaded_tcia_series_by_patient,
        raw_root=raw_root,
        project_root=ROOT,
        source_name=source_name,
        split_name=str(cptac_cfg.split_name),
        show_progress=True,
        progress_desc="Preparing CPTAC registry rows",
    )

    staging_root = Path(str(cfg.data.staging_root))
    staging_path = staging_root / f"{source_name}.parquet"
    write_registry_parquet(source_df, staging_path, validate=False)

    unified_path = Path(str(cfg.data.unified_registry_path))
    unified_df = read_parquet_or_empty(unified_path)
    merged_df = replace_source_slice(unified_df, source_df, source_name=source_name)
    expected_source_counts = expected_source_row_counts_after_replace(
        unified_df,
        source_df,
        source_name=source_name,
    )
    write_registry_parquet(merged_df, unified_path, validate=False)
    written_source_counts = source_row_counts(read_parquet_or_empty(unified_path))
    if written_source_counts != expected_source_counts:
        raise RuntimeError(
            "Unified registry source counts changed unexpectedly after CPTAC write. "
            f"Expected {expected_source_counts}, found {written_source_counts}."
        )

    manifest_path = write_run_manifest(
        manifests_root=Path(str(cfg.data.manifests_root)),
        repo_root=ROOT,
        source_name=source_name,
        source_row_count=len(source_df),
        staging_path=staging_path,
        unified_path=unified_path,
        extra={
            "primary_sites": primary_sites,
            "tcia_collections": tcia_collections,
            "download_enabled": download_enabled,
            "endpoint_filters": {
                "gdc_program": "CPTAC",
                "gdc_sample_types": _string_list(cptac_cfg.gdc.sample_types),
                "rna_bulk_workflow_types": _string_list(cptac_cfg.gdc.rna_bulk.workflow_types),
                "dna_methylation_workflow_types": _string_list(cptac_cfg.gdc.dna_methylation.workflow_types),
                "mutation_workflow_types": _string_list(cptac_cfg.gdc.mutation.workflow_types),
                "tcia_collections": tcia_collections,
            },
            "api_counts": {
                "cases": len(cases),
                "rna_bulk_files": len(rna_bulk_files),
                "dna_methylation_files": len(dnam_files),
                "mutation_files": len(mutation_files),
                "report_files": len(report_files),
                "radiology_patients": len(tcia_studies_by_patient),
                "tcia_series_patients": len(tcia_series_by_patient),
                "tcia_series_records": sum(len(entries) for entries in tcia_series_by_patient.values()),
            },
            "download_counts": {
                "rna_bulk_files": rna_bulk_download_count,
                "dna_methylation_files": dnam_download_count,
                "mutation_files": mutation_download_count,
                "radiology_series": radiology_download_count,
            },
            "notes": (
                "CPTAC external validation source refresh for kidney, lung, and uterus. "
                "GDC report endpoint is queried, but current checked result is zero PDF reports."
            ),
        },
    )

    print(f"CPTAC source registry upsert complete: {source_name}")
    print(f"Primary sites: {primary_sites}")
    print(f"Cases pulled: {len(cases)}")
    print(f"RNA STAR-count files pulled: {len(rna_bulk_files)}")
    print(f"DNAm SeSAMe beta files pulled: {len(dnam_files)}")
    print(f"Masked MAF mutation files pulled: {len(mutation_files)}")
    print(f"Pathology/clinical report PDF files pulled: {len(report_files)}")
    print(f"Radiology patients pulled: {len(tcia_studies_by_patient)}")
    print(f"TCIA series metadata records pulled: {sum(len(entries) for entries in tcia_series_by_patient.values())}")
    print(f"Rows written: {len(source_df)}")
    print(f"Staging parquet: {staging_path}")
    print(f"Unified parquet: {unified_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
