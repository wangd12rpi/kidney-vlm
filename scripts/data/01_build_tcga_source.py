#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.data.manifest import write_run_manifest
from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.sources.tcga import (
    APIQueryError,
    GDCClient,
    TCIAClient,
    build_tcga_registry_rows,
    index_ssm_hits_by_case_and_patient,
)
from kidney_vlm.data.unified_registry import replace_source_slice

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg(source_name: str = "tcga", overrides: list[str] | None = None) -> DictConfig:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        base_cfg = compose(config_name="config")
    OmegaConf.set_struct(base_cfg, False)

    source_cfg_path = conf_dir / "data" / "sources" / f"{source_name}.yaml"
    if not source_cfg_path.exists():
        raise FileNotFoundError(f"Missing source config: {source_cfg_path}")

    source_cfg = OmegaConf.load(source_cfg_path)
    merged = OmegaConf.merge(base_cfg, source_cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, override_cfg)

    OmegaConf.set_struct(merged, False)
    merged.project.root_dir = str(ROOT)
    return merged


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _split_ratios(tcga_cfg: DictConfig) -> dict[str, float]:
    split_cfg = OmegaConf.to_container(tcga_cfg.split_ratios, resolve=True) or {}
    train_ratio = float(split_cfg.get("train", 0.9))
    test_ratio = float(split_cfg.get("test", 0.1))
    val_ratio = float(split_cfg.get("val", 0.0))
    return {"train": train_ratio, "val": val_ratio, "test": test_ratio}


def _fetch_tcga_payloads(
    tcga_cfg: DictConfig,
    gdc_client: GDCClient,
    tcia_client: TCIAClient | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, str]]],
    list[dict[str, Any]],
]:
    project_ids = [str(project_id) for project_id in list(tcga_cfg.project_ids)]
    if not project_ids:
        raise ValueError("No TCGA projects configured. Set data.source.tcga.project_ids in tcga.yaml.")

    cases = gdc_client.fetch_cases(
        project_ids=project_ids,
        max_cases=_optional_int(tcga_cfg.gdc.max_cases),
    )

    pathology_files = gdc_client.fetch_pathology_files(
        project_ids=project_ids,
        data_formats=[str(x) for x in list(tcga_cfg.gdc.pathology_data_formats)],
        data_types=[str(x) for x in list(tcga_cfg.gdc.pathology_data_types)],
        max_files=_optional_int(tcga_cfg.gdc.max_pathology_files),
    )

    case_ids = [str(case.get("case_id", "")).strip() for case in cases if str(case.get("case_id", "")).strip()]
    report_files = gdc_client.fetch_report_files(
        project_ids=project_ids,
        case_ids=case_ids,
        data_formats=[str(x) for x in list(tcga_cfg.gdc.report_data_formats)],
        data_types=[str(x) for x in list(tcga_cfg.gdc.report_data_types)],
        data_categories=[str(x) for x in list(tcga_cfg.gdc.report_data_categories)],
        max_files=_optional_int(tcga_cfg.gdc.max_report_files),
    )

    tcia_studies_by_patient: dict[str, list[dict[str, Any]]] = {}
    tcia_series_by_patient: dict[str, list[dict[str, str]]] = {}
    if tcia_client is not None and bool(tcga_cfg.tcia.enabled):
        configured_collections = [str(x) for x in list(tcga_cfg.tcia.collections)]
        collections = configured_collections or project_ids
        tcia_studies_by_patient = tcia_client.fetch_studies_by_patient(
            collections=collections,
            max_studies_per_collection=_optional_int(tcga_cfg.tcia.max_studies_per_collection),
        )
        if bool(tcga_cfg.tcia.fetch_series_metadata):
            tcia_series_by_patient = tcia_client.fetch_series_by_patient(
                studies_by_patient=tcia_studies_by_patient,
                max_series_per_study=_optional_int(tcga_cfg.tcia.max_series_per_study_metadata),
            )

    ssm_hits: list[dict[str, Any]] = []
    if bool(tcga_cfg.gdc.fetch_ssm_mutations):
        mutation_gene_panel = [str(gene) for gene in list(tcga_cfg.gdc.mutation_gene_panel)]
        ssm_hits = gdc_client.fetch_ssm_hits(
            project_ids=project_ids,
            gene_symbols=mutation_gene_panel,
            max_hits=_optional_int(tcga_cfg.gdc.max_ssm_hits),
        )

    return cases, pathology_files, report_files, tcia_studies_by_patient, tcia_series_by_patient, ssm_hits


def _first_linked_case(file_hit: dict[str, Any]) -> tuple[str, str, str]:
    linked_cases = file_hit.get("cases", [])
    if not isinstance(linked_cases, list):
        return "", "", ""
    for linked in linked_cases:
        if not isinstance(linked, dict):
            continue
        case_id = str(linked.get("case_id", "")).strip()
        patient_id = str(linked.get("submitter_id", "")).strip()
        project_id = str((linked.get("project") or {}).get("project_id", "")).strip()
        if patient_id or case_id:
            return case_id, patient_id, project_id
    return "", "", ""


def _build_gdc_download_plan(
    file_hits: list[dict[str, Any]],
    *,
    raw_root: Path,
    source_name: str,
    subfolder: str,
) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    seen_file_ids: set[str] = set()

    for file_hit in file_hits:
        file_id = str(file_hit.get("file_id", "")).strip()
        file_name = str(file_hit.get("file_name", "")).strip()
        if not file_id or not file_name or file_id in seen_file_ids:
            continue

        _case_id, patient_id, project_id = _first_linked_case(file_hit)
        if not patient_id:
            patient_id = "unknown_patient"
        if not project_id:
            project_id = "unknown_project"

        output_path = raw_root / source_name / subfolder / project_id / patient_id / file_name
        plan.append(
            {
                "file_id": file_id,
                "file_name": file_name,
                "output_path": str(output_path),
                "project_id": project_id,
                "patient_id": patient_id,
            }
        )
        seen_file_ids.add(file_id)

    return plan


def _download_gdc_plan(
    gdc_client: GDCClient,
    plan: list[dict[str, str]],
    *,
    skip_existing: bool,
    max_downloads: int | None = None,
) -> tuple[dict[str, str], int]:
    downloaded: dict[str, str] = {}
    completed = 0

    for item in plan:
        if max_downloads is not None and completed >= max_downloads:
            break
        file_id = item["file_id"]
        output_path = Path(item["output_path"])
        resolved = gdc_client.download_data_file(
            file_id=file_id,
            output_path=output_path,
            skip_existing=skip_existing,
        )
        downloaded[file_id] = str(resolved)
        completed += 1

    return downloaded, completed


def _download_tcia_series(
    tcia_client: TCIAClient,
    *,
    tcia_studies_by_patient: dict[str, list[dict[str, Any]]],
    patient_ids: set[str],
    raw_root: Path,
    source_name: str,
    skip_existing: bool,
    max_series_per_study: int | None,
    max_series_total: int | None,
) -> tuple[dict[str, list[dict[str, str]]], int]:
    downloaded_by_patient: dict[str, list[dict[str, str]]] = {}
    total_downloaded = 0

    for patient_id, studies in tcia_studies_by_patient.items():
        if patient_id not in patient_ids:
            continue

        for study in studies:
            if max_series_total is not None and total_downloaded >= max_series_total:
                return downloaded_by_patient, total_downloaded

            collection = str(study.get("collection", "")).strip() or "unknown_collection"
            study_uid = str(study.get("study_instance_uid", "")).strip()
            if not study_uid:
                continue

            series_list = tcia_client.fetch_series_for_study(study_uid, max_series=max_series_per_study)
            for series in series_list:
                if max_series_total is not None and total_downloaded >= max_series_total:
                    return downloaded_by_patient, total_downloaded

                series_uid = str(series.get("SeriesInstanceUID", "")).strip()
                if not series_uid:
                    continue
                modality = str(series.get("Modality", "")).strip()

                output_path = (
                    raw_root
                    / source_name
                    / "radiology"
                    / collection
                    / patient_id
                    / study_uid
                    / f"{series_uid}.zip"
                )
                resolved = tcia_client.download_series_zip(
                    series_instance_uid=series_uid,
                    output_path=output_path,
                    skip_existing=skip_existing,
                )
                downloaded_by_patient.setdefault(patient_id, []).append(
                    {
                        "collection": collection,
                        "patient_id": patient_id,
                        "study_instance_uid": study_uid,
                        "series_instance_uid": series_uid,
                        "modality": modality,
                        "local_path": str(resolved),
                    }
                )
                total_downloaded += 1

    return downloaded_by_patient, total_downloaded


def main() -> None:
    overrides = sys.argv[1:]
    cfg = load_cfg("tcga", overrides=overrides)

    source_name = str(cfg.data.source.name)
    tcga_cfg = cfg.data.source.tcga
    project_ids = [str(project_id) for project_id in list(tcga_cfg.project_ids)]

    gdc_client = GDCClient(
        base_url=str(tcga_cfg.gdc.base_url),
        timeout_seconds=int(tcga_cfg.gdc.timeout_seconds),
        page_size=int(tcga_cfg.gdc.page_size),
        max_retries=int(tcga_cfg.gdc.max_retries),
        retry_backoff_seconds=float(tcga_cfg.gdc.retry_backoff_seconds),
    )
    tcia_client = None
    if bool(tcga_cfg.tcia.enabled):
        tcia_client = TCIAClient(
            base_url=str(tcga_cfg.tcia.base_url),
            api_version=str(tcga_cfg.tcia.api_version),
            timeout_seconds=int(tcga_cfg.tcia.timeout_seconds),
            max_retries=int(tcga_cfg.tcia.max_retries),
            retry_backoff_seconds=float(tcga_cfg.tcia.retry_backoff_seconds),
        )

    print(f"Pulling metadata for projects: {project_ids}")
    cases, pathology_files, report_files, tcia_studies_by_patient, tcia_series_by_patient, ssm_hits = (
        _fetch_tcga_payloads(
            tcga_cfg=tcga_cfg,
            gdc_client=gdc_client,
            tcia_client=tcia_client,
        )
    )
    ssm_mutations_by_case_id, ssm_mutations_by_patient_id = index_ssm_hits_by_case_and_patient(ssm_hits)
    mutation_gene_panel = [str(gene) for gene in list(tcga_cfg.gdc.mutation_gene_panel)]

    download_cfg = cfg.data.source.download
    download_enabled = bool(download_cfg.enabled)
    skip_existing = bool(download_cfg.skip_existing)

    downloaded_pathology_by_file_id: dict[str, str] = {}
    downloaded_reports_by_file_id: dict[str, str] = {}
    downloaded_tcia_series_by_patient: dict[str, list[dict[str, str]]] = {}
    pathology_download_count = 0
    report_download_count = 0
    radiology_download_count = 0

    if download_enabled:
        raw_root = Path(str(cfg.data.raw_root))
        print("Download stage enabled. Starting payload downloads...")

        if bool(download_cfg.include.pathology):
            pathology_plan = _build_gdc_download_plan(
                pathology_files,
                raw_root=raw_root,
                source_name=source_name,
                subfolder="pathology",
            )
            downloaded_pathology_by_file_id, pathology_download_count = _download_gdc_plan(
                gdc_client,
                pathology_plan,
                skip_existing=skip_existing,
                max_downloads=_optional_int(download_cfg.max_pathology_downloads),
            )
            print(f"Pathology files downloaded/resolved: {pathology_download_count}")

        if bool(download_cfg.include.reports):
            report_plan = _build_gdc_download_plan(
                report_files,
                raw_root=raw_root,
                source_name=source_name,
                subfolder="reports",
            )
            downloaded_reports_by_file_id, report_download_count = _download_gdc_plan(
                gdc_client,
                report_plan,
                skip_existing=skip_existing,
                max_downloads=_optional_int(download_cfg.max_report_downloads),
            )
            print(f"Report PDF files downloaded/resolved: {report_download_count}")

        if bool(download_cfg.include.radiology) and tcia_client is not None:
            patient_ids = {
                str(case.get("submitter_id", "")).strip()
                for case in cases
                if str(case.get("submitter_id", "")).strip()
            }
            downloaded_tcia_series_by_patient, radiology_download_count = _download_tcia_series(
                tcia_client,
                tcia_studies_by_patient=tcia_studies_by_patient,
                patient_ids=patient_ids,
                raw_root=raw_root,
                source_name=source_name,
                skip_existing=skip_existing,
                max_series_per_study=_optional_int(download_cfg.max_series_per_study),
                max_series_total=_optional_int(download_cfg.max_radiology_series_downloads),
            )
            print(f"Radiology series zip files downloaded/resolved: {radiology_download_count}")

    source_df = build_tcga_registry_rows(
        cases=cases,
        pathology_files=pathology_files,
        report_files=report_files,
        tcia_studies_by_patient=tcia_studies_by_patient,
        tcia_series_by_patient=tcia_series_by_patient,
        downloaded_pathology_by_file_id=downloaded_pathology_by_file_id,
        downloaded_reports_by_file_id=downloaded_reports_by_file_id,
        ssm_mutations_by_case_id=ssm_mutations_by_case_id,
        ssm_mutations_by_patient_id=ssm_mutations_by_patient_id,
        mutation_gene_panel=mutation_gene_panel,
        downloaded_tcia_series_by_patient=downloaded_tcia_series_by_patient,
        raw_root=Path(str(cfg.data.raw_root)),
        source_name=source_name,
        split_ratios=_split_ratios(tcga_cfg),
        show_progress=True,
        progress_desc="Building tcga rows",
    )

    staging_root = Path(str(cfg.data.staging_root))
    staging_path = staging_root / f"{source_name}.parquet"
    write_registry_parquet(source_df, staging_path, validate=False)

    unified_path = Path(str(cfg.data.unified_registry_path))
    unified_df = read_parquet_or_empty(unified_path)
    merged_df = replace_source_slice(unified_df, source_df, source_name=source_name)
    write_registry_parquet(merged_df, unified_path, validate=False)

    manifest_path = write_run_manifest(
        manifests_root=Path(str(cfg.data.manifests_root)),
        repo_root=ROOT,
        source_name=source_name,
        source_row_count=len(source_df),
        staging_path=staging_path,
        unified_path=unified_path,
        extra={
            "projects": project_ids,
            "download_enabled": download_enabled,
            "api_counts": {
                "cases": len(cases),
                "pathology_files": len(pathology_files),
                "report_files": len(report_files),
                "radiology_patients": len(tcia_studies_by_patient),
                "tcia_series_patients": len(tcia_series_by_patient),
                "tcia_series_records": sum(len(entries) for entries in tcia_series_by_patient.values()),
                "ssm_hits": len(ssm_hits),
                "mutation_case_count": len(ssm_mutations_by_case_id),
                "mutation_patient_count": len(ssm_mutations_by_patient_id),
            },
            "download_counts": {
                "pathology_files": pathology_download_count,
                "report_files": report_download_count,
                "radiology_series": radiology_download_count,
            },
            "notes": "TCGA kidney build with optional payload downloads (pathology, radiology, reports).",
        },
    )

    print(f"TCGA source build complete: {source_name}")
    print(f"Projects: {project_ids}")
    print(f"Cases pulled: {len(cases)}")
    print(f"Pathology files pulled: {len(pathology_files)}")
    print(f"Report PDF files pulled: {len(report_files)}")
    print(f"Radiology patients pulled: {len(tcia_studies_by_patient)}")
    print(f"TCIA series metadata records pulled: {sum(len(entries) for entries in tcia_series_by_patient.values())}")
    print(f"SSM mutation hits pulled: {len(ssm_hits)}")
    print(f"Mutation case index size: {len(ssm_mutations_by_case_id)}")
    print(f"Mutation patient index size: {len(ssm_mutations_by_patient_id)}")
    print(f"Rows written: {len(source_df)}")
    print(f"Staging parquet: {staging_path}")
    print(f"Unified parquet: {unified_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
