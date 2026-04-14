#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.data.manifest import write_run_manifest
from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.sources.tcga import (
    APIQueryError,
    DEFAULT_TCIA_RADIOLOGY_MODALITIES,
    GDCClient,
    TCIAClient,
    build_tcga_registry_rows,
    index_ssm_hits_by_case_and_patient,
    normalize_tcia_modality,
    select_tcia_radiology_cohort,
)
from kidney_vlm.data.sources.tcga_genomics import (
    build_tcga_genomics_by_patient_id,
    download_tcga_pancan_atlas,
    write_tcga_genomics_jsonl,
)
from kidney_vlm.data.unified_registry import replace_source_slice, upsert_source_slice

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


MSI_METADATA_PAYLOAD_KEY = "msisensor_scores"


GENOMICS_RAW_PAYLOAD_SPECS: dict[str, dict[str, Any]] = {
    "masked_somatic_mutation": {
        "subfolder": "genomics/masked_somatic_mutation",
        "data_categories": ["Simple Nucleotide Variation"],
        "data_types": ["Masked Somatic Mutation"],
        "data_formats": ["MAF", "TXT"],
    },
    "gene_expression_quantification": {
        "subfolder": "genomics/gene_expression_quantification",
        "query_variants": [
            {
                "data_categories": ["Transcriptome Profiling"],
                "data_types": ["Gene Expression Quantification"],
                "data_formats": ["TSV"],
                "workflow_types": ["STAR - Counts"],
                "experimental_strategies": ["RNA-Seq"],
            },
            {
                "data_categories": ["Transcriptome Profiling"],
                "data_types": ["Gene Expression Quantification"],
                "data_formats": ["TXT", "TSV"],
                "experimental_strategies": ["RNA-Seq"],
            },
            {
                "data_categories": ["Transcriptome Profiling"],
                "data_types": ["Gene Expression Quantification"],
                "data_formats": ["TXT", "TSV"],
            },
        ],
    },
    "copy_number_segments": {
        "subfolder": "genomics/copy_number_segments",
        "data_categories": ["Copy Number Variation"],
        "data_types": ["Copy Number Segment"],
        "data_formats": ["TXT", "TSV"],
    },
    "masked_copy_number_segments": {
        "subfolder": "genomics/masked_copy_number_segments",
        "data_categories": ["Copy Number Variation"],
        "data_types": ["Masked Copy Number Segment"],
        "data_formats": ["TXT", "TSV"],
    },
    "gene_level_copy_number": {
        "subfolder": "genomics/gene_level_copy_number",
        "data_categories": ["Copy Number Variation"],
        "data_types": ["Gene Level Copy Number"],
        "data_formats": ["TXT", "TSV"],
    },
    "mirna_expression_quantification": {
        "subfolder": "genomics/mirna_expression_quantification",
        "query_variants": [
            {
                "data_categories": ["Transcriptome Profiling"],
                "data_types": ["miRNA Expression Quantification"],
                "data_formats": ["TXT", "TSV"],
                "workflow_types": ["BCGSC miRNA Profiling"],
                "experimental_strategies": ["miRNA-Seq"],
            },
            {
                "data_categories": ["Transcriptome Profiling"],
                "data_types": ["miRNA Expression Quantification"],
                "data_formats": ["TXT", "TSV"],
                "experimental_strategies": ["miRNA-Seq"],
            },
            {
                "data_categories": ["Transcriptome Profiling"],
                "data_types": ["miRNA Expression Quantification"],
                "data_formats": ["TXT", "TSV"],
            },
        ],
    },
    "clinical_supplement": {
        "subfolder": "genomics/clinical_supplement",
        "data_categories": ["Clinical"],
        "data_types": ["Clinical Supplement"],
        "data_formats": ["BCR OMF XML", "BCR Biotab", "XML", "TXT", "TSV"],
    },
    "biospecimen_supplement": {
        "subfolder": "genomics/biospecimen_supplement",
        "data_categories": ["Biospecimen"],
        "data_types": ["Biospecimen Supplement"],
        "data_formats": [
            "BCR Biotab",
            "BCR SSF XML",
            "BCR PPS XML",
            "BCR Auxiliary XML",
            "FoundationOne XML",
            "XML",
            "TXT",
            "TSV",
        ],
    },
    "methylation_beta_value": {
        "subfolder": "genomics/methylation_beta_value",
        "query_variants": [
            {
                "data_categories": ["DNA Methylation"],
                "data_types": ["Methylation Beta Value"],
                "data_formats": ["TXT", "TSV"],
                "experimental_strategies": ["Methylation Array"],
            },
            {
                "data_categories": ["DNA Methylation"],
                "data_types": ["Methylation Beta Value"],
                "data_formats": ["TXT", "TSV"],
            },
        ],
    },
}


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


def _normalized_string_list(values: Any) -> list[str]:
    items: list[str] = []
    for value in list(values or []):
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _resolve_tcga_project_ids(tcga_cfg: DictConfig, gdc_client: GDCClient) -> list[str]:
    legacy_project_ids = _normalized_string_list(tcga_cfg.get("project_ids", []))
    exclude_project_ids = set(_normalized_string_list(tcga_cfg.get("exclude_project_ids", [])))

    if legacy_project_ids:
        print("[warning] data.source.tcga.project_ids is deprecated; prefer exclude_project_ids.")
        selected_project_ids = legacy_project_ids
    else:
        project_hits = gdc_client.fetch_projects(project_id_pattern="TCGA-*")
        selected_project_ids = []
        for project_hit in project_hits:
            project_id = str(project_hit.get("project_id", "")).strip()
            if project_id:
                selected_project_ids.append(project_id)

    selected_project_ids = [project_id for project_id in selected_project_ids if project_id not in exclude_project_ids]
    if not selected_project_ids:
        raise ValueError("No TCGA projects selected after applying exclude_project_ids.")
    return selected_project_ids


def _describe_enabled_download_payloads(download_cfg: DictConfig, tcia_client: TCIAClient | None) -> list[str]:
    payloads: list[str] = []
    if bool(download_cfg.include.pathology):
        payloads.append("pathology slide files")
    if bool(download_cfg.include.reports):
        payloads.append("report PDFs")
    if bool(download_cfg.include.radiology) and tcia_client is not None:
        payloads.append("radiology series zip files")
    return payloads


def _normalized_tcia_modalities(values: Any) -> list[str]:
    items: list[str] = []
    for value in list(values or []):
        modality = normalize_tcia_modality(value)
        if modality and modality not in items:
            items.append(modality)
    return items


def _resolve_tcia_collections(
    tcga_cfg: DictConfig,
    *,
    project_ids: list[str],
    tcia_client: TCIAClient,
) -> list[str]:
    configured_collections = _normalized_string_list(tcga_cfg.tcia.get("collections", []))
    if configured_collections:
        return configured_collections

    try:
        available_collections = set(tcia_client.fetch_collection_values())
    except APIQueryError as exc:
        print(f"[warning] TCIA collection discovery failed; falling back to TCGA project ids. {exc}")
        return list(project_ids)

    selected_collections = [project_id for project_id in project_ids if project_id in available_collections]
    if not selected_collections:
        print("[warning] No TCIA collections matched the selected TCGA projects.")
    return selected_collections


def _filter_mapping_by_patient_ids(
    records_by_patient: dict[str, list[dict[str, Any]]],
    patient_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    if not patient_ids:
        return records_by_patient
    return {
        patient_id: list(entries)
        for patient_id, entries in records_by_patient.items()
        if patient_id in patient_ids
    }


def _resolve_selected_patient_ids(
    tcga_cfg: DictConfig,
    *,
    available_patient_ids: set[str],
) -> list[str]:
    selected_ids = _normalized_string_list(tcga_cfg.get("patient_subset_ids", []))
    if selected_ids:
        return [patient_id for patient_id in selected_ids if patient_id in available_patient_ids]

    chunk_cfg = tcga_cfg.get("patient_chunk")
    if chunk_cfg is None:
        return []

    chunk_size = _optional_int(chunk_cfg.get("size"))
    if chunk_size is None:
        return []
    if chunk_size <= 0:
        raise ValueError("data.source.tcga.patient_chunk.size must be a positive integer when set.")

    chunk_index = int(chunk_cfg.get("index", 0))
    if chunk_index < 0:
        raise ValueError("data.source.tcga.patient_chunk.index must be >= 0.")

    sorted_ids = sorted(available_patient_ids)
    start = chunk_index * chunk_size
    if start >= len(sorted_ids):
        raise ValueError(
            "Requested TCGA patient chunk is empty. "
            f"index={chunk_index}, size={chunk_size}, available_patients={len(sorted_ids)}."
        )
    return sorted_ids[start : start + chunk_size]


def _fetch_tcga_payloads(
    *,
    tcga_cfg: DictConfig,
    project_ids: list[str],
    gdc_client: GDCClient,
    tcia_client: TCIAClient | None,
    tcia_collections: list[str] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, str]]],
    list[dict[str, Any]],
    bool,
    list[str],
]:
    restrict_to_radiology_cases = bool(tcga_cfg.tcia.get("restrict_to_radiology_cases", True))
    if restrict_to_radiology_cases and (tcia_client is None or not bool(tcga_cfg.tcia.enabled)):
        raise ValueError("Radiology-first TCGA discovery requires TCIA to be enabled.")

    selected_tcia_collections: list[str] = list(tcia_collections or [])
    tcia_studies_by_patient: dict[str, list[dict[str, Any]]] = {}
    tcia_series_by_patient: dict[str, list[dict[str, str]]] = {}
    radiology_patient_ids: set[str] = set()
    if tcia_client is not None and bool(tcga_cfg.tcia.enabled):
        selected_tcia_collections = list(tcia_collections or [])
        raw_tcia_studies_by_patient = tcia_client.fetch_studies_by_patient(
            collections=selected_tcia_collections,
            max_studies_per_collection=_optional_int(tcga_cfg.tcia.max_studies_per_collection),
        )
        raw_tcia_series_by_patient: dict[str, list[dict[str, str]]] = {}
        if bool(tcga_cfg.tcia.fetch_series_metadata):
            raw_tcia_series_by_patient = tcia_client.fetch_series_by_patient(
                studies_by_patient=raw_tcia_studies_by_patient,
                max_series_per_study=_optional_int(tcga_cfg.tcia.max_series_per_study_metadata),
            )
        qualifying_modalities = _normalized_tcia_modalities(tcga_cfg.tcia.get("qualifying_modalities", []))
        (
            radiology_patient_ids,
            tcia_studies_by_patient,
            tcia_series_by_patient,
        ) = select_tcia_radiology_cohort(
            raw_tcia_studies_by_patient,
            series_by_patient=raw_tcia_series_by_patient,
            qualifying_modalities=qualifying_modalities,
            default_modalities=list(DEFAULT_TCIA_RADIOLOGY_MODALITIES),
        )

    available_patient_ids = set(tcia_studies_by_patient)
    selected_patient_ids = _resolve_selected_patient_ids(
        tcga_cfg,
        available_patient_ids=available_patient_ids,
    )
    selected_patient_id_set = set(selected_patient_ids)
    if selected_patient_id_set:
        radiology_patient_ids = selected_patient_id_set.intersection(radiology_patient_ids)
        tcia_studies_by_patient = _filter_mapping_by_patient_ids(tcia_studies_by_patient, selected_patient_id_set)
        tcia_series_by_patient = _filter_mapping_by_patient_ids(tcia_series_by_patient, selected_patient_id_set)

    gdc_case_submitter_filter = (
        sorted(radiology_patient_ids)
        if restrict_to_radiology_cases and radiology_patient_ids
        else None
    )
    if selected_patient_ids:
        gdc_case_submitter_filter = list(selected_patient_ids)
    if restrict_to_radiology_cases and not radiology_patient_ids:
        raise ValueError(
            "No TCIA radiology patients matched the selected collections/modalities while "
            "data.source.tcga.tcia.restrict_to_radiology_cases=true. "
            "Aborting to avoid replacing the TCGA registry slice with an empty cohort."
        )

    cases = gdc_client.fetch_cases(
        project_ids=project_ids,
        submitter_ids=gdc_case_submitter_filter,
        max_cases=_optional_int(tcga_cfg.gdc.max_cases),
    )
    case_ids = [str(case.get("case_id", "")).strip() for case in cases if str(case.get("case_id", "")).strip()]
    fetched_case_submitter_ids = [
        str(case.get("submitter_id", "")).strip()
        for case in cases
        if str(case.get("submitter_id", "")).strip()
    ]

    pathology_files = gdc_client.fetch_pathology_files(
        project_ids=project_ids,
        case_ids=case_ids,
        submitter_ids=fetched_case_submitter_ids,
        data_formats=[str(x) for x in list(tcga_cfg.gdc.pathology_data_formats)],
        data_types=[str(x) for x in list(tcga_cfg.gdc.pathology_data_types)],
        max_files=_optional_int(tcga_cfg.gdc.max_pathology_files),
    )

    report_files = gdc_client.fetch_report_files(
        project_ids=project_ids,
        case_ids=case_ids,
        data_formats=[str(x) for x in list(tcga_cfg.gdc.report_data_formats)],
        data_types=[str(x) for x in list(tcga_cfg.gdc.report_data_types)],
        data_categories=[str(x) for x in list(tcga_cfg.gdc.report_data_categories)],
        max_files=_optional_int(tcga_cfg.gdc.max_report_files),
    )

    ssm_hits: list[dict[str, Any]] = []
    mutation_query_succeeded = False
    if bool(tcga_cfg.gdc.fetch_ssm_mutations):
        mutation_gene_panel = [str(gene) for gene in list(tcga_cfg.gdc.mutation_gene_panel)]
        try:
            ssm_hits = gdc_client.fetch_ssm_hits(
                project_ids=project_ids,
                gene_symbols=mutation_gene_panel,
                max_hits=_optional_int(tcga_cfg.gdc.max_ssm_hits),
            )
            mutation_query_succeeded = True
        except APIQueryError as exc:
            print(f"[warning] GDC SSM query failed; mutation fields will be null. {exc}")

    return (
        cases,
        pathology_files,
        report_files,
        tcia_studies_by_patient,
        tcia_series_by_patient,
        ssm_hits,
        mutation_query_succeeded,
        selected_tcia_collections,
    )


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
    include_file_id_dir: bool = False,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
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

        output_path = raw_root / source_name / subfolder / project_id / patient_id
        if include_file_id_dir:
            output_path = output_path / file_id
        output_path = output_path / file_name
        plan.append(
            {
                "file_id": file_id,
                "file_name": file_name,
                "output_path": str(output_path),
                "project_id": project_id,
                "patient_id": patient_id,
                "expected_size": _optional_int(file_hit.get("file_size")),
            }
        )
        seen_file_ids.add(file_id)

    return plan


def _download_gdc_plan(
    gdc_client: GDCClient,
    plan: list[dict[str, Any]],
    *,
    skip_existing: bool,
    max_downloads: int | None = None,
    progress_desc: str,
) -> tuple[dict[str, str], int]:
    downloaded: dict[str, str] = {}
    completed = 0
    effective_plan = plan[:max_downloads] if max_downloads is not None else plan

    loop = tqdm(effective_plan, total=len(effective_plan), desc=progress_desc, unit="file", leave=False)
    for item in loop:
        file_id = item["file_id"]
        output_path = Path(item["output_path"])
        resolved = gdc_client.download_data_file(
            file_id=file_id,
            output_path=output_path,
            skip_existing=skip_existing,
            expected_size=_optional_int(item.get("expected_size")),
        )
        downloaded[file_id] = str(resolved)
        completed += 1

    return downloaded, completed


def _project_relative_or_absolute(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _is_genomics_payload_enabled(genomics_cfg: DictConfig, payload_key: str) -> bool:
    raw_payloads_cfg = genomics_cfg.get("raw_payloads")
    if raw_payloads_cfg is None:
        return True
    value = raw_payloads_cfg.get(payload_key)
    if value is None:
        return True
    return bool(value)


def _fetch_genomics_file_hits_for_payload(
    *,
    gdc_client: GDCClient,
    project_ids: list[str],
    case_ids: list[str],
    submitter_ids: list[str],
    payload_key: str,
    payload_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    query_variants = list(payload_spec.get("query_variants", []))
    if not query_variants:
        query_variants = [payload_spec]

    for variant in query_variants:
        hits = gdc_client.fetch_genomics_files(
            project_ids=project_ids,
            case_ids=case_ids,
            submitter_ids=submitter_ids,
            data_categories=[str(x) for x in list(variant.get("data_categories", []))],
            data_types=[str(x) for x in list(variant.get("data_types", []))],
            data_formats=[str(x) for x in list(variant.get("data_formats", []))],
            workflow_types=[str(x) for x in list(variant.get("workflow_types", []))],
            experimental_strategies=[str(x) for x in list(variant.get("experimental_strategies", []))],
        )
        if hits:
            return hits

    print(f"[warning] No open-access genomics files found for payload '{payload_key}' in the selected TCGA cohort.")
    return []


def _index_downloaded_genomics_entries(
    *,
    payload_key: str,
    file_hits: list[dict[str, Any]],
    downloaded_by_file_id: dict[str, str],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    by_case_id: dict[str, list[dict[str, str]]] = {}
    by_patient_id: dict[str, list[dict[str, str]]] = {}
    manifest_entries: list[dict[str, str]] = []
    seen_links: set[tuple[str, str, str, str]] = set()

    for file_hit in file_hits:
        file_id = str(file_hit.get("file_id", "")).strip()
        local_path = str(downloaded_by_file_id.get(file_id, "")).strip()
        if not file_id or not local_path:
            continue

        file_name = str(file_hit.get("file_name", "")).strip()
        data_category = str(file_hit.get("data_category", "")).strip()
        data_type = str(file_hit.get("data_type", "")).strip()
        data_format = str(file_hit.get("data_format", "")).strip()
        workflow_type = str((file_hit.get("analysis") or {}).get("workflow_type", "")).strip()
        project_relative_path = _project_relative_or_absolute(local_path)

        linked_cases = file_hit.get("cases", [])
        if not isinstance(linked_cases, list) or not linked_cases:
            fallback_case_id, fallback_patient_id, fallback_project_id = _first_linked_case(file_hit)
            linked_cases = [
                {
                    "case_id": fallback_case_id,
                    "submitter_id": fallback_patient_id,
                    "project": {"project_id": fallback_project_id},
                }
            ]

        for linked_case in linked_cases:
            if not isinstance(linked_case, dict):
                continue
            case_id = str(linked_case.get("case_id", "")).strip()
            patient_id = str(linked_case.get("submitter_id", "")).strip()
            project_id = str((linked_case.get("project") or {}).get("project_id", "")).strip()
            link_key = (payload_key, file_id, case_id, patient_id)
            if link_key in seen_links:
                continue
            seen_links.add(link_key)

            entry = {
                "payload_key": payload_key,
                "file_id": file_id,
                "file_name": file_name,
                "data_category": data_category,
                "data_type": data_type,
                "data_format": data_format,
                "workflow_type": workflow_type,
                "local_path": local_path,
                "relative_path": project_relative_path,
                "case_id": case_id,
                "patient_id": patient_id,
                "project_id": project_id,
            }
            if case_id:
                by_case_id.setdefault(case_id, []).append(entry)
            if patient_id:
                by_patient_id.setdefault(patient_id, []).append(entry)
            manifest_entries.append(entry)

    return by_case_id, by_patient_id, manifest_entries


def _write_tcga_genomics_download_manifest(entries: list[dict[str, str]], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for entry in sorted(
            entries,
            key=lambda item: (
                item.get("patient_id", ""),
                item.get("payload_key", ""),
                item.get("file_name", ""),
                item.get("file_id", ""),
            ),
        ):
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return destination


def _make_temp_tcga_genomics_cache_dir(*, raw_root: Path, source_name: str) -> Path:
    parent = raw_root / source_name / "genomics" / "_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="pancan_", dir=str(parent)))


def _write_tcga_genomics_text_files(
    genomics_by_patient_id: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for patient_id in sorted(genomics_by_patient_id):
        text = str(genomics_by_patient_id[patient_id].get("genomics_text", "")).strip()
        if not text:
            continue
        path = destination / f"{patient_id}.txt"
        path.write_text(text + "\n", encoding="utf-8")
        written[patient_id] = path
    return written


def _entry_identity(entry: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        str(entry.get("payload_key", "")).strip(),
        str(entry.get("file_id", "")).strip(),
        str(entry.get("file_name", "")).strip(),
        str(entry.get("case_id", "")).strip(),
        str(entry.get("patient_id", "")).strip(),
        str(entry.get("relative_path") or entry.get("local_path", "")).strip(),
    )


def _merge_genomics_download_entry_indexes(
    base: dict[str, list[dict[str, str]]],
    incoming: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    merged: dict[str, list[dict[str, str]]] = {
        key: [dict(entry) for entry in entries]
        for key, entries in base.items()
    }
    for key, entries in incoming.items():
        existing = merged.setdefault(key, [])
        seen = {_entry_identity(entry) for entry in existing}
        for entry in entries:
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            existing.append(dict(entry))
            seen.add(identity)
    return merged


def _merge_genomics_manifest_entries(
    *entry_groups: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for entries in entry_groups:
        for entry in entries:
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            merged.append(dict(entry))
            seen.add(identity)
    return merged


def _build_tcga_genomics_artifact_index(
    *,
    cases: list[dict[str, Any]],
    genomics_text_paths: dict[str, Path],
    output_manifest_path: str | Path | None = None,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
    Path | None,
    list[dict[str, str]],
]:
    case_id_by_patient: dict[str, str] = {}
    project_id_by_patient: dict[str, str] = {}
    for case in cases:
        patient_id = str(case.get("submitter_id", "")).strip()
        if not patient_id:
            continue
        case_id_by_patient.setdefault(patient_id, str(case.get("case_id", "")).strip())
        project_id_by_patient.setdefault(patient_id, str((case.get("project") or {}).get("project_id", "")).strip())

    by_case_id: dict[str, list[dict[str, str]]] = {}
    by_patient_id: dict[str, list[dict[str, str]]] = {}
    manifest_entries: list[dict[str, str]] = []

    for patient_id, path in sorted(genomics_text_paths.items()):
        case_id = case_id_by_patient.get(patient_id, "")
        project_id = project_id_by_patient.get(patient_id, "")
        entry = {
            "payload_key": "genomics_text",
            "file_id": "",
            "file_name": path.name,
            "data_category": "Derived Genomics",
            "data_type": "Patient Genomics Text",
            "data_format": "TXT",
            "workflow_type": "derived",
            "local_path": str(path),
            "relative_path": _project_relative_or_absolute(path),
            "case_id": case_id,
            "patient_id": patient_id,
            "project_id": project_id,
        }
        by_patient_id.setdefault(patient_id, []).append(entry)
        if case_id:
            by_case_id.setdefault(case_id, []).append(entry)
        manifest_entries.append(entry)

    manifest_path = None
    if output_manifest_path is not None and manifest_entries:
        manifest_path = _write_tcga_genomics_download_manifest(manifest_entries, output_manifest_path)
    return by_case_id, by_patient_id, manifest_path, manifest_entries


def _write_tcga_msi_metadata_sidecars(
    *,
    msi_hits: list[dict[str, Any]],
    raw_root: Path,
    source_name: str,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
    list[dict[str, str]],
    int,
]:
    grouped_rows: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for hit in msi_hits:
        if not isinstance(hit, dict):
            continue
        linked_cases = hit.get("cases", [])
        if not isinstance(linked_cases, list):
            continue
        for linked_case in linked_cases:
            if not isinstance(linked_case, dict):
                continue
            case_id = str(linked_case.get("case_id", "")).strip()
            patient_id = str(linked_case.get("submitter_id", "")).strip()
            project_id = str((linked_case.get("project") or {}).get("project_id", "")).strip()
            if not patient_id:
                continue
            grouped_rows.setdefault((case_id, patient_id, project_id), []).append(
                {
                    "file_id": str(hit.get("file_id", "")).strip(),
                    "file_name": str(hit.get("file_name", "")).strip(),
                    "data_type": str(hit.get("data_type", "")).strip(),
                    "data_format": str(hit.get("data_format", "")).strip(),
                    "experimental_strategy": str(hit.get("experimental_strategy", "")).strip(),
                    "access": str(hit.get("access", "")).strip(),
                    "msi_score": str(hit.get("msi_score", "")).strip(),
                    "msi_status": str(hit.get("msi_status", "")).strip(),
                }
            )

    by_case_id: dict[str, list[dict[str, str]]] = {}
    by_patient_id: dict[str, list[dict[str, str]]] = {}
    manifest_entries: list[dict[str, str]] = []

    for (case_id, patient_id, project_id), rows in sorted(grouped_rows.items()):
        output_dir = raw_root / source_name / "genomics" / MSI_METADATA_PAYLOAD_KEY / project_id / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "msi_scores.tsv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file_id",
                    "file_name",
                    "data_type",
                    "data_format",
                    "experimental_strategy",
                    "access",
                    "msi_score",
                    "msi_status",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        entry = {
            "payload_key": MSI_METADATA_PAYLOAD_KEY,
            "file_id": "",
            "file_name": output_path.name,
            "data_category": "Derived Genomics Metadata",
            "data_type": "MSIsensor Score",
            "data_format": "TSV",
            "workflow_type": "gdc_api_metadata",
            "local_path": str(output_path),
            "relative_path": _project_relative_or_absolute(output_path),
            "case_id": case_id,
            "patient_id": patient_id,
            "project_id": project_id,
        }
        if case_id:
            by_case_id.setdefault(case_id, []).append(entry)
        by_patient_id.setdefault(patient_id, []).append(entry)
        manifest_entries.append(entry)

    return by_case_id, by_patient_id, manifest_entries, len(manifest_entries)


def _download_tcga_genomics_raw_payloads(
    *,
    gdc_client: GDCClient,
    genomics_cfg: DictConfig,
    cases: list[dict[str, Any]],
    project_ids: list[str],
    raw_root: Path,
    source_name: str,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
    list[dict[str, str]],
    dict[str, int],
    int,
]:
    case_ids = [str(case.get("case_id", "")).strip() for case in cases if str(case.get("case_id", "")).strip()]
    submitter_ids = [
        str(case.get("submitter_id", "")).strip()
        for case in cases
        if str(case.get("submitter_id", "")).strip()
    ]
    if not case_ids or not submitter_ids:
        return {}, {}, [], {}, 0

    skip_existing = bool(genomics_cfg.get("skip_existing", True))
    downloaded_by_case_id: dict[str, list[dict[str, str]]] = {}
    downloaded_by_patient_id: dict[str, list[dict[str, str]]] = {}
    manifest_entries: list[dict[str, str]] = []
    counts_by_payload: dict[str, int] = {}
    total_downloaded = 0
    previous_timeout = getattr(gdc_client, "timeout_seconds", None)
    if previous_timeout is not None:
        gdc_client.timeout_seconds = int(genomics_cfg.get("timeout_seconds", previous_timeout))
    try:
        for payload_key, payload_spec in GENOMICS_RAW_PAYLOAD_SPECS.items():
            if not _is_genomics_payload_enabled(genomics_cfg, payload_key):
                continue

            file_hits = _fetch_genomics_file_hits_for_payload(
                gdc_client=gdc_client,
                project_ids=project_ids,
                case_ids=case_ids,
                submitter_ids=submitter_ids,
                payload_key=payload_key,
                payload_spec=payload_spec,
            )
            if not file_hits:
                counts_by_payload[payload_key] = 0
                continue

            download_plan = _build_gdc_download_plan(
                file_hits,
                raw_root=raw_root,
                source_name=source_name,
                subfolder=str(payload_spec["subfolder"]),
                include_file_id_dir=True,
            )
            downloaded_by_file_id, payload_download_count = _download_gdc_plan(
                gdc_client,
                download_plan,
                skip_existing=skip_existing,
                progress_desc=f"Downloading {payload_key.replace('_', ' ')}",
            )
            payload_case_index, payload_patient_index, payload_manifest_entries = _index_downloaded_genomics_entries(
                payload_key=payload_key,
                file_hits=file_hits,
                downloaded_by_file_id=downloaded_by_file_id,
            )
            for case_id, entries in payload_case_index.items():
                downloaded_by_case_id.setdefault(case_id, []).extend(entries)
            for patient_id, entries in payload_patient_index.items():
                downloaded_by_patient_id.setdefault(patient_id, []).extend(entries)
            manifest_entries.extend(payload_manifest_entries)
            counts_by_payload[payload_key] = payload_download_count
            total_downloaded += payload_download_count

        if _is_genomics_payload_enabled(genomics_cfg, MSI_METADATA_PAYLOAD_KEY):
            msi_hits = gdc_client.fetch_msi_metadata(
                project_ids=project_ids,
                case_ids=case_ids,
                submitter_ids=submitter_ids,
                experimental_strategies=["WXS"],
            )
            if not msi_hits:
                msi_hits = gdc_client.fetch_msi_metadata(
                    project_ids=project_ids,
                    case_ids=case_ids,
                    submitter_ids=submitter_ids,
                )
            (
                msi_case_index,
                msi_patient_index,
                msi_manifest_entries,
                msi_sidecar_count,
            ) = _write_tcga_msi_metadata_sidecars(
                msi_hits=msi_hits,
                raw_root=raw_root,
                source_name=source_name,
            )
            downloaded_by_case_id = _merge_genomics_download_entry_indexes(downloaded_by_case_id, msi_case_index)
            downloaded_by_patient_id = _merge_genomics_download_entry_indexes(downloaded_by_patient_id, msi_patient_index)
            manifest_entries = _merge_genomics_manifest_entries(manifest_entries, msi_manifest_entries)
            counts_by_payload[MSI_METADATA_PAYLOAD_KEY] = msi_sidecar_count
            total_downloaded += msi_sidecar_count
    finally:
        if previous_timeout is not None:
            gdc_client.timeout_seconds = previous_timeout

    return downloaded_by_case_id, downloaded_by_patient_id, manifest_entries, counts_by_payload, total_downloaded


def _series_metadata_limit_for_download(tcga_cfg: DictConfig, download_cfg: DictConfig) -> int | None:
    metadata_limit = _optional_int(tcga_cfg.tcia.get("max_series_per_study_metadata"))
    if metadata_limit is not None:
        return metadata_limit
    return _optional_int(download_cfg.max_series_per_study)


def _ensure_tcia_series_metadata_for_download(
    *,
    tcga_cfg: DictConfig,
    download_cfg: DictConfig,
    tcia_client: TCIAClient,
    tcia_studies_by_patient: dict[str, list[dict[str, Any]]],
    tcia_series_by_patient: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    if tcia_series_by_patient or not tcia_studies_by_patient:
        return tcia_series_by_patient

    raw_series_by_patient = tcia_client.fetch_series_by_patient(
        studies_by_patient=tcia_studies_by_patient,
        max_series_per_study=_series_metadata_limit_for_download(tcga_cfg, download_cfg),
    )
    qualifying_modalities = _normalized_tcia_modalities(tcga_cfg.tcia.get("qualifying_modalities", []))
    _eligible_patients, _filtered_studies, filtered_series_by_patient = select_tcia_radiology_cohort(
        tcia_studies_by_patient,
        series_by_patient=raw_series_by_patient,
        qualifying_modalities=qualifying_modalities,
        default_modalities=list(DEFAULT_TCIA_RADIOLOGY_MODALITIES),
    )
    return filtered_series_by_patient


def _download_tcia_series(
    tcia_client: TCIAClient,
    *,
    tcia_series_by_patient: dict[str, list[dict[str, str]]],
    patient_ids: set[str],
    raw_root: Path,
    source_name: str,
    skip_existing: bool,
    max_series_per_study: int | None,
    max_series_total: int | None,
) -> tuple[dict[str, list[dict[str, str]]], int]:
    downloaded_by_patient: dict[str, list[dict[str, str]]] = {}
    candidate_series: list[dict[str, str]] = []

    for patient_id, series_entries in tcia_series_by_patient.items():
        if patient_id not in patient_ids:
            continue
        downloaded_per_study: dict[str, int] = {}
        for entry in series_entries:
            study_uid = str(entry.get("study_instance_uid", "")).strip()
            if max_series_per_study is not None:
                current_count = downloaded_per_study.get(study_uid, 0)
                if current_count >= max_series_per_study:
                    continue
                downloaded_per_study[study_uid] = current_count + 1
            candidate_series.append(entry)

    if max_series_total is not None:
        candidate_series = candidate_series[:max_series_total]

    progress = tqdm(
        candidate_series,
        total=len(candidate_series),
        desc="Downloading radiology series",
        unit="series",
        leave=False,
    )

    total_downloaded = 0
    for entry in progress:
        patient_id = str(entry.get("patient_id", "")).strip()
        collection = str(entry.get("collection", "")).strip() or "unknown_collection"
        study_uid = str(entry.get("study_instance_uid", "")).strip()
        series_uid = str(entry.get("series_instance_uid", "")).strip()
        modality = str(entry.get("modality", "")).strip()
        if not patient_id or not study_uid or not series_uid:
            continue

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

    progress.close()
    return downloaded_by_patient, total_downloaded


def main() -> None:
    overrides = sys.argv[1:]
    cfg = load_cfg("tcga", overrides=overrides)

    source_name = str(cfg.data.source.name)
    tcga_cfg = cfg.data.source.tcga
    exclude_project_ids = _normalized_string_list(tcga_cfg.get("exclude_project_ids", []))

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

    project_ids = _resolve_tcga_project_ids(tcga_cfg, gdc_client)
    selected_tcia_collections = (
        _resolve_tcia_collections(tcga_cfg, project_ids=project_ids, tcia_client=tcia_client)
        if tcia_client is not None and bool(tcga_cfg.tcia.enabled)
        else []
    )
    print(f"Pulling metadata for projects: {project_ids}")
    if exclude_project_ids:
        print(f"Excluding projects: {exclude_project_ids}")
    if selected_tcia_collections:
        print(f"TCIA collections selected: {selected_tcia_collections}")
    (
        cases,
        pathology_files,
        report_files,
        tcia_studies_by_patient,
        tcia_series_by_patient,
        ssm_hits,
        mutation_query_succeeded,
        selected_tcia_collections,
    ) = (
        _fetch_tcga_payloads(
            tcga_cfg=tcga_cfg,
            project_ids=project_ids,
            gdc_client=gdc_client,
            tcia_client=tcia_client,
            tcia_collections=selected_tcia_collections,
        )
    )
    ssm_mutations_by_case_id, ssm_mutations_by_patient_id = index_ssm_hits_by_case_and_patient(ssm_hits)
    mutation_gene_panel = [str(gene) for gene in list(tcga_cfg.gdc.mutation_gene_panel)]

    download_cfg = cfg.data.source.download
    download_enabled = bool(download_cfg.enabled)
    skip_existing = bool(download_cfg.skip_existing)
    raw_root = Path(str(cfg.data.raw_root))

    downloaded_pathology_by_file_id: dict[str, str] = {}
    downloaded_reports_by_file_id: dict[str, str] = {}
    downloaded_tcia_series_by_patient: dict[str, list[dict[str, str]]] = {}
    genomics_by_patient_id: dict[str, dict[str, Any]] = {}
    downloaded_genomics_by_case_id: dict[str, list[dict[str, str]]] = {}
    downloaded_genomics_by_patient_id: dict[str, list[dict[str, str]]] = {}
    genomics_raw_manifest_path: Path | None = None
    genomics_raw_download_counts: dict[str, int] = {}
    genomics_pancan_downloaded_files: dict[str, Path] = {}
    genomics_text_paths: dict[str, Path] = {}
    genomics_sidecar_path: Path | None = None
    genomics_cache_dir: Path | None = None
    genomics_temp_cache_cleaned = False
    raw_genomics_manifest_entries: list[dict[str, str]] = []
    text_genomics_manifest_entries: list[dict[str, str]] = []
    pathology_download_count = 0
    report_download_count = 0
    radiology_download_count = 0
    genomics_raw_download_count = 0
    genomics_pancan_source_count = 0

    if download_enabled:
        enabled_payloads = _describe_enabled_download_payloads(download_cfg, tcia_client)
        if enabled_payloads:
            print(
                "Download stage enabled. Resolving/downloading: "
                + ", ".join(enabled_payloads)
                + "."
            )
        else:
            print("Download stage enabled, but no payload types are selected.")

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
                progress_desc="Downloading pathology slide files",
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
                progress_desc="Downloading report PDFs",
            )
            print(f"Report PDF files downloaded/resolved: {report_download_count}")

        if bool(download_cfg.include.radiology) and tcia_client is not None:
            if not tcia_series_by_patient and tcia_studies_by_patient:
                print("Fetching TCIA series metadata on demand for radiology downloads.")
            tcia_series_by_patient = _ensure_tcia_series_metadata_for_download(
                tcga_cfg=tcga_cfg,
                download_cfg=download_cfg,
                tcia_client=tcia_client,
                tcia_studies_by_patient=tcia_studies_by_patient,
                tcia_series_by_patient=tcia_series_by_patient,
            )
            patient_ids = {
                str(case.get("submitter_id", "")).strip()
                for case in cases
                if str(case.get("submitter_id", "")).strip()
            }
            downloaded_tcia_series_by_patient, radiology_download_count = _download_tcia_series(
                tcia_client,
                tcia_series_by_patient=tcia_series_by_patient,
                patient_ids=patient_ids,
                raw_root=raw_root,
                source_name=source_name,
                skip_existing=skip_existing,
                max_series_per_study=_optional_int(download_cfg.max_series_per_study),
                max_series_total=_optional_int(download_cfg.max_radiology_series_downloads),
            )
            print(f"Radiology series zip files downloaded/resolved: {radiology_download_count}")

    genomics_cfg = tcga_cfg.get("genomics")
    genomics_enabled = bool(genomics_cfg and bool(genomics_cfg.get("enabled", False)))
    download_raw_genomics = bool(genomics_cfg and bool(genomics_cfg.get("download_raw", False)))
    build_text_from_pancan = bool(genomics_cfg and bool(genomics_cfg.get("build_text_from_pancan", True)))
    if genomics_enabled:
        if download_raw_genomics:
            (
                downloaded_genomics_by_case_id,
                downloaded_genomics_by_patient_id,
                raw_genomics_manifest_entries,
                genomics_raw_download_counts,
                genomics_raw_download_count,
            ) = _download_tcga_genomics_raw_payloads(
                gdc_client=gdc_client,
                genomics_cfg=genomics_cfg,
                cases=cases,
                project_ids=project_ids,
                raw_root=raw_root,
                source_name=source_name,
            )
            print(f"TCGA raw genomics source files downloaded/resolved: {genomics_raw_download_count}")

        if build_text_from_pancan:
            cleanup_temp_cache = bool(genomics_cfg.get("cleanup_temp_cache", True))
            use_temp_cache = cleanup_temp_cache or not str(genomics_cfg.get("pancan_cache_dir", "")).strip()
            genomics_cache_dir = (
                _make_temp_tcga_genomics_cache_dir(raw_root=raw_root, source_name=source_name)
                if use_temp_cache
                else Path(str(genomics_cfg.pancan_cache_dir))
            )
            print(f"Downloading/resolving TCGA PanCancer genomics cache: {genomics_cache_dir}")
            try:
                genomics_pancan_downloaded_files = download_tcga_pancan_atlas(
                    genomics_cache_dir,
                    skip_existing=bool(genomics_cfg.get("skip_existing", True)),
                    timeout_seconds=int(genomics_cfg.get("timeout_seconds", 300)),
                )
                genomics_pancan_source_count = len(genomics_pancan_downloaded_files)
                genomics_by_patient_id = build_tcga_genomics_by_patient_id(
                    cases=cases,
                    data_dir=genomics_cache_dir,
                )
                print(f"TCGA genomics text blocks prepared: {len(genomics_by_patient_id)}")

                if bool(genomics_cfg.get("write_jsonl_sidecar", True)):
                    genomics_sidecar_path = write_tcga_genomics_jsonl(
                        genomics_by_patient_id,
                        Path(str(genomics_cfg.sidecar_jsonl_path)),
                    )
                    print(f"TCGA genomics sidecar JSONL: {genomics_sidecar_path}")

                if bool(genomics_cfg.get("write_text_files", True)):
                    genomics_text_paths = _write_tcga_genomics_text_files(
                        genomics_by_patient_id,
                        Path(str(genomics_cfg.text_output_dir)),
                    )
                    print(f"TCGA genomics patient text files written: {len(genomics_text_paths)}")

                (
                    text_genomics_by_case_id,
                    text_genomics_by_patient_id,
                    _text_genomics_manifest_path,
                    text_genomics_manifest_entries,
                ) = _build_tcga_genomics_artifact_index(
                    cases=cases,
                    genomics_text_paths=genomics_text_paths,
                    output_manifest_path=None,
                )
                del _text_genomics_manifest_path
                downloaded_genomics_by_case_id = _merge_genomics_download_entry_indexes(
                    downloaded_genomics_by_case_id,
                    text_genomics_by_case_id,
                )
                downloaded_genomics_by_patient_id = _merge_genomics_download_entry_indexes(
                    downloaded_genomics_by_patient_id,
                    text_genomics_by_patient_id,
                )
            finally:
                if cleanup_temp_cache and genomics_cache_dir.exists():
                    shutil.rmtree(genomics_cache_dir, ignore_errors=True)
                    genomics_temp_cache_cleaned = True

        combined_genomics_manifest_entries = _merge_genomics_manifest_entries(
            raw_genomics_manifest_entries,
            text_genomics_manifest_entries,
        )
        if combined_genomics_manifest_entries and bool(genomics_cfg.get("write_download_manifest", True)):
            genomics_raw_manifest_path = _write_tcga_genomics_download_manifest(
                combined_genomics_manifest_entries,
                Path(str(genomics_cfg.download_manifest_jsonl_path)),
            )
            print(f"TCGA genomics artifact manifest JSONL: {genomics_raw_manifest_path}")

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
        genomics_by_patient_id=genomics_by_patient_id,
        downloaded_genomics_by_case_id=downloaded_genomics_by_case_id,
        downloaded_genomics_by_patient_id=downloaded_genomics_by_patient_id,
        genomics_download_manifest_path=(
            _project_relative_or_absolute(genomics_raw_manifest_path)
            if genomics_raw_manifest_path is not None
            else ""
        ),
        raw_root=raw_root,
        project_root=ROOT,
        mutation_query_succeeded=mutation_query_succeeded,
        source_name=source_name,
        split_ratios=_split_ratios(tcga_cfg),
        show_progress=True,
        progress_desc="Preparing tcga registry rows",
    )

    staging_root = Path(str(cfg.data.staging_root))
    staging_path = staging_root / f"{source_name}.parquet"
    upsert_mode = str(tcga_cfg.get("upsert_mode", "replace")).strip().lower()

    staging_df = read_parquet_or_empty(staging_path)
    unified_path = Path(str(cfg.data.unified_registry_path))
    unified_df = read_parquet_or_empty(unified_path)
    if upsert_mode == "incremental":
        merged_staging_df = upsert_source_slice(
            staging_df,
            source_df,
            source_name=source_name,
            key_columns=("sample_id",),
        )
        merged_df = upsert_source_slice(
            unified_df,
            source_df,
            source_name=source_name,
            key_columns=("sample_id",),
        )
    elif upsert_mode == "replace":
        merged_staging_df = source_df
        merged_df = replace_source_slice(unified_df, source_df, source_name=source_name)
    else:
        raise ValueError("data.source.tcga.upsert_mode must be one of: replace, incremental")
    write_registry_parquet(merged_staging_df, staging_path, validate=False)
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
            "excluded_projects": exclude_project_ids,
            "upsert_mode": upsert_mode,
            "patient_subset_ids": _normalized_string_list(tcga_cfg.get("patient_subset_ids", [])),
            "patient_chunk": {
                "index": int(tcga_cfg.get("patient_chunk", {}).get("index", 0)),
                "size": _optional_int(tcga_cfg.get("patient_chunk", {}).get("size")),
            },
            "tcia_collections": selected_tcia_collections,
            "download_enabled": download_enabled,
            "genomics_enabled": genomics_enabled,
            "api_counts": {
                "cases": len(cases),
                "pathology_files": len(pathology_files),
                "report_files": len(report_files),
                "radiology_patients": len(tcia_studies_by_patient),
                "radiology_eligible_patients": len(tcia_studies_by_patient),
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
                "genomics_source_files": genomics_raw_download_count,
                "genomics_pancan_source_files": genomics_pancan_source_count,
                "genomics_patient_text_files": len(genomics_text_paths),
            },
            "genomics": {
                "cache_dir": (
                    str(genomics_cache_dir)
                    if genomics_cache_dir is not None and not genomics_temp_cache_cleaned
                    else ""
                ),
                "temp_cache_dir": (
                    str(genomics_cache_dir)
                    if genomics_cache_dir is not None and genomics_temp_cache_cleaned
                    else ""
                ),
                "temp_cache_cleaned": genomics_temp_cache_cleaned,
                "download_manifest_path": str(genomics_raw_manifest_path) if genomics_raw_manifest_path is not None else "",
                "sidecar_jsonl_path": str(genomics_sidecar_path) if genomics_sidecar_path is not None else "",
                "text_output_dir": str(Path(str(genomics_cfg.text_output_dir))) if genomics_cfg is not None else "",
                "downloaded_patient_count": len(downloaded_genomics_by_patient_id),
                "patient_count": len(genomics_by_patient_id),
                "download_counts_by_payload": dict(genomics_raw_download_counts),
                "downloaded_source_keys": sorted(genomics_pancan_downloaded_files),
            },
            "notes": "TCGA radiology-first source refresh with text-first genomics derivation and source-slice upsert.",
        },
    )

    print(f"TCGA source registry upsert complete: {source_name}")
    print(f"Projects: {project_ids}")
    print(f"Cases pulled: {len(cases)}")
    print(f"Pathology files pulled: {len(pathology_files)}")
    print(f"Report PDF files pulled: {len(report_files)}")
    print(f"Radiology patients pulled: {len(tcia_studies_by_patient)}")
    print(f"TCIA series metadata records pulled: {sum(len(entries) for entries in tcia_series_by_patient.values())}")
    print(f"SSM mutation hits pulled: {len(ssm_hits)}")
    print(f"Mutation case index size: {len(ssm_mutations_by_case_id)}")
    print(f"Mutation patient index size: {len(ssm_mutations_by_patient_id)}")
    print(f"Genomics artifact patient count: {len(downloaded_genomics_by_patient_id)}")
    print(f"Genomics raw source files downloaded/resolved: {genomics_raw_download_count}")
    print(f"Genomics PanCancer source files downloaded/resolved: {genomics_pancan_source_count}")
    print(f"Genomics patient text files written: {len(genomics_text_paths)}")
    print(f"Genomics text blocks prepared: {len(genomics_by_patient_id)}")
    print(f"Rows written: {len(source_df)}")
    print(f"Staging parquet: {staging_path}")
    print(f"Unified parquet: {unified_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
