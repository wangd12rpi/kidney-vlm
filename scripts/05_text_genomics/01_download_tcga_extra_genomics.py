#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.sources.tcga import GDCClient
from kidney_vlm.genomics.case_filter import load_patient_ids_from_json
from kidney_vlm.genomics.download_specs import EXTRA_GENOMICS_SPECS, EXTRA_GENOMICS_SPEC_BY_KEY
from kidney_vlm.genomics.extra_downloads import (
    build_extra_genomics_download_plan,
    fetch_extra_genomics_files,
    write_extra_genomics_manifest,
)
from kidney_vlm.genomics.registry_integration import update_registry_with_extra_genomics_manifest
from kidney_vlm.repo_root import find_repo_root

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

    merged = OmegaConf.merge(base_cfg, OmegaConf.load(source_cfg_path))
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
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


def _normalized_string_list(values: Any) -> list[str]:
    items: list[str] = []
    for value in list(values or []):
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _resolve_project_ids(tcga_cfg: DictConfig, gdc_client: GDCClient) -> list[str]:
    legacy_project_ids = _normalized_string_list(tcga_cfg.get("project_ids", []))
    exclude_project_ids = set(_normalized_string_list(tcga_cfg.get("exclude_project_ids", [])))
    if legacy_project_ids:
        print("[warning] data.source.tcga.project_ids is deprecated; prefer exclude_project_ids.")
        selected = legacy_project_ids
    else:
        selected = [
            str(project.get("project_id", "")).strip()
            for project in gdc_client.fetch_projects(project_id_pattern="TCGA-*")
            if str(project.get("project_id", "")).strip()
        ]
    selected = [project_id for project_id in selected if project_id not in exclude_project_ids]
    if not selected:
        raise ValueError("No TCGA projects selected after applying exclude_project_ids.")
    return selected


def _selected_specs(extra_cfg: DictConfig) -> list[Any]:
    requested = _normalized_string_list(extra_cfg.get("modalities", []))
    if not requested:
        return list(EXTRA_GENOMICS_SPECS)
    missing = sorted(set(requested).difference(EXTRA_GENOMICS_SPEC_BY_KEY))
    if missing:
        raise ValueError(f"Unknown extra genomics modalities requested: {missing}")
    return [EXTRA_GENOMICS_SPEC_BY_KEY[key] for key in requested]


def _case_subset_patient_filter(extra_cfg: DictConfig) -> tuple[list[str], Path | None]:
    case_subset = str(extra_cfg.get("case_subset", "") or "").strip().lower()
    case_cfg = extra_cfg.get("pathology_cases", {})

    # Backward compatibility for the previous hidden switch.
    if not case_subset and bool(case_cfg.get("enabled", False)):
        case_subset = "pathology_cases"
    case_subset = case_subset or "all"

    if case_subset == "all":
        return [], None
    if case_subset != "pathology_cases":
        raise ValueError(
            "data.source.extra_genomics.case_subset must be one of "
            "['all', 'pathology_cases']."
        )

    json_path = Path(str(case_cfg.get("path", "pathology_cases.json"))).expanduser()
    if not json_path.is_absolute():
        json_path = ROOT / json_path
    patient_ids = load_patient_ids_from_json(json_path)
    if not patient_ids:
        raise RuntimeError(f"No patient IDs found in case subset JSON: {json_path}")
    return patient_ids, json_path


def main() -> None:
    cfg = load_cfg("tcga", overrides=sys.argv[1:])
    source_name = str(cfg.data.source.name)
    tcga_cfg = cfg.data.source.tcga
    raw_root = Path(str(cfg.data.raw_root))
    extra_cfg = cfg.data.source.get("extra_genomics", {})

    gdc_client = GDCClient(
        base_url=str(tcga_cfg.gdc.base_url),
        timeout_seconds=int(tcga_cfg.gdc.timeout_seconds),
        page_size=int(tcga_cfg.gdc.page_size),
        max_retries=int(tcga_cfg.gdc.max_retries),
        retry_backoff_seconds=float(tcga_cfg.gdc.retry_backoff_seconds),
    )

    project_ids = _resolve_project_ids(tcga_cfg, gdc_client)
    patient_submitter_ids, patient_subset_path = _case_subset_patient_filter(extra_cfg)
    selected_specs = _selected_specs(extra_cfg)
    print(f"[extra-genomics] Projects: {project_ids}")
    if patient_submitter_ids:
        print(
            "[extra-genomics] Case subset: "
            f"{patient_subset_path or 'pathology_cases'} ({len(patient_submitter_ids)} patients)"
        )
    else:
        print("[extra-genomics] Case subset: all")
    print(f"[extra-genomics] Modalities: {[spec.key for spec in selected_specs]}")

    download_cfg = cfg.data.source.download
    download_enabled = bool(download_cfg.enabled)
    skip_existing = bool(download_cfg.skip_existing)
    per_modality_download_max = _optional_int(
        download_cfg.get("max_extra_genomics_downloads_per_modality")
    )
    manifests_by_key: dict[str, list[dict[str, Any]]] = {}
    failed_downloads: list[dict[str, Any]] = []

    for spec in selected_specs:
        print(f"\n[extra-genomics] === {spec.key} === ({spec.description})")
        file_hits = fetch_extra_genomics_files(
            gdc_client=gdc_client,
            project_ids=project_ids,
            patient_submitter_ids=patient_submitter_ids or None,
            spec=spec,
            max_files=_optional_int(tcga_cfg.gdc.get(f"max_{spec.key}_files")),
        )
        print(f"[extra-genomics] Files indexed: {len(file_hits)}")

        plan = build_extra_genomics_download_plan(
            file_hits=file_hits,
            raw_root=raw_root,
            source_name=source_name,
            subfolder=spec.subfolder,
        )

        if download_enabled and plan:
            effective_plan = plan[:per_modality_download_max] if per_modality_download_max is not None else plan
            loop = tqdm(
                effective_plan,
                total=len(effective_plan),
                desc=f"Downloading {spec.key}",
                unit="file",
                leave=False,
            )
            completed = 0
            completed_plan: list[dict[str, Any]] = []
            for item in loop:
                try:
                    gdc_client.download_data_file(
                        file_id=item["file_id"],
                        output_path=Path(item["output_path"]),
                        skip_existing=skip_existing,
                    )
                    completed += 1
                    completed_plan.append(item)
                except Exception as exc:  # noqa: BLE001
                    failed_downloads.append(
                        {
                            "modality": spec.key,
                            "file_id": item.get("file_id", ""),
                            "file_name": item.get("file_name", ""),
                            "project_id": item.get("project_id", ""),
                            "patient_id": item.get("patient_id", ""),
                            "output_path": item.get("output_path", ""),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    tqdm.write(
                        "[extra-genomics] download failed; continuing: "
                        f"{item.get('file_id', '')} {item.get('file_name', '')} "
                        f"({type(exc).__name__}: {exc})"
                    )
            manifests_by_key[spec.key] = completed_plan
            failed = len(effective_plan) - completed
            print(f"[extra-genomics] Downloaded/resolved {completed} files for {spec.key}; failed: {failed}")
        elif not download_enabled:
            manifests_by_key[spec.key] = plan
            print(f"[extra-genomics] data.source.download.enabled=false; writing manifest only.")
        else:
            manifests_by_key[spec.key] = []

    manifest_path = write_extra_genomics_manifest(
        manifests_by_key=manifests_by_key,
        manifests_root=Path(str(cfg.data.manifests_root)),
        source_name=source_name,
    )
    print(f"\n[extra-genomics] Manifest written: {manifest_path}")
    if failed_downloads:
        import pandas as pd

        failed_path = manifest_path.with_name(manifest_path.stem + "_failed_downloads.parquet")
        pd.DataFrame(failed_downloads).to_parquet(failed_path, index=False)
        print(f"[extra-genomics] Failed-download manifest written: {failed_path}")

    update_cfg = extra_cfg.get("registry_update", {})
    if bool(update_cfg.get("enabled", True)):
        registry_path = Path(str(cfg.data.unified_registry_path))
        registry_df = read_parquet_or_empty(registry_path)
        if registry_df.empty:
            print(f"[extra-genomics] Registry is empty or missing; skipping registry update: {registry_path}")
            return
        import pandas as pd

        manifest_df = pd.read_parquet(manifest_path)
        updated_df, stats = update_registry_with_extra_genomics_manifest(
            registry_df,
            manifest_df,
            repo_root=ROOT,
            source_name=source_name,
            allowed_patient_ids=set(patient_submitter_ids) if patient_submitter_ids else None,
            clear_existing=bool(update_cfg.get("clear_existing", False)),
        )
        write_registry_parquet(updated_df, registry_path, validate=True)
        print("[extra-genomics] Registry updated.")
        print(f"[extra-genomics] Matched registry rows: {stats.matched_registry_rows}")
        print(f"[extra-genomics] Updated registry rows: {stats.updated_registry_rows}")
        print(f"[extra-genomics] Unmatched manifest cases: {stats.unmatched_manifest_cases}")


if __name__ == "__main__":
    main()
