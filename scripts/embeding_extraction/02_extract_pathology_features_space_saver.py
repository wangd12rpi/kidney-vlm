#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.sources.tcga import GDCClient
from kidney_vlm.pathology.feature_registry import register_existing_pathology_features
from kidney_vlm.pathology.trident_adapter import TridentAdapter
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


@dataclass(frozen=True)
class PendingRemoteSlideCandidate:
    row_idx: Any
    sample_id: str
    source: str
    project_id: str
    slide_path: str
    slide_stem: str
    pathology_file_ids: tuple[str, ...]
    needs_patch: bool
    needs_slide: bool


@dataclass(frozen=True)
class PendingRemoteSlideJob:
    row_idx: Any
    sample_id: str
    source: str
    project_id: str
    slide_path: str
    slide_stem: str
    file_id: str
    needs_patch: bool
    needs_slide: bool


@dataclass(frozen=True)
class PrefetchedRemoteSlideDownload:
    job: PendingRemoteSlideJob
    temp_slide_path: Path
    future: Future[Path]


@dataclass(frozen=True)
class RemoteExtractionDiscoveryStats:
    cases_scanned: int
    selected_slide_refs: int
    local_raw_available: int
    already_extracted: int
    unsupported_source: int
    missing_file_ids: int
    todo_candidates: int


def load_cfg():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    return cfg


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_local_path(path_value: str) -> str:
    text = str(path_value).strip().strip("'").strip('"')
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def _to_registry_relative_path(path_value: str | Path) -> str:
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
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix().lstrip("/")


def _local_wsi_paths(value: Any) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in _as_list(value):
        if "://" in raw_path:
            continue
        normalized = _normalize_local_path(raw_path)
        if normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return paths


def _tcga_sample_type_code(slide_stem: str) -> str:
    match = re.search(r"-([0-9]{2}[A-Z])-[0-9]{2}-", str(slide_stem).upper())
    if match is None:
        return ""
    return str(match.group(1))


def _is_normal_tcga_slide(slide_stem: str) -> bool:
    sample_type_code = _tcga_sample_type_code(slide_stem)
    return sample_type_code.startswith("11")


def _slide_kind_rank(slide_stem: str) -> int:
    upper_stem = str(slide_stem).upper()
    if "-DX" in upper_stem:
        return 0
    if "-TS" in upper_stem:
        return 1
    return 2


def _slide_kind(slide_stem: str) -> str:
    upper_stem = str(slide_stem).upper()
    if "-DX" in upper_stem:
        return "DX"
    if "-TS" in upper_stem:
        return "TS"
    if "-BS" in upper_stem:
        return "BS"
    return ""


def _resolve_allowed_slide_kinds(value: Any) -> set[str]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
            value = [part.strip().strip("'").strip('"') for part in text.split(",") if part.strip()]
    allowed = {str(item).strip().upper() for item in _as_list(value) if str(item).strip()}
    return {item for item in allowed if item}


def _filter_wsi_paths_by_allowed_kinds(paths: list[str], allowed_slide_kinds: set[str]) -> list[str]:
    if not allowed_slide_kinds:
        return paths
    filtered: list[str] = []
    for normalized_wsi_path in paths:
        if _slide_kind(Path(normalized_wsi_path).stem) in allowed_slide_kinds:
            filtered.append(normalized_wsi_path)
    return filtered


def _is_valid_h5(path: Path, required_keys: tuple[str, ...]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            return all(key in handle for key in required_keys)
    except Exception:
        return False


def _is_valid_patch_features(path: Path) -> bool:
    return _is_valid_h5(path, ("features", "coords"))


def _is_valid_slide_features(path: Path) -> bool:
    return _is_valid_h5(path, ("features",))


def _is_valid_coords(path: Path) -> bool:
    return _is_valid_h5(path, ("coords",))


def _is_valid_patch_tensor(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_patch_count(coords_path: Path, patch_features_path: Path, save_format: str) -> int:
    if _is_valid_coords(coords_path):
        try:
            import h5py

            with h5py.File(coords_path, "r") as handle:
                if "coords" in handle:
                    return int(handle["coords"].shape[0])
        except Exception:
            pass

    if save_format == "h5" and _is_valid_patch_features(patch_features_path):
        try:
            import h5py

            with h5py.File(patch_features_path, "r") as handle:
                if "features" in handle:
                    return int(handle["features"].shape[0])
        except Exception:
            return 0

    if save_format == "pt" and _is_valid_patch_tensor(patch_features_path):
        try:
            tensor = torch.load(patch_features_path, map_location="cpu")
            if hasattr(tensor, "shape") and len(tensor.shape) > 0:
                return int(tensor.shape[0])
            if isinstance(tensor, list):
                return len(tensor)
        except Exception:
            return 0

    return 0


def _existing_patch_count_for_slide(
    slide_stem: str,
    *,
    patch_features_dir: Path,
    coords_root: Path,
    save_format: str,
) -> int | None:
    patch_features_path = patch_features_dir / f"{slide_stem}.{save_format}"
    coords_path = coords_root / "patches" / f"{slide_stem}_patches.h5"

    patch_ready = _is_valid_patch_features(patch_features_path) if save_format == "h5" else _is_valid_patch_tensor(patch_features_path)
    if not patch_ready:
        return None

    return _read_patch_count(coords_path, patch_features_path, save_format)


def _select_case_wsi_paths(
    case_wsi_paths: list[str],
    *,
    patch_features_dir: Path,
    coords_root: Path,
    save_format: str,
) -> list[str]:
    if not case_wsi_paths:
        return []

    ranked_paths: list[tuple[tuple[Any, ...], str]] = []
    for normalized_wsi_path in case_wsi_paths:
        slide_path = Path(normalized_wsi_path)
        slide_stem = slide_path.stem
        existing_patch_count = _existing_patch_count_for_slide(
            slide_stem,
            patch_features_dir=patch_features_dir,
            coords_root=coords_root,
            save_format=save_format,
        )
        sort_key = (
            1 if _is_normal_tcga_slide(slide_stem) else 0,
            _slide_kind_rank(slide_stem),
            0 if existing_patch_count is not None else 1,
            -(existing_patch_count or 0),
            slide_path.name.upper(),
        )
        ranked_paths.append((sort_key, normalized_wsi_path))

    ranked_paths.sort(key=lambda item: item[0])
    return [ranked_paths[0][1]]


def _filter_existing_embedding_paths(paths: list[str], selected_slide_stem: str | None) -> list[str]:
    if not selected_slide_stem:
        return paths
    filtered_paths: list[str] = []
    for path_value in paths:
        if Path(str(path_value)).stem == selected_slide_stem:
            filtered_paths.append(str(path_value))
    return filtered_paths


def _resolve_device(device: str) -> str:
    requested = str(device).strip() or "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested '{requested}' but CUDA is unavailable; falling back to 'cpu'.")
        return "cpu"
    return requested


def _resolve_trident_root(configured_root: str) -> Path:
    root = Path(configured_root).expanduser()
    if root.exists():
        return root

    fallbacks = [
        ROOT / "external" / "TRIDENT",
        ROOT / "external" / "trident",
    ]
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    return root


def _resolve_registry_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _build_gdc_client(cfg: Any) -> GDCClient:
    return GDCClient(
        base_url=str(cfg.base_url),
        timeout_seconds=int(cfg.timeout_seconds),
        page_size=int(cfg.page_size),
        max_retries=int(cfg.max_retries),
        retry_backoff_seconds=float(cfg.retry_backoff_seconds),
    )


def _resolve_tcga_gdc_cfg(cfg: Any) -> Any:
    direct_cfg = OmegaConf.select(cfg, "data.source.tcga.gdc")
    if direct_cfg is not None:
        return direct_cfg

    tcga_cfg_path = ROOT / "conf" / "data" / "sources" / "tcga.yaml"
    if not tcga_cfg_path.exists():
        raise FileNotFoundError(
            "TCGA data source config is missing, cannot build GDC client: "
            f"{tcga_cfg_path}"
        )

    fallback_cfg = OmegaConf.load(tcga_cfg_path)
    fallback_gdc_cfg = OmegaConf.select(fallback_cfg, "data.source.tcga.gdc")
    if fallback_gdc_cfg is None:
        raise KeyError(
            "TCGA source config does not define data.source.tcga.gdc: "
            f"{tcga_cfg_path}"
        )
    return fallback_gdc_cfg


def _fetch_file_name_by_id(
    gdc_client: GDCClient,
    file_ids: list[str],
    *,
    chunk_size: int,
) -> dict[str, str]:
    normalized_ids = [str(file_id).strip() for file_id in file_ids if str(file_id).strip()]
    if not normalized_ids:
        return {}

    file_name_by_id: dict[str, str] = {}
    for start in range(0, len(normalized_ids), max(chunk_size, 1)):
        chunk_ids = normalized_ids[start : start + max(chunk_size, 1)]
        hits = gdc_client.fetch_files_by_ids(chunk_ids, fields=["file_id", "file_name"])
        for hit in hits:
            file_id = str(hit.get("file_id", "")).strip()
            file_name = str(hit.get("file_name", "")).strip()
            if file_id and file_name:
                file_name_by_id[file_id] = file_name
    return file_name_by_id


def _match_candidate_to_file_id(candidate: PendingRemoteSlideCandidate, file_name_by_id: dict[str, str]) -> str:
    target_name = Path(candidate.slide_path).name
    matched_ids = [
        file_id
        for file_id in candidate.pathology_file_ids
        if str(file_name_by_id.get(file_id, "")).strip() == target_name
    ]
    if len(matched_ids) == 1:
        return matched_ids[0]
    if len(matched_ids) > 1:
        return sorted(matched_ids)[0]
    if len(candidate.pathology_file_ids) == 1:
        return candidate.pathology_file_ids[0]
    return ""


def _build_remote_slide_jobs(
    candidates: list[PendingRemoteSlideCandidate],
    file_name_by_id: dict[str, str],
) -> tuple[list[PendingRemoteSlideJob], int]:
    jobs: list[PendingRemoteSlideJob] = []
    unresolved = 0
    for candidate in candidates:
        matched_file_id = _match_candidate_to_file_id(candidate, file_name_by_id)
        if not matched_file_id:
            unresolved += 1
            continue
        jobs.append(
            PendingRemoteSlideJob(
                row_idx=candidate.row_idx,
                sample_id=candidate.sample_id,
                source=candidate.source,
                project_id=candidate.project_id,
                slide_path=candidate.slide_path,
                slide_stem=candidate.slide_stem,
                file_id=matched_file_id,
                needs_patch=candidate.needs_patch,
                needs_slide=candidate.needs_slide,
            )
        )
    return jobs, unresolved


def _build_remote_slide_candidates(
    registry_df: pd.DataFrame,
    *,
    patch_features_dir: Path,
    slide_features_dir: Path,
    coords_root: Path,
    save_format: str,
    extract_patch_only: bool,
    overwrite_existing: bool,
    allowed_slide_kinds: set[str],
) -> tuple[list[PendingRemoteSlideCandidate], RemoteExtractionDiscoveryStats]:
    candidates: list[PendingRemoteSlideCandidate] = []
    selected_slide_refs = 0
    local_raw_available = 0
    already_extracted = 0
    unsupported_source = 0
    missing_file_ids = 0

    for row_idx in registry_df.index.tolist():
        row = registry_df.loc[row_idx]
        case_wsi_paths = _filter_wsi_paths_by_allowed_kinds(
            _local_wsi_paths(row.get("pathology_wsi_paths")),
            allowed_slide_kinds,
        )
        for normalized_wsi_path in case_wsi_paths:
            selected_slide_refs += 1
            slide_path = Path(normalized_wsi_path)
            slide_stem = slide_path.stem
            final_patch_path = patch_features_dir / f"{slide_stem}.{save_format}"
            final_slide_path = slide_features_dir / f"{slide_stem}.h5"

            patch_ready = _is_valid_patch_features(final_patch_path) if save_format == "h5" else _is_valid_patch_tensor(final_patch_path)
            slide_ready = True if extract_patch_only else _is_valid_slide_features(final_slide_path)
            needs_patch = overwrite_existing or not patch_ready
            needs_slide = (not extract_patch_only) and (overwrite_existing or not slide_ready)

            if not needs_patch and not needs_slide:
                already_extracted += 1
                continue

            if slide_path.exists():
                local_raw_available += 1
                continue

            source = str(row.get("source", "")).strip().lower()
            if source != "tcga":
                unsupported_source += 1
                continue

            pathology_file_ids = tuple(_as_list(row.get("pathology_file_ids")))
            if not pathology_file_ids:
                missing_file_ids += 1
                continue

            candidates.append(
                PendingRemoteSlideCandidate(
                    row_idx=row_idx,
                    sample_id=str(row.get("sample_id", "")).strip(),
                    source=str(row.get("source", "")).strip(),
                    project_id=str(row.get("project_id", "")).strip(),
                    slide_path=normalized_wsi_path,
                    slide_stem=slide_stem,
                    pathology_file_ids=pathology_file_ids,
                    needs_patch=needs_patch,
                    needs_slide=needs_slide,
                )
            )

    stats = RemoteExtractionDiscoveryStats(
        cases_scanned=len(registry_df),
        selected_slide_refs=selected_slide_refs,
        local_raw_available=local_raw_available,
        already_extracted=already_extracted,
        unsupported_source=unsupported_source,
        missing_file_ids=missing_file_ids,
        todo_candidates=len(candidates),
    )
    return candidates, stats


def _count_registry_slide_refs(registry_df: pd.DataFrame, *, allowed_slide_kinds: set[str]) -> int:
    return sum(
        len(_filter_wsi_paths_by_allowed_kinds(_local_wsi_paths(row.get("pathology_wsi_paths")), allowed_slide_kinds))
        for _, row in registry_df.iterrows()
    )


def _existing_registry_feature_paths(registry_df: pd.DataFrame, *, save_format: str) -> set[str]:
    existing_paths: set[str] = set()
    for _, row in registry_df.iterrows():
        for feature_path in _as_list(row.get("pathology_tile_embedding_paths")):
            local_path = Path(_normalize_local_path(feature_path))
            valid = _is_valid_patch_features(local_path) if save_format == "h5" else _is_valid_patch_tensor(local_path)
            if valid:
                existing_paths.add(_to_registry_relative_path(local_path))
    return existing_paths


def _sync_registry_with_saved_patch_features(
    registry_df: pd.DataFrame,
    *,
    registry_path: Path,
    patch_features_dir: Path,
    coords_root: Path,
    save_format: str,
    patch_size: int,
    target_mag: int,
) -> tuple[pd.DataFrame, int, int]:
    before_paths = _existing_registry_feature_paths(registry_df, save_format=save_format)
    updated_df, _ = register_existing_pathology_features(
        registry_df,
        patch_features_dir=patch_features_dir,
        coords_root=coords_root,
        save_format=save_format,
        patch_size=patch_size,
        target_mag=target_mag,
        root_dir=ROOT,
        progress=False,
    )
    after_paths = _existing_registry_feature_paths(updated_df, save_format=save_format)
    inserted_paths = after_paths - before_paths
    if inserted_paths or not updated_df.equals(registry_df):
        write_registry_parquet(updated_df, registry_path, validate=False)
    return updated_df, len(inserted_paths), len(after_paths)


def _rebuild_case_slide_embedding_paths(row: pd.Series, *, slide_features_dir: Path) -> list[str]:
    slide_paths: list[str] = []
    seen: set[str] = set()
    for normalized_wsi_path in _local_wsi_paths(row.get("pathology_wsi_paths")):
        slide_stem = Path(normalized_wsi_path).stem
        slide_feature_path = slide_features_dir / f"{slide_stem}.h5"
        if not _is_valid_slide_features(slide_feature_path):
            continue
        relative_path = _to_registry_relative_path(slide_feature_path)
        if relative_path in seen:
            continue
        seen.add(relative_path)
        slide_paths.append(relative_path)
    return slide_paths


def _print_step_line(step_no: int, label: str, *metrics: tuple[str, Any]) -> None:
    parts = [f"Step {step_no}", label]
    for name, value in metrics:
        parts.append(f"{name}={value}")
    print("\t".join(parts))


def _print_timing_line(label: str, *metrics: tuple[str, Any]) -> None:
    parts = [label]
    for name, value in metrics:
        if isinstance(value, float):
            parts.append(f"{name}={value:.2f}")
        else:
            parts.append(f"{name}={value}")
    print("\t".join(parts))


def _canonicalize_generated_artifact(
    generated_path: Path,
    canonical_path: Path,
    *,
    overwrite_existing: bool,
) -> Path:
    if generated_path.resolve() == canonical_path.resolve():
        return generated_path

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    if canonical_path.exists():
        if overwrite_existing:
            canonical_path.unlink()
        else:
            if generated_path.exists() and generated_path.resolve() != canonical_path.resolve():
                generated_path.unlink()
            return canonical_path

    generated_path.replace(canonical_path)
    return canonical_path


def _build_temp_slide_download_path(temp_dir: Path, *, job: PendingRemoteSlideJob, job_index: int) -> Path:
    file_name = Path(job.slide_path).name.strip() or f"{job.slide_stem}.svs"
    safe_file_id = re.sub(r"[^A-Za-z0-9._-]+", "-", str(job.file_id).strip()) or "unknown-file"
    return temp_dir / f"{job_index:05d}-{safe_file_id}-{file_name}"


def _submit_prefetch_download(
    *,
    executor: ThreadPoolExecutor,
    gdc_client: GDCClient,
    job: PendingRemoteSlideJob,
    temp_dir: Path,
    job_index: int,
) -> PrefetchedRemoteSlideDownload:
    temp_slide_path = _build_temp_slide_download_path(temp_dir, job=job, job_index=job_index)
    if temp_slide_path.exists():
        temp_slide_path.unlink()

    future = executor.submit(
        gdc_client.download_data_file,
        file_id=job.file_id,
        output_path=temp_slide_path,
        skip_existing=False,
    )
    return PrefetchedRemoteSlideDownload(
        job=job,
        temp_slide_path=temp_slide_path,
        future=future,
    )


def main() -> None:
    cfg = load_cfg()

    pathology_cfg = cfg.embeding_extraction.pathology
    registry_path = _resolve_registry_path(cfg.data.unified_registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Unified registry not found at '{registry_path}'. "
            "Build a source first before extracting pathology features."
        )

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        print(f"Registry is empty at '{registry_path}'. Nothing to extract.")
        return

    patch_encoder_name = str(pathology_cfg.patch_encoder)
    slide_encoder_name = str(pathology_cfg.slide_encoder)
    extract_patch_only = bool(pathology_cfg.get("extract_patch_only", False))

    patch_encoder_kwargs = OmegaConf.to_container(pathology_cfg.get("patch_encoder_kwargs", {}), resolve=True) or {}
    slide_encoder_kwargs = OmegaConf.to_container(pathology_cfg.get("slide_encoder_kwargs", {}), resolve=True) or {}

    overwrite_existing = bool(pathology_cfg.get("overwrite_existing", False))
    skip_errors = bool(pathology_cfg.get("skip_errors", True))
    device = _resolve_device(str(pathology_cfg.get("device", "cpu")))
    save_format = str(pathology_cfg.get("save_format", "h5")).lower()
    if save_format not in {"h5", "pt"}:
        raise ValueError("save_format must be one of: h5, pt")
    if not extract_patch_only and save_format != "h5":
        raise ValueError("Slide feature extraction requires save_format='h5' for patch features.")

    target_mag = int(pathology_cfg.get("target_magnification", 20))
    patch_size = int(pathology_cfg.get("patch_size", 512))
    patch_overlap = int(pathology_cfg.get("patch_overlap", 0))
    min_tissue_proportion = float(pathology_cfg.get("min_tissue_proportion", 0.0))
    batch_limit = int(pathology_cfg.get("batch_limit", 512))
    allowed_slide_kinds = _resolve_allowed_slide_kinds(pathology_cfg.get("allowed_slide_kinds"))
    patch_loader_num_workers_cfg = pathology_cfg.get("patch_loader_num_workers")
    patch_loader_num_workers = None if patch_loader_num_workers_cfg in (None, "", "null") else int(patch_loader_num_workers_cfg)
    verbose_inner = bool(pathology_cfg.get("verbose", False))
    timing_debug = bool(pathology_cfg.get("timing_debug", False))

    features_root = Path(str(pathology_cfg.get("features_root", ROOT / "data" / "features"))).expanduser()
    if not features_root.is_absolute():
        features_root = (ROOT / features_root).resolve()
    else:
        features_root = features_root.resolve()
    features_root.mkdir(parents=True, exist_ok=True)
    patch_features_dir = features_root / f"features_{patch_encoder_name}"
    slide_features_dir = features_root / f"slide_features_{slide_encoder_name}"
    patch_features_dir.mkdir(parents=True, exist_ok=True)
    if not extract_patch_only:
        slide_features_dir.mkdir(parents=True, exist_ok=True)

    coords_root = Path(
        str(
            pathology_cfg.get(
                "coords_root",
                features_root / f"coords_{target_mag}x_{patch_size}px_{patch_overlap}px_overlap",
            )
        )
    ).expanduser()
    if not coords_root.is_absolute():
        coords_root = (ROOT / coords_root).resolve()
    else:
        coords_root = coords_root.resolve()

    total_registry_slide_refs = _count_registry_slide_refs(registry_df, allowed_slide_kinds=allowed_slide_kinds)
    _print_step_line(1, "registry pathology slide refs", ("slides", total_registry_slide_refs))

    indexed_registry_feature_paths = _existing_registry_feature_paths(registry_df, save_format=save_format)
    _print_step_line(2, "registry-indexed feature files on disk", ("slides", len(indexed_registry_feature_paths)))

    registry_df, inserted_feature_paths, indexed_feature_paths_after_sync = _sync_registry_with_saved_patch_features(
        registry_df,
        registry_path=registry_path,
        patch_features_dir=patch_features_dir,
        coords_root=coords_root,
        save_format=save_format,
        patch_size=patch_size,
        target_mag=target_mag,
    )
    _print_step_line(
        3,
        "sync saved feature files into registry",
        ("inserted", inserted_feature_paths),
        ("indexed_after_sync", indexed_feature_paths_after_sync),
    )

    candidates, discovery_stats = _build_remote_slide_candidates(
        registry_df,
        patch_features_dir=patch_features_dir,
        slide_features_dir=slide_features_dir,
        coords_root=coords_root,
        save_format=save_format,
        extract_patch_only=extract_patch_only,
        overwrite_existing=overwrite_existing,
        allowed_slide_kinds=allowed_slide_kinds,
    )

    _print_step_line(5, "remote slides needing temp-download extraction", ("slides", discovery_stats.todo_candidates))
    _print_timing_line(
        "Config",
        ("allowed_slide_kinds", ",".join(sorted(allowed_slide_kinds)) if allowed_slide_kinds else "all"),
        ("batch_limit", batch_limit),
        ("patch_loader_num_workers", patch_loader_num_workers if patch_loader_num_workers is not None else "auto"),
        ("timing_debug", timing_debug),
    )

    if not candidates:
        print("No missing remote pathology slides require extraction.")
        return

    file_metadata_chunk_size = int(pathology_cfg.get("file_metadata_chunk_size", 200) or 200)
    gdc_client = _build_gdc_client(_resolve_tcga_gdc_cfg(cfg))
    unique_file_ids = sorted({file_id for candidate in candidates for file_id in candidate.pathology_file_ids})
    file_name_by_id = _fetch_file_name_by_id(gdc_client, unique_file_ids, chunk_size=file_metadata_chunk_size)
    jobs, unresolved_jobs = _build_remote_slide_jobs(candidates, file_name_by_id)

    print(f"Unique pathology file IDs queried: {len(unique_file_ids)}")
    print(f"Resolved remote extraction jobs: {len(jobs)}")
    print(f"Unresolved remote jobs after file-name matching: {unresolved_jobs}")

    if not jobs:
        print("No remote pathology slides could be resolved to GDC file IDs. Nothing to extract.")
        return

    trident_root = _resolve_trident_root(str(pathology_cfg.trident_root))
    adapter = TridentAdapter(trident_root=trident_root)
    adapter.ensure_on_path()
    adapter.import_core()

    from trident import OpenSlideWSI
    from trident.patch_encoder_models import encoder_factory as patch_encoder_factory

    patch_encoder = patch_encoder_factory(patch_encoder_name, **patch_encoder_kwargs)
    patch_encoder.eval()
    patch_encoder.to(device)

    slide_encoder = None
    if not extract_patch_only:
        from trident.slide_encoder_models import encoder_factory as slide_encoder_factory

        slide_encoder = slide_encoder_factory(slide_encoder_name, **slide_encoder_kwargs)
        slide_encoder.eval()
        slide_encoder.to(device)

    temp_download_root_value = str(pathology_cfg.get("temp_download_root", "")).strip()
    temp_download_root: Path | None = None
    if temp_download_root_value:
        temp_download_root = Path(temp_download_root_value).expanduser()
        if not temp_download_root.is_absolute():
            temp_download_root = (ROOT / temp_download_root).resolve()
        else:
            temp_download_root = temp_download_root.resolve()
        temp_download_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    extracted_patch = 0
    extracted_slide = 0
    failed = 0
    registry_rows_written = 0

    with tempfile.TemporaryDirectory(
        dir=str(temp_download_root) if temp_download_root is not None else None,
        prefix="kidney-vlm-pathology-",
    ) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        job_loop = tqdm(jobs, total=len(jobs), desc="Extracting missing pathology embeddings")
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pathology-download-prefetch") as download_executor:
            next_prefetch: PrefetchedRemoteSlideDownload | None = None
            for job_index, job in enumerate(job_loop):
                slide_total_start = time.perf_counter()
                current_prefetch = next_prefetch
                if current_prefetch is None:
                    current_prefetch = _submit_prefetch_download(
                        executor=download_executor,
                        gdc_client=gdc_client,
                        job=job,
                        temp_dir=temp_dir,
                        job_index=job_index,
                    )

                next_prefetch = None
                if job_index + 1 < len(jobs):
                    next_prefetch = _submit_prefetch_download(
                        executor=download_executor,
                        gdc_client=gdc_client,
                        job=jobs[job_index + 1],
                        temp_dir=temp_dir,
                        job_index=job_index + 1,
                    )

                row = registry_df.loc[job.row_idx]
                final_patch_path = patch_features_dir / f"{job.slide_stem}.{save_format}"
                final_slide_path = slide_features_dir / f"{job.slide_stem}.h5"
                coords_path = coords_root / "patches" / f"{job.slide_stem}_patches.h5"
                temp_slide_path = current_prefetch.temp_slide_path
                download_wait_s = 0.0
                slide_open_s = 0.0
                coords_extract_s = 0.0
                patch_extract_s = 0.0
                slide_extract_s = 0.0
                registry_write_s = 0.0

                if overwrite_existing:
                    stale_paths = [coords_path, final_patch_path]
                    if not extract_patch_only:
                        stale_paths.append(final_slide_path)
                    for stale_path in stale_paths:
                        if stale_path.exists():
                            stale_path.unlink()

                slide = None
                try:
                    download_wait_start = time.perf_counter()
                    downloaded_path = Path(current_prefetch.future.result())
                    download_wait_s = time.perf_counter() - download_wait_start
                    downloaded += 1
                    slide_open_start = time.perf_counter()
                    slide = OpenSlideWSI(slide_path=str(downloaded_path), lazy_init=False)
                    slide_open_s = time.perf_counter() - slide_open_start

                    if job.needs_patch:
                        coords_ok = _is_valid_coords(coords_path)
                        if not coords_ok:
                            coords_start = time.perf_counter()
                            coords_path = Path(
                                slide.extract_tissue_coords(
                                    target_mag=target_mag,
                                    patch_size=patch_size,
                                    save_coords=str(coords_root),
                                    overlap=patch_overlap,
                                    min_tissue_proportion=min_tissue_proportion,
                                )
                            )
                            coords_path = _canonicalize_generated_artifact(
                                coords_path,
                                coords_root / "patches" / f"{job.slide_stem}_patches.h5",
                                overwrite_existing=overwrite_existing,
                            )
                            coords_extract_s = time.perf_counter() - coords_start

                        patch_start = time.perf_counter()
                        generated_patch_path = Path(
                            slide.extract_patch_features(
                                patch_encoder=patch_encoder,
                                coords_path=str(coords_path),
                                save_features=str(patch_features_dir),
                                device=device,
                                saveas=save_format,
                                batch_limit=batch_limit,
                                verbose=verbose_inner,
                                num_workers=patch_loader_num_workers,
                                timing_verbose=timing_debug,
                            )
                        )
                        generated_patch_path = _canonicalize_generated_artifact(
                            generated_patch_path,
                            patch_features_dir / f"{job.slide_stem}.{save_format}",
                            overwrite_existing=overwrite_existing,
                        )
                        patch_extract_s = time.perf_counter() - patch_start
                        final_patch_path = generated_patch_path
                        valid_patch = _is_valid_patch_features(final_patch_path) if save_format == "h5" else _is_valid_patch_tensor(final_patch_path)
                        if not valid_patch:
                            raise RuntimeError(f"Generated patch features are invalid: {final_patch_path}")
                        extracted_patch += 1

                    if job.needs_slide:
                        if not _is_valid_patch_features(final_patch_path):
                            raise FileNotFoundError(
                                f"Patch features are missing or invalid, cannot run slide encoder: {final_patch_path}"
                            )
                        slide_extract_start = time.perf_counter()
                        generated_slide_path = Path(
                            slide.extract_slide_features(
                                patch_features_path=str(final_patch_path),
                                slide_encoder=slide_encoder,
                                save_features=str(slide_features_dir),
                                device=device,
                            )
                        )
                        generated_slide_path = _canonicalize_generated_artifact(
                            generated_slide_path,
                            slide_features_dir / f"{job.slide_stem}.h5",
                            overwrite_existing=overwrite_existing,
                        )
                        slide_extract_s = time.perf_counter() - slide_extract_start
                        final_slide_path = generated_slide_path
                        if not _is_valid_slide_features(final_slide_path):
                            raise RuntimeError(f"Generated slide features are invalid: {final_slide_path}")
                        extracted_slide += 1

                    if torch.cuda.is_available() and device.startswith("cuda"):
                        torch.cuda.empty_cache()
                except Exception as exc:
                    failed += 1
                    job_loop.write(f"[error] {Path(job.slide_path).name}: {exc}")
                    if timing_debug:
                        job_loop.write(
                            "\t".join(
                                [
                                    "[timing][remote_slide]",
                                    f"slide={job.slide_stem}",
                                    "status=error",
                                    f"download_wait_s={download_wait_s:.2f}",
                                    f"open_s={slide_open_s:.2f}",
                                    f"coords_s={coords_extract_s:.2f}",
                                    f"patch_s={patch_extract_s:.2f}",
                                    f"slide_s={slide_extract_s:.2f}",
                                    f"registry_s={registry_write_s:.2f}",
                                    f"total_s={(time.perf_counter() - slide_total_start):.2f}",
                                ]
                            )
                        )
                    if not skip_errors:
                        raise
                finally:
                    if slide is not None:
                        try:
                            slide.release()
                        except Exception:
                            pass
                    if temp_slide_path.exists():
                        temp_slide_path.unlink()

                patch_ready = _is_valid_patch_features(final_patch_path) if save_format == "h5" else _is_valid_patch_tensor(final_patch_path)
                slide_ready = True if extract_patch_only else _is_valid_slide_features(final_slide_path)
                if not patch_ready:
                    continue

                registry_write_start = time.perf_counter()
                single_row_df = registry_df.loc[[job.row_idx]].copy()
                updated_row_df, _ = register_existing_pathology_features(
                    single_row_df,
                    patch_features_dir=patch_features_dir,
                    coords_root=coords_root,
                    save_format=save_format,
                    patch_size=patch_size,
                    target_mag=target_mag,
                    root_dir=ROOT,
                    progress=False,
                )
                updated_row = updated_row_df.iloc[0]
                registry_df.at[job.row_idx, "pathology_tile_embedding_paths"] = updated_row["pathology_tile_embedding_paths"]
                registry_df.at[job.row_idx, "pathology_tile_embedding_patch_counts"] = updated_row["pathology_tile_embedding_patch_counts"]
                registry_df.at[job.row_idx, "pathology_embedding_patch_size"] = updated_row["pathology_embedding_patch_size"]
                registry_df.at[job.row_idx, "pathology_embedding_magnification"] = updated_row["pathology_embedding_magnification"]

                if not extract_patch_only:
                    registry_df.at[job.row_idx, "pathology_slide_embedding_paths"] = _rebuild_case_slide_embedding_paths(
                        registry_df.loc[job.row_idx],
                        slide_features_dir=slide_features_dir,
                    )

                write_registry_parquet(registry_df, registry_path, validate=False)
                registry_rows_written += 1
                registry_write_s = time.perf_counter() - registry_write_start

                if timing_debug:
                    patch_counts = updated_row.get("pathology_tile_embedding_patch_counts", []) or []
                    patch_count = patch_counts[0] if patch_counts else 0
                    job_loop.write(
                        "\t".join(
                            [
                                "[timing][remote_slide]",
                                f"slide={job.slide_stem}",
                                "status=ok",
                                f"patches={patch_count}",
                                f"download_wait_s={download_wait_s:.2f}",
                                f"open_s={slide_open_s:.2f}",
                                f"coords_s={coords_extract_s:.2f}",
                                f"patch_s={patch_extract_s:.2f}",
                                f"slide_s={slide_extract_s:.2f}",
                                f"registry_s={registry_write_s:.2f}",
                                f"total_s={(time.perf_counter() - slide_total_start):.2f}",
                            ]
                        )
                    )

    print("Remote extraction complete.")
    print(f"Slides downloaded to temp storage this run: {downloaded}")
    print(f"Patch embeddings extracted this run: {extracted_patch}")
    print(f"Slide embeddings extracted this run: {extracted_slide}")
    print(f"Slides failed: {failed}")
    print(f"Registry rows written this run: {registry_rows_written}")
    print(f"Unified registry written: {registry_path}")


if __name__ == "__main__":
    main()
