#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.sources.tcga import GDCClient
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

THUMBNAIL_COLUMN = "pathology_png_thumbnail_paths"


@dataclass(frozen=True)
class PathologyPngJob:
    row_idx: int
    sample_id: str
    patient_id: str
    project_id: str
    slide_path: str
    slide_stem: str
    file_id: str
    output_dir: Path


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="01_pathology_png/01_extract_pathology_pngs.yaml",
        overrides=sys.argv[1:],
    )


def _as_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _to_project_relative_path(path_value: str | Path) -> str:
    path = Path(path_value).expanduser().resolve()
    return Path(os.path.relpath(path, start=ROOT)).as_posix()


def _safe_name(value: str, *, default: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip(".-_")
    return safe or default


def _slide_kind(slide_stem: str) -> str:
    match = re.search(r"-([A-Z]{2})[0-9](?:[._-]|$)", str(slide_stem).upper())
    return match.group(1) if match else ""


def _slide_kind_allowed(slide_stem: str, allowed_slide_kinds: set[str]) -> bool:
    if not allowed_slide_kinds:
        return True
    return _slide_kind(slide_stem) in allowed_slide_kinds


def _thumbnail_output_path(output_dir: Path, slide_stem: str) -> Path:
    return output_dir / f"{slide_stem}__thumbnail.png"


def _valid_png(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def _slide_outputs_ready(output_dir: Path, slide_stem: str) -> bool:
    return _valid_png(_thumbnail_output_path(output_dir, slide_stem))


def _existing_slide_output_paths(output_dir: Path, slide_stem: str) -> list[Path]:
    thumbnail_path = _thumbnail_output_path(output_dir, slide_stem)
    return [thumbnail_path] if _valid_png(thumbnail_path) else []


def _remove_slide_paths(existing_paths: list[str], slide_stem: str) -> list[str]:
    prefix = f"{slide_stem}__"
    return [path for path in existing_paths if not Path(path).name.startswith(prefix)]


def _register_slide_thumbnail(
    registry_df: pd.DataFrame,
    *,
    row_idx: int,
    slide_stem: str,
    thumbnail_paths: list[Path],
) -> None:
    if THUMBNAIL_COLUMN not in registry_df.columns:
        registry_df[THUMBNAIL_COLUMN] = [[] for _ in range(len(registry_df))]

    existing_thumbnails = _remove_slide_paths(_as_list(registry_df.at[row_idx, THUMBNAIL_COLUMN]), slide_stem)
    registry_df.at[row_idx, THUMBNAIL_COLUMN] = existing_thumbnails + [
        _to_project_relative_path(path) for path in thumbnail_paths
    ]


def _fetch_file_name_by_id(gdc_client: GDCClient, file_ids: list[str], *, chunk_size: int) -> dict[str, str]:
    normalized_ids = [str(file_id).strip() for file_id in file_ids if str(file_id).strip()]
    if not normalized_ids:
        return {}
    file_name_by_id: dict[str, str] = {}
    step = max(1, int(chunk_size))
    for start in tqdm(range(0, len(normalized_ids), step), desc="Fetching GDC file names", unit="batch"):
        chunk_ids = normalized_ids[start : start + step]
        hits = gdc_client.fetch_files_by_ids(chunk_ids, fields=["file_id", "file_name"])
        for hit in hits:
            file_id = str(hit.get("file_id", "")).strip()
            file_name = str(hit.get("file_name", "")).strip()
            if file_id and file_name:
                file_name_by_id[file_id] = file_name
    return file_name_by_id


def _match_file_id_for_slide(slide_path: str, file_ids: list[str], file_name_by_id: dict[str, str]) -> str:
    target_name = Path(slide_path).name
    matched = [file_id for file_id in file_ids if file_name_by_id.get(file_id, "") == target_name]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        return sorted(matched)[0]
    if len(file_ids) == 1:
        return file_ids[0]
    return ""


def _extract_slide_thumbnail(
    *,
    slide_path: Path,
    output_dir: Path,
    slide_stem: str,
    thumbnail_max_size_px: int,
    overwrite: bool,
) -> list[Path]:
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError("openslide-python is required for pathology thumbnail extraction.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = _thumbnail_output_path(output_dir, slide_stem)

    with openslide.OpenSlide(str(slide_path)) as slide:
        thumbnail = slide.get_thumbnail((int(thumbnail_max_size_px), int(thumbnail_max_size_px))).convert("RGB")
        if overwrite or not _valid_png(thumbnail_path):
            thumbnail.save(thumbnail_path)

    return [thumbnail_path]


def _build_jobs(
    registry_df: pd.DataFrame,
    *,
    output_root: Path,
    file_name_by_id: dict[str, str],
    allowed_slide_kinds: set[str],
    source_filter: str,
    project_ids: set[str],
    max_cases: int | None,
    max_slides: int | None,
    overwrite: bool,
) -> tuple[list[PathologyPngJob], int]:
    jobs: list[PathologyPngJob] = []
    ready_count = 0
    cases_seen = 0

    for row_idx in registry_df.index.tolist():
        row = registry_df.loc[row_idx]
        source = str(row.get("source", "")).strip().lower()
        if source_filter and source != source_filter:
            continue
        project_id = str(row.get("project_id", "")).strip()
        if project_ids and project_id not in project_ids:
            continue

        slide_paths = _as_list(row.get("pathology_wsi_paths"))
        file_ids = _as_list(row.get("pathology_file_ids"))
        if not slide_paths or not file_ids:
            continue

        cases_seen += 1
        if max_cases is not None and cases_seen > max_cases:
            break

        patient_id = str(row.get("patient_id", "")).strip()
        case_dir = output_root / _safe_name(patient_id or str(row.get("sample_id", "")), default=f"row-{row_idx}")
        for slide_path in slide_paths:
            slide_stem = Path(slide_path).stem
            if not _slide_kind_allowed(slide_stem, allowed_slide_kinds):
                continue
            if not overwrite and _slide_outputs_ready(case_dir, slide_stem):
                thumbnail_paths = _existing_slide_output_paths(case_dir, slide_stem)
                _register_slide_thumbnail(
                    registry_df,
                    row_idx=row_idx,
                    slide_stem=slide_stem,
                    thumbnail_paths=thumbnail_paths,
                )
                ready_count += 1
                continue

            file_id = _match_file_id_for_slide(slide_path, file_ids, file_name_by_id)
            if not file_id:
                continue
            jobs.append(
                PathologyPngJob(
                    row_idx=int(row_idx),
                    sample_id=str(row.get("sample_id", "")).strip(),
                    patient_id=patient_id,
                    project_id=project_id,
                    slide_path=slide_path,
                    slide_stem=slide_stem,
                    file_id=file_id,
                    output_dir=case_dir,
                )
            )
            if max_slides is not None and len(jobs) >= max_slides:
                return jobs, ready_count

    return jobs, ready_count


def _collect_candidate_file_ids(
    registry_df: pd.DataFrame,
    *,
    output_root: Path,
    allowed_slide_kinds: set[str],
    source_filter: str,
    project_ids: set[str],
    max_cases: int | None,
    overwrite: bool,
) -> list[str]:
    file_ids: set[str] = set()
    cases_seen = 0
    for row_idx in registry_df.index.tolist():
        row = registry_df.loc[row_idx]
        source = str(row.get("source", "")).strip().lower()
        if source_filter and source != source_filter:
            continue
        project_id = str(row.get("project_id", "")).strip()
        if project_ids and project_id not in project_ids:
            continue
        slide_paths = _as_list(row.get("pathology_wsi_paths"))
        row_file_ids = _as_list(row.get("pathology_file_ids"))
        if not slide_paths or not row_file_ids:
            continue

        cases_seen += 1
        if max_cases is not None and cases_seen > max_cases:
            break

        patient_id = str(row.get("patient_id", "")).strip()
        case_dir = output_root / _safe_name(patient_id or str(row.get("sample_id", "")), default=f"row-{row_idx}")
        for slide_path in slide_paths:
            slide_stem = Path(slide_path).stem
            if not _slide_kind_allowed(slide_stem, allowed_slide_kinds):
                continue
            if not overwrite and _slide_outputs_ready(case_dir, slide_stem):
                continue
            file_ids.update(row_file_ids)
    return sorted(file_ids)


def _download_temp_slide_path(staging_root: Path, job: PathologyPngJob, job_index: int) -> Path:
    safe_file_id = _safe_name(job.file_id, default="unknown-file")
    file_name = Path(job.slide_path).name.strip() or f"{job.slide_stem}.svs"
    return staging_root / f"{job_index:06d}-{safe_file_id}-{file_name}"


def _optional_positive_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def main() -> None:
    cfg = load_cfg()
    png_cfg = cfg.pathology_png
    registry_path = _resolve_path(png_cfg.registry_path)
    output_root = _resolve_path(png_cfg.output_root)
    staging_root = _resolve_path(png_cfg.staging_root)
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        raise RuntimeError(f"Unified registry is empty: {registry_path}")
    if THUMBNAIL_COLUMN not in registry_df.columns:
        registry_df[THUMBNAIL_COLUMN] = [[] for _ in range(len(registry_df))]

    allowed_slide_kinds = {str(value).strip().upper() for value in list(png_cfg.allowed_slide_kinds or []) if str(value).strip()}
    project_ids = {str(value).strip() for value in list(png_cfg.project_ids or []) if str(value).strip()}
    source_filter = str(png_cfg.get("source_filter", "tcga")).strip().lower()
    max_cases = _optional_positive_int(png_cfg.get("max_cases"))
    max_slides = _optional_positive_int(png_cfg.get("max_slides"))
    overwrite = bool(png_cfg.overwrite)

    candidate_file_ids = _collect_candidate_file_ids(
        registry_df,
        output_root=output_root,
        allowed_slide_kinds=allowed_slide_kinds,
        source_filter=source_filter,
        project_ids=project_ids,
        max_cases=max_cases,
        overwrite=overwrite,
    )
    gdc_client = GDCClient(
        base_url=str(png_cfg.download.base_url),
        timeout_seconds=int(png_cfg.download.timeout_seconds),
        page_size=int(png_cfg.download.page_size),
        max_retries=int(png_cfg.download.max_retries),
        retry_backoff_seconds=float(png_cfg.download.retry_backoff_seconds),
    )
    file_name_by_id = _fetch_file_name_by_id(
        gdc_client,
        candidate_file_ids,
        chunk_size=int(png_cfg.download.file_metadata_chunk_size),
    )
    jobs, ready_count = _build_jobs(
        registry_df,
        output_root=output_root,
        file_name_by_id=file_name_by_id,
        allowed_slide_kinds=allowed_slide_kinds,
        source_filter=source_filter,
        project_ids=project_ids,
        max_cases=max_cases,
        max_slides=max_slides,
        overwrite=overwrite,
    )

    print("Pathology thumbnail PNG extraction")
    print(f"Registry: {registry_path}")
    print(f"Output root: {output_root}")
    print(f"Staging root: {staging_root}")
    print(f"Allowed slide kinds: {sorted(allowed_slide_kinds) if allowed_slide_kinds else ['ALL']}")
    print(f"Already ready slides registered: {ready_count}")
    print(f"Slides to download/process: {len(jobs)}")

    if ready_count > 0 and not jobs:
        write_registry_parquet(registry_df, registry_path, validate=False)
        print("No new slides required processing; registry refreshed with existing PNG paths.")
        return
    if not jobs:
        print("No pathology slides require PNG extraction.")
        return

    write_every = max(1, int(png_cfg.write_registry_every))
    processed = 0
    failed = 0
    slide_loop = tqdm(jobs, desc="Pathology thumbnail slides", unit="slide")
    for job_index, job in enumerate(slide_loop, start=1):
        slide_loop.set_postfix(processed=processed, failed=failed)
        temp_slide_path = _download_temp_slide_path(staging_root, job, job_index)
        try:
            if temp_slide_path.exists():
                temp_slide_path.unlink()
            gdc_client.download_data_file(file_id=job.file_id, output_path=temp_slide_path, skip_existing=False)
            thumbnail_paths = _extract_slide_thumbnail(
                slide_path=temp_slide_path,
                output_dir=job.output_dir,
                slide_stem=job.slide_stem,
                thumbnail_max_size_px=int(png_cfg.thumbnail.max_size_px),
                overwrite=overwrite,
            )
            _register_slide_thumbnail(
                registry_df,
                row_idx=job.row_idx,
                slide_stem=job.slide_stem,
                thumbnail_paths=thumbnail_paths,
            )
            processed += 1
            if processed % write_every == 0:
                write_registry_parquet(registry_df, registry_path, validate=False)
        except Exception as exc:
            failed += 1
            print(f"Failed pathology PNG extraction for {job.slide_stem} ({job.file_id}): {exc}", file=sys.stderr)
        finally:
            if bool(png_cfg.delete_downloaded_svs) and temp_slide_path.exists():
                temp_slide_path.unlink()

    write_registry_parquet(registry_df, registry_path, validate=False)
    print(f"Pathology thumbnail slides processed: {processed}")
    print(f"Pathology thumbnail slides failed: {failed}")
    print(f"Updated registry column: {THUMBNAIL_COLUMN}")


if __name__ == "__main__":
    main()
