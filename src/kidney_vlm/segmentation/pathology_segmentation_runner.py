from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

import openslide

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.sources.tcga import GDCClient
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


@dataclass(frozen=True)
class PendingPathologySegmentationCandidate:
    row_idx: Any
    sample_id: str
    source: str
    project_id: str
    slide_path: str
    slide_stem: str
    pathology_file_ids: tuple[str, ...]
    local_slide_path: str | None


@dataclass(frozen=True)
class PendingPathologySegmentationJob:
    row_idx: Any
    sample_id: str
    source: str
    project_id: str
    slide_path: str
    slide_stem: str
    local_slide_path: str | None
    file_id: str | None


@dataclass(frozen=True)
class PrefetchedRemotePathologySlideDownload:
    job: PendingPathologySegmentationJob
    temp_slide_path: Path
    future: Future[Path]


@dataclass(frozen=True)
class PathologySegmentationDiscoveryStats:
    cases_scanned: int
    selected_slide_refs: int
    local_candidates: int
    remote_candidates: int
    already_segmented: int
    unsupported_source: int
    missing_file_ids: int
    todo_candidates: int


def load_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
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
    allowed = {str(item).strip().upper() for item in _as_list(value) if str(item).strip()}
    return {item for item in allowed if item}


def _filter_wsi_paths_by_allowed_kinds(paths: list[str], allowed_slide_kinds: set[str]) -> list[str]:
    if not allowed_slide_kinds:
        return paths
    return [path for path in paths if _slide_kind(Path(path).stem) in allowed_slide_kinds]


def _resolve_registry_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_optional_path(path_value: Any, *, default: Path) -> Path:
    text = str(path_value or "").strip()
    path = Path(text).expanduser() if text else default
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_python_executable(path_value: Any) -> str:
    text = str(path_value or "").strip()
    if not text:
        return str((ROOT / ".venv" / "bin" / "python").resolve())
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = (ROOT / path).resolve()
    if candidate.exists():
        return str(candidate)
    return text


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
    fallback_cfg = OmegaConf.load(tcga_cfg_path)
    fallback_gdc_cfg = OmegaConf.select(fallback_cfg, "data.source.tcga.gdc")
    if fallback_gdc_cfg is None:
        raise KeyError(f"Missing data.source.tcga.gdc in {tcga_cfg_path}")
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
    step = max(1, int(chunk_size))
    for start in range(0, len(normalized_ids), step):
        chunk_ids = normalized_ids[start : start + step]
        hits = gdc_client.fetch_files_by_ids(chunk_ids, fields=["file_id", "file_name"])
        for hit in hits:
            file_id = str(hit.get("file_id", "")).strip()
            file_name = str(hit.get("file_name", "")).strip()
            if file_id and file_name:
                file_name_by_id[file_id] = file_name
    return file_name_by_id


def _match_candidate_to_file_id(
    candidate: PendingPathologySegmentationCandidate,
    file_name_by_id: dict[str, str],
) -> str:
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


def _mask_output_path(mask_dir: Path, slide_stem: str) -> Path:
    return mask_dir / f"{slide_stem}.png"


def _slide_image_output_path(slide_image_dir: Path, slide_stem: str) -> Path:
    return slide_image_dir / f"{slide_stem}.png"


def _overlay_output_path(overlay_dir: Path, slide_stem: str) -> Path:
    return overlay_dir / f"{slide_stem}.png"


def _metadata_output_path(metadata_dir: Path, slide_stem: str) -> Path:
    return metadata_dir / f"{slide_stem}.json"


def _is_valid_png(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as handle:
            handle.verify()
        return True
    except Exception:
        return False


def _outputs_ready(
    slide_stem: str,
    *,
    mask_dir: Path,
    slide_image_dir: Path,
    overlay_dir: Path,
    metadata_dir: Path,
    save_slide_image: bool,
    save_overlay_image: bool,
    save_metadata_json: bool,
) -> bool:
    if not _is_valid_png(_mask_output_path(mask_dir, slide_stem)):
        return False
    if save_slide_image and not _is_valid_png(_slide_image_output_path(slide_image_dir, slide_stem)):
        return False
    if save_overlay_image and not _is_valid_png(_overlay_output_path(overlay_dir, slide_stem)):
        return False
    if save_metadata_json:
        metadata_path = _metadata_output_path(metadata_dir, slide_stem)
        if not metadata_path.exists() or metadata_path.stat().st_size <= 0:
            return False
    return True


def _build_segmentation_candidates(
    registry_df: pd.DataFrame,
    *,
    mask_dir: Path,
    slide_image_dir: Path,
    overlay_dir: Path,
    metadata_dir: Path,
    overwrite_existing: bool,
    save_slide_image: bool,
    save_overlay_image: bool,
    save_metadata_json: bool,
    allowed_slide_kinds: set[str],
) -> tuple[list[PendingPathologySegmentationCandidate], PathologySegmentationDiscoveryStats]:
    candidates: list[PendingPathologySegmentationCandidate] = []
    selected_slide_refs = 0
    local_candidates = 0
    remote_candidates = 0
    already_segmented = 0
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
            slide_stem = Path(normalized_wsi_path).stem
            if (
                not overwrite_existing
                and _outputs_ready(
                    slide_stem,
                    mask_dir=mask_dir,
                    slide_image_dir=slide_image_dir,
                    overlay_dir=overlay_dir,
                    metadata_dir=metadata_dir,
                    save_slide_image=save_slide_image,
                    save_overlay_image=save_overlay_image,
                    save_metadata_json=save_metadata_json,
                )
            ):
                already_segmented += 1
                continue

            local_slide_path = normalized_wsi_path if Path(normalized_wsi_path).exists() else None
            pathology_file_ids = tuple(_as_list(row.get("pathology_file_ids")))
            if local_slide_path is None:
                source = str(row.get("source", "")).strip().lower()
                if source != "tcga":
                    unsupported_source += 1
                    continue
                if not pathology_file_ids:
                    missing_file_ids += 1
                    continue
                remote_candidates += 1
            else:
                local_candidates += 1

            candidates.append(
                PendingPathologySegmentationCandidate(
                    row_idx=row_idx,
                    sample_id=str(row.get("sample_id", "")).strip(),
                    source=str(row.get("source", "")).strip(),
                    project_id=str(row.get("project_id", "")).strip(),
                    slide_path=normalized_wsi_path,
                    slide_stem=slide_stem,
                    pathology_file_ids=pathology_file_ids,
                    local_slide_path=local_slide_path,
                )
            )

    stats = PathologySegmentationDiscoveryStats(
        cases_scanned=len(registry_df),
        selected_slide_refs=selected_slide_refs,
        local_candidates=local_candidates,
        remote_candidates=remote_candidates,
        already_segmented=already_segmented,
        unsupported_source=unsupported_source,
        missing_file_ids=missing_file_ids,
        todo_candidates=len(candidates),
    )
    return candidates, stats


def _sort_and_limit_candidates(
    candidates: list[PendingPathologySegmentationCandidate],
    *,
    top_k_slides: int | None,
) -> list[PendingPathologySegmentationCandidate]:
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            str(item.project_id).upper(),
            str(item.sample_id).upper(),
            str(item.slide_stem).upper(),
        ),
    )
    if top_k_slides is None or int(top_k_slides) <= 0:
        return sorted_candidates
    return sorted_candidates[: int(top_k_slides)]


def _build_segmentation_jobs(
    candidates: list[PendingPathologySegmentationCandidate],
    *,
    file_name_by_id: dict[str, str],
) -> tuple[list[PendingPathologySegmentationJob], int]:
    jobs: list[PendingPathologySegmentationJob] = []
    unresolved = 0
    for candidate in candidates:
        file_id = None
        if candidate.local_slide_path is None:
            file_id = _match_candidate_to_file_id(candidate, file_name_by_id)
            if not file_id:
                unresolved += 1
                continue
        jobs.append(
            PendingPathologySegmentationJob(
                row_idx=candidate.row_idx,
                sample_id=candidate.sample_id,
                source=candidate.source,
                project_id=candidate.project_id,
                slide_path=candidate.slide_path,
                slide_stem=candidate.slide_stem,
                local_slide_path=candidate.local_slide_path,
                file_id=file_id,
            )
        )
    return jobs, unresolved


def _build_temp_slide_download_path(temp_dir: Path, *, job: PendingPathologySegmentationJob, job_index: int) -> Path:
    file_name = Path(job.slide_path).name.strip() or f"{job.slide_stem}.svs"
    safe_file_id = re.sub(r"[^A-Za-z0-9._-]+", "-", str(job.file_id).strip()) or "unknown-file"
    return temp_dir / f"{job_index:05d}-{safe_file_id}-{file_name}"


def _submit_prefetch_download(
    *,
    executor: ThreadPoolExecutor,
    gdc_client: GDCClient,
    job: PendingPathologySegmentationJob,
    temp_dir: Path,
    job_index: int,
) -> PrefetchedRemotePathologySlideDownload:
    temp_slide_path = _build_temp_slide_download_path(temp_dir, job=job, job_index=job_index)
    if temp_slide_path.exists():
        temp_slide_path.unlink()

    future = executor.submit(
        gdc_client.download_data_file,
        file_id=str(job.file_id),
        output_path=temp_slide_path,
        skip_existing=False,
    )
    return PrefetchedRemotePathologySlideDownload(
        job=job,
        temp_slide_path=temp_slide_path,
        future=future,
    )


def _save_png(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def _load_slide_rgb_at_magnification(
    slide_path: Path,
    *,
    target_magnification: float | None,
    max_rendered_slide_side: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        full_width, full_height = slide.dimensions
        objective_power = _resolve_objective_power(slide)
        requested_mag = float(target_magnification) if target_magnification not in (None, "", "null") else None

        scaled_width = full_width
        scaled_height = full_height
        base_scale = None
        if objective_power is not None and requested_mag is not None and objective_power > 0 and requested_mag > 0:
            base_scale = min(1.0, requested_mag / objective_power)
            scaled_width = max(1, int(round(full_width * base_scale)))
            scaled_height = max(1, int(round(full_height * base_scale)))

        final_width = scaled_width
        final_height = scaled_height
        if max_rendered_slide_side > 0:
            longest_side = max(final_width, final_height)
            if longest_side > max_rendered_slide_side:
                cap_scale = max_rendered_slide_side / float(longest_side)
                final_width = max(1, int(round(final_width * cap_scale)))
                final_height = max(1, int(round(final_height * cap_scale)))

        thumbnail = slide.get_thumbnail((final_width, final_height)).convert("RGB")
        rgb = np.asarray(thumbnail)
        metadata = {
            "full_width_px": int(full_width),
            "full_height_px": int(full_height),
            "objective_power": objective_power,
            "target_magnification": requested_mag,
            "base_scale_from_magnification": base_scale,
            "render_width_px": int(rgb.shape[1]),
            "render_height_px": int(rgb.shape[0]),
            "max_rendered_slide_side": int(max_rendered_slide_side),
        }
        return rgb, metadata
    finally:
        slide.close()


def _run_sampath_predict_on_rgb_image(
    *,
    rgb_image: np.ndarray,
    slide_stem: str,
    sampath_root: Path,
    sampath_python_executable: str,
    sampath_config_module: str,
    sampath_pretrained_path: Path,
    sampath_devices: str,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="kidney-vlm-sampath-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_image_path = input_dir / f"{slide_stem}.png"
        _save_png(input_image_path, rgb_image)

        command = [
            str(sampath_python_executable),
            str(sampath_root / "predict.py"),
            "--config",
            str(sampath_config_module),
            "--pretrained",
            str(sampath_pretrained_path),
            "--input_dir",
            str(input_dir),
            "--data_ext",
            ".png",
            "--output_dir",
            str(output_dir),
            "--devices",
            str(sampath_devices).strip() or "0",
        ]
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = f"{sampath_root}{os.pathsep}{pythonpath}" if pythonpath else str(sampath_root)
        subprocess.run(
            command,
            cwd=str(sampath_root),
            env=env,
            check=True,
        )

        predicted_mask_path = output_dir / f"{slide_stem}_mask.png"
        predicted_mask = np.asarray(Image.open(predicted_mask_path))
        if predicted_mask.ndim == 3:
            predicted_mask = predicted_mask[:, :, 0]
        if predicted_mask.shape != rgb_image.shape[:2]:
            predicted_mask = cv2.resize(
                predicted_mask.astype(np.uint8),
                (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return predicted_mask.astype(np.uint8)


def _resolve_objective_power(slide: openslide.OpenSlide) -> float | None:
    property_keys = [
        getattr(openslide, "PROPERTY_NAME_OBJECTIVE_POWER", ""),
        "aperio.AppMag",
        "openslide.objective-power",
    ]
    for key in property_keys:
        if not key:
            continue
        raw_value = str(slide.properties.get(key, "")).strip()
        if not raw_value:
            continue
        try:
            return float(raw_value)
        except ValueError:
            continue
    return None


def _segment_thumbnail_rgb(
    rgb_image: np.ndarray,
    *,
    gaussian_blur_kernel_size: int,
    saturation_threshold: int,
    background_intensity_threshold: int,
    morphology_kernel_size: int,
    min_component_area_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {tuple(rgb_image.shape)}")

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    blur_size = max(1, int(gaussian_blur_kernel_size))
    if blur_size % 2 == 0:
        blur_size += 1
    blurred_gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0) if blur_size > 1 else gray
    otsu_threshold, inverted_otsu_mask = cv2.threshold(
        255 - blurred_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    non_background = gray < int(background_intensity_threshold)
    colorful = saturation >= int(saturation_threshold)
    combined_mask = np.logical_and(non_background, np.logical_or(inverted_otsu_mask > 0, colorful))
    combined_u8 = (combined_mask.astype(np.uint8) * 255)

    morph_size = max(1, int(morphology_kernel_size))
    if morph_size % 2 == 0:
        morph_size += 1
    if morph_size > 1:
        kernel = np.ones((morph_size, morph_size), dtype=np.uint8)
        combined_u8 = cv2.morphologyEx(combined_u8, cv2.MORPH_OPEN, kernel)
        combined_u8 = cv2.morphologyEx(combined_u8, cv2.MORPH_CLOSE, kernel)

    component_input = (combined_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(component_input, connectivity=8)
    filtered_mask = np.zeros_like(combined_u8)
    min_area = max(1, int(min_component_area_px))
    kept_components = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        filtered_mask[labels == label_idx] = 255
        kept_components += 1

    tissue_fraction = float(np.count_nonzero(filtered_mask)) / float(filtered_mask.size or 1)
    stats_dict = {
        "tissue_fraction": tissue_fraction,
        "kept_components": kept_components,
        "otsu_threshold": float(otsu_threshold),
    }
    return filtered_mask, stats_dict


def _normalize_color_palette(value: Any) -> list[np.ndarray]:
    default_palette = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([255.0, 70.0, 70.0], dtype=np.float32),
        np.array([70.0, 160.0, 255.0], dtype=np.float32),
        np.array([80.0, 200.0, 120.0], dtype=np.float32),
        np.array([255.0, 200.0, 70.0], dtype=np.float32),
        np.array([180.0, 90.0, 255.0], dtype=np.float32),
    ]
    if value is None:
        return default_palette
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, list):
        return default_palette

    palette: list[np.ndarray] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            palette.append(np.array([float(item[0]), float(item[1]), float(item[2])], dtype=np.float32))
    return palette or default_palette


def _build_overlay_rgb(
    rgb_image: np.ndarray,
    mask_image: np.ndarray,
    *,
    class_palette_rgb: list[np.ndarray],
    overlay_alpha: float,
) -> np.ndarray:
    overlay = rgb_image.astype(np.float32).copy()
    alpha = float(max(0.0, min(1.0, overlay_alpha)))
    for label_id in sorted(int(label) for label in np.unique(mask_image)):
        if label_id <= 0:
            continue
        class_mask = mask_image == label_id
        if not np.any(class_mask):
            continue
        color = class_palette_rgb[min(label_id, len(class_palette_rgb) - 1)].astype(np.float32)
        overlay[class_mask] = ((1.0 - alpha) * overlay[class_mask]) + (alpha * color)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _show_segmentation_plot(
    *,
    slide_stem: str,
    rgb_image: np.ndarray,
    mask_image: np.ndarray,
    overlay_image: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(rgb_image)
    axes[0].set_title(f"{slide_stem} slide")
    axes[1].imshow(mask_image, cmap="gray")
    axes[1].set_title("Mask")
    axes[2].imshow(overlay_image)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def _rebuild_case_pathology_mask_paths(
    row: pd.Series,
    *,
    mask_dir: Path,
    allowed_slide_kinds: set[str],
) -> list[str]:
    return _rebuild_case_pathology_output_paths(
        row,
        allowed_slide_kinds=allowed_slide_kinds,
        output_path_builder=lambda slide_stem: _mask_output_path(mask_dir, slide_stem),
        path_is_ready=_is_valid_png,
    )


def _rebuild_case_pathology_slide_image_paths(
    row: pd.Series,
    *,
    slide_image_dir: Path,
    allowed_slide_kinds: set[str],
) -> list[str]:
    return _rebuild_case_pathology_output_paths(
        row,
        allowed_slide_kinds=allowed_slide_kinds,
        output_path_builder=lambda slide_stem: _slide_image_output_path(slide_image_dir, slide_stem),
        path_is_ready=_is_valid_png,
    )


def _rebuild_case_pathology_overlay_paths(
    row: pd.Series,
    *,
    overlay_dir: Path,
    allowed_slide_kinds: set[str],
) -> list[str]:
    return _rebuild_case_pathology_output_paths(
        row,
        allowed_slide_kinds=allowed_slide_kinds,
        output_path_builder=lambda slide_stem: _overlay_output_path(overlay_dir, slide_stem),
        path_is_ready=_is_valid_png,
    )


def _rebuild_case_pathology_metadata_paths(
    row: pd.Series,
    *,
    metadata_dir: Path,
    allowed_slide_kinds: set[str],
) -> list[str]:
    return _rebuild_case_pathology_output_paths(
        row,
        allowed_slide_kinds=allowed_slide_kinds,
        output_path_builder=lambda slide_stem: _metadata_output_path(metadata_dir, slide_stem),
        path_is_ready=lambda path: path.exists() and path.stat().st_size > 0,
    )


def _rebuild_case_pathology_output_paths(
    row: pd.Series,
    *,
    allowed_slide_kinds: set[str],
    output_path_builder: Any,
    path_is_ready: Any,
) -> list[str]:
    output_paths: list[str] = []
    seen: set[str] = set()
    for normalized_wsi_path in _filter_wsi_paths_by_allowed_kinds(
        _local_wsi_paths(row.get("pathology_wsi_paths")),
        allowed_slide_kinds,
    ):
        slide_stem = Path(normalized_wsi_path).stem
        output_path = output_path_builder(slide_stem)
        if not path_is_ready(output_path):
            continue
        relative_path = _to_registry_relative_path(output_path)
        if relative_path in seen:
            continue
        seen.add(relative_path)
        output_paths.append(relative_path)
    return output_paths


def _process_segmentation_job(
    *,
    job: PendingPathologySegmentationJob,
    slide_path: Path,
    registry_df: pd.DataFrame,
    registry_path: Path,
    mask_dir: Path,
    slide_image_dir: Path,
    overlay_dir: Path,
    metadata_dir: Path,
    sampath_root: Path,
    sampath_python_executable: str,
    sampath_config_module: str,
    sampath_pretrained_path: Path,
    sampath_devices: str,
    target_magnification: float | None,
    max_rendered_slide_side: int,
    class_palette_rgb: list[np.ndarray],
    overlay_alpha: float,
    save_slide_image: bool,
    save_overlay_image: bool,
    save_metadata_json: bool,
    show_plots: bool,
    should_show_plot: bool,
    allowed_slide_kinds: set[str],
    overwrite_existing: bool,
) -> dict[str, Any]:
    mask_path = _mask_output_path(mask_dir, job.slide_stem)
    slide_image_path = _slide_image_output_path(slide_image_dir, job.slide_stem)
    overlay_path = _overlay_output_path(overlay_dir, job.slide_stem)
    metadata_path = _metadata_output_path(metadata_dir, job.slide_stem)

    if overwrite_existing:
        for output_path in (mask_path, slide_image_path, overlay_path, metadata_path):
            if output_path.exists():
                output_path.unlink()

    rgb_image, render_metadata = _load_slide_rgb_at_magnification(
        slide_path,
        target_magnification=target_magnification,
        max_rendered_slide_side=max_rendered_slide_side,
    )
    mask_image = _run_sampath_predict_on_rgb_image(
        rgb_image=rgb_image,
        slide_stem=job.slide_stem,
        sampath_root=sampath_root,
        sampath_python_executable=sampath_python_executable,
        sampath_config_module=sampath_config_module,
        sampath_pretrained_path=sampath_pretrained_path,
        sampath_devices=sampath_devices,
    )
    unique_labels, label_counts = np.unique(mask_image, return_counts=True)
    foreground_fraction = float(np.count_nonzero(mask_image > 0)) / float(mask_image.size or 1)
    segmentation_stats = {
        "foreground_fraction": foreground_fraction,
        "labels_present": [int(label) for label in unique_labels.tolist()],
        "label_pixel_counts": {str(int(label)): int(count) for label, count in zip(unique_labels.tolist(), label_counts.tolist(), strict=True)},
    }
    overlay_image = _build_overlay_rgb(
        rgb_image,
        mask_image,
        class_palette_rgb=class_palette_rgb,
        overlay_alpha=overlay_alpha,
    )

    _save_png(mask_path, mask_image)
    if save_slide_image:
        _save_png(slide_image_path, rgb_image)
    if save_overlay_image:
        _save_png(overlay_path, overlay_image)

    combined_metadata = {
        "sample_id": job.sample_id,
        "project_id": job.project_id,
        "source": job.source,
        "slide_stem": job.slide_stem,
        "slide_path": job.slide_path,
        "render": render_metadata,
        "segmentation": segmentation_stats,
        "class_palette_rgb": [[int(round(channel)) for channel in color.tolist()] for color in class_palette_rgb],
        "overlay_alpha": float(overlay_alpha),
        "mask_path": _to_registry_relative_path(mask_path),
        "slide_image_path": _to_registry_relative_path(slide_image_path) if save_slide_image else "",
        "overlay_path": _to_registry_relative_path(overlay_path) if save_overlay_image else "",
    }
    if save_metadata_json:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(combined_metadata, indent=2))

    if show_plots and should_show_plot:
        _show_segmentation_plot(
            slide_stem=job.slide_stem,
            rgb_image=rgb_image,
            mask_image=mask_image,
            overlay_image=overlay_image,
        )

    registry_df.at[job.row_idx, "pathology_mask_paths"] = _rebuild_case_pathology_mask_paths(
        registry_df.loc[job.row_idx],
        mask_dir=mask_dir,
        allowed_slide_kinds=allowed_slide_kinds,
    )
    registry_df.at[job.row_idx, "pathology_segmentation_slide_image_paths"] = _rebuild_case_pathology_slide_image_paths(
        registry_df.loc[job.row_idx],
        slide_image_dir=slide_image_dir,
        allowed_slide_kinds=allowed_slide_kinds,
    )
    registry_df.at[job.row_idx, "pathology_segmentation_overlay_paths"] = _rebuild_case_pathology_overlay_paths(
        registry_df.loc[job.row_idx],
        overlay_dir=overlay_dir,
        allowed_slide_kinds=allowed_slide_kinds,
    )
    registry_df.at[job.row_idx, "pathology_segmentation_metadata_paths"] = _rebuild_case_pathology_metadata_paths(
        registry_df.loc[job.row_idx],
        metadata_dir=metadata_dir,
        allowed_slide_kinds=allowed_slide_kinds,
    )
    write_registry_parquet(registry_df, registry_path, validate=False)

    return {
        "mask_path": mask_path,
        "foreground_fraction": float(segmentation_stats["foreground_fraction"]),
        "labels_present": list(segmentation_stats["labels_present"]),
        "render_width_px": int(render_metadata["render_width_px"]),
        "render_height_px": int(render_metadata["render_height_px"]),
    }


def _print_metrics_line(label: str, *metrics: tuple[str, Any]) -> None:
    parts = [label]
    for name, value in metrics:
        if isinstance(value, float):
            parts.append(f"{name}={value:.4f}")
        else:
            parts.append(f"{name}={value}")
    print("\t".join(parts))


def main() -> None:
    cfg = load_cfg()
    pathology_cfg = cfg.embeding_extraction.segmentation.pathology

    registry_path = _resolve_registry_path(pathology_cfg.get("registry_path", cfg.data.unified_registry_path))
    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        print(f"Registry is empty at '{registry_path}'. Nothing to segment.")
        return

    allowed_project_ids = [str(value).strip() for value in list(pathology_cfg.get("allowed_project_ids", []) or []) if str(value).strip()]
    if allowed_project_ids and "project_id" in registry_df.columns:
        registry_df = registry_df[registry_df["project_id"].astype(str).isin(allowed_project_ids)].copy()
    if registry_df.empty:
        print("No registry rows remain after filtering allowed_project_ids.")
        return

    allowed_slide_kinds = _resolve_allowed_slide_kinds(pathology_cfg.get("allowed_slide_kinds"))
    output_root = _resolve_optional_path(
        pathology_cfg.get("output_root"),
        default=ROOT / "data" / "pathology_segmentation",
    )
    mask_dir = _resolve_optional_path(
        pathology_cfg.get("mask_dir"),
        default=output_root / "masks",
    )
    slide_image_dir = _resolve_optional_path(
        pathology_cfg.get("slide_image_dir"),
        default=output_root / "slide_images",
    )
    overlay_dir = _resolve_optional_path(
        pathology_cfg.get("overlay_dir"),
        default=output_root / "overlays",
    )
    metadata_dir = _resolve_optional_path(
        pathology_cfg.get("metadata_dir"),
        default=output_root / "metadata",
    )
    for directory in (output_root, mask_dir, slide_image_dir, overlay_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resumable = bool(pathology_cfg.get("resumable", True))
    overwrite_existing = not resumable
    skip_errors = bool(pathology_cfg.get("skip_errors", True))
    save_slide_image = bool(pathology_cfg.get("save_slide_image", True))
    save_overlay_image = bool(pathology_cfg.get("save_overlay_image", True))
    save_metadata_json = bool(pathology_cfg.get("save_metadata_json", True))
    show_plots = bool(pathology_cfg.get("show_plots", False))
    max_plots_to_show = max(0, int(pathology_cfg.get("max_plots_to_show", 0) or 0))
    top_k_cfg = pathology_cfg.get("top_k_slides")
    top_k_slides = None if top_k_cfg in (None, "", "null") else int(top_k_cfg)
    sampath_root = _resolve_optional_path(
        pathology_cfg.get("sampath_root"),
        default=ROOT / "external" / "SAMPath",
    )
    sampath_python_executable = _resolve_python_executable(pathology_cfg.get("sampath_python_executable"))
    sampath_config_module = str(pathology_cfg.get("sampath_config_module", "configs.BCSS")).strip() or "configs.BCSS"
    sampath_pretrained_path = _resolve_optional_path(
        pathology_cfg.get("sampath_pretrained_path"),
        default=sampath_root / "weights" / "model.ckpt",
    )
    sampath_devices = str(pathology_cfg.get("sampath_devices", "0")).strip() or "0"
    target_magnification_cfg = pathology_cfg.get("target_magnification")
    target_magnification = None if target_magnification_cfg in (None, "", "null") else float(target_magnification_cfg)
    max_rendered_slide_side = int(pathology_cfg.get("max_rendered_slide_side", 2048))
    class_palette_rgb = _normalize_color_palette(pathology_cfg.get("class_palette_rgb"))
    overlay_alpha = float(pathology_cfg.get("overlay_alpha", 0.45))
    file_metadata_chunk_size = int(pathology_cfg.get("file_metadata_chunk_size", 200))
    temp_download_root = _resolve_optional_path(
        pathology_cfg.get("temp_download_root"),
        default=ROOT / "data" / "staging" / "pathology_segmentation_downloads",
    )
    temp_download_root.mkdir(parents=True, exist_ok=True)

    candidates, stats = _build_segmentation_candidates(
        registry_df,
        mask_dir=mask_dir,
        slide_image_dir=slide_image_dir,
        overlay_dir=overlay_dir,
        metadata_dir=metadata_dir,
        overwrite_existing=overwrite_existing,
        save_slide_image=save_slide_image,
        save_overlay_image=save_overlay_image,
        save_metadata_json=save_metadata_json,
        allowed_slide_kinds=allowed_slide_kinds,
    )
    candidates = _sort_and_limit_candidates(candidates, top_k_slides=top_k_slides)

    print("Pathology segmentation runner")
    _print_metrics_line(
        "Config",
        ("name", str(pathology_cfg.get("name", "pathology_segmentation"))),
        ("method_name", str(pathology_cfg.get("method_name", "sampath"))),
        ("sampath_config", sampath_config_module),
        ("allowed_slide_kinds", ",".join(sorted(allowed_slide_kinds)) if allowed_slide_kinds else "all"),
        ("target_magnification", target_magnification if target_magnification is not None else "auto"),
        ("max_rendered_slide_side", max_rendered_slide_side),
        ("top_k_slides", top_k_slides if top_k_slides is not None else "all"),
        ("resumable", resumable),
        ("save_outputs", "mask+slide+overlay"),
    )
    _print_metrics_line(
        "Discovery",
        ("cases_scanned", stats.cases_scanned),
        ("selected_slide_refs", stats.selected_slide_refs),
        ("local_candidates", stats.local_candidates),
        ("remote_candidates", stats.remote_candidates),
        ("already_segmented", stats.already_segmented),
        ("unsupported_source", stats.unsupported_source),
        ("missing_file_ids", stats.missing_file_ids),
        ("todo_candidates", stats.todo_candidates),
        ("selected_for_run", len(candidates)),
    )

    if not candidates:
        print("No pathology slides require segmentation.")
        return

    remote_candidates = [candidate for candidate in candidates if candidate.local_slide_path is None]
    file_name_by_id: dict[str, str] = {}
    if remote_candidates:
        gdc_client = _build_gdc_client(_resolve_tcga_gdc_cfg(cfg))
        unique_file_ids = sorted({file_id for candidate in remote_candidates for file_id in candidate.pathology_file_ids})
        file_name_by_id = _fetch_file_name_by_id(gdc_client, unique_file_ids, chunk_size=file_metadata_chunk_size)
    else:
        gdc_client = None

    jobs, unresolved_jobs = _build_segmentation_jobs(candidates, file_name_by_id=file_name_by_id)
    if unresolved_jobs:
        print(f"Unresolved remote pathology segmentation jobs after file-name matching: {unresolved_jobs}")
    if not jobs:
        print("No pathology segmentation jobs remain after remote file matching.")
        return

    downloaded = 0
    segmented = 0
    failed = 0
    registry_rows_written = 0
    shown_plots = 0

    local_jobs = [job for job in jobs if job.local_slide_path is not None]
    remote_jobs = [job for job in jobs if job.local_slide_path is None]
    job_loop = tqdm(local_jobs, total=len(local_jobs), desc="Segmenting local pathology slides")
    for job in job_loop:
        slide_total_start = time.perf_counter()
        try:
            result = _process_segmentation_job(
                job=job,
                slide_path=Path(str(job.local_slide_path)),
                registry_df=registry_df,
                registry_path=registry_path,
                mask_dir=mask_dir,
                slide_image_dir=slide_image_dir,
                overlay_dir=overlay_dir,
                metadata_dir=metadata_dir,
                sampath_root=sampath_root,
                sampath_python_executable=sampath_python_executable,
                sampath_config_module=sampath_config_module,
                sampath_pretrained_path=sampath_pretrained_path,
                sampath_devices=sampath_devices,
                target_magnification=target_magnification,
                max_rendered_slide_side=max_rendered_slide_side,
                class_palette_rgb=class_palette_rgb,
                overlay_alpha=overlay_alpha,
                save_slide_image=save_slide_image,
                save_overlay_image=save_overlay_image,
                save_metadata_json=save_metadata_json,
                show_plots=show_plots,
                should_show_plot=shown_plots < max_plots_to_show,
                allowed_slide_kinds=allowed_slide_kinds,
                overwrite_existing=overwrite_existing,
            )
            segmented += 1
            registry_rows_written += 1
            if show_plots and shown_plots < max_plots_to_show:
                shown_plots += 1
            _print_metrics_line(
                "[local_slide]",
                ("slide", job.slide_stem),
                ("foreground_fraction", result["foreground_fraction"]),
                ("labels_present", ",".join(str(label) for label in result["labels_present"])),
                ("render_width_px", result["render_width_px"]),
                ("render_height_px", result["render_height_px"]),
                ("total_s", time.perf_counter() - slide_total_start),
            )
        except Exception as exc:
            failed += 1
            job_loop.write(f"[error][local] {job.slide_stem}: {exc}")
            if not skip_errors:
                raise

    if remote_jobs and gdc_client is not None:
        with tempfile.TemporaryDirectory(dir=str(temp_download_root), prefix="kidney-vlm-pathology-segmentation-") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            remote_loop = tqdm(remote_jobs, total=len(remote_jobs), desc="Segmenting remote pathology slides")
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pathology-segmentation-download-prefetch") as download_executor:
                next_prefetch: PrefetchedRemotePathologySlideDownload | None = None
                for job_index, job in enumerate(remote_loop):
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
                    if job_index + 1 < len(remote_jobs):
                        next_prefetch = _submit_prefetch_download(
                            executor=download_executor,
                            gdc_client=gdc_client,
                            job=remote_jobs[job_index + 1],
                            temp_dir=temp_dir,
                            job_index=job_index + 1,
                        )

                    temp_slide_path = current_prefetch.temp_slide_path
                    try:
                        downloaded_path = Path(current_prefetch.future.result())
                        downloaded += 1
                        result = _process_segmentation_job(
                            job=job,
                            slide_path=downloaded_path,
                            registry_df=registry_df,
                            registry_path=registry_path,
                            mask_dir=mask_dir,
                            slide_image_dir=slide_image_dir,
                            overlay_dir=overlay_dir,
                            metadata_dir=metadata_dir,
                            sampath_root=sampath_root,
                            sampath_python_executable=sampath_python_executable,
                            sampath_config_module=sampath_config_module,
                            sampath_pretrained_path=sampath_pretrained_path,
                            sampath_devices=sampath_devices,
                            target_magnification=target_magnification,
                            max_rendered_slide_side=max_rendered_slide_side,
                            class_palette_rgb=class_palette_rgb,
                            overlay_alpha=overlay_alpha,
                            save_slide_image=save_slide_image,
                            save_overlay_image=save_overlay_image,
                            save_metadata_json=save_metadata_json,
                            show_plots=show_plots,
                            should_show_plot=shown_plots < max_plots_to_show,
                            allowed_slide_kinds=allowed_slide_kinds,
                            overwrite_existing=overwrite_existing,
                        )
                        segmented += 1
                        registry_rows_written += 1
                        if show_plots and shown_plots < max_plots_to_show:
                            shown_plots += 1
                        _print_metrics_line(
                            "[remote_slide]",
                            ("slide", job.slide_stem),
                            ("foreground_fraction", result["foreground_fraction"]),
                            ("labels_present", ",".join(str(label) for label in result["labels_present"])),
                            ("render_width_px", result["render_width_px"]),
                            ("render_height_px", result["render_height_px"]),
                            ("total_s", time.perf_counter() - slide_total_start),
                        )
                    except Exception as exc:
                        failed += 1
                        remote_loop.write(f"[error][remote] {job.slide_stem}: {exc}")
                        if not skip_errors:
                            raise
                    finally:
                        if temp_slide_path.exists():
                            temp_slide_path.unlink()

    print("Pathology segmentation complete.")
    print(f"Slides downloaded to temp storage this run: {downloaded}")
    print(f"Slides segmented this run: {segmented}")
    print(f"Registry rows updated this run: {registry_rows_written}")
    print(f"Failed slides this run: {failed}")
