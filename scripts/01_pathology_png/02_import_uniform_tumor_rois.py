#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

ROI_COLUMN = "pathology_png_roi_paths"
ROI_MARKER = "__uniform_tumor_8k__"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CLEAR_EXISTING_ROI_PATHS_BEFORE_REGISTER = True
FAIL_ON_UNMATCHED_ROIS = True

TCGA_SLIDE_KEY_PATTERN = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]-[0-9]{2}-[A-Z0-9]{3}[0-9]?)")


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="01_pathology_png/02_import_uniform_tumor_rois.yaml",
        overrides=sys.argv[1:],
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
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


def _extract_slide_key(text: Any) -> str:
    match = TCGA_SLIDE_KEY_PATTERN.search(str(text).upper())
    return match.group(1) if match else ""


def _to_project_relative_path(path_value: str | Path) -> str:
    path = Path(path_value).expanduser().resolve()
    return Path(os.path.relpath(path, start=ROOT)).as_posix()


def _build_registry_matches(registry_df: pd.DataFrame) -> dict[str, int]:
    slide_matches: dict[str, int] = {}
    for row_idx in registry_df.index.tolist():
        row = registry_df.loc[row_idx]
        for wsi_path in _as_list(row.get("pathology_wsi_paths")):
            slide_key = _extract_slide_key(Path(wsi_path).stem)
            if slide_key:
                slide_matches.setdefault(slide_key, int(row_idx))
    return slide_matches


def iter_local_roi_paths(roi_root: Path) -> list[Path]:
    if not roi_root.exists():
        raise FileNotFoundError(f"Pathology ROI root not found: {roi_root}")
    return sorted(
        path
        for path in roi_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and ROI_MARKER in path.name
    )


def register_local_roi_paths(registry_df: pd.DataFrame, roi_paths: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = registry_df.copy()
    if ROI_COLUMN not in out.columns:
        out[ROI_COLUMN] = [[] for _ in range(len(out))]
    if CLEAR_EXISTING_ROI_PATHS_BEFORE_REGISTER:
        out[ROI_COLUMN] = [[] for _ in range(len(out))]

    slide_matches = _build_registry_matches(out)
    registered = 0
    skipped_duplicate = 0
    missing_slide_key_examples: list[str] = []
    missing_match_examples: list[str] = []

    for roi_path in tqdm(roi_paths, desc="Registering local pathology ROI PNGs", unit="roi"):
        slide_key = _extract_slide_key(roi_path.name)
        if not slide_key:
            if len(missing_slide_key_examples) < 10:
                missing_slide_key_examples.append(_to_project_relative_path(roi_path))
            continue

        row_idx = slide_matches.get(slide_key)
        if row_idx is None:
            if len(missing_match_examples) < 10:
                missing_match_examples.append(_to_project_relative_path(roi_path))
            continue

        portable_path = _to_project_relative_path(roi_path)
        existing_paths = _as_list(out.at[row_idx, ROI_COLUMN])
        if portable_path in existing_paths:
            skipped_duplicate += 1
            continue
        out.at[row_idx, ROI_COLUMN] = existing_paths + [portable_path]
        registered += 1

    stats: dict[str, Any] = {
        "roi_files_seen": len(roi_paths),
        "registered_roi_paths": registered,
        "skipped_duplicate": skipped_duplicate,
        "skipped_missing_slide_key": len(missing_slide_key_examples),
        "skipped_missing_match": len(missing_match_examples),
        "missing_slide_key_examples": missing_slide_key_examples,
        "missing_match_examples": missing_match_examples,
    }
    return out, stats


def _raise_for_unmatched_rois(stats: dict[str, Any]) -> None:
    if not FAIL_ON_UNMATCHED_ROIS:
        return
    if int(stats["skipped_missing_slide_key"]) == 0 and int(stats["skipped_missing_match"]) == 0:
        return
    details: list[str] = []
    if stats["missing_slide_key_examples"]:
        details.append(f"missing slide key examples: {stats['missing_slide_key_examples']}")
    if stats["missing_match_examples"]:
        details.append(f"missing registry match examples: {stats['missing_match_examples']}")
    raise RuntimeError(
        "Some local UniformTumor ROI files could not be registered. "
        "The ROI folder should already be complete and match the unified registry. "
        + "; ".join(details)
    )


def main() -> None:
    cfg = load_cfg()
    png_cfg = cfg.pathology_png
    registry_path = _resolve_path(png_cfg.registry_path)
    roi_root = _resolve_path(png_cfg.roi_root)

    if not registry_path.exists():
        raise FileNotFoundError(f"Unified registry not found: {registry_path}")
    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        raise RuntimeError(f"Unified registry is empty: {registry_path}")

    roi_paths = iter_local_roi_paths(roi_root)
    if not roi_paths:
        raise RuntimeError(f"No local UniformTumor ROI images found under: {roi_root}")

    print("Registering local UniformTumor ROI PNGs")
    print(f"Registry path: {registry_path}")
    print(f"ROI root: {roi_root}")
    print(f"ROI files found: {len(roi_paths)}")

    updated_df, stats = register_local_roi_paths(registry_df, roi_paths)
    _raise_for_unmatched_rois(stats)
    write_registry_parquet(updated_df, registry_path, validate=False)

    print("Local UniformTumor ROI registration complete.")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
