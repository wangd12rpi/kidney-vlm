#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.pathology.feature_registry import register_existing_pathology_features
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


CONFIG_RELATIVE_PATH = "data/05_register_uni_paths_into_registry.yaml"

# Stable script behavior. These are not exposed in YAML because they should not
# be changed during routine registry rebuilds.
REGISTRY_PATH = ROOT / "data" / "registry" / "unified.parquet"
SAVE_FORMAT = "h5"
PATCH_SIZE = 256
TARGET_MAGNIFICATION = 20
COORDS_ROOT = ROOT / "data" / "features" / "coords_uni_unused"
CLEAR_EXISTING_PATHOLOGY_PATCH_EMBEDDINGS_BEFORE_REGISTER = True
ALLOWED_PROJECT_IDS: list[str] = []


def _normalized_string_list(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _count_cases_with_patch_embeddings(frame) -> int:
    def _has_paths(value) -> bool:
        if value is None:
            return False
        if hasattr(value, "tolist") and not isinstance(value, str):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return any(str(item).strip() for item in value)
        return bool(str(value).strip())

    if "pathology_tile_embedding_paths" not in frame.columns:
        return 0
    return int(frame["pathology_tile_embedding_paths"].map(_has_paths).sum())


def _clear_patch_embedding_fields(frame):
    out = frame.copy()
    out["pathology_tile_embedding_paths"] = [[] for _ in range(len(out))]
    if "pathology_tile_embedding_patch_counts" in out.columns:
        out["pathology_tile_embedding_patch_counts"] = [[] for _ in range(len(out))]
    if "pathology_embedding_patch_size" in out.columns:
        out["pathology_embedding_patch_size"] = None
    if "pathology_embedding_magnification" in out.columns:
        out["pathology_embedding_magnification"] = None
    return out


def _build_enabled_jobs(sources_cfg: Any) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for label, source_cfg in sources_cfg.items():
        if not bool(source_cfg.enabled):
            continue
        jobs.append(
            {
                "label": str(label),
                "source_filter": str(label),
                "patch_features_dir": Path(str(source_cfg.features_dir)),
                "match_patient_id_when_no_wsi_paths": True,
            }
        )
    if not jobs:
        raise ValueError("No UNI registration sources are enabled in the YAML config.")
    return jobs


def _register_uni_job(
    registry_df,
    *,
    job: dict[str, object],
    allowed_project_ids: list[str],
    clear_existing_pathology_patch_embeddings_before_register: bool,
    coords_root: Path,
    save_format: str,
    patch_size: int,
    target_magnification: int,
):
    label = str(job["label"])
    patch_features_dir = Path(job["patch_features_dir"])
    source_filter = str(job["source_filter"]).strip()
    match_patient_id_when_no_wsi_paths = bool(job["match_patient_id_when_no_wsi_paths"])

    if not patch_features_dir.exists():
        raise FileNotFoundError(f"UNI features dir not found for {label}: {patch_features_dir}")

    if source_filter:
        selected_mask = registry_df["source"].fillna("").astype(str).str.lower().eq(source_filter.lower())
    else:
        selected_mask = registry_df.index.to_series().map(lambda _idx: True)
    normalized_allowed_project_ids = _normalized_string_list(allowed_project_ids)
    if normalized_allowed_project_ids and "project_id" in registry_df.columns:
        selected_mask = selected_mask & registry_df["project_id"].fillna("").astype(str).isin(
            set(normalized_allowed_project_ids)
        )

    selected_registry_df = registry_df.loc[selected_mask].copy()
    if selected_registry_df.empty:
        raise RuntimeError(f"No registry rows selected for UNI registration job: {label}")

    print(f"\nUNI registration job: {label}")
    print(f"Registry rows selected: {len(selected_registry_df)}")
    print(f"UNI features dir: {patch_features_dir}")
    print(f"Match by patient_id when WSI paths are absent: {match_patient_id_when_no_wsi_paths}")
    print(f"Rows with patch embeddings before: {_count_cases_with_patch_embeddings(selected_registry_df)}")

    feature_paths = sorted(patch_features_dir.glob(f"*.{save_format}"))
    print(f"UNI feature files found: {len(feature_paths)}")
    if not feature_paths:
        raise RuntimeError(f"No .{save_format} UNI feature files found under {patch_features_dir}")

    working_df = selected_registry_df
    if clear_existing_pathology_patch_embeddings_before_register:
        print(f"Clearing existing pathology patch embedding fields before UNI registration ({label})...")
        working_df = _clear_patch_embedding_fields(working_df)

    updated_selected_df, stats = register_existing_pathology_features(
        working_df,
        patch_features_dir=patch_features_dir,
        coords_root=coords_root,
        save_format=save_format,
        patch_size=patch_size,
        target_mag=target_magnification,
        root_dir=ROOT,
        progress=True,
        match_patient_id_when_no_wsi_paths=match_patient_id_when_no_wsi_paths,
    )

    out = registry_df.copy()
    for column in updated_selected_df.columns:
        if column not in out.columns:
            out[column] = None
        try:
            out.loc[updated_selected_df.index, column] = updated_selected_df[column]
        except (TypeError, ValueError):
            out[column] = out[column].astype("object")
            out.loc[updated_selected_df.index, column] = updated_selected_df[column].astype("object")

    print("UNI registry insertion job complete.")
    print(f"Cases scanned: {stats.cases_scanned}")
    print(f"Cases with slide paths: {stats.cases_with_slide_paths}")
    print(f"Cases with matched features: {stats.cases_with_matches}")
    print(f"Matched feature paths written: {stats.matched_feature_paths}")
    print(f"Feature files indexed: {stats.feature_files_indexed}")
    print(f"Invalid feature files skipped: {stats.invalid_feature_files}")
    print(f"Rows with patch embeddings after: {_count_cases_with_patch_embeddings(updated_selected_df)}")
    return out


def main() -> None:
    cfg = load_script_cfg(repo_root=ROOT, config_relative_path=CONFIG_RELATIVE_PATH, overrides=sys.argv[1:])
    uni_cfg = cfg.uni_registration
    jobs = _build_enabled_jobs(uni_cfg.sources)

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Unified registry not found: {REGISTRY_PATH}")

    registry_df = read_parquet_or_empty(REGISTRY_PATH)
    if registry_df.empty:
        raise RuntimeError(f"Unified registry is empty: {REGISTRY_PATH}")

    print(f"Registry path: {REGISTRY_PATH}")
    print(f"Save format: {SAVE_FORMAT}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Target magnification: {TARGET_MAGNIFICATION}")
    print(f"Allowed project ids: {_normalized_string_list(ALLOWED_PROJECT_IDS) or ['ALL']}")
    print(f"Enabled sources: {[job['label'] for job in jobs]}")

    final_registry_df = registry_df
    for job in jobs:
        final_registry_df = _register_uni_job(
            final_registry_df,
            job=job,
            allowed_project_ids=ALLOWED_PROJECT_IDS,
            clear_existing_pathology_patch_embeddings_before_register=CLEAR_EXISTING_PATHOLOGY_PATCH_EMBEDDINGS_BEFORE_REGISTER,
            coords_root=COORDS_ROOT,
            save_format=SAVE_FORMAT,
            patch_size=PATCH_SIZE,
            target_magnification=TARGET_MAGNIFICATION,
        )

    write_registry_parquet(final_registry_df, REGISTRY_PATH, validate=True)
    print("\nUNI registry insertion complete.")
    print(f"Rows with patch embeddings after: {_count_cases_with_patch_embeddings(final_registry_df)}")


if __name__ == "__main__":
    main()
