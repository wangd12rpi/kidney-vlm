from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from kidney_vlm.data.registry_schema import normalize_registry_df


@dataclass(frozen=True)
class RadiologySeriesArtifactRecord:
    series_dir: str
    png_dir: str
    embedding_ref: str = ""
    slice_count: int = 0
    source_zip_path: str = ""
    mask_paths: tuple[str, ...] = ()
    mask_manifest_path: str = ""


@dataclass(frozen=True)
class RadiologyFeatureRegistrationStats:
    series_artifacts_indexed: int
    cases_scanned: int
    cases_with_series_paths: int
    cases_with_matches: int
    matched_series_refs: int


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


def _normalize_local_path(root_dir: Path, path_value: str | Path) -> str:
    text = str(path_value).strip().strip("'").strip('"')
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((root_dir / path).resolve())


def _to_registry_relative_path(root_dir: Path, path_value: str | Path) -> str:
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
        return resolved.relative_to(root_dir).as_posix()
    except ValueError:
        return resolved.as_posix().lstrip("/")


def _build_series_artifact_dir(
    *,
    root_dir: Path,
    raw_root: Path,
    artifact_root: Path,
    series_dir: str | Path,
) -> Path:
    series_path = Path(_normalize_local_path(root_dir, series_dir))
    raw_root = raw_root.expanduser().resolve()
    try:
        relative = series_path.relative_to(raw_root)
    except ValueError:
        relative = Path(_to_registry_relative_path(root_dir, series_path))
        if len(relative.parts) >= 2 and relative.parts[0] == "data" and relative.parts[1] == "raw":
            relative = Path(*relative.parts[2:])
    if len(relative.parts) >= 2 and relative.parts[1] == "radiology":
        relative = Path(*relative.parts[2:])
    return (artifact_root / relative).resolve()


def build_png_series_dir(
    *,
    root_dir: Path,
    raw_root: Path,
    png_root: Path,
    series_dir: str | Path,
) -> Path:
    return _build_series_artifact_dir(
        root_dir=root_dir,
        raw_root=raw_root,
        artifact_root=png_root,
        series_dir=series_dir,
    )


def build_mask_series_dir(
    *,
    root_dir: Path,
    raw_root: Path,
    mask_root: Path,
    series_dir: str | Path,
) -> Path:
    return _build_series_artifact_dir(
        root_dir=root_dir,
        raw_root=raw_root,
        artifact_root=mask_root,
        series_dir=series_dir,
    )


def format_series_embedding_ref(
    *,
    root_dir: Path,
    store_path: str | Path,
    series_dir: str | Path,
) -> str:
    store_rel = _to_registry_relative_path(root_dir, store_path)
    series_rel = _to_registry_relative_path(root_dir, series_dir)
    return f"{store_rel}::series={series_rel}"


def register_radiology_series_artifacts(
    registry_df: pd.DataFrame,
    *,
    root_dir: Path,
    artifacts_by_series_dir: Mapping[str, RadiologySeriesArtifactRecord],
) -> tuple[pd.DataFrame, RadiologyFeatureRegistrationStats]:
    out = normalize_registry_df(registry_df.copy())
    if "radiology_png_dirs" not in out.columns:
        out["radiology_png_dirs"] = [[] for _ in range(len(out))]
    if "radiology_series_slice_counts" not in out.columns:
        out["radiology_series_slice_counts"] = [[] for _ in range(len(out))]
    if "radiology_mask_manifest_paths" not in out.columns:
        out["radiology_mask_manifest_paths"] = [[] for _ in range(len(out))]
    if "radiology_download_paths" not in out.columns:
        out["radiology_download_paths"] = [[] for _ in range(len(out))]

    cases_with_series_paths = 0
    cases_with_matches = 0
    matched_series_refs = 0
    artifacts_by_png_dir = {
        _normalize_local_path(root_dir, artifact.png_dir): artifact
        for artifact in artifacts_by_series_dir.values()
        if str(artifact.png_dir).strip()
    }
    artifacts_by_zip_path = {
        _normalize_local_path(root_dir, artifact.source_zip_path): artifact
        for artifact in artifacts_by_series_dir.values()
        if str(artifact.source_zip_path).strip()
    }

    for row_idx, row in out.iterrows():
        series_paths = _as_list(row.get("radiology_image_paths"))
        download_paths = _as_list(row.get("radiology_download_paths"))
        if series_paths:
            cases_with_series_paths += 1

        matched_embedding_refs: list[str] = []
        matched_png_dirs: list[str] = []
        matched_slice_counts: list[int] = []
        matched_mask_paths: list[str] = []
        matched_mask_manifest_paths: list[str] = []
        seen_mask_paths: set[str] = set()
        seen_mask_manifest_paths: set[str] = set()

        candidate_paths = download_paths + series_paths
        for series_path in candidate_paths:
            if "://" in series_path:
                continue
            normalized_series_path = _normalize_local_path(root_dir, series_path)
            artifact = artifacts_by_series_dir.get(normalized_series_path)
            if artifact is None:
                artifact = artifacts_by_png_dir.get(normalized_series_path)
            if artifact is None:
                artifact = artifacts_by_zip_path.get(normalized_series_path)
            if artifact is None:
                continue
            embedding_ref = str(artifact.embedding_ref).strip()
            if embedding_ref:
                matched_embedding_refs.append(embedding_ref)
            png_dir = str(artifact.png_dir).strip()
            if png_dir:
                matched_png_dirs.append(_to_registry_relative_path(root_dir, png_dir))
            if int(artifact.slice_count) > 0:
                matched_slice_counts.append(int(artifact.slice_count))
            for mask_path in artifact.mask_paths:
                mask_relpath = _to_registry_relative_path(root_dir, mask_path)
                if mask_relpath and mask_relpath not in seen_mask_paths:
                    seen_mask_paths.add(mask_relpath)
                    matched_mask_paths.append(mask_relpath)
            mask_manifest_path = str(artifact.mask_manifest_path).strip()
            if mask_manifest_path:
                manifest_relpath = _to_registry_relative_path(root_dir, mask_manifest_path)
                if manifest_relpath and manifest_relpath not in seen_mask_manifest_paths:
                    seen_mask_manifest_paths.add(manifest_relpath)
                    matched_mask_manifest_paths.append(manifest_relpath)

        if matched_embedding_refs or matched_png_dirs or matched_mask_paths or matched_mask_manifest_paths:
            cases_with_matches += 1
            matched_series_refs += len(matched_embedding_refs)
            out.at[row_idx, "radiology_embedding_paths"] = matched_embedding_refs
            out.at[row_idx, "radiology_png_dirs"] = matched_png_dirs
            out.at[row_idx, "radiology_series_slice_counts"] = matched_slice_counts
            out.at[row_idx, "radiology_mask_paths"] = matched_mask_paths
            out.at[row_idx, "radiology_mask_manifest_paths"] = matched_mask_manifest_paths

    stats = RadiologyFeatureRegistrationStats(
        series_artifacts_indexed=len(artifacts_by_series_dir),
        cases_scanned=len(out),
        cases_with_series_paths=cases_with_series_paths,
        cases_with_matches=cases_with_matches,
        matched_series_refs=matched_series_refs,
    )
    return out, stats
