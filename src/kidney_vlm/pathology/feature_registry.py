from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from kidney_vlm.data.registry_schema import normalize_registry_df


@dataclass(frozen=True)
class ExistingFeatureRegistrationStats:
    feature_files_indexed: int
    cases_scanned: int
    cases_with_slide_paths: int
    cases_with_matches: int
    matched_feature_paths: int
    invalid_feature_files: int


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


def _normalize_local_path(root_dir: Path, path_value: str) -> str:
    text = str(path_value).strip().strip("'").strip('"')
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
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


def _local_wsi_paths(root_dir: Path, value: Any) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in _as_list(value):
        if "://" in raw_path:
            continue
        normalized = _normalize_local_path(root_dir, raw_path)
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


def _is_valid_patch_tensor(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _is_valid_coords(path: Path) -> bool:
    return _is_valid_h5(path, ("coords",))


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
            import torch

            tensor = torch.load(patch_features_path, map_location="cpu")
            if hasattr(tensor, "shape") and len(tensor.shape) > 0:
                return int(tensor.shape[0])
            if isinstance(tensor, list):
                return len(tensor)
        except Exception:
            return 0

    return 0


def _match_feature_path(
    slide_stem: str,
    *,
    feature_by_stem: dict[str, Path],
    coords_root: Path,
    save_format: str,
    patch_count_cache: dict[str, int | None],
    invalid_stems: set[str],
) -> tuple[Path | None, int | None]:
    feature_path = feature_by_stem.get(slide_stem)
    if feature_path is None:
        return None, None

    if slide_stem in patch_count_cache:
        patch_count = patch_count_cache[slide_stem]
        return feature_path, patch_count

    if save_format == "h5":
        valid = _is_valid_patch_features(feature_path)
    else:
        valid = _is_valid_patch_tensor(feature_path)
    if not valid:
        patch_count_cache[slide_stem] = None
        invalid_stems.add(slide_stem)
        return None, None

    coords_path = coords_root / "patches" / f"{slide_stem}_patches.h5"
    patch_count = _read_patch_count(coords_path, feature_path, save_format)
    patch_count_cache[slide_stem] = patch_count
    return feature_path, patch_count


def register_existing_pathology_features(
    registry_df: pd.DataFrame,
    *,
    patch_features_dir: Path,
    coords_root: Path,
    save_format: str,
    patch_size: int,
    target_mag: int,
    root_dir: Path,
    progress: bool = False,
) -> tuple[pd.DataFrame, ExistingFeatureRegistrationStats]:
    if save_format not in {"h5", "pt"}:
        raise ValueError("save_format must be one of: h5, pt")

    feature_by_stem = {path.stem: path.resolve() for path in sorted(patch_features_dir.glob(f"*.{save_format}"))}
    patch_count_cache: dict[str, int | None] = {}
    invalid_stems: set[str] = set()

    out = registry_df.copy()
    if "pathology_tile_embedding_patch_counts" not in out.columns:
        out["pathology_tile_embedding_patch_counts"] = [[] for _ in range(len(out))]
    if "pathology_embedding_patch_size" not in out.columns:
        out["pathology_embedding_patch_size"] = None
    if "pathology_embedding_magnification" not in out.columns:
        out["pathology_embedding_magnification"] = None
    cases_with_slide_paths = 0
    cases_with_matches = 0
    matched_feature_paths = 0

    row_indices = out.index.tolist()
    row_iter = tqdm(row_indices, total=len(row_indices), desc="Registering existing pathology embeddings") if progress else row_indices

    for row_idx in row_iter:
        case_wsi_paths = _local_wsi_paths(root_dir, out.at[row_idx, "pathology_wsi_paths"])
        if case_wsi_paths:
            cases_with_slide_paths += 1

        ranked_paths: list[tuple[tuple[Any, ...], str]] = []
        for normalized_wsi_path in case_wsi_paths:
            slide_stem = Path(normalized_wsi_path).stem
            feature_path, patch_count = _match_feature_path(
                slide_stem,
                feature_by_stem=feature_by_stem,
                coords_root=coords_root,
                save_format=save_format,
                patch_count_cache=patch_count_cache,
                invalid_stems=invalid_stems,
            )
            sort_key = (
                1 if _is_normal_tcga_slide(slide_stem) else 0,
                _slide_kind_rank(slide_stem),
                0 if feature_path is not None else 1,
                -(patch_count or 0),
                Path(normalized_wsi_path).name.upper(),
            )
            ranked_paths.append((sort_key, normalized_wsi_path))

        ranked_paths.sort(key=lambda item: item[0])

        case_tile_paths: list[str] = []
        case_patch_counts: list[int] = []
        seen_tile_paths: set[str] = set()
        for _, normalized_wsi_path in ranked_paths:
            slide_stem = Path(normalized_wsi_path).stem
            feature_path, patch_count = _match_feature_path(
                slide_stem,
                feature_by_stem=feature_by_stem,
                coords_root=coords_root,
                save_format=save_format,
                patch_count_cache=patch_count_cache,
                invalid_stems=invalid_stems,
            )
            if feature_path is None or patch_count is None:
                continue
            relative_path = _to_registry_relative_path(root_dir, feature_path)
            if relative_path in seen_tile_paths:
                continue
            seen_tile_paths.add(relative_path)
            case_tile_paths.append(relative_path)
            case_patch_counts.append(int(patch_count))

        if not case_tile_paths:
            continue

        out.at[row_idx, "pathology_tile_embedding_paths"] = case_tile_paths
        out.at[row_idx, "pathology_embedding_patch_size"] = patch_size
        out.at[row_idx, "pathology_embedding_magnification"] = target_mag
        out.at[row_idx, "pathology_tile_embedding_patch_counts"] = case_patch_counts

        cases_with_matches += 1
        matched_feature_paths += len(case_tile_paths)

    out = normalize_registry_df(out)
    stats = ExistingFeatureRegistrationStats(
        feature_files_indexed=len(feature_by_stem),
        cases_scanned=len(out),
        cases_with_slide_paths=cases_with_slide_paths,
        cases_with_matches=cases_with_matches,
        matched_feature_paths=matched_feature_paths,
        invalid_feature_files=len(invalid_stems),
    )
    return out, stats
