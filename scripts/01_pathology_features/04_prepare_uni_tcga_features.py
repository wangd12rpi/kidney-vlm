#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm.auto import tqdm

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


def _archive_label(archive_path: Path) -> str:
    name = archive_path.name
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")]
    return archive_path.stem


def _string_list(values: Any) -> list[str]:
    items: list[str] = []
    for value in list(values or []):
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _source_value(mapping: Any, source_name: str) -> Any:
    if source_name not in mapping:
        raise KeyError(f"Missing pathology_features.uni value for source '{source_name}'.")
    return mapping[source_name]


def _selected_archives(*, archive_dir: Path, selected_archives: list[str]) -> list[Path]:
    if selected_archives:
        return [archive_dir / name for name in selected_archives]
    return sorted(path for path in archive_dir.glob("*.tar.gz") if path.is_file())


def _normalize_feature_array(features: np.ndarray) -> np.ndarray:
    if features.ndim == 2:
        return features
    if features.ndim == 3 and features.shape[0] == 1:
        return features[0]
    raise ValueError(f"Unsupported UNI feature shape: {tuple(features.shape)}")


def _normalize_coords_array(coords: np.ndarray) -> np.ndarray:
    if coords.ndim == 2:
        return coords
    if coords.ndim == 3 and coords.shape[0] == 1:
        return coords[0]
    raise ValueError(f"Unsupported UNI coords shape: {tuple(coords.shape)}")


def _dataset_kwargs(compression: str) -> dict[str, Any]:
    normalized = str(compression).strip().lower()
    if normalized in {"", "none"}:
        return {}
    if normalized == "gzip":
        return {"compression": "gzip", "compression_opts": 1}
    raise ValueError(f"Unsupported compression: {compression}")


def _slide_kind(file_stem: str) -> str:
    upper_stem = str(file_stem).upper()
    if "-DX" in upper_stem:
        return "DX"
    if "-TS" in upper_stem:
        return "TS"
    if "-BS" in upper_stem:
        return "BS"
    return ""


def _filter_h5_paths_by_allowed_slide_kinds(h5_paths: list[Path], allowed_slide_kinds: list[str]) -> list[Path]:
    allowed = {str(kind).strip().upper() for kind in allowed_slide_kinds if str(kind).strip()}
    if not allowed:
        return h5_paths
    return [path for path in h5_paths if _slide_kind(path.stem) in allowed]


def _safe_members(archive: tarfile.TarFile, destination_dir: Path) -> list[tarfile.TarInfo]:
    destination_dir = destination_dir.resolve()
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        member_path = (destination_dir / member.name).resolve()
        if not str(member_path).startswith(str(destination_dir)):
            raise RuntimeError(f"Unsafe tar member path detected: {member.name}")
        if member.isfile():
            members.append(member)
    return members


def extract_archive(archive_path: Path, destination_dir: Path) -> list[Path]:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths: list[Path] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _safe_members(archive, destination_dir)
        loop = tqdm(members, total=len(members), desc=f"Extracting {_archive_label(archive_path)}", unit="file", leave=False)
        for member in loop:
            archive.extract(member, path=destination_dir)
            extracted_paths.append(destination_dir / member.name)
    return sorted(path for path in extracted_paths if path.suffix == ".h5")


def convert_uni_h5_file(
    input_path: Path,
    *,
    output_path: Path,
    feature_dtype: np.dtype,
    compression: str,
    overwrite: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    if output_path.exists() and not overwrite:
        with h5py.File(output_path, "r") as handle:
            stored_shape = tuple(handle["features"].shape) if "features" in handle else ()
            stored_dtype = str(handle["features"].dtype) if "features" in handle else ""
        return stored_shape, stored_shape, stored_dtype

    with h5py.File(input_path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"Missing 'features' dataset in {input_path}")
        if "coords" not in handle:
            raise KeyError(f"Missing 'coords' dataset in {input_path}")
        original_features = np.asarray(handle["features"])
        original_coords = np.asarray(handle["coords"])

    features = _normalize_feature_array(original_features).astype(feature_dtype, copy=False)
    coords = _normalize_coords_array(original_coords)
    if features.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Patch count mismatch after normalization for {input_path}: "
            f"features={tuple(features.shape)} coords={tuple(coords.shape)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temp_output_path.exists():
        temp_output_path.unlink()

    kwargs = _dataset_kwargs(compression)
    with h5py.File(temp_output_path, "w") as handle:
        handle.create_dataset("features", data=features, **kwargs)
        handle.create_dataset("coords", data=coords, **kwargs)
        handle.attrs["source_format"] = "uni2"
        handle.attrs["converted_layout"] = "conch_like"
        handle.attrs["original_features_shape"] = np.asarray(original_features.shape, dtype=np.int64)
        handle.attrs["original_features_dtype"] = str(original_features.dtype)
        handle.attrs["stored_features_dtype"] = str(features.dtype)

    temp_output_path.replace(output_path)
    return tuple(original_features.shape), tuple(features.shape), str(features.dtype)


def _convert_archive_h5s(
    extracted_h5_paths: list[Path],
    archive_path: Path,
    *,
    output_features_dir: Path,
    feature_dtype: np.dtype,
    compression: str,
    overwrite_existing: bool,
    allowed_slide_kinds: list[str],
) -> None:
    selected_h5_paths = _filter_h5_paths_by_allowed_slide_kinds(extracted_h5_paths, allowed_slide_kinds)
    loop = tqdm(
        selected_h5_paths,
        total=len(selected_h5_paths),
        desc=f"Converting {_archive_label(archive_path)}",
        unit="file",
        leave=False,
    )
    for extracted_h5_path in loop:
        output_path = output_features_dir / extracted_h5_path.name
        convert_uni_h5_file(
            extracted_h5_path,
            output_path=output_path,
            feature_dtype=feature_dtype,
            compression=compression,
            overwrite=overwrite_existing,
        )


def _clear_existing_pathology_embeddings(registry_df: Any, row_mask: Any) -> Any:
    out = registry_df.copy()
    for row_idx in out.index[row_mask]:
        out.at[row_idx, "pathology_tile_embedding_paths"] = []
        if "pathology_tile_embedding_patch_counts" in out.columns:
            out.at[row_idx, "pathology_tile_embedding_patch_counts"] = []
        if "pathology_embedding_patch_size" in out.columns:
            out.at[row_idx, "pathology_embedding_patch_size"] = None
        if "pathology_embedding_magnification" in out.columns:
            out.at[row_idx, "pathology_embedding_magnification"] = None
    return out


def _register_output_features(
    *,
    registry_path: Path,
    output_features_dir: Path,
    coords_root: Path,
    save_format: str,
    patch_size: int,
    target_magnification: int,
    source_filter: str | None,
    clear_existing: bool,
    match_patient_id_when_no_wsi_paths: bool,
    register_enabled: bool,
) -> None:
    if not register_enabled:
        return
    if not registry_path.exists():
        raise FileNotFoundError(f"Unified registry not found: {registry_path}")

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        raise RuntimeError(f"Unified registry is empty: {registry_path}")

    if source_filter:
        row_mask = registry_df["source"].fillna("").astype(str).str.lower().eq(source_filter.lower())
    else:
        row_mask = registry_df.index.to_series().map(lambda _idx: True)
    if not bool(row_mask.any()):
        raise RuntimeError(f"No registry rows selected for UNI registration source_filter={source_filter!r}.")

    if clear_existing:
        label = source_filter or "all sources"
        print(f"Clearing existing pathology patch embedding fields before UNI registration ({label})...")
        registry_df = _clear_existing_pathology_embeddings(registry_df, row_mask)

    print("Registering converted UNI features into the unified registry...")
    target_df = registry_df.loc[row_mask].copy()
    updated_df, stats = register_existing_pathology_features(
        target_df,
        patch_features_dir=output_features_dir,
        coords_root=coords_root,
        save_format=save_format,
        patch_size=patch_size,
        target_mag=target_magnification,
        root_dir=ROOT,
        progress=True,
        match_patient_id_when_no_wsi_paths=match_patient_id_when_no_wsi_paths,
    )

    merged_df = registry_df.copy()
    for column in updated_df.columns:
        if column not in merged_df.columns:
            merged_df[column] = None
        merged_df.loc[updated_df.index, column] = updated_df[column]
    write_registry_parquet(merged_df, registry_path, validate=True)

    print("UNI registry registration complete.")
    print(f"Cases scanned: {stats.cases_scanned}")
    print(f"Cases with slide paths: {stats.cases_with_slide_paths}")
    print(f"Cases with matched features: {stats.cases_with_matches}")
    print(f"Matched feature paths written: {stats.matched_feature_paths}")
    print(f"Feature files indexed: {stats.feature_files_indexed}")
    print(f"Invalid feature files skipped: {stats.invalid_feature_files}")


def main() -> None:
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="01_pathology_features/default.yaml",
        overrides=sys.argv[1:],
    )
    feature_cfg = cfg.pathology_features
    uni_cfg = feature_cfg.uni
    source_name = str(uni_cfg.source).strip().lower()
    dataset_subfolder = str(_source_value(uni_cfg.dataset_subfolders, source_name))
    archive_root = Path(str(_source_value(uni_cfg.archive_roots, source_name)))
    archive_dir = archive_root / dataset_subfolder
    temp_extract_root = Path(str(_source_value(uni_cfg.extract_roots, source_name)))
    output_features_dir = Path(str(_source_value(uni_cfg.output_feature_dirs, source_name)))
    selected_archives = _string_list(_source_value(uni_cfg.selected_archives, source_name))
    allowed_slide_kinds = _string_list(_source_value(uni_cfg.allowed_slide_kinds, source_name))
    feature_dtype = np.dtype(str(uni_cfg.feature_dtype))
    hdf5_compression = str(uni_cfg.hdf5_compression)
    register_cfg = uni_cfg.register
    register_source_filter = register_cfg.get("source_filter")
    source_filter = str(register_source_filter).strip() if register_source_filter else source_name
    match_patient_id_when_no_wsi_paths = bool(
        _source_value(register_cfg.match_patient_id_when_no_wsi_paths, source_name)
    )

    archives = _selected_archives(archive_dir=archive_dir, selected_archives=selected_archives)
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive dir not found: {archive_dir}")
    if not archives:
        raise RuntimeError(f"No .tar.gz UNI archives selected under: {archive_dir}")

    temp_extract_root.mkdir(parents=True, exist_ok=True)
    output_features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source_name}")
    print(f"Archive dir: {archive_dir}")
    print(f"Temp extract root: {temp_extract_root}")
    print(f"Output features dir: {output_features_dir}")
    print(f"Archives selected: {len(archives)}")
    print(f"Feature dtype: {feature_dtype}")
    print(f"HDF5 compression: {hdf5_compression}")
    print(f"Allowed slide kinds: {allowed_slide_kinds if allowed_slide_kinds else ['ALL']}")
    print(f"Registry source filter: {source_filter or 'ALL'}")
    print(f"Match by patient_id when no WSI paths: {match_patient_id_when_no_wsi_paths}")
    print(f"Delete archives after processing: {bool(uni_cfg.delete_archives_after_processing)}")

    archive_loop = tqdm(archives, total=len(archives), desc=f"Preparing UNI {source_name.upper()} archives", unit="archive")
    for archive_path in archive_loop:
        archive_loop.set_postfix_str(archive_path.name)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        extract_dir = temp_extract_root / _archive_label(archive_path)
        extracted_h5_paths = extract_archive(archive_path, extract_dir)
        _convert_archive_h5s(
            extracted_h5_paths,
            archive_path,
            output_features_dir=output_features_dir,
            feature_dtype=feature_dtype,
            compression=hdf5_compression,
            overwrite_existing=bool(uni_cfg.overwrite_existing_feature_files),
            allowed_slide_kinds=allowed_slide_kinds,
        )

        if bool(uni_cfg.delete_extracted_files_after_processing) and extract_dir.exists():
            shutil.rmtree(extract_dir)
        if bool(uni_cfg.delete_archives_after_processing) and archive_path.exists():
            archive_path.unlink()

    _register_output_features(
        registry_path=Path(str(register_cfg.registry_path)),
        output_features_dir=output_features_dir,
        coords_root=Path(str(register_cfg.coords_root)),
        save_format=str(feature_cfg.save_format),
        patch_size=int(register_cfg.patch_size),
        target_magnification=int(register_cfg.target_magnification),
        source_filter=source_filter,
        clear_existing=bool(register_cfg.clear_existing_pathology_patch_embeddings_before_register),
        match_patient_id_when_no_wsi_paths=match_patient_id_when_no_wsi_paths,
        register_enabled=bool(register_cfg.enabled),
    )
    print("UNI archive preparation complete.")


if __name__ == "__main__":
    main()
