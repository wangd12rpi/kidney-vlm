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

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


# Input/output locations
ARCHIVE_DIR = ROOT / "data" / "staging" / "uni2_tcga_archives" / "TCGA"
TEMP_EXTRACT_ROOT = ROOT / "data" / "staging" / "uni2_tcga_extracted"
OUTPUT_FEATURES_DIR = ROOT / "data" / "features" / "features_uni"
REGISTRY_PATH = ROOT / "data" / "registry" / "unified.parquet"

# Archive selection
# If empty, process every .tar.gz under ARCHIVE_DIR.
PROJECT_ARCHIVES: list[str] = [
    # "TCGA-KICH.tar.gz",
    # "TCGA-KIRC.tar.gz",
    # "TCGA-KIRP.tar.gz",
]

# Conversion settings
FEATURE_DTYPE = np.float16
HDF5_COMPRESSION = "none"  # "none" or "gzip"
OVERWRITE_EXISTING_FEATURE_FILES = False
ALLOWED_SLIDE_KINDS: list[str] = ["DX"]

# Registry settings
REGISTER_INTO_REGISTRY = True
UNI_PATCH_SIZE = 256
UNI_TARGET_MAGNIFICATION = 20
UNI_COORDS_ROOT = ROOT / "data" / "features" / "coords_uni_unused"
CLEAR_EXISTING_PATHOLOGY_PATCH_EMBEDDINGS_BEFORE_REGISTER = True

# Cleanup settings
DELETE_ARCHIVES_AFTER_PROCESSING = True
DELETE_EXTRACTED_FILES_AFTER_PROCESSING = True


def _archive_label(archive_path: Path) -> str:
    name = archive_path.name
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")]
    return archive_path.stem


def _selected_archives() -> list[Path]:
    if PROJECT_ARCHIVES:
        return [ARCHIVE_DIR / str(name).strip() for name in PROJECT_ARCHIVES if str(name).strip()]
    return sorted(path for path in ARCHIVE_DIR.glob("*.tar.gz") if path.is_file())


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


def _filter_h5_paths_by_allowed_slide_kinds(h5_paths: list[Path]) -> list[Path]:
    allowed = {str(kind).strip().upper() for kind in ALLOWED_SLIDE_KINDS if str(kind).strip()}
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


def _convert_archive_h5s(extracted_h5_paths: list[Path], archive_path: Path) -> None:
    selected_h5_paths = _filter_h5_paths_by_allowed_slide_kinds(extracted_h5_paths)
    loop = tqdm(
        selected_h5_paths,
        total=len(selected_h5_paths),
        desc=f"Converting {_archive_label(archive_path)}",
        unit="file",
        leave=False,
    )
    for extracted_h5_path in loop:
        output_path = OUTPUT_FEATURES_DIR / extracted_h5_path.name
        convert_uni_h5_file(
            extracted_h5_path,
            output_path=output_path,
            feature_dtype=np.dtype(FEATURE_DTYPE),
            compression=HDF5_COMPRESSION,
            overwrite=OVERWRITE_EXISTING_FEATURE_FILES,
        )


def _register_output_features() -> None:
    if not REGISTER_INTO_REGISTRY:
        return
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Unified registry not found: {REGISTRY_PATH}")

    registry_df = read_parquet_or_empty(REGISTRY_PATH)
    if registry_df.empty:
        raise RuntimeError(f"Unified registry is empty: {REGISTRY_PATH}")

    if CLEAR_EXISTING_PATHOLOGY_PATCH_EMBEDDINGS_BEFORE_REGISTER:
        print("Clearing existing pathology patch embedding fields before UNI registration...")
        registry_df = registry_df.copy()
        registry_df["pathology_tile_embedding_paths"] = [[] for _ in range(len(registry_df))]
        if "pathology_tile_embedding_patch_counts" in registry_df.columns:
            registry_df["pathology_tile_embedding_patch_counts"] = [[] for _ in range(len(registry_df))]
        if "pathology_embedding_patch_size" in registry_df.columns:
            registry_df["pathology_embedding_patch_size"] = None
        if "pathology_embedding_magnification" in registry_df.columns:
            registry_df["pathology_embedding_magnification"] = None

    print("Registering converted UNI features into the unified registry...")
    updated_df, stats = register_existing_pathology_features(
        registry_df,
        patch_features_dir=OUTPUT_FEATURES_DIR,
        coords_root=UNI_COORDS_ROOT,
        save_format="h5",
        patch_size=int(UNI_PATCH_SIZE),
        target_mag=int(UNI_TARGET_MAGNIFICATION),
        root_dir=ROOT,
        progress=True,
    )
    write_registry_parquet(updated_df, REGISTRY_PATH, validate=True)

    print("UNI registry registration complete.")
    print(f"Cases scanned: {stats.cases_scanned}")
    print(f"Cases with slide paths: {stats.cases_with_slide_paths}")
    print(f"Cases with matched features: {stats.cases_with_matches}")
    print(f"Matched feature paths written: {stats.matched_feature_paths}")
    print(f"Feature files indexed: {stats.feature_files_indexed}")
    print(f"Invalid feature files skipped: {stats.invalid_feature_files}")


def main() -> None:
    archives = _selected_archives()
    if not ARCHIVE_DIR.exists():
        raise FileNotFoundError(f"Archive dir not found: {ARCHIVE_DIR}")
    if not archives:
        raise RuntimeError(f"No .tar.gz UNI archives selected under: {ARCHIVE_DIR}")

    TEMP_EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Archive dir: {ARCHIVE_DIR}")
    print(f"Temp extract root: {TEMP_EXTRACT_ROOT}")
    print(f"Output features dir: {OUTPUT_FEATURES_DIR}")
    print(f"Archives selected: {len(archives)}")
    print(f"Feature dtype: {np.dtype(FEATURE_DTYPE)}")
    print(f"HDF5 compression: {HDF5_COMPRESSION}")
    print(f"Allowed slide kinds: {ALLOWED_SLIDE_KINDS if ALLOWED_SLIDE_KINDS else ['ALL']}")
    print(f"Delete archives after processing: {DELETE_ARCHIVES_AFTER_PROCESSING}")

    archive_loop = tqdm(archives, total=len(archives), desc="Preparing UNI TCGA archives", unit="archive")
    for archive_path in archive_loop:
        archive_loop.set_postfix_str(archive_path.name)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        extract_dir = TEMP_EXTRACT_ROOT / _archive_label(archive_path)
        extracted_h5_paths = extract_archive(archive_path, extract_dir)
        _convert_archive_h5s(extracted_h5_paths, archive_path)

        if DELETE_EXTRACTED_FILES_AFTER_PROCESSING and extract_dir.exists():
            shutil.rmtree(extract_dir)
        if DELETE_ARCHIVES_AFTER_PROCESSING and archive_path.exists():
            archive_path.unlink()

    _register_output_features()
    print("UNI archive preparation complete.")


if __name__ == "__main__":
    main()
