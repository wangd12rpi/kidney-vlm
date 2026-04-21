from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "01_pathology_features" / "04_prepare_uni_tcga_features.py"
    spec = importlib.util.spec_from_file_location("prepare_uni_tcga_features_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_label_strips_tar_gz_suffix() -> None:
    module = _load_script_module()
    archive_path = Path("/tmp/TCGA-KIRC.tar.gz")

    assert module._archive_label(archive_path) == "TCGA-KIRC"


def test_convert_uni_h5_file_flattens_leading_batch_axis_and_casts_dtype(tmp_path: Path) -> None:
    module = _load_script_module()

    input_path = tmp_path / "uni_input.h5"
    output_path = tmp_path / "uni_output.h5"
    features = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
    coords = np.arange(8, dtype=np.int64).reshape(1, 4, 2)

    with h5py.File(input_path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)
        handle.create_dataset("annots", data=np.zeros((1, 4), dtype=np.int64))

    original_shape, converted_shape, converted_dtype = module.convert_uni_h5_file(
        input_path,
        output_path=output_path,
        feature_dtype=np.dtype(np.float16),
        compression="none",
        overwrite=False,
    )

    assert original_shape == (1, 4, 6)
    assert converted_shape == (4, 6)
    assert converted_dtype == "float16"

    with h5py.File(output_path, "r") as handle:
        assert tuple(handle["features"].shape) == (4, 6)
        assert tuple(handle["coords"].shape) == (4, 2)
        assert str(handle["features"].dtype) == "float16"
        assert handle.attrs["source_format"] == "uni2"
        assert handle.attrs["converted_layout"] == "conch_like"


def test_convert_uni_h5_file_keeps_existing_flat_arrays(tmp_path: Path) -> None:
    module = _load_script_module()

    input_path = tmp_path / "uni_flat_input.h5"
    output_path = tmp_path / "uni_flat_output.h5"
    features = np.arange(18, dtype=np.float32).reshape(3, 6)
    coords = np.arange(6, dtype=np.int64).reshape(3, 2)

    with h5py.File(input_path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)

    _original_shape, converted_shape, converted_dtype = module.convert_uni_h5_file(
        input_path,
        output_path=output_path,
        feature_dtype=np.dtype(np.float16),
        compression="gzip",
        overwrite=False,
    )

    assert converted_shape == (3, 6)
    assert converted_dtype == "float16"

    with h5py.File(output_path, "r") as handle:
        assert tuple(handle["features"].shape) == (3, 6)
        assert tuple(handle["coords"].shape) == (3, 2)
        assert handle["features"].compression == "gzip"


def test_register_output_features_filters_to_selected_source(tmp_path: Path) -> None:
    module = _load_script_module()

    registry_path = tmp_path / "unified.parquet"
    patch_features_dir = tmp_path / "features_uni_cptac"
    patch_features_dir.mkdir(parents=True)
    feature_stem = "C3N-00001-21-001"
    feature_path = patch_features_dir / f"{feature_stem}.h5"
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=np.zeros((3, 4), dtype=np.float16))
        handle.create_dataset("coords", data=np.zeros((3, 2), dtype=np.int64))

    pd.DataFrame(
        [
            {
                "sample_id": "tcga-1",
                "source": "tcga",
                "patient_id": "TCGA-AA-0001",
                "study_id": "tcga-study",
                "split": "train",
                "pathology_wsi_paths": [],
                "pathology_tile_embedding_paths": ["keep-tcga-feature.h5"],
                "radiology_image_paths": [],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": [],
                "biomarkers_text": "",
                "question": "",
                "answer": "",
            },
            {
                "sample_id": "cptac-1",
                "source": "cptac",
                "patient_id": "C3N-00001",
                "study_id": "cptac-study",
                "split": "cptac_external_test",
                "pathology_wsi_paths": [],
                "pathology_tile_embedding_paths": ["old-cptac-feature.h5"],
                "radiology_image_paths": [],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": [],
                "biomarkers_text": "",
                "question": "",
                "answer": "",
            },
        ]
    ).to_parquet(registry_path, index=False)

    module._register_output_features(
        registry_path=registry_path,
        output_features_dir=patch_features_dir,
        coords_root=tmp_path / "unused_coords",
        save_format="h5",
        patch_size=256,
        target_magnification=20,
        source_filter="cptac",
        clear_existing=True,
        match_patient_id_when_no_wsi_paths=True,
        register_enabled=True,
    )

    updated = pd.read_parquet(registry_path)
    tcga_row = updated[updated["source"] == "tcga"].iloc[0]
    cptac_row = updated[updated["source"] == "cptac"].iloc[0]
    assert tcga_row["pathology_tile_embedding_paths"] == ["keep-tcga-feature.h5"]
    assert cptac_row["pathology_tile_embedding_paths"] == [str(feature_path).lstrip("/")]
    assert cptac_row["pathology_tile_embedding_patch_counts"] == [3]
