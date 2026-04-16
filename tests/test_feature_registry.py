from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

from kidney_vlm.pathology.feature_registry import register_existing_pathology_features


def _write_patch_features(path: Path, rows: int, cols: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=[[0.0] * cols for _ in range(rows)])
        handle.create_dataset("coords", data=[[0, 0] for _ in range(rows)])


def test_register_existing_pathology_features_ranks_dx_first_and_records_patch_counts(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    patch_features_dir = root_dir / "data" / "features" / "features_conch_v15"
    coords_root = root_dir / "data" / "features" / "coords_20x_512px_0px_overlap"

    dx_stem = "TCGA-AA-0001-01Z-00-DX1.dx-uuid"
    ts_stem = "TCGA-AA-0001-01A-01-TS1.ts-uuid"
    _write_patch_features(patch_features_dir / f"{dx_stem}.h5", rows=8)
    _write_patch_features(patch_features_dir / f"{ts_stem}.h5", rows=5)

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "tcga-case-1",
                "source": "tcga",
                "patient_id": "TCGA-AA-0001",
                "study_id": "case-1",
                "split": "train",
                "pathology_wsi_paths": [
                    f"data/raw/tcga/pathology/TCGA-KIRC/TCGA-AA-0001/{ts_stem}.svs",
                    f"data/raw/tcga/pathology/TCGA-KIRC/TCGA-AA-0001/{dx_stem}.svs",
                ],
                "radiology_image_paths": [],
                "radiology_image_modalities": [],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
                "pathology_tile_embedding_paths": [],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": [],
                "biomarkers_text": "",
                "question": "",
                "answer": "",
            }
        ]
    )

    updated_df, stats = register_existing_pathology_features(
        registry_df,
        patch_features_dir=patch_features_dir,
        coords_root=coords_root,
        save_format="h5",
        patch_size=512,
        target_mag=20,
        root_dir=root_dir,
        progress=False,
    )

    row = updated_df.iloc[0]
    assert row["pathology_tile_embedding_paths"] == [
        f"data/features/features_conch_v15/{dx_stem}.h5",
        f"data/features/features_conch_v15/{ts_stem}.h5",
    ]
    assert row["pathology_tile_embedding_patch_counts"] == [8, 5]
    assert row["pathology_embedding_patch_size"] == 512
    assert row["pathology_embedding_magnification"] == 20
    assert stats.feature_files_indexed == 2
    assert stats.cases_with_matches == 1
    assert stats.matched_feature_paths == 2


def test_register_existing_pathology_features_skips_invalid_files(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    patch_features_dir = root_dir / "data" / "features" / "features_conch_v15"
    coords_root = root_dir / "data" / "features" / "coords_20x_512px_0px_overlap"
    patch_features_dir.mkdir(parents=True, exist_ok=True)

    bad_stem = "TCGA-BB-0002-01Z-00-DX1.bad-uuid"
    bad_path = patch_features_dir / f"{bad_stem}.h5"
    bad_path.write_text("not-an-h5", encoding="utf-8")

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "tcga-case-2",
                "source": "tcga",
                "patient_id": "TCGA-BB-0002",
                "study_id": "case-2",
                "split": "train",
                "pathology_wsi_paths": [f"data/raw/tcga/pathology/TCGA-KIRC/TCGA-BB-0002/{bad_stem}.svs"],
                "radiology_image_paths": [],
                "radiology_image_modalities": [],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
                "pathology_tile_embedding_paths": [],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": [],
                "biomarkers_text": "",
                "question": "",
                "answer": "",
            }
        ]
    )

    updated_df, stats = register_existing_pathology_features(
        registry_df,
        patch_features_dir=patch_features_dir,
        coords_root=coords_root,
        save_format="h5",
        patch_size=512,
        target_mag=20,
        root_dir=root_dir,
        progress=False,
    )

    row = updated_df.iloc[0]
    assert row["pathology_tile_embedding_paths"] == []
    assert row["pathology_tile_embedding_patch_counts"] == []
    assert stats.cases_with_matches == 0
    assert stats.invalid_feature_files == 1
