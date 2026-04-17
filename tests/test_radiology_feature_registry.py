from __future__ import annotations

from pathlib import Path

import pandas as pd

from kidney_vlm.radiology.feature_registry import (
    RadiologySeriesArtifactRecord,
    build_png_series_dir,
    format_series_embedding_ref,
    register_radiology_series_artifacts,
)


def test_build_png_series_dir_prefers_path_relative_to_raw_root(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    raw_root = root_dir / "data" / "raw"
    png_root = root_dir / "data" / "processes" / "radiology" / "chunk1" / "pngs"
    series_dir = raw_root / "tcga" / "radiology" / "TCGA-KIRC" / "TCGA-AA-0001" / "1.2.3" / "4.5.6"

    resolved = build_png_series_dir(
        root_dir=root_dir,
        raw_root=raw_root,
        png_root=png_root,
        series_dir=series_dir,
    )

    assert resolved == png_root / "TCGA-KIRC" / "TCGA-AA-0001" / "1.2.3" / "4.5.6"


def test_format_series_embedding_ref_uses_registry_relative_paths(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    store_path = root_dir / "data" / "processes" / "radiology" / "chunk1" / "features_medsiglip448" / "chunk1.h5"
    series_dir = root_dir / "data" / "processes" / "radiology" / "chunk1" / "pngs" / "TCGA-KIRC" / "TCGA-AA-0001" / "1.2.3" / "4.5.6"

    ref = format_series_embedding_ref(
        root_dir=root_dir,
        store_path=store_path,
        series_dir=series_dir,
    )

    assert ref == (
        "data/processes/radiology/chunk1/features_medsiglip448/chunk1.h5"
        "::series=data/processes/radiology/chunk1/pngs/TCGA-KIRC/TCGA-AA-0001/1.2.3/4.5.6"
    )


def test_register_radiology_series_artifacts_updates_registry_with_series_refs(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    series_dir = root_dir / "data" / "processes" / "radiology" / "chunk1" / "pngs" / "TCGA-KIRC" / "TCGA-AA-0001" / "1.2.3" / "4.5.6"
    png_dir = series_dir
    store_path = root_dir / "data" / "processes" / "radiology" / "chunk1" / "features_medsiglip448" / "chunk1.h5"
    embedding_ref = format_series_embedding_ref(
        root_dir=root_dir,
        store_path=store_path,
        series_dir=series_dir,
    )

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "tcga-case-1",
                "source": "tcga",
                "patient_id": "TCGA-AA-0001",
                "study_id": "case-1",
                "split": "train",
                "pathology_wsi_paths": [],
                "radiology_image_paths": [
                    "data/processes/radiology/chunk1/pngs/TCGA-KIRC/TCGA-AA-0001/1.2.3/4.5.6",
                ],
                "radiology_image_modalities": ["CT"],
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

    updated_df, stats = register_radiology_series_artifacts(
        registry_df,
        root_dir=root_dir,
        artifacts_by_series_dir={
            str(series_dir.resolve()): RadiologySeriesArtifactRecord(
                series_dir=str(series_dir.resolve()),
                png_dir=str(png_dir.resolve()),
                embedding_ref=embedding_ref,
                slice_count=42,
            )
        },
    )

    row = updated_df.iloc[0]
    assert row["radiology_embedding_paths"] == [embedding_ref]
    assert row["radiology_png_dirs"] == [
        "data/processes/radiology/chunk1/pngs/TCGA-KIRC/TCGA-AA-0001/1.2.3/4.5.6"
    ]
    assert row["radiology_series_slice_counts"] == [42]
    assert stats.series_artifacts_indexed == 1
    assert stats.cases_with_series_paths == 1
    assert stats.cases_with_matches == 1
    assert stats.matched_series_refs == 1


def test_register_radiology_series_artifacts_matches_zip_paths_and_masks(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    zip_path = root_dir / "data" / "raw" / "tcga" / "radiology" / "TCGA-KIRC" / "TCGA-AA-0001" / "1.2.3" / "4.5.6.zip"
    series_dir = root_dir / "data" / "processes" / "radiology" / "pngs" / "TCGA-KIRC" / "TCGA-AA-0001" / "1.2.3" / "4.5.6"
    mask_path = root_dir / "data" / "processes" / "radiology" / "masks" / "TCGA-KIRC" / "TCGA-AA-0001" / "1.2.3" / "4.5.6" / "slice.mask.png"
    manifest_path = mask_path.parent / "series_manifest.json"
    embedding_ref = "data/processes/radiology/features/features_tcga.h5::series=data/processes/radiology/pngs/TCGA-KIRC/TCGA-AA-0001/1.2.3/4.5.6"

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "tcga-case-1",
                "source": "tcga",
                "patient_id": "TCGA-AA-0001",
                "study_id": "case-1",
                "split": "train",
                "pathology_wsi_paths": [],
                "radiology_image_paths": [],
                "radiology_download_paths": [
                    "data/raw/tcga/radiology/TCGA-KIRC/TCGA-AA-0001/1.2.3/4.5.6.zip",
                ],
                "radiology_image_modalities": ["CT"],
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

    updated_df, _stats = register_radiology_series_artifacts(
        registry_df,
        root_dir=root_dir,
        artifacts_by_series_dir={
            str(zip_path.resolve()): RadiologySeriesArtifactRecord(
                series_dir=str(series_dir.resolve()),
                png_dir=str(series_dir.resolve()),
                embedding_ref=embedding_ref,
                slice_count=8,
                source_zip_path=str(zip_path.resolve()),
                mask_paths=(str(mask_path.resolve()),),
                mask_manifest_path=str(manifest_path.resolve()),
            )
        },
    )

    row = updated_df.iloc[0]
    assert row["radiology_embedding_paths"] == [embedding_ref]
    assert row["radiology_mask_paths"] == [
        "data/processes/radiology/masks/TCGA-KIRC/TCGA-AA-0001/1.2.3/4.5.6/slice.mask.png"
    ]
    assert row["radiology_mask_manifest_paths"] == [
        "data/processes/radiology/masks/TCGA-KIRC/TCGA-AA-0001/1.2.3/4.5.6/series_manifest.json"
    ]
