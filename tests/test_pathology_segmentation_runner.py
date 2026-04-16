from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from kidney_vlm.segmentation.pathology_segmentation_runner import (
    PendingPathologySegmentationCandidate,
    _build_overlay_rgb,
    _normalize_color_palette,
    _rebuild_case_pathology_metadata_paths,
    _rebuild_case_pathology_mask_paths,
    _rebuild_case_pathology_overlay_paths,
    _rebuild_case_pathology_slide_image_paths,
    _sort_and_limit_candidates,
)


def test_build_overlay_rgb_uses_multiclass_palette() -> None:
    rgb_image = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask_image = np.zeros((8, 8), dtype=np.uint8)
    mask_image[:, 4:] = 2
    palette = _normalize_color_palette(
        [
            [0, 0, 0],
            [255, 70, 70],
            [70, 160, 255],
        ]
    )

    overlay = _build_overlay_rgb(
        rgb_image,
        mask_image,
        class_palette_rgb=palette,
        overlay_alpha=0.5,
    )

    assert overlay.shape == rgb_image.shape
    assert np.array_equal(overlay[0, 0], np.array([255, 255, 255], dtype=np.uint8))
    assert not np.array_equal(overlay[0, 6], np.array([255, 255, 255], dtype=np.uint8))


def test_sort_and_limit_candidates_applies_deterministic_top_k() -> None:
    candidates = [
        PendingPathologySegmentationCandidate(
            row_idx=2,
            sample_id="sample-b",
            source="tcga",
            project_id="TCGA-KIRC",
            slide_path="b",
            slide_stem="slide-b",
            pathology_file_ids=(),
            local_slide_path="/tmp/b.svs",
        ),
        PendingPathologySegmentationCandidate(
            row_idx=1,
            sample_id="sample-a",
            source="tcga",
            project_id="TCGA-KICH",
            slide_path="a",
            slide_stem="slide-a",
            pathology_file_ids=(),
            local_slide_path="/tmp/a.svs",
        ),
        PendingPathologySegmentationCandidate(
            row_idx=3,
            sample_id="sample-c",
            source="tcga",
            project_id="TCGA-KIRC",
            slide_path="c",
            slide_stem="slide-c",
            pathology_file_ids=(),
            local_slide_path="/tmp/c.svs",
        ),
    ]

    selected = _sort_and_limit_candidates(candidates, top_k_slides=2)

    assert [candidate.sample_id for candidate in selected] == ["sample-a", "sample-b"]


def test_rebuild_case_pathology_mask_paths_uses_slide_order_and_kind_filter(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(mask_dir / "TCGA-AB-1234-01Z-00-DX1.png")

    row = pd.Series(
        {
            "pathology_wsi_paths": [
                "data/raw/pathology/TCGA-AB-1234-01Z-00-DX1.svs",
                "data/raw/pathology/TCGA-AB-1234-01Z-00-TS1.svs",
            ]
        }
    )

    mask_paths = _rebuild_case_pathology_mask_paths(
        row,
        mask_dir=mask_dir,
        allowed_slide_kinds={"DX"},
    )

    assert len(mask_paths) == 1
    assert mask_paths[0].endswith("TCGA-AB-1234-01Z-00-DX1.png")


def test_rebuild_case_pathology_segmentation_auxiliary_paths_follow_slide_order(tmp_path: Path) -> None:
    slide_image_dir = tmp_path / "slide_images"
    overlay_dir = tmp_path / "overlays"
    metadata_dir = tmp_path / "metadata"
    for directory in (slide_image_dir, overlay_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)).save(slide_image_dir / "TCGA-AB-1234-01Z-00-DX1.png")
    Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8)).save(overlay_dir / "TCGA-AB-1234-01Z-00-DX1.png")
    (metadata_dir / "TCGA-AB-1234-01Z-00-DX1.json").write_text("{\"ok\": true}")

    row = pd.Series(
        {
            "pathology_wsi_paths": [
                "data/raw/pathology/TCGA-AB-1234-01Z-00-DX1.svs",
                "data/raw/pathology/TCGA-AB-1234-01Z-00-TS1.svs",
            ]
        }
    )

    slide_image_paths = _rebuild_case_pathology_slide_image_paths(
        row,
        slide_image_dir=slide_image_dir,
        allowed_slide_kinds={"DX"},
    )
    overlay_paths = _rebuild_case_pathology_overlay_paths(
        row,
        overlay_dir=overlay_dir,
        allowed_slide_kinds={"DX"},
    )
    metadata_paths = _rebuild_case_pathology_metadata_paths(
        row,
        metadata_dir=metadata_dir,
        allowed_slide_kinds={"DX"},
    )

    assert len(slide_image_paths) == 1
    assert slide_image_paths[0].endswith("TCGA-AB-1234-01Z-00-DX1.png")
    assert len(overlay_paths) == 1
    assert overlay_paths[0].endswith("TCGA-AB-1234-01Z-00-DX1.png")
    assert len(metadata_paths) == 1
    assert metadata_paths[0].endswith("TCGA-AB-1234-01Z-00-DX1.json")
