from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from kidney_vlm.data.sources.pmc_oa import (
    build_pmc_oa_caption_rows,
    build_pmc_oa_registry_rows,
    normalize_pmc_oa_split_name,
)


def test_normalize_pmc_oa_split_name_maps_validation_to_val() -> None:
    assert normalize_pmc_oa_split_name("validation") == "val"
    assert normalize_pmc_oa_split_name("valid") == "val"
    assert normalize_pmc_oa_split_name("train") == "train"
    assert normalize_pmc_oa_split_name("test") == "test"


def _make_local_tmpdir(name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_root = repo_root / ".test_artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    path = artifacts_root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_build_pmc_oa_caption_rows_and_registry_rows() -> None:
    root_dir = _make_local_tmpdir("pmc_oa_source")
    try:
        image_root_dir = root_dir / "data" / "raw" / "pmc_oa" / "images"
        image_root_dir.mkdir(parents=True, exist_ok=True)
        (image_root_dir / "PMC1_fig1.jpg").write_bytes(b"jpg")

        caption_frame = pd.DataFrame(
            [
                {
                    "image": "PMC1_fig1.jpg",
                    "caption": "Example radiology caption.",
                    "split": "validation",
                    "pmcid": "PMC1",
                    "url_name": "article-1",
                }
            ]
        )
        feature_index_by_image_name = {
            "PMC1_fig1.jpg": {
                "sample_id": "pmc_oa-sample-a",
                "image_key": "image_0",
                "embedding_ref": "data/processes/pmc_oa/pmc_oa_features.h5::sample=pmc_oa-sample-a::image=image_0",
            }
        }

        caption_rows, missing_feature_images, missing_image_files = build_pmc_oa_caption_rows(
            caption_frame,
            root_dir=root_dir,
            feature_index_by_image_name=feature_index_by_image_name,
            image_root_dir=image_root_dir,
            require_existing_image_files=True,
            default_instruction="Describe the radiology image.",
        )

        assert missing_feature_images == []
        assert missing_image_files == []
        assert len(caption_rows) == 1
        assert caption_rows[0]["split"] == "val"
        assert caption_rows[0]["radiology_image_paths"] == ["data/raw/pmc_oa/images/PMC1_fig1.jpg"]

        registry_df = build_pmc_oa_registry_rows(caption_rows)

        assert registry_df["sample_id"].tolist() == ["pmc_oa-sample-a"]
        assert registry_df["source"].tolist() == ["pmc_oa"]
        assert registry_df["split"].tolist() == ["val"]
        assert registry_df["radiology_embedding_paths"].tolist() == [
            ["data/processes/pmc_oa/pmc_oa_features.h5::sample=pmc_oa-sample-a::image=image_0"]
        ]
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)
