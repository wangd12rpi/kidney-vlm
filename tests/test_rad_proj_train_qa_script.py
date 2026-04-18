from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rad_proj_train" / "01_build_rad_proj_train_qa.py"
    spec = importlib.util.spec_from_file_location("build_rad_proj_train_qa_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_split_name_maps_valid_to_validation() -> None:
    module = _load_script_module()

    assert module._normalize_split_name("valid") == "validation"
    assert module._normalize_split_name("test") == "test"
    assert module._normalize_split_name("train") == "train"


def test_build_training_rows_joins_by_image_name(tmp_path: Path) -> None:
    module = _load_script_module()

    caption_frame = pd.DataFrame(
        [
            {
                "image": "PMC1_fig1.jpg",
                "caption": "Caption one.",
                "pmcid": "PMC1",
                "url_name": "url-1.jpg",
                "split": "train",
            },
            {
                "image": "PMC2_fig2.jpg",
                "caption": "Caption two.",
                "pmcid": "PMC2",
                "url_name": "url-2.jpg",
                "split": "validation",
            },
        ]
    )
    feature_lookup = {
        "PMC1_fig1.jpg": {
            "sample_id": "pmc_oa-sample-a",
            "embedding_ref": "data/processes/pmc_oa/pmc_oa_features.h5::sample=pmc_oa-sample-a::image=image_0",
        },
        "PMC2_fig2.jpg": {
            "sample_id": "pmc_oa-sample-b",
            "embedding_ref": "data/processes/pmc_oa/pmc_oa_features.h5::sample=pmc_oa-sample-b::image=image_0",
        },
    }

    rows, missing_features, missing_images = module._build_training_rows(
        caption_frame,
        feature_index_by_image_name=feature_lookup,
        image_root_dir=module.ROOT / "images",
        require_existing_image_files=False,
        default_instruction="Describe the radiology image.",
    )

    assert missing_features == []
    assert missing_images == []
    assert rows[0]["qa_row_id"] == "pmc_oa-sample-a"
    assert rows[0]["radiology_image_paths"] == ["images/PMC1_fig1.jpg"]
    assert rows[0]["radiology_embedding_paths"] == [
        "data/processes/pmc_oa/pmc_oa_features.h5::sample=pmc_oa-sample-a::image=image_0"
    ]
    assert rows[1]["split"] == "validation"
    assert rows[1]["answer"] == "Caption two."


def test_build_output_frame_deduplicates_on_qa_row_id() -> None:
    module = _load_script_module()

    existing_output = pd.DataFrame(
        [
            {"qa_row_id": "row-1", "caption": "old"},
            {"qa_row_id": "row-2", "caption": "keep"},
        ]
    )
    generated_rows = [
        {"qa_row_id": "row-1", "caption": "new"},
        {"qa_row_id": "row-3", "caption": "added"},
    ]

    final_df = module._build_output_frame(
        existing_output=existing_output,
        generated_rows=generated_rows,
        overwrite_output=False,
    )

    assert final_df["qa_row_id"].tolist() == ["row-2", "row-1", "row-3"]
    assert final_df["caption"].tolist() == ["keep", "new", "added"]
