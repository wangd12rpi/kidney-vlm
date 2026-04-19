from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "02_radiology_proj" / "02_build_radiology_proj_train_qa.py"
    spec = importlib.util.spec_from_file_location("build_radiology_proj_train_qa_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_training_rows_prefers_registry_split() -> None:
    module = _load_script_module()

    registry_rows = [
        {
            "sample_id": "pmc_oa-sample-a",
            "source": "pmc_oa",
            "project_id": "pmc_oa",
            "patient_id": "PMC1",
            "study_id": "article-1",
            "split": "train",
            "radiology_image_paths": ["data/raw/pmc_oa/images/PMC1_fig1.jpg"],
            "radiology_image_modalities": ["figure"],
            "radiology_embedding_paths": [
                "data/processes/pmc_oa/pmc_oa_features.h5::sample=pmc_oa-sample-a::image=image_0"
            ],
        }
    ]
    caption_rows = [
        {
            "radiology_caption_row_id": "pmc_oa-sample-a::caption-1",
            "sample_id": "pmc_oa-sample-a",
            "source": "pmc_oa",
            "split": "validation",
            "caption_variant_index": 0,
            "image_name": "PMC1_fig1.jpg",
            "caption": "caption one",
            "answer": "caption one",
            "caption_model": "pmc_oa_human",
        }
    ]

    training_rows, split_mismatch_count = module._build_training_rows(
        registry_rows,
        caption_rows,
        default_instruction="Describe the radiology image.",
    )

    assert split_mismatch_count == 1
    assert training_rows[0]["qa_row_id"] == "pmc_oa-sample-a::radiology-qa-1"
    assert training_rows[0]["split"] == "train"
    assert training_rows[0]["series_stem"] == "PMC1_fig1"


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
