from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import h5py
import pandas as pd


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "01_pathology_proj" / "03_build_path_proj_train_qa.py"
    spec = importlib.util.spec_from_file_location("path_proj_train_qa_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expand_case_row_to_slide_rows_uses_single_embedding_per_slide(tmp_path: Path) -> None:
    module = _load_script_module()

    existing_tile_path = tmp_path / "TCGA-AA-0001-01Z-00-DX1.dx-uuid.h5"
    existing_tile_path.write_bytes(b"feature-bytes")
    missing_tile_path = tmp_path / "TCGA-AA-0001-01A-01-TS1.ts-uuid.h5"

    row = {
        "sample_id": "tcga-case-1",
        "pathology_wsi_paths": [
            "data/raw/tcga/pathology/TCGA-KIRC/TCGA-AA-0001/TCGA-AA-0001-01Z-00-DX1.dx-uuid.svs",
            "data/raw/tcga/pathology/TCGA-KIRC/TCGA-AA-0001/TCGA-AA-0001-01A-01-TS1.ts-uuid.svs",
        ],
        "pathology_tile_embedding_paths": [str(existing_tile_path), str(missing_tile_path)],
        "pathology_slide_embedding_paths": [
            "data/features/slide_features_titan/TCGA-AA-0001-01Z-00-DX1.dx-uuid.h5",
            "data/features/slide_features_titan/TCGA-AA-0001-01A-01-TS1.ts-uuid.h5",
        ],
    }

    slide_rows = module._expand_case_row_to_slide_rows(
        row,
        require_existing_patch_embedding_files=True,
    )

    assert len(slide_rows) == 1
    assert slide_rows[0]["slide_stem"] == "TCGA-AA-0001-01Z-00-DX1.dx-uuid"
    assert slide_rows[0]["pathology_tile_embedding_paths"] == [str(existing_tile_path)]
    assert slide_rows[0]["pathology_wsi_paths"] == [
        "data/raw/tcga/pathology/TCGA-KIRC/TCGA-AA-0001/TCGA-AA-0001-01Z-00-DX1.dx-uuid.svs"
    ]
    assert slide_rows[0]["pathology_slide_embedding_paths"] == [
        "data/features/slide_features_titan/TCGA-AA-0001-01Z-00-DX1.dx-uuid.h5"
    ]


def test_expand_slide_rows_to_training_rows_cartesian_products_case_captions() -> None:
    module = _load_script_module()

    slide_rows = [
        {
            "sample_id": "tcga-case-1",
            "source": "tcga",
            "project_id": "TCGA-KIRC",
            "patient_id": "patient-1",
            "study_id": "study-1",
            "split": "train",
            "slide_stem": "slide-a",
            "slide_index": 0,
            "pathology_tile_embedding_paths": ["a.h5"],
            "pathology_wsi_paths": ["a.svs"],
            "pathology_slide_embedding_paths": ["a-slide.h5"],
        },
        {
            "sample_id": "tcga-case-1",
            "source": "tcga",
            "project_id": "TCGA-KIRC",
            "patient_id": "patient-1",
            "study_id": "study-1",
            "split": "train",
            "slide_stem": "slide-b",
            "slide_index": 1,
            "pathology_tile_embedding_paths": ["b.h5"],
            "pathology_wsi_paths": ["b.svs"],
            "pathology_slide_embedding_paths": ["b-slide.h5"],
        },
    ]
    case_caption_rows = [
        {
            "case_caption_row_id": "tcga-case-1::caption-1",
            "sample_id": "tcga-case-1",
            "source": "tcga",
            "caption_variant_index": 0,
            "caption_prompt_variant": "variant 1",
            "caption_length_instruction": "4-6 sentences",
            "report_pdf_paths": ["report-1.pdf"],
            "instruction": "Describe the pathology case.",
            "caption": "caption one",
            "answer": "caption one",
            "caption_model": "gpt-test",
        },
        {
            "case_caption_row_id": "tcga-case-1::caption-2",
            "sample_id": "tcga-case-1",
            "source": "tcga",
            "caption_variant_index": 1,
            "caption_prompt_variant": "variant 2",
            "caption_length_instruction": "4-6 sentences",
            "report_pdf_paths": ["report-1.pdf"],
            "instruction": "Describe the pathology case.",
            "caption": "caption two",
            "answer": "caption two",
            "caption_model": "gpt-test",
        },
    ]

    training_rows = module._expand_slide_rows_to_training_rows(
        slide_rows,
        case_caption_rows,
        default_instruction="Describe the pathology image.",
    )

    assert len(training_rows) == 4
    assert training_rows[0]["qa_row_id"] == "tcga-case-1::slide-a::caption-1"
    assert training_rows[1]["qa_row_id"] == "tcga-case-1::slide-a::caption-2"
    assert training_rows[2]["qa_row_id"] == "tcga-case-1::slide-b::caption-1"
    assert training_rows[3]["qa_row_id"] == "tcga-case-1::slide-b::caption-2"
    assert training_rows[3]["caption"] == "caption two"


def test_existing_output_row_id_falls_back_to_slide_and_variant_columns() -> None:
    module = _load_script_module()

    row = {
        "sample_id": "tcga-case-1",
        "pathology_tile_embedding_paths": ["data/features/features_conch_v15/slide-a.h5"],
        "caption_variant_index": 2,
    }

    row_id = module._existing_output_row_id(row)

    assert row_id == "tcga-case-1::slide-a::caption-3"


def test_build_output_frame_deduplicates_on_qa_row_id() -> None:
    module = _load_script_module()

    existing_output = module.pd.DataFrame(
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


def test_prepare_training_frame_excludes_normal_tcga_slides() -> None:
    module = _load_script_module()

    frame = pd.DataFrame(
        [
            {
                "source": "tcga",
                "slide_stem": "TCGA-AA-0001-11A-01-BS1.normal",
                "pathology_patch_count": 100,
            },
            {
                "source": "tcga",
                "slide_stem": "TCGA-AA-0001-01A-01-BS1.tumor",
                "pathology_patch_count": 120,
            },
            {
                "source": "other",
                "slide_stem": "OTHER-CASE-11A-slide",
                "pathology_patch_count": 140,
            },
        ]
    )

    filtered, stats = module._prepare_training_frame(
        frame,
        exclude_normal_tcga_slides=True,
        patch_count_lower_quantile=None,
        patch_count_upper_quantile=None,
    )

    assert filtered["slide_stem"].tolist() == [
        "TCGA-AA-0001-01A-01-BS1.tumor",
        "OTHER-CASE-11A-slide",
    ]
    assert stats["rows_removed_normal_tcga"] == 1


def test_prepare_training_frame_keeps_middle_patch_count_quantiles(tmp_path: Path) -> None:
    module = _load_script_module()

    patch_counts = [10, 20, 30, 40, 50]
    rows = []
    for idx, patch_count in enumerate(patch_counts, start=1):
        feature_path = tmp_path / f"slide-{idx}.h5"
        with h5py.File(feature_path, "w") as handle:
            handle.create_dataset("features", data=[[0.0] * 4 for _ in range(patch_count)])
            handle.create_dataset("coords", data=[[0, 0] for _ in range(patch_count)])
        rows.append(
            {
                "source": "tcga",
                "slide_stem": f"TCGA-AA-000{idx}-01A-01-BS1.slide-{idx}",
                "pathology_tile_embedding_paths": [str(feature_path)],
            }
        )

    frame = pd.DataFrame(rows)
    filtered, stats = module._prepare_training_frame(
        frame,
        exclude_normal_tcga_slides=False,
        patch_count_lower_quantile=0.05,
        patch_count_upper_quantile=0.95,
    )

    assert filtered["pathology_patch_count"].tolist() == [20, 30, 40]
    assert stats["rows_removed_patch_count_outliers"] == 2


def test_filter_missing_pathology_report_form_rows_excludes_sample_ids(monkeypatch) -> None:
    module = _load_script_module()

    def fake_sample_ids_with_missing_pathology_report_forms(
        rows,
        *,
        repo_root,
        sample_id_key="sample_id",
        report_paths_key="report_pdf_paths",
        progress_desc=None,
        total=None,
    ):
        return {"tcga-bad"}

    monkeypatch.setattr(
        module,
        "sample_ids_with_missing_pathology_report_forms",
        fake_sample_ids_with_missing_pathology_report_forms,
    )

    frame = module.pd.DataFrame(
        [
            {"sample_id": "tcga-good", "report_pdf_paths": ["good.pdf"]},
            {"sample_id": "tcga-bad", "report_pdf_paths": ["bad.pdf"]},
        ]
    )

    filtered, bad_sample_ids = module._filter_missing_pathology_report_form_rows(frame)

    assert bad_sample_ids == {"tcga-bad"}
    assert filtered["sample_id"].tolist() == ["tcga-good"]
