from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from kidney_vlm.vqa.schema import VQA_COLUMNS


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "05_vqa_question_generation" / "fix_caption_qa.py"
    spec = importlib.util.spec_from_file_location("fix_caption_qa_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_row(case_id: str, question_type: str, task_id: str = "CO3") -> dict:
    return {
        "case_id": case_id,
        "project_id": "TCGA-TEST",
        "question_id": 1,
        "base_question_id": 1,
        "split": "train",
        "question_type": question_type,
        "generation_type": "from_caption",
        "task_category": "cross_modal_synthesis",
        "task_id": task_id,
        "modality_combination_name": "bad_legacy_value",
        "use_pathology": False,
        "use_radiology": False,
        "use_dnam": False,
        "use_rna": False,
        "question": "How do the findings correlate?",
        "option_a": "",
        "option_b": "",
        "option_c": "",
        "option_d": "",
        "answer": "Pathology and molecular findings are concordant.",
        "answer_label": "",
        "caption_id": f"caption-{case_id}",
        "ground_truth_source": "legacy_caption",
        "radiology_biomarker": "",
        "pathology_feature_paths": [],
        "radiology_feature_paths": [],
        "dnam_feature_path": "",
        "rna_feature_path": "",
        "pathology_roi_png_dir": "",
        "radiology_view_png_dir": "",
        "dnam_text_summary": "",
        "rna_text_summary": "",
    }


def _registry_frame(split: str = "train") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": split,
                "pathology_tile_embedding_paths": ["features/path-1.h5"],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": [
                    "features/rad.h5::series=data/processes/radiology/chunk/pngs/TCGA-TEST/TCGA-AA-0001/study/series"
                ],
                "genomics_dna_methylation_feature_path": "features/dnam-1.pt",
                "genomics_rna_bulk_feature_path": "features/rna-1.pt",
                "pathology_png_roi_paths": [
                    "data/pathology_png/TCGA-AA-0001/TCGA-AA-0001__roi.png"
                ],
            },
            {
                "patient_id": "TCGA-AA-0002",
                "project_id": "TCGA-TEST",
                "split": split,
                "pathology_tile_embedding_paths": ["features/path-2.h5"],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": [],
                "genomics_dna_methylation_feature_path": "",
                "genomics_rna_bulk_feature_path": "",
                "pathology_png_roi_paths": [
                    "data/pathology_png/TCGA-AA-0002/TCGA-AA-0002__roi.png"
                ],
            },
        ]
    )


def _cfg(*, require_test_artifacts: bool = False, sampling_enabled: bool = False) -> dict:
    return {
        "seed": 42,
        "show_progress": False,
        "radiology_png_root": "data/radiology_png",
        "max_test_radiology_images": 3,
        "populate_test_genomics_text_summaries": False,
        "require_test_pathology_roi_png_dir": require_test_artifacts,
        "require_test_radiology_view_png_dir": require_test_artifacts,
        "require_test_genomics_text_summaries": False,
        "sampling": {
            "enabled": sampling_enabled,
            "seed": 42,
            "split": "train",
            "modality_keep_ratios": {
                "all_available": 1.0,
                "path_only": 0.0,
                "radiology_only": 1.0,
            },
        },
    }


def test_fix_caption_qa_uses_qa_only_and_rebuilds_modalities_from_unified() -> None:
    module = _load_script_module()
    legacy = pd.DataFrame(
        [
            _legacy_row("TCGA-AA-0001", "qa"),
            _legacy_row("TCGA-AA-0001", "mcq"),
            _legacy_row("TCGA-AA-0002", "qa", task_id="CO4"),
        ],
        columns=VQA_COLUMNS,
    )

    frame, stats = module.build_fixed_caption_qa_frame(
        legacy_vqa_df=legacy,
        registry_df=_registry_frame(),
        cfg=_cfg(),
    )

    assert list(frame.columns) == VQA_COLUMNS
    assert stats["legacy_qa_rows"] == 2
    assert frame["question_type"].eq("qa").all()
    assert frame["generation_type"].eq("from_caption").all()
    assert set(frame["case_id"]) == {"TCGA-AA-0001", "TCGA-AA-0002"}
    case1 = frame[frame["case_id"].eq("TCGA-AA-0001")]
    assert sorted(case1["modality_combination_name"].tolist()) == [
        "all_available",
        "path_only",
        "radiology_only",
    ]
    all_row = case1[case1["modality_combination_name"].eq("all_available")].iloc[0]
    assert bool(all_row["use_pathology"]) is True
    assert bool(all_row["use_radiology"]) is True
    assert bool(all_row["use_dnam"]) is True
    assert bool(all_row["use_rna"]) is True
    path_row = frame[
        frame["case_id"].eq("TCGA-AA-0002")
        & frame["modality_combination_name"].eq("path_only")
    ].iloc[0]
    assert bool(path_row["use_pathology"]) is True
    assert bool(path_row["use_radiology"]) is False
    assert path_row["pathology_feature_paths"] == ["features/path-2.h5"]
    assert path_row["radiology_feature_paths"] == []


def test_test_fallback_fields_keep_schema_string_conventions(monkeypatch, tmp_path) -> None:
    module = _load_script_module()
    monkeypatch.setenv("KIDNEY_VLM_ROOT", str(tmp_path))
    series_dir = tmp_path / "data/radiology_png/TCGA-TEST/TCGA-AA-0001/study/series"
    series_dir.mkdir(parents=True)
    for index in range(5):
        (series_dir / f"{index:08d}.png").write_bytes(b"png")
    legacy = pd.DataFrame([_legacy_row("TCGA-AA-0001", "qa")], columns=VQA_COLUMNS)

    frame, _ = module.build_fixed_caption_qa_frame(
        legacy_vqa_df=legacy,
        registry_df=_registry_frame(split="test"),
        cfg=_cfg(require_test_artifacts=True),
    )

    row = frame[frame["modality_combination_name"].eq("all_available")].iloc[0]
    assert row["pathology_roi_png_dir"] == "data/pathology_png/TCGA-AA-0001"
    assert isinstance(row["radiology_view_png_dir"], str)
    pngs = json.loads(row["radiology_view_png_dir"])
    assert len(pngs) == 3
    assert all(path.startswith("data/radiology_png/TCGA-TEST/TCGA-AA-0001/study/series/") for path in pngs)


def test_sampling_only_applies_to_train_and_each_modality_ratio() -> None:
    module = _load_script_module()
    train = pd.DataFrame([_legacy_row("TCGA-AA-0001", "qa")], columns=VQA_COLUMNS)
    full_train, _ = module.build_fixed_caption_qa_frame(
        legacy_vqa_df=train,
        registry_df=_registry_frame(split="train"),
        cfg=_cfg(),
    )
    test = pd.DataFrame([_legacy_row("TCGA-AA-0001", "qa")], columns=VQA_COLUMNS)
    full_test, _ = module.build_fixed_caption_qa_frame(
        legacy_vqa_df=test,
        registry_df=_registry_frame(split="test"),
        cfg=_cfg(require_test_artifacts=False),
    )
    full = pd.concat([full_train, full_test], ignore_index=True)

    sampled, stats = module.sample_train_rows_by_modality(full, cfg=_cfg(sampling_enabled=True))

    train_rows = sampled[sampled["split"].eq("train")]
    test_rows = sampled[sampled["split"].eq("test")]
    assert "path_only" not in set(train_rows["modality_combination_name"])
    assert "path_only" in set(test_rows["modality_combination_name"])
    assert stats["sampled_out_rows"] == 1
