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
    script_path = repo_root / "scripts" / "10_mcq_from_caption_new" / "02_generate_vqa_from_condensed.py"
    spec = importlib.util.spec_from_file_location("generate_vqa_from_condensed_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(tmp_path: Path, *, require_test_artifacts: bool = False):
    return {
        "seed": 123,
        "show_progress": False,
        "radiology_png_root": "data/radiology_png",
        "max_test_radiology_images": 3,
        "populate_test_genomics_text_summaries": False,
        "require_test_pathology_roi_png_dir": require_test_artifacts,
        "require_test_radiology_view_png_dir": require_test_artifacts,
        "require_test_genomics_text_summaries": False,
        "distractors": {
            "pool_mode": "same_project",
            "use_bertscore": True,
            "candidate_pull_count": 3,
            "required_wrong_options": 3,
        },
        "sampling": {
            "enabled": False,
            "seed": 123,
            "split": "train",
            "modality_keep_ratios": {
                "all_available": 1.0,
                "path_only": 1.0,
                "radiology_only": 1.0,
            },
        },
        "bert_score": {"model_type": "dummy", "num_layers": 1},
        "tasks": [
            {
                "enabled": True,
                "section_key": "pathology_findings",
                "task_category": "caption_pathology",
                "task_id": "caption_pathology_findings",
                "question": "Which pathology finding best matches this case?",
            }
        ],
    }


def _condensed_frame():
    return pd.DataFrame(
        [
            {
                "case_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "radiology_findings": "Target radiology.",
                "pathology_findings": "Target discohesive tumor cords.",
                "genomic_findings": "Target PIK3CA mutation.",
                "integrated_interpretation": "Target integrated interpretation.",
            },
            {
                "case_id": "TCGA-AA-0002",
                "project_id": "TCGA-TEST",
                "radiology_findings": "Other radiology 2.",
                "pathology_findings": "Wrong solid nests.",
                "genomic_findings": "Wrong TP53.",
                "integrated_interpretation": "Wrong integrated 2.",
            },
            {
                "case_id": "TCGA-AA-0003",
                "project_id": "TCGA-TEST",
                "radiology_findings": "Other radiology 3.",
                "pathology_findings": "Wrong papillary fronds.",
                "genomic_findings": "Wrong CDH1.",
                "integrated_interpretation": "Wrong integrated 3.",
            },
            {
                "case_id": "TCGA-AA-0004",
                "project_id": "TCGA-TEST",
                "radiology_findings": "Other radiology 4.",
                "pathology_findings": "Wrong cribriform glands.",
                "genomic_findings": "Wrong KRAS.",
                "integrated_interpretation": "Wrong integrated 4.",
            },
        ]
    )


def _registry_frame(split: str = "train"):
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
                "radiology_embedding_paths": ["features/rad-2.h5::series=x"],
                "genomics_dna_methylation_feature_path": "",
                "genomics_rna_bulk_feature_path": "",
                "pathology_png_roi_paths": [],
            },
            {
                "patient_id": "TCGA-AA-0003",
                "project_id": "TCGA-TEST",
                "split": split,
                "pathology_tile_embedding_paths": ["features/path-3.h5"],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": ["features/rad-3.h5::series=x"],
                "genomics_dna_methylation_feature_path": "",
                "genomics_rna_bulk_feature_path": "",
                "pathology_png_roi_paths": [],
            },
            {
                "patient_id": "TCGA-AA-0004",
                "project_id": "TCGA-TEST",
                "split": split,
                "pathology_tile_embedding_paths": ["features/path-4.h5"],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": ["features/rad-4.h5::series=x"],
                "genomics_dna_methylation_feature_path": "",
                "genomics_rna_bulk_feature_path": "",
                "pathology_png_roi_paths": [],
            },
        ]
    )


def test_build_caption_condensed_mcq_frame_uses_exact_vqa_schema_and_option_labels(monkeypatch, tmp_path):
    module = _load_script_module()
    monkeypatch.setattr(module, "_bertscore_f1_pairs", lambda correct, candidates, cfg: [0.3, 0.1, 0.2])

    frame, stats = module.build_caption_condensed_mcq_frame(
        condensed_df=_condensed_frame(),
        registry_df=_registry_frame(),
        cfg=_cfg(tmp_path),
    )

    assert list(frame.columns) == VQA_COLUMNS
    assert stats["generated_semantic_questions"] == 4
    assert frame["question_type"].eq("mcq").all()
    assert frame["generation_type"].eq("from_caption").all()
    assert frame["caption_id"].eq(frame["case_id"]).all()
    assert frame["ground_truth_source"].eq("condensed_caption:pathology_findings").all()
    for _, row in frame.iterrows():
        options = [row["option_a"], row["option_b"], row["option_c"], row["option_d"]]
        assert row["answer"] in options
        assert row["answer_label"] == "ABCD"[options.index(row["answer"])]
        assert isinstance(row["pathology_feature_paths"], list)
        assert isinstance(row["radiology_feature_paths"], list)
        assert isinstance(row["dnam_feature_path"], str)
        assert isinstance(row["rna_feature_path"], str)


def test_empty_condensed_section_is_skipped(monkeypatch, tmp_path):
    module = _load_script_module()
    monkeypatch.setattr(module, "_bertscore_f1_pairs", lambda correct, candidates, cfg: [0.1, 0.2, 0.3])
    condensed = _condensed_frame()
    condensed.loc[0, "pathology_findings"] = ""

    frame, stats = module.build_caption_condensed_mcq_frame(
        condensed_df=condensed,
        registry_df=_registry_frame(),
        cfg=_cfg(tmp_path),
    )

    assert "TCGA-AA-0001" not in set(frame["case_id"])
    assert stats["skipped_empty_section"] == 1


def test_same_project_distractor_pool_excludes_target_and_other_projects(tmp_path):
    module = _load_script_module()
    condensed = pd.concat(
        [
            _condensed_frame(),
            pd.DataFrame(
                [
                    {
                        "case_id": "TCGA-BB-0001",
                        "project_id": "TCGA-OTHER",
                        "radiology_findings": "",
                        "pathology_findings": "Wrong other project.",
                        "genomic_findings": "",
                        "integrated_interpretation": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    candidates = module._sample_candidate_distractors(
        condensed_df=condensed,
        target_case_id="TCGA-AA-0001",
        target_project_id="TCGA-TEST",
        section_key="pathology_findings",
        correct_answer="Target discohesive tumor cords.",
        pool_mode="same_project",
        candidate_pull_count=10,
        seed=1,
    )

    assert "Target discohesive tumor cords." not in candidates
    assert "Wrong other project." not in candidates
    assert set(candidates) == {"Wrong solid nests.", "Wrong papillary fronds.", "Wrong cribriform glands."}


def test_lowest_bertscore_candidates_are_selected(monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(module, "_bertscore_f1_pairs", lambda correct, candidates, cfg: [0.9, 0.1, 0.4, 0.2])

    selected = module._select_wrong_options(
        correct_answer="correct",
        candidates=["too close", "best wrong", "middle", "second wrong"],
        cfg={},
        required_count=3,
        use_bertscore=True,
    )

    assert selected == ["best wrong", "second wrong", "middle"]


def test_bertscore_can_be_disabled_for_fast_testing(monkeypatch):
    module = _load_script_module()

    def _raise_if_called(correct, candidates, cfg):
        raise AssertionError("BERTScore should not be called when disabled.")

    monkeypatch.setattr(module, "_bertscore_f1_pairs", _raise_if_called)
    selected = module._select_wrong_options(
        correct_answer="correct",
        candidates=["first", "second", "third", "fourth"],
        cfg={},
        required_count=3,
        use_bertscore=False,
    )

    assert selected == ["first", "second", "third"]


def test_test_fallback_fields_use_schema_string_conventions(monkeypatch, tmp_path):
    module = _load_script_module()
    monkeypatch.setenv("KIDNEY_VLM_ROOT", str(tmp_path))
    monkeypatch.setattr(module, "_bertscore_f1_pairs", lambda correct, candidates, cfg: [0.1, 0.2, 0.3])
    series_dir = tmp_path / "data/radiology_png/TCGA-TEST/TCGA-AA-0001/study/series"
    series_dir.mkdir(parents=True)
    for index in range(5):
        (series_dir / f"{index:08d}.png").write_bytes(b"png")

    frame, _ = module.build_caption_condensed_mcq_frame(
        condensed_df=_condensed_frame(),
        registry_df=_registry_frame(split="test"),
        cfg=_cfg(tmp_path, require_test_artifacts=True),
    )

    row = frame[(frame["case_id"].eq("TCGA-AA-0001")) & (frame["modality_combination_name"].eq("all_available"))].iloc[0]
    assert row["pathology_roi_png_dir"] == "data/pathology_png/TCGA-AA-0001"
    assert isinstance(row["radiology_view_png_dir"], str)
    pngs = json.loads(row["radiology_view_png_dir"])
    assert len(pngs) == 3
    assert all(path.startswith("data/radiology_png/TCGA-TEST/TCGA-AA-0001/study/series/") for path in pngs)


def test_sampling_outputs_train_subset_by_modality_and_keeps_test(monkeypatch, tmp_path):
    module = _load_script_module()
    monkeypatch.setattr(module, "_bertscore_f1_pairs", lambda correct, candidates, cfg: [0.1, 0.2, 0.3])
    train_frame, _ = module.build_caption_condensed_mcq_frame(
        condensed_df=_condensed_frame(),
        registry_df=_registry_frame(split="train"),
        cfg=_cfg(tmp_path),
    )
    test_frame, _ = module.build_caption_condensed_mcq_frame(
        condensed_df=_condensed_frame(),
        registry_df=_registry_frame(split="test"),
        cfg=_cfg(tmp_path),
    )
    full = pd.concat([train_frame, test_frame], ignore_index=True)
    cfg = _cfg(tmp_path)
    cfg["sampling"] = {
        "enabled": True,
        "seed": 123,
        "split": "train",
        "modality_keep_ratios": {
            "all_available": 1.0,
            "path_only": 0.0,
            "radiology_only": 1.0,
        },
    }

    sampled, stats = module.sample_train_rows_by_modality(full, cfg=cfg)

    train_rows = sampled[sampled["split"].eq("train")]
    test_rows = sampled[sampled["split"].eq("test")]
    assert "path_only" not in set(train_rows["modality_combination_name"])
    assert "path_only" in set(test_rows["modality_combination_name"])
    assert stats["sampled_out_rows"] == int(train_frame["modality_combination_name"].eq("path_only").sum())
