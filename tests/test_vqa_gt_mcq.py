from __future__ import annotations

import pandas as pd
import pytest

from kidney_vlm.vqa import genomics_text_summary
from kidney_vlm.vqa.gt_mcq import build_ground_truth_mcq_frame


def test_build_ground_truth_mcq_frame_creates_paired_modality_rows() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "test",
                "task_stage_label": "Stage II",
                "pathology_tile_embedding_paths": [
                    "features/path-a.h5",
                    "features/path-b.h5",
                ],
                "radiology_embedding_paths": [
                    "features/rad-a.h5::series=a",
                    "features/rad-b.h5::series=b",
                ],
                "radiology_biomarker": "largest lesion diameter: 3.1 cm",
                "genomics_dna_methylation_feature_path": "features/dnam-a.pt",
                "genomics_rna_bulk_feature_path": "features/rna-a.pt",
                "pathology_png_roi_paths": [
                    "data/pathology_png/TCGA-AA-0001/TCGA-AA-0001-01Z-00-DX1__uniform_tumor_8k__roi.png"
                ],
            }
        ]
    )
    cfg = {
        "choice_count": 4,
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "ground_truth_source": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "modality_combination_overrides": [
                    {
                        "name": "path_only",
                        "use_pathology": "must_have",
                        "use_radiology": "not_include",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                    {
                        "name": "path_dnam_rna",
                        "use_pathology": "must_have",
                        "use_radiology": "not_include",
                        "use_dnam": "must_have",
                        "use_rna": "must_have",
                    },
                    {
                        "name": "radiology_only",
                        "use_pathology": "not_include",
                        "use_radiology": "must_have",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                ],
            }
        ],
        "boolean_tasks": [],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert len(frame) == 3
    assert stats["semantic_questions"] == 1
    assert frame["base_question_id"].nunique() == 1
    assert frame["question_id"].nunique() == 3
    assert frame["answer"].tolist() == ["Stage II", "Stage II", "Stage II"]
    assert frame["answer_label"].tolist() == ["B", "B", "B"]
    assert frame[
        ["option_a", "option_b", "option_c", "option_d"]
    ].drop_duplicates().values.tolist() == [
        ["Stage I", "Stage II", "Stage III", "Stage IV"]
    ]
    assert frame["ground_truth_source"].tolist() == [
        "task_stage_label",
        "task_stage_label",
        "task_stage_label",
    ]
    assert sorted(
        zip(
            frame["use_pathology"],
            frame["use_radiology"],
            frame["use_dnam"],
            frame["use_rna"],
        )
    ) == [
        (False, True, False, False),
        (True, False, False, False),
        (True, False, True, True),
    ]
    for _, row in frame[frame["use_pathology"]].iterrows():
        assert row["pathology_feature_paths"] == [
            "features/path-a.h5",
            "features/path-b.h5",
        ]
        assert row["pathology_roi_png_dir"] == "data/pathology_png/TCGA-AA-0001"
    for _, row in frame[frame["use_radiology"]].iterrows():
        assert row["radiology_feature_paths"] == [
            "features/rad-a.h5::series=a",
            "features/rad-b.h5::series=b",
        ]
        assert row["radiology_biomarker"] == "largest lesion diameter: 3.1 cm"
    for _, row in frame.iterrows():
        assert row["answer"] in [
            row["option_a"],
            row["option_b"],
            row["option_c"],
            row["option_d"],
        ]


def test_build_ground_truth_mcq_frame_applies_modality_requirement_states() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage I",
                "pathology_tile_embedding_paths": ["features/path-a.h5"],
            },
            {
                "sample_id": "TCGA-AA-0002",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage II",
                "pathology_tile_embedding_paths": ["features/path-b.h5"],
                "radiology_embedding_paths": ["features/rad-b.h5"],
            },
            {
                "sample_id": "TCGA-AA-0003",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage III",
                "radiology_embedding_paths": ["features/rad-c.h5"],
            },
        ]
    )
    cfg = {
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "modality_combination_overrides": [
                    {
                        "name": "path_rad_if_available",
                        "use_pathology": "must_have",
                        "use_radiology": "use_if_avail",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                    {
                        "name": "radiology_only",
                        "use_pathology": "not_include",
                        "use_radiology": "must_have",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                    {
                        "name": "any_available",
                        "use_pathology": "use_if_avail",
                        "use_radiology": "use_if_avail",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                ],
            }
        ],
        "boolean_tasks": [],
    }

    frame, _ = build_ground_truth_mcq_frame(registry, cfg)

    assert sorted(
        zip(
            frame["case_id"],
            frame["modality_combination_name"],
            frame["use_pathology"],
            frame["use_radiology"],
        )
    ) == [
        ("TCGA-AA-0001", "any_available", True, False),
        ("TCGA-AA-0001", "path_rad_if_available", True, False),
        ("TCGA-AA-0002", "any_available", True, True),
        ("TCGA-AA-0002", "path_rad_if_available", True, True),
        ("TCGA-AA-0002", "radiology_only", False, True),
        ("TCGA-AA-0003", "any_available", False, True),
        ("TCGA-AA-0003", "radiology_only", False, True),
    ]
    assert frame["question_id"].is_unique


def test_build_ground_truth_mcq_sampling_protects_radiology_base_questions() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage I",
                "pathology_tile_embedding_paths": ["features/path-a.h5"],
            },
            {
                "sample_id": "TCGA-AA-0002",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage II",
                "pathology_tile_embedding_paths": ["features/path-b.h5"],
                "radiology_embedding_paths": ["features/rad-b.h5"],
            },
            {
                "sample_id": "TCGA-AA-0003",
                "project_id": "TCGA-TEST",
                "split": "val",
                "task_stage_label": "Stage III",
                "pathology_tile_embedding_paths": ["features/path-c.h5"],
            },
        ]
    )
    cfg = {
        "sampling": {
            "enabled": True,
            "seed": 123,
            "splits": ["train"],
            "protect_radiology_questions": True,
            "task_category_modality_keep_ratios": {
                "stage": {
                    "all_available": 0.0,
                    "path_only": 0.0,
                    "radiology_only": 0.0,
                }
            },
        },
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "modality_combination_overrides": [
                    {
                        "name": "all_available",
                        "use_pathology": "must_have",
                        "use_radiology": "use_if_avail",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                    {
                        "name": "path_only",
                        "use_pathology": "must_have",
                        "use_radiology": "not_include",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                    {
                        "name": "radiology_only",
                        "use_pathology": "not_include",
                        "use_radiology": "must_have",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                ],
            }
        ],
        "boolean_tasks": [],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert set(frame["case_id"]) == {"TCGA-AA-0002", "TCGA-AA-0003"}
    assert sorted(
        frame.loc[
            frame["case_id"].eq("TCGA-AA-0002"),
            "modality_combination_name",
        ]
    ) == [
        "all_available",
        "path_only",
        "radiology_only",
    ]
    assert stats["sampling"]["sampled_out_semantic_questions"] == 1
    assert stats["sampling"]["sampling_protected_radiology_questions"] == 1


def test_build_ground_truth_mcq_sampling_supports_modality_ratios() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage I",
                "pathology_tile_embedding_paths": ["features/path-a.h5"],
                "radiology_embedding_paths": ["features/rad-a.h5"],
            }
        ]
    )
    cfg = {
        "sampling": {
            "enabled": True,
            "seed": 123,
            "splits": ["train"],
            "protect_radiology_questions": False,
            "task_category_modality_keep_ratios": {
                "stage": {
                    "all_available": 1.0,
                    "path_only": 0.0,
                    "radiology_only": 1.0,
                }
            },
        },
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "modality_combination_overrides": [
                    {
                        "name": "all_available",
                        "use_pathology": "must_have",
                        "use_radiology": "use_if_avail",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                    {
                        "name": "path_only",
                        "use_pathology": "must_have",
                        "use_radiology": "not_include",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                    {
                        "name": "radiology_only",
                        "use_pathology": "not_include",
                        "use_radiology": "must_have",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                ],
            }
        ],
        "boolean_tasks": [],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert set(frame["modality_combination_name"]) == {
        "all_available",
        "radiology_only",
    }
    assert stats["sampling"]["sampled_out_rows"] == 1
    assert stats["sampling"]["sampled_out_semantic_questions"] == 0


def test_build_ground_truth_mcq_sampling_rejects_old_ratio_keys() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage I",
                "pathology_tile_embedding_paths": ["features/path-a.h5"],
            }
        ]
    )
    cfg = {
        "sampling": {
            "enabled": True,
            "seed": 123,
            "splits": ["train"],
            "protect_radiology_questions": False,
            "task_category_keep_ratios": {"stage": 0.0},
        },
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "modality_combination_overrides": [
                    {
                        "name": "path_only",
                        "use_pathology": "must_have",
                        "use_radiology": "not_include",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    }
                ],
            }
        ],
        "boolean_tasks": [],
    }

    with pytest.raises(ValueError, match="task_category_modality_keep_ratios"):
        build_ground_truth_mcq_frame(registry, cfg)


def test_build_ground_truth_mcq_frame_requires_modality_combination_names() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage I",
                "pathology_tile_embedding_paths": ["features/path-a.h5"],
            }
        ]
    )
    cfg = {
        "default_modality_combinations": [
            {
                "use_pathology": "must_have",
                "use_radiology": "not_include",
                "use_dnam": "not_include",
                "use_rna": "not_include",
            }
        ],
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
            }
        ],
        "boolean_tasks": [],
    }

    with pytest.raises(ValueError, match="must define a non-empty name"):
        build_ground_truth_mcq_frame(registry, cfg)


def test_build_ground_truth_mcq_frame_rejects_legacy_modality_booleans() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage I",
                "pathology_tile_embedding_paths": ["features/path-a.h5"],
            }
        ]
    )
    cfg = {
        "default_modality_combinations": [
            {
                "name": "legacy_bool",
                "use_pathology": True,
                "use_radiology": False,
                "use_dnam": False,
                "use_rna": False,
            }
        ],
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
            }
        ],
        "boolean_tasks": [],
    }

    with pytest.raises(ValueError, match="Unsupported modality requirement"):
        build_ground_truth_mcq_frame(registry, cfg)


def test_build_ground_truth_mcq_frame_supports_binary_mutation_tasks() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "train",
                "mutation_tp53": True,
                "genomics_dna_methylation_feature_path": "features/dnam-a.pt",
                "genomics_rna_bulk_feature_path": "features/rna-a.pt",
            },
            {
                "sample_id": "TCGA-AA-0002",
                "project_id": "TCGA-TEST",
                "split": "train",
                "mutation_tp53": False,
                "genomics_dna_methylation_feature_path": "features/dnam-b.pt",
                "genomics_rna_bulk_feature_path": "features/rna-b.pt",
            },
        ]
    )
    cfg = {
        "choice_count": 4,
        "default_modality_combinations": [
            {
                "name": "dnam_rna",
                "use_pathology": "not_include",
                "use_radiology": "not_include",
                "use_dnam": "must_have",
                "use_rna": "must_have",
            }
        ],
        "categorical_tasks": [],
        "boolean_tasks": [
            {
                "task_category": "mutation",
                "task_id_template": "{source_column}",
                "source_columns": ["mutation_tp53"],
                "question_template": "What is the {gene} mutation status for this case?",
                "true_answer_template": "{gene} mutation present",
                "false_answer_template": "{gene} mutation absent",
                "choice_count": 2,
            }
        ],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert len(frame) == 2
    assert set(frame["answer"]) == {"TP53 mutation present", "TP53 mutation absent"}
    assert set(zip(frame["answer"], frame["answer_label"], strict=True)) == {
        ("TP53 mutation present", "A"),
        ("TP53 mutation absent", "B"),
    }
    assert set(frame["task_id"]) == {"mutation_tp53"}
    assert frame["pathology_feature_paths"].tolist() == [[], []]
    assert frame["radiology_feature_paths"].tolist() == [[], []]
    assert frame["radiology_biomarker"].tolist() == ["", ""]
    assert frame["option_c"].tolist() == ["", ""]
    assert frame["option_d"].tolist() == ["", ""]
    assert stats["semantic_questions"] == 2


def test_build_ground_truth_mcq_frame_merges_and_skips_categorical_answers() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage IA",
                "genomics_dna_methylation_feature_path": "features/dnam-a.pt",
            },
            {
                "sample_id": "TCGA-AA-0002",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage IVB",
                "genomics_dna_methylation_feature_path": "features/dnam-b.pt",
            },
            {
                "sample_id": "TCGA-AA-0003",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage 0",
            },
            {
                "sample_id": "TCGA-AA-0004",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Not A Stage",
            },
        ]
    )
    cfg = {
        "default_modality_combinations": [
            {
                "name": "dnam_rna_if_available",
                "use_pathology": "not_include",
                "use_radiology": "not_include",
                "use_dnam": "use_if_avail",
                "use_rna": "use_if_avail",
            }
        ],
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "value_map": {"Stage IA": "Stage I", "Stage IVB": "Stage IV"},
                "skip_values": ["Stage 0"],
            }
        ],
        "boolean_tasks": [],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert frame["answer"].tolist() == ["Stage I", "Stage IV"]
    assert frame["answer_label"].tolist() == ["A", "D"]
    assert frame[
        ["option_a", "option_b", "option_c", "option_d"]
    ].drop_duplicates().values.tolist() == [
        ["Stage I", "Stage II", "Stage III", "Stage IV"]
    ]
    assert stats["task_stats"]["pathologic_stage"]["skipped_empty_answer"] == 1
    assert stats["task_stats"]["pathologic_stage"]["skipped_answer_not_in_options"] == 1


def test_build_ground_truth_mcq_frame_applies_minimum_before_modality_expansion() -> (
    None
):
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-SMALL",
                "split": "train",
                "task_stage_label": "Stage I",
            },
            {
                "sample_id": "TCGA-AA-0002",
                "project_id": "TCGA-SMALL",
                "split": "train",
                "task_stage_label": "Stage II",
            },
            {
                "sample_id": "TCGA-BB-0001",
                "project_id": "TCGA-KEEP",
                "split": "train",
                "task_stage_label": "Stage I",
                "pathology_tile_embedding_paths": ["features/path-keep-a.h5"],
                "genomics_dna_methylation_feature_path": "features/dnam-keep-a.pt",
                "genomics_rna_bulk_feature_path": "features/rna-keep-a.pt",
            },
            {
                "sample_id": "TCGA-BB-0002",
                "project_id": "TCGA-KEEP",
                "split": "train",
                "task_stage_label": "Stage II",
                "pathology_tile_embedding_paths": ["features/path-keep-b.h5"],
                "genomics_dna_methylation_feature_path": "features/dnam-keep-b.pt",
                "genomics_rna_bulk_feature_path": "features/rna-keep-b.pt",
            },
            {
                "sample_id": "TCGA-BB-0003",
                "project_id": "TCGA-KEEP",
                "split": "train",
                "task_stage_label": "Stage III",
                "pathology_tile_embedding_paths": ["features/path-keep-c.h5"],
                "genomics_dna_methylation_feature_path": "features/dnam-keep-c.pt",
                "genomics_rna_bulk_feature_path": "features/rna-keep-c.pt",
            },
        ]
    )
    cfg = {
        "min_semantic_questions_per_project_task": 3,
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "modality_combination_overrides": [
                    {
                        "name": "path_only",
                        "use_pathology": "must_have",
                        "use_radiology": "not_include",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                    {
                        "name": "path_genomics_if_available",
                        "use_pathology": "must_have",
                        "use_radiology": "not_include",
                        "use_dnam": "use_if_avail",
                        "use_rna": "use_if_avail",
                    },
                ],
            }
        ],
        "boolean_tasks": [],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert set(frame["project_id"]) == {"TCGA-KEEP"}
    assert stats["semantic_questions"] == 3
    assert len(frame) == 6
    assert stats["task_stats"]["pathologic_stage"]["skipped_minimum"] == 2


def test_build_ground_truth_mcq_frame_can_require_test_pathology_roi_only_for_test_rows() -> (
    None
):
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "test",
                "task_stage_label": "Stage I",
                "pathology_tile_embedding_paths": ["features/path-test.h5"],
                "genomics_dna_methylation_feature_path": "features/dnam-test.pt",
                "genomics_rna_bulk_feature_path": "features/rna-test.pt",
                "pathology_png_roi_paths": [],
            },
            {
                "sample_id": "TCGA-AA-0002",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage II",
                "pathology_tile_embedding_paths": ["features/path-train.h5"],
                "genomics_dna_methylation_feature_path": "features/dnam-train.pt",
                "genomics_rna_bulk_feature_path": "features/rna-train.pt",
                "pathology_png_roi_paths": [],
            },
        ]
    )
    cfg = {
        "require_test_pathology_roi_png_dir": True,
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "modality_combination_overrides": [
                    {
                        "name": "path_only",
                        "use_pathology": "must_have",
                        "use_radiology": "not_include",
                        "use_dnam": "not_include",
                        "use_rna": "not_include",
                    },
                    {
                        "name": "dnam_rna",
                        "use_pathology": "not_include",
                        "use_radiology": "not_include",
                        "use_dnam": "must_have",
                        "use_rna": "must_have",
                    },
                ],
            }
        ],
        "boolean_tasks": [],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert stats["semantic_questions"] == 2
    assert len(frame) == 3
    assert sorted(
        zip(
            frame["case_id"],
            frame["use_pathology"],
            frame["use_dnam"],
            frame["use_rna"],
        )
    ) == [
        ("TCGA-AA-0001", False, True, True),
        ("TCGA-AA-0002", False, True, True),
        ("TCGA-AA-0002", True, False, False),
    ]
    assert frame.loc[frame["split"].eq("train"), "pathology_roi_png_dir"].tolist() == [
        "",
        "",
    ]


def test_build_ground_truth_mcq_frame_populates_test_genomics_text_from_raw_files(
    tmp_path,
) -> None:
    dnam_raw = tmp_path / "sample.level3betas.txt"
    dnam_raw.write_text(
        "\n".join(
            [
                "cg00000001\t0.05",
                "cg00000002\t0.50",
                "cg00000003\t0.95",
                "cg00000004\tNA",
            ]
        ),
        encoding="utf-8",
    )
    rna_raw = tmp_path / "sample.rna_seq.augmented_star_gene_counts.tsv"
    rna_raw.write_text(
        "\n".join(
            [
                "# gene-model: test",
                "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded",
                "N_unmapped\t\t\t10\t10\t10\t\t\t",
                "N_multimapping\t\t\t2\t2\t2\t\t\t",
                "ENSG000001\tGENEA\tprotein_coding\t100\t0\t0\t12.5\t0\t0",
                "ENSG000002\tGENEB\tprotein_coding\t20\t0\t0\t0.8\t0\t0",
                "ENSG000003\tGENEC\tprotein_coding\t50\t0\t0\t3.2\t0\t0",
                "ENSG000004\tLNCA\tlncRNA\t500\t0\t0\t100\t0\t0",
            ]
        ),
        encoding="utf-8",
    )
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "test",
                "task_stage_label": "Stage I",
                "genomics_dna_methylation_feature_path": "features/dnam-test.pt",
                "genomics_rna_bulk_feature_path": "features/rna-test.pt",
                "genomics_dna_methylation_paths": [str(dnam_raw)],
                "genomics_rna_bulk_paths": [str(rna_raw)],
            },
            {
                "sample_id": "TCGA-AA-0002",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage II",
                "genomics_dna_methylation_feature_path": "features/dnam-train.pt",
                "genomics_rna_bulk_feature_path": "features/rna-train.pt",
                "genomics_dna_methylation_paths": [str(dnam_raw)],
                "genomics_rna_bulk_paths": [str(rna_raw)],
            },
        ]
    )
    cfg = {
        "populate_test_genomics_text_summaries": True,
        "require_test_genomics_text_summaries": True,
        "default_modality_combinations": [
            {
                "name": "dnam_rna",
                "use_pathology": "not_include",
                "use_radiology": "not_include",
                "use_dnam": "must_have",
                "use_rna": "must_have",
            }
        ],
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
            }
        ],
        "boolean_tasks": [],
    }

    frame, _ = build_ground_truth_mcq_frame(registry, cfg)

    test_row = frame.loc[frame["split"].eq("test")].iloc[0]
    train_row = frame.loc[frame["split"].eq("train")].iloc[0]
    assert test_row["dnam_text_summary"].startswith("DNA methylation raw beta summary")
    assert "mean beta" in test_row["dnam_text_summary"]
    assert test_row["rna_text_summary"].startswith("RNA-seq raw expression summary")
    assert "GENEA (TPM 12.5)" in test_row["rna_text_summary"]
    assert train_row["dnam_text_summary"] == ""
    assert train_row["rna_text_summary"] == ""

    dnam_summary = genomics_text_summary.build_dnam_text_summary(
        {
            "project_id": "TCGA-KIRC",
            "genomics_dna_methylation_paths": [str(dnam_raw)],
        },
        panel_genes=["VHL"],
    )
    assert "Promoter methylation for benchmark mutation panel genes" in dnam_summary
    assert "VHL promoter beta not_assessed" in dnam_summary

    rna_summary = genomics_text_summary.build_rna_text_summary(
        {
            "project_id": "TCGA-TEST",
            "genomics_rna_bulk_paths": [str(rna_raw)],
        },
        panel_genes=["GENEA", "GENEC", "TP53"],
    )
    assert "RNA expression for benchmark mutation panel genes" in rna_summary
    assert "GENEA TPM 12.5" in rna_summary
    assert "GENEC TPM 3.2" in rna_summary
    assert "TP53 TPM not_assessed" in rna_summary
    assert "RNA pathway and microenvironment signatures" in rna_summary


def test_build_ground_truth_mcq_frame_can_skip_test_genomics_text_summaries(
    tmp_path,
) -> None:
    raw_path = tmp_path / "sample.level3betas.txt"
    raw_path.write_text("cg00000001\t0.05\n", encoding="utf-8")
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "test",
                "task_stage_label": "Stage I",
                "genomics_dna_methylation_feature_path": "features/dnam-test.pt",
                "genomics_dna_methylation_paths": [str(raw_path)],
            }
        ]
    )
    cfg = {
        "populate_test_genomics_text_summaries": False,
        "require_test_genomics_text_summaries": True,
        "default_modality_combinations": [
            {
                "name": "dnam_only",
                "use_pathology": "not_include",
                "use_radiology": "not_include",
                "use_dnam": "must_have",
                "use_rna": "not_include",
            }
        ],
        "categorical_tasks": [
            {
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
            }
        ],
        "boolean_tasks": [],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert frame.empty
    assert stats["generated_rows"] == 0


def test_genomics_text_summary_resolves_raw_paths_from_repo_root(
    tmp_path, monkeypatch
) -> None:
    fake_repo_root = tmp_path / "repo"
    fake_raw_dir = fake_repo_root / "raw"
    fake_raw_dir.mkdir(parents=True)
    dnam_raw = fake_raw_dir / "sample.level3betas.txt"
    dnam_raw.write_text("cg00000001\t0.05\ncg00000002\t0.95\n", encoding="utf-8")

    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(genomics_text_summary, "REPO_ROOT", fake_repo_root)

    summary = genomics_text_summary.build_dnam_text_summary(
        {"genomics_dna_methylation_paths": ["raw/sample.level3betas.txt"]}
    )

    assert summary.startswith("DNA methylation raw beta summary:")


def test_genomics_text_summary_raises_for_missing_raw_paths() -> None:
    with pytest.raises(FileNotFoundError):
        genomics_text_summary.build_dnam_text_summary(
            {"genomics_dna_methylation_paths": ["missing/sample.level3betas.txt"]}
        )


def test_build_ground_truth_mcq_frame_uses_project_specific_gene_panels() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-ONE",
                "split": "train",
                "mutation_tp53": True,
                "mutation_hla-b": True,
                "genomics_dna_methylation_feature_path": "features/dnam-a.pt",
                "genomics_rna_bulk_feature_path": "features/rna-a.pt",
            },
            {
                "sample_id": "TCGA-AA-0002",
                "project_id": "TCGA-ONE",
                "split": "train",
                "mutation_tp53": False,
                "mutation_hla-b": True,
                "genomics_dna_methylation_feature_path": "features/dnam-b.pt",
                "genomics_rna_bulk_feature_path": "features/rna-b.pt",
            },
            {
                "sample_id": "TCGA-BB-0001",
                "project_id": "TCGA-DLBC",
                "split": "train",
                "mutation_tp53": True,
                "mutation_hla-b": True,
                "genomics_dna_methylation_feature_path": "features/dnam-c.pt",
                "genomics_rna_bulk_feature_path": "features/rna-c.pt",
            },
            {
                "sample_id": "TCGA-BB-0002",
                "project_id": "TCGA-DLBC",
                "split": "train",
                "mutation_tp53": True,
                "mutation_hla-b": False,
                "genomics_dna_methylation_feature_path": "features/dnam-d.pt",
                "genomics_rna_bulk_feature_path": "features/rna-d.pt",
            },
        ]
    )
    cfg = {
        "default_modality_combinations": [
            {
                "name": "dnam_rna",
                "use_pathology": "not_include",
                "use_radiology": "not_include",
                "use_dnam": "must_have",
                "use_rna": "must_have",
            }
        ],
        "categorical_tasks": [],
        "boolean_tasks": [
            {
                "task_category": "mutation",
                "task_id_template": "{source_column}",
                "gene_panel_by_project": {"TCGA-ONE": ["TP53"], "TCGA-DLBC": ["HLA-B"]},
                "question_template": "What is the {gene} mutation status for this case?",
                "true_answer_template": "{gene} mutation present",
                "false_answer_template": "{gene} mutation absent",
                "choice_count": 2,
            }
        ],
    }

    frame, _ = build_ground_truth_mcq_frame(registry, cfg)

    assert len(frame) == 4
    assert set(frame.loc[frame["project_id"].eq("TCGA-ONE"), "task_id"]) == {
        "mutation_tp53"
    }
    assert set(frame.loc[frame["project_id"].eq("TCGA-DLBC"), "task_id"]) == {
        "mutation_hla-b"
    }
    assert set(frame.loc[frame["task_id"].eq("mutation_hla-b"), "answer"]) == {
        "HLA-B mutation present",
        "HLA-B mutation absent",
    }
    assert frame.loc[
        frame["task_id"].eq("mutation_hla-b"), ["option_a", "option_b"]
    ].drop_duplicates().values.tolist() == [
        ["HLA-B mutation present", "HLA-B mutation absent"]
    ]


def test_build_ground_truth_mcq_frame_applies_mutation_minimum_after_false_downsampling() -> (
    None
):
    registry = pd.DataFrame(
        [
            {
                "sample_id": f"TCGA-AA-{index:04d}",
                "project_id": "TCGA-ONE",
                "split": "train",
                "mutation_tp53": index < 2,
            }
            for index in range(12)
        ]
    )
    cfg = {
        "min_semantic_questions_per_project_task": 5,
        "categorical_tasks": [],
        "boolean_tasks": [
            {
                "task_category": "mutation",
                "task_id_template": "{source_column}",
                "gene_panel_by_project": {"TCGA-ONE": ["TP53"]},
                "question_template": "What is the {gene} mutation status for this case?",
                "true_answer_template": "{gene} mutation present",
                "false_answer_template": "{gene} mutation absent",
                "choice_count": 2,
                "max_false_per_true": 1.0,
            }
        ],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert frame.empty
    mutation_stats = stats["task_stats"]["mutation"]
    assert mutation_stats["downsampled_false"] == 8
    assert mutation_stats["skipped_minimum"] == 4


def test_build_ground_truth_mcq_frame_downsamples_mutation_false_in_all_splits() -> (
    None
):
    records = []
    for split, positives, negatives in [
        ("train", 2, 10),
        ("val", 1, 5),
        ("test", 0, 5),
    ]:
        for index in range(positives + negatives):
            records.append(
                {
                    "sample_id": f"TCGA-AA-{len(records):04d}",
                    "project_id": "TCGA-ONE",
                    "split": split,
                    "mutation_tp53": index < positives,
                    "genomics_dna_methylation_feature_path": f"features/dnam-{len(records)}.pt",
                    "genomics_rna_bulk_feature_path": f"features/rna-{len(records)}.pt",
                }
            )
    registry = pd.DataFrame(records)
    cfg = {
        "default_modality_combinations": [
            {
                "name": "dnam_rna",
                "use_pathology": "not_include",
                "use_radiology": "not_include",
                "use_dnam": "must_have",
                "use_rna": "must_have",
            }
        ],
        "categorical_tasks": [],
        "boolean_tasks": [
            {
                "task_category": "mutation",
                "task_id_template": "{source_column}",
                "gene_panel_by_project": {"TCGA-ONE": ["TP53"]},
                "question_template": "What is the {gene} mutation status for this case?",
                "true_answer_template": "{gene} mutation present",
                "false_answer_template": "{gene} mutation absent",
                "choice_count": 2,
                "max_false_per_true": 1.0,
            }
        ],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    mutation_stats = stats["task_stats"]["mutation"]
    assert mutation_stats["downsampled_false"] == 17
    assert mutation_stats["dropped_false_without_positive"] == 5
    assert frame.groupby(["split", "answer"]).size().to_dict() == {
        ("train", "TP53 mutation absent"): 2,
        ("train", "TP53 mutation present"): 2,
        ("val", "TP53 mutation absent"): 1,
        ("val", "TP53 mutation present"): 1,
    }


def test_build_ground_truth_mcq_frame_supports_stage_mutation_joint_profile() -> None:
    registry = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "project_id": "TCGA-TEST",
                "split": "train",
                "task_stage_label": "Stage IIA",
                "mutation_tp53": True,
                "mutation_pik3ca": False,
                "pathology_tile_embedding_paths": ["features/path-a.h5"],
                "radiology_embedding_paths": ["features/rad-a.h5"],
                "genomics_dna_methylation_feature_path": "features/dnam-a.pt",
                "genomics_rna_bulk_feature_path": "features/rna-a.pt",
            }
        ]
    )
    cfg = {
        "default_modality_combinations": [
            {
                "name": "all_available",
                "use_pathology": "must_have",
                "use_radiology": "use_if_avail",
                "use_dnam": "use_if_avail",
                "use_rna": "use_if_avail",
            },
            {
                "name": "path_only",
                "use_pathology": "must_have",
                "use_radiology": "not_include",
                "use_dnam": "not_include",
                "use_rna": "not_include",
            },
            {
                "name": "radiology_only",
                "use_pathology": "not_include",
                "use_radiology": "must_have",
                "use_dnam": "not_include",
                "use_rna": "not_include",
            },
        ],
        "categorical_tasks": [
            {
                "enabled": False,
                "task_category": "stage",
                "task_id": "pathologic_stage",
                "source_column": "task_stage_label",
                "question_template": "What is the AJCC pathologic stage for this case?",
                "options": ["Stage I", "Stage II", "Stage III", "Stage IV"],
                "value_map": {"Stage IIA": "Stage II"},
            }
        ],
        "boolean_tasks": [
            {
                "enabled": False,
                "task_category": "mutation",
                "task_id_template": "{source_column}",
                "gene_panel_by_project": {"TCGA-TEST": ["TP53", "PIK3CA"]},
                "question_template": "What is the {gene} mutation status for this case?",
                "true_answer_template": "{gene} mutation present",
                "false_answer_template": "{gene} mutation absent",
                "choice_count": 2,
            }
        ],
        "joint_profile_tasks": [
            {
                "enabled": True,
                "task_category": "joint_profile",
                "task_id": "stage_mutation",
                "task_id_template": "stage_{source_column}",
                "stage_task_id": "pathologic_stage",
                "mutation_task_category": "mutation",
                "max_questions_per_case": 1,
                "question_template": "Which combined AJCC pathologic stage and mutation profile matches this case?",
            }
        ],
    }

    frame, stats = build_ground_truth_mcq_frame(registry, cfg)

    assert len(frame) == 3
    assert stats["semantic_questions"] == 1
    assert (
        stats["task_stats"]["stage_mutation"]["skipped_case_question_limit"] == 1
    )
    assert set(frame["modality_combination_name"]) == {
        "all_available",
        "path_only",
        "radiology_only",
    }
    assert set(frame["task_category"]) == {"joint_profile"}
    assert frame["task_id"].nunique() == 1
    assert set(frame["task_id"]).issubset(
        {"stage_mutation_tp53", "stage_mutation_pik3ca"}
    )
    assert frame["ground_truth_source"].nunique() == 1
    assert set(frame["ground_truth_source"]).issubset(
        {"task_stage_label|mutation_tp53", "task_stage_label|mutation_pik3ca"}
    )
    assert frame["answer"].nunique() == 1
    first = frame.iloc[0]
    choices = [first["option_a"], first["option_b"], first["option_c"], first["option_d"]]
    assert len(set(choices)) == 4
    assert str(first["answer"]).startswith("Stage II + ")
    assert choices.count(first["answer"]) == 1
    assert first["answer"] in choices
    assert sum(choice.startswith("Stage II +") for choice in choices) == 2
    assert frame["answer_label"].map(
        {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}
    ).notna().all()
    for _, row in frame.iterrows():
        assert row["answer"] == row[
            {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}[
                row["answer_label"]
            ]
        ]
