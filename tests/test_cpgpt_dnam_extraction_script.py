from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "03_dnam_features" / "01_extract_cpgpt_dnam_features.py"
    spec = importlib.util.spec_from_file_location("extract_cpgpt_dnam_features_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_extraction_tasks_selects_missing_dnam_features(tmp_path: Path) -> None:
    module = _load_script_module()
    root_dir = tmp_path / "repo"
    raw_path = root_dir / "data" / "raw" / "tcga" / "dna_methylation" / "TCGA-KIRC" / "TCGA-AA-0001" / "beta.txt"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("cg00000029\t0.5\n", encoding="utf-8")

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "tcga-1",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "split": "train",
                "genomics_dna_methylation_paths": [
                    "data/raw/tcga/dna_methylation/TCGA-KIRC/TCGA-AA-0001/beta.txt"
                ],
                "genomics_dna_methylation_feature_path": "",
            },
            {
                "sample_id": "tcga-2",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-BB-0002",
                "split": "train",
                "genomics_dna_methylation_paths": [
                    "data/raw/tcga/dna_methylation/TCGA-KIRC/TCGA-BB-0002/beta.txt"
                ],
                "genomics_dna_methylation_feature_path": "data/features/existing.pt",
            },
        ]
    )

    tasks = module.build_extraction_tasks(
        registry_df,
        root_dir=root_dir,
        output_root=root_dir / "data" / "features" / "features_cpgpt_dnam",
        raw_paths_column="genomics_dna_methylation_paths",
        feature_path_column="genomics_dna_methylation_feature_path",
        sources=["tcga"],
        project_ids=[],
        splits=[],
        selected_raw_path_index=0,
        overwrite_existing_registry_paths=False,
        fail_on_missing_raw_file=True,
        max_rows=None,
    )

    assert len(tasks) == 1
    task = tasks[0]
    assert task.sample_id == "tcga-1"
    assert task.raw_beta_path == raw_path
    assert task.feature_path_value == (
        "data/features/features_cpgpt_dnam/tcga/TCGA-KIRC/"
        "TCGA-AA-0001__beta.pt"
    )


def test_build_extraction_tasks_treats_null_feature_path_as_missing(tmp_path: Path) -> None:
    module = _load_script_module()
    root_dir = tmp_path / "repo"
    raw_path = root_dir / "data" / "raw" / "cptac" / "dna_methylation" / "CPTAC-3" / "C3N-00001" / "beta.txt"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("cg00000029\t0.5\n", encoding="utf-8")
    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "cptac-1",
                "source": "cptac",
                "project_id": "CPTAC-3",
                "patient_id": "C3N-00001",
                "split": "cptac_external_test",
                "genomics_dna_methylation_paths": [raw_path.relative_to(root_dir).as_posix()],
                "genomics_dna_methylation_feature_path": np.nan,
            }
        ]
    )

    tasks = module.build_extraction_tasks(
        registry_df,
        root_dir=root_dir,
        output_root=root_dir / "data" / "features" / "features_cpgpt_dnam",
        raw_paths_column="genomics_dna_methylation_paths",
        feature_path_column="genomics_dna_methylation_feature_path",
        sources=["cptac"],
        project_ids=[],
        splits=[],
        selected_raw_path_index=0,
        overwrite_existing_registry_paths=False,
        fail_on_missing_raw_file=True,
        max_rows=None,
    )

    assert len(tasks) == 1
    assert tasks[0].sample_id == "cptac-1"


def test_update_registry_with_manifest_respects_overwrite_flag() -> None:
    module = _load_script_module()
    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "case-1",
                "genomics_dna_methylation_feature_path": "",
            },
            {
                "sample_id": "case-2",
                "genomics_dna_methylation_feature_path": "data/features/old.pt",
            },
        ]
    )
    manifest_df = pd.DataFrame(
        [
            {"sample_id": "case-1", "feature_path": "data/features/new-1.pt"},
            {"sample_id": "case-2", "feature_path": "data/features/new-2.pt"},
        ]
    )

    updated = module.update_registry_with_manifest(
        registry_df,
        manifest_df,
        feature_path_column="genomics_dna_methylation_feature_path",
        overwrite_existing_registry_paths=False,
    )
    assert updated.loc[0, "genomics_dna_methylation_feature_path"] == "data/features/new-1.pt"
    assert updated.loc[1, "genomics_dna_methylation_feature_path"] == "data/features/old.pt"

    overwritten = module.update_registry_with_manifest(
        registry_df,
        manifest_df,
        feature_path_column="genomics_dna_methylation_feature_path",
        overwrite_existing_registry_paths=True,
    )
    assert overwritten.loc[1, "genomics_dna_methylation_feature_path"] == "data/features/new-2.pt"


def test_string_list_handles_hydra_list_config() -> None:
    module = _load_script_module()

    assert module._string_list(OmegaConf.create(["tcga", "cptac"])) == ["tcga", "cptac"]
    assert module._string_list(OmegaConf.create([])) == []
