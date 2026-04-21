from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "data" / "05_register_uni_paths_into_registry.py"
    spec = importlib.util.spec_from_file_location("register_uni_paths_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_register_uni_job_matches_patient_when_wsi_paths_absent(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)

    feature_dir = tmp_path / "data" / "features" / "features_uni_cptac"
    feature_dir.mkdir(parents=True)
    feature_path = feature_dir / "C3N-00001__slide.h5"
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=np.zeros((3, 4), dtype=np.float16))
        handle.create_dataset("coords", data=np.zeros((3, 2), dtype=np.int32))

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "cptac-a",
                "source": "cptac",
                "patient_id": "C3N-00001",
                "study_id": "case-a",
                "split": "cptac_external_test",
                "pathology_wsi_paths": [],
                "pathology_tile_embedding_paths": [],
            }
        ]
    )

    updated_df = module._register_uni_job(
        registry_df,
        job={
            "label": "cptac",
            "source_filter": "cptac",
            "patch_features_dir": feature_dir,
            "match_patient_id_when_no_wsi_paths": True,
        },
        allowed_project_ids=[],
        clear_existing_pathology_patch_embeddings_before_register=False,
        coords_root=tmp_path / "coords",
        save_format="h5",
        patch_size=256,
        target_magnification=20,
    )

    assert updated_df.at[0, "pathology_tile_embedding_paths"] == [
        "data/features/features_uni_cptac/C3N-00001__slide.h5"
    ]
    assert updated_df.at[0, "pathology_tile_embedding_patch_counts"] == [3]


def test_register_uni_config_uses_explicit_source_switches() -> None:
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[1]

    cfg = module.load_script_cfg(repo_root=repo_root, config_relative_path=module.CONFIG_RELATIVE_PATH)
    jobs = module._build_enabled_jobs(cfg.uni_registration.sources)

    assert [job["label"] for job in jobs] == ["tcga"]
    assert bool(cfg.uni_registration.sources.tcga.enabled)
    assert not bool(cfg.uni_registration.sources.cptac.enabled)
