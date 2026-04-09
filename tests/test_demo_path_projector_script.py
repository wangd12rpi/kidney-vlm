from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pytest


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "demo" / "demo_path_projector.py"
    spec = importlib.util.spec_from_file_location("demo_path_projector_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_feature_path_uses_patch_encoder_folder(tmp_path: Path) -> None:
    module = _load_script_module()

    cfg = module.OmegaConf.create(
        {
            "embeding_extraction": {
                "pathology": {
                    "patch_encoder": "conch_v15",
                    "save_format": "h5",
                    "features_root": str(tmp_path / "features"),
                }
            }
        }
    )
    feature_dir = tmp_path / "features" / "features_conch_v15"
    feature_dir.mkdir(parents=True)
    expected = feature_dir / "slide-a.h5"
    expected.write_bytes(b"x")

    resolved = module._resolve_feature_path(cfg, "slide-a")

    assert resolved == expected.resolve()


def test_load_h5_patch_features_downsamples_to_max_patch_tokens(tmp_path: Path) -> None:
    module = _load_script_module()

    path = tmp_path / "slide-a.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.arange(30, dtype=np.float32).reshape(10, 3))
        handle.create_dataset("coords", data=np.arange(20, dtype=np.int64).reshape(10, 2))

    tensor = module._load_h5_patch_features(path, max_patch_tokens=4)

    assert tuple(tensor.shape) == (4, 3)


def test_generate_response_requires_transformers_for_tokenizer_import():
    pytest.importorskip("transformers")
