from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "demo" / "demo_projector.py"
    spec = importlib.util.spec_from_file_location("demo_projector_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_feature_path_uses_patch_encoder_folder(tmp_path: Path) -> None:
    module = _load_script_module()

    cfg = module.OmegaConf.create(
        {
            "pathology_features": {
                "patch_encoder": "conch_v15",
                "save_format": "h5",
                "features_root": str(tmp_path / "features"),
            }
        }
    )
    feature_dir = tmp_path / "features" / "features_conch_v15"
    feature_dir.mkdir(parents=True)
    expected = feature_dir / "slide-a.h5"
    expected.write_bytes(b"x")

    modality_cfg = module.OmegaConf.create({"feature_path_field": "pathology_tile_embedding_paths"})
    resolved = module._resolve_feature_path(cfg, "pathology", modality_cfg, "slide-a")

    assert resolved == expected.resolve()


def test_resolve_feature_path_prefers_projector_parquet_mapping(tmp_path: Path) -> None:
    module = _load_script_module()

    cfg = module.OmegaConf.create(
        {
            "pathology_features": {
                "patch_encoder": "conch_v15",
                "save_format": "h5",
                "features_root": str(tmp_path / "features"),
            }
        }
    )
    uni_feature = tmp_path / "data" / "features" / "features_uni" / "slide-a.h5"
    uni_feature.parent.mkdir(parents=True)
    uni_feature.write_bytes(b"x")

    resolved = module._resolve_feature_path(
        cfg,
        "pathology",
        module.OmegaConf.create({"feature_path_field": "pathology_tile_embedding_paths"}),
        "slide-a",
        feature_paths_by_sample_key={"slide-a": str(uni_feature)},
    )

    assert resolved == uni_feature.resolve()


def test_resolve_feature_path_supports_dnam_parquet_mapping(tmp_path: Path) -> None:
    module = _load_script_module()

    feature_path = tmp_path / "data" / "features" / "features_cpgpt_dnam" / "case-a.pt"
    feature_path.parent.mkdir(parents=True)
    torch.save({"embedding": torch.randn(1, 8)}, feature_path)

    resolved = module._resolve_feature_path(
        module.OmegaConf.create({}),
        "dnam",
        module.OmegaConf.create({"feature_path_field": "genomics_dna_methylation_feature_path"}),
        "case-a",
        feature_paths_by_sample_key={"case-a": str(feature_path)},
    )

    assert resolved == feature_path.resolve()


def test_build_chat_prompt_input_ids_uses_fixed_demo_prompt() -> None:
    module = _load_script_module()

    class DummyTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "what is this. caption: "}]
            assert kwargs["tokenize"] is True
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
            return [11, 12, 13]

    input_ids = module._build_chat_prompt_input_ids(DummyTokenizer(), device=torch.device("cpu"))

    assert input_ids.tolist() == [[11, 12, 13]]


def test_load_h5_patch_features_downsamples_to_max_patch_tokens(tmp_path: Path) -> None:
    module = _load_script_module()

    path = tmp_path / "slide-a.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.arange(30, dtype=np.float32).reshape(10, 3))
        handle.create_dataset("coords", data=np.arange(20, dtype=np.int64).reshape(10, 2))

    tensor = module._load_h5_patch_features(path, max_patch_tokens=4)

    assert tuple(tensor.shape) == (4, 3)


def test_load_h5_patch_features_supports_spatial_mean_pooling(tmp_path: Path) -> None:
    module = _load_script_module()

    path = tmp_path / "slide-a.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "features",
            data=np.array(
                [
                    [1.0, 1.0],
                    [3.0, 3.0],
                    [5.0, 5.0],
                    [7.0, 7.0],
                ],
                dtype=np.float32,
            ),
        )
        handle.create_dataset(
            "coords",
            data=np.array(
                [
                    [0, 0],
                    [0, 512],
                    [512, 0],
                    [512, 512],
                ],
                dtype=np.int64,
            ),
        )

    tensor = module._load_h5_patch_features(
        path,
        max_patch_tokens=0,
        compression_method="mean_pool",
        compression_kernel_size=2,
    )

    assert torch.allclose(tensor, torch.tensor([[4.0, 4.0]], dtype=torch.float32))


def test_resolve_checkpoint_path_finds_nested_best_checkpoint(tmp_path: Path) -> None:
    module = _load_script_module()

    nested_ckpt = tmp_path / "outputs" / "projectors" / "qwen3_5_9b" / "pathology" / "run_a" / "best.ckpt"
    nested_ckpt.parent.mkdir(parents=True)
    nested_ckpt.write_bytes(b"ckpt")

    resolved = module._resolve_checkpoint_path(nested_ckpt.parents[3])

    assert resolved == nested_ckpt.resolve()


def test_generate_response_requires_transformers_for_tokenizer_import():
    pytest.importorskip("transformers")
