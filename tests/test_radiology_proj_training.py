from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

pytest.importorskip("torch")
pytest.importorskip("omegaconf")


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "02_radiology_proj" / "03_train_radiology_projectors.py"
    spec = importlib.util.spec_from_file_location("train_radiology_projectors_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_artifacts_are_saved_inside_timestamped_run_dir() -> None:
    import torch
    from omegaconf import OmegaConf
    module = _load_script_module()

    class DummyProjectors:
        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    class DummyModel:
        radiology_projectors = DummyProjectors()
        hidden_size = 3584

        @staticmethod
        def trainable_parameter_count() -> int:
            return 1

        @staticmethod
        def total_parameter_count() -> int:
            return 2

    class DummyTokenizer:
        def save_pretrained(self, path: Path) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}")

    cfg = OmegaConf.create(
        {
            "radiology_proj": {
                "modality_tag": "radiology",
                "modality_dir_name": "radiology",
                "model_name_or_path": "Qwen/Qwen3.5-9B",
                "radiology_embedding_dim": 1152,
                "projector_type": "mlp",
                "projector_num_latents": 64,
                "projector_depth": 2,
                "projector_num_heads": 8,
                "projector_mlp_ratio": 4.0,
                "projector_dropout": 0.05,
                "max_slice_tokens": 32,
                "save_tokenizer_snapshot": False,
            }
        }
    )

    repo_root = Path(__file__).resolve().parents[1]
    artifacts_root = repo_root / ".test_artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    tmp_path = artifacts_root / f"radiology_proj_training_{uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=False)

    try:
        run_output_dir = module._build_run_output_dir(
            output_root=tmp_path,
            llm_tag="qwen3_5_9b",
            modality_dir_name="radiology",
            modality_tag="radiology",
            projector_type="mlp",
        )
        state_path = module._save_artifacts(
            run_output_dir=run_output_dir,
            checkpoint_name="epoch_01.ckpt",
            cfg=cfg,
            model=DummyModel(),
            tokenizer=DummyTokenizer(),
            global_step=3,
            epoch=1,
            validation_loss=0.25,
        )
        metadata_path = module._write_run_metadata(
            run_output_dir=run_output_dir,
            cfg=cfg,
            model=DummyModel(),
            global_step=3,
            epoch_checkpoint_paths=[str(state_path)],
            best_checkpoint_path=run_output_dir / "best.ckpt",
            best_epoch=1,
            best_validation_loss=0.25,
        )

        assert run_output_dir.parent == tmp_path / "qwen3_5_9b" / "radiology"
        assert run_output_dir.name.startswith("radiology_mlp_")
        assert run_output_dir.name.endswith("_EST")
        assert state_path.name == "epoch_01.ckpt"
        assert state_path.exists()
        assert (run_output_dir / "config.yaml").exists()
        assert not (run_output_dir / "tokenizer").exists()

        metadata = json.loads(metadata_path.read_text())
        assert metadata["epoch_checkpoint_paths"] == [state_path.resolve().relative_to(repo_root).as_posix()]
        assert metadata["config_path"] == (run_output_dir / "config.yaml").resolve().relative_to(repo_root).as_posix()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
