from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf


@lru_cache(maxsize=1)
def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rad_proj_train" / "02_train_rad_projectors.py"
    spec = importlib.util.spec_from_file_location("train_rad_projectors_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_train_validation_keeps_all_rows_for_a_series_together() -> None:
    module = _load_script_module()

    frame = pd.DataFrame(
        [
            {"sample_id": "a", "series_stem": "series-1"},
            {"sample_id": "b", "series_stem": "series-1"},
            {"sample_id": "c", "series_stem": "series-2"},
            {"sample_id": "d", "series_stem": "series-2"},
            {"sample_id": "e", "series_stem": "series-3"},
            {"sample_id": "f", "series_stem": "series-3"},
        ]
    )

    train_frame, validation_frame = module._split_train_validation(frame, seed=42, validation_fraction=1 / 3)

    train_stems = set(train_frame["series_stem"].tolist())
    validation_stems = set(validation_frame["series_stem"].tolist())

    assert train_stems
    assert validation_stems
    assert train_stems.isdisjoint(validation_stems)
    assert len(validation_stems) == 1
    assert len(validation_frame) == 2


def test_partition_prefers_explicit_validation_split_when_available() -> None:
    module = _load_script_module()

    frame = pd.DataFrame(
        [
            {"sample_id": "train-1", "series_stem": "series-1", "split": "train"},
            {"sample_id": "train-2", "series_stem": "series-2", "split": "train"},
            {"sample_id": "val-1", "series_stem": "series-3", "split": "validation"},
            {"sample_id": "test-1", "series_stem": "series-4", "split": "test"},
        ]
    )

    train_frame, validation_frame, test_frame, mode = module._partition_by_split_or_holdout(
        frame,
        seed=42,
        validation_fraction=0.5,
    )

    assert mode == "explicit"
    assert train_frame["sample_id"].tolist() == ["train-1", "train-2"]
    assert validation_frame["sample_id"].tolist() == ["val-1"]
    assert test_frame["sample_id"].tolist() == ["test-1"]


def test_run_artifacts_are_saved_inside_timestamped_run_dir(tmp_path: Path) -> None:
    module = _load_script_module()

    class DummyProjectors:
        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    class DummyModel:
        rad_projectors = DummyProjectors()
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
            "rad_proj_train": {
                "modality_tag": "rad",
                "model_name_or_path": "Qwen/Qwen3.5-9B",
                "radiology_embedding_dim": 1152,
                "projector_type": "mlp",
                "projector_num_latents": 32,
                "projector_depth": 2,
                "projector_num_heads": 8,
                "projector_mlp_ratio": 4.0,
                "projector_dropout": 0.05,
                "max_slice_tokens": 32,
            }
        }
    )

    run_output_dir = module._build_run_output_dir(
        output_root=tmp_path,
        modality_tag="rad",
        projector_type="mlp",
    )
    state_path = module._save_artifacts(
        run_output_dir=run_output_dir,
        checkpoint_name="epoch_001.ckpt",
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

    assert run_output_dir.parent == tmp_path
    assert run_output_dir.name.startswith("rad_mlp_")
    assert run_output_dir.name.endswith("_EST")
    assert state_path.name == "epoch_001.ckpt"
    assert state_path.exists()
    assert (run_output_dir / "config.yaml").exists()
    assert (run_output_dir / "tokenizer" / "tokenizer.json").exists()

    metadata = metadata_path.read_text()
    assert str(state_path) in metadata
    assert str(run_output_dir / "config.yaml") in metadata
