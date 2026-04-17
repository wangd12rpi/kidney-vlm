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
    script_path = repo_root / "scripts" / "01_pathology_proj" / "04_train_path_projectors.py"
    spec = importlib.util.spec_from_file_location("train_path_projectors_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_train_validation_keeps_all_caption_rows_for_a_slide_together() -> None:
    module = _load_script_module()

    frame = pd.DataFrame(
        [
            {"sample_id": "a", "slide_stem": "slide-1", "caption_variant_index": 0},
            {"sample_id": "a", "slide_stem": "slide-1", "caption_variant_index": 1},
            {"sample_id": "b", "slide_stem": "slide-2", "caption_variant_index": 0},
            {"sample_id": "b", "slide_stem": "slide-2", "caption_variant_index": 1},
            {"sample_id": "c", "slide_stem": "slide-3", "caption_variant_index": 0},
            {"sample_id": "c", "slide_stem": "slide-3", "caption_variant_index": 1},
        ]
    )

    train_frame, validation_frame = module._split_train_validation(frame, seed=42, validation_fraction=1 / 3)

    train_stems = set(train_frame["slide_stem"].tolist())
    validation_stems = set(validation_frame["slide_stem"].tolist())

    assert train_stems
    assert validation_stems
    assert train_stems.isdisjoint(validation_stems)
    assert len(validation_stems) == 1
    assert len(validation_frame) == 2


def test_top_validation_slide_stems_returns_unique_stems_in_order() -> None:
    module = _load_script_module()

    frame = pd.DataFrame(
        [
            {"slide_stem": "slide-3"},
            {"slide_stem": "slide-3"},
            {"slide_stem": "slide-1"},
            {"slide_stem": "slide-2"},
            {"slide_stem": "slide-4"},
            {"slide_stem": "slide-5"},
            {"slide_stem": "slide-6"},
        ]
    )

    stems = module._top_validation_slide_stems(frame, limit=5)

    assert stems == ["slide-3", "slide-1", "slide-2", "slide-4", "slide-5"]


def test_run_artifacts_are_saved_inside_timestamped_run_dir(tmp_path: Path) -> None:
    module = _load_script_module()

    class DummyProjectors:
        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    class DummyModel:
        path_projectors = DummyProjectors()
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
            "pathology_proj": {
                "modality_tag": "path",
                "modality_dir_name": "pathology",
                "model_name_or_path": "Qwen/Qwen3.5-9B",
                "pathology_embedding_dim": 768,
                "projector_type": "resampler",
                "projector_num_latents": 64,
                "projector_depth": 2,
                "projector_num_heads": 8,
                "projector_mlp_ratio": 4.0,
                "projector_dropout": 0.1,
                "max_patch_tokens": 1024,
                "save_tokenizer_snapshot": False,
            }
        }
    )

    run_output_dir = module._build_run_output_dir(
        output_root=tmp_path,
        llm_tag="qwen3_5_9b",
        modality_dir_name="pathology",
        modality_tag="path",
        projector_type="resampler",
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

    assert run_output_dir.parent == tmp_path / "qwen3_5_9b" / "pathology"
    assert run_output_dir.name.startswith("path_resampler_")
    assert run_output_dir.name.endswith("_EST")
    assert state_path.name == "epoch_001.ckpt"
    assert state_path.exists()
    assert (run_output_dir / "config.yaml").exists()
    assert not (run_output_dir / "tokenizer").exists()

    metadata = metadata_path.read_text()
    assert str(state_path) in metadata
    assert str(run_output_dir / "config.yaml") in metadata


def test_compute_total_optimizer_steps_and_warmup_resolution() -> None:
    module = _load_script_module()

    total_steps = module._compute_total_optimizer_steps(
        num_batches_per_epoch=11,
        num_epochs=3,
        gradient_accumulation_steps=4,
    )
    warmup_steps = module._resolve_warmup_steps(
        total_optimizer_steps=total_steps,
        warmup_steps_cfg=None,
        warmup_ratio=0.1,
    )

    assert total_steps == 9
    assert warmup_steps == 1
