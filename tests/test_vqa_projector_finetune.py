from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from torch import nn


def _load_train_script(repo_root: Path):
    script_path = repo_root / "scripts" / "06_vqa_train" / "train_vqa_lora.py"
    spec = importlib.util.spec_from_file_location(
        "train_vqa_lora_projector_finetune_test", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_projector_finetune_profile_uses_live_features_and_no_cot() -> None:
    pytest.importorskip("hydra")
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)
    module = _load_train_script(repo_root)

    cfg = module.load_cfg_from_overrides(
        ["method=sft", "profile=projector_ft"]
    ).vqa_train

    assert cfg.name == "vqa_sft_projector_ft"
    assert cfg.post_train_method == "sft"
    assert cfg.cot.enabled is False
    assert cfg.prefix_cache.enabled is False
    assert cfg.projectors.pathology.trainable is True
    assert cfg.projectors.radiology.trainable is False
    assert cfg.projectors.dnam.trainable is False
    assert cfg.projectors.rna.trainable is False
    assert cfg.num_epochs == 4
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.projector_learning_rate == pytest.approx(1e-5)
    assert str(cfg.dataset.vqa_parquet_path).endswith(
        "caption_mcq_all_available_pathology_findings_cot.parquet"
    )


def test_all_caption_mcq_projector_finetune_profile_expands_scope() -> None:
    pytest.importorskip("hydra")
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)
    module = _load_train_script(repo_root)

    cfg = module.load_cfg_from_overrides(
        ["method=sft", "profile=projector_ft_all_caption_mcq"]
    ).vqa_train

    assert cfg.name == "vqa_sft_projector_ft_all_caption_mcq"
    assert str(cfg.dataset.vqa_parquet_path).endswith("data/vqa/merged_vqa.parquet")
    assert list(cfg.dataset.question_types) == ["mcq"]
    assert list(cfg.dataset.generation_types) == ["from_caption"]
    assert list(cfg.dataset.modality_combination_names) == []
    assert list(cfg.dataset.task_categories) == []
    assert list(cfg.dataset.task_ids) == []
    assert cfg.cot.enabled is False
    assert cfg.prefix_cache.enabled is False
    assert all(
        cfg.projectors[modality].trainable
        for modality in ("pathology", "radiology", "dnam", "rna")
    )
    assert str(cfg.sft.init_lora_adapter_path).endswith("best/lora_adapter")
    assert all(
        str(cfg.projectors[modality].checkpoint_path).endswith(
            "best/projectors.ckpt"
        )
        for modality in ("pathology", "radiology", "dnam", "rna")
    )
    assert cfg.num_epochs == 4
    assert cfg.save_every_epoch is True
    assert cfg.wandb.enabled is True
    assert cfg.wandb.mode == "online"


def test_projector_finetune_optimizer_uses_discriminative_learning_rates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_train_script(repo_root)

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language = nn.Linear(2, 2)
            self.path_projectors = nn.ModuleDict({"pathology": nn.Linear(2, 2)})

        def projector_modules(self):
            return [("pathology", self.path_projectors)]

    model = TinyModel()
    optimizer, trainable = module._build_sft_optimizer(
        model=model,
        train_model=model,
        stage_cfg={
            "learning_rate": 1e-4,
            "projector_learning_rate": 1e-5,
            "weight_decay": 0.0,
        },
    )

    assert len(trainable) == 4
    assert {group["group_name"]: group["lr"] for group in optimizer.param_groups} == {
        "language_lora": pytest.approx(1e-4),
        "projectors": pytest.approx(1e-5),
    }


def test_saved_joint_projector_checkpoint_can_be_reloaded(tmp_path: Path) -> None:
    from kidney_vlm.modeling.path_projectors import ModalityProjector
    from kidney_vlm.vqa.modeling import (
        OncoVLMVQASFTModel,
        build_projector_module,
        save_vqa_model_artifacts,
    )

    class DummyLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 4, "use_cache": True})()
            self.weight = nn.Parameter(torch.ones(1))

        def save_pretrained(self, output_dir: Path) -> None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    class DummyTokenizer:
        def save_pretrained(self, output_dir: Path) -> None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    pathology_projector = nn.ModuleDict(
        {
            "pathology": ModalityProjector(
                in_dim=3,
                out_dim=4,
                projector_type="mlp",
                num_latents=2,
                depth=1,
                num_heads=1,
                mlp_ratio=2.0,
                dropout=0.0,
            )
        }
    )
    model = OncoVLMVQASFTModel(
        language_model=DummyLanguageModel(),
        projectors={"pathology": pathology_projector},
        projector_metadata={
            "pathology": {
                "trainable": True,
                "embedding_dim": 3,
                "hidden_size": 4,
                "projector_type": "mlp",
                "projector_num_latents": 2,
                "projector_depth": 1,
                "projector_num_heads": 1,
                "projector_mlp_ratio": 2.0,
                "projector_dropout": 0.0,
                "prefix_tokens": 0,
                "prefix_expander_mlp_ratio": 1.0,
            }
        },
    )
    artifacts = save_vqa_model_artifacts(
        artifact_dir=tmp_path / "artifact",
        stage_cfg={
            "model_name_or_path": "dummy/model",
            "save_tokenizer_snapshot": False,
        },
        model=model,
        tokenizer=DummyTokenizer(),
        global_step=3,
        epoch=1,
        validation_loss=1.0,
    )

    checkpoint_path = Path(artifacts["projectors_checkpoint"])
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["hidden_size"] == 4

    reloaded, metadata = build_projector_module(
        repo_root=tmp_path,
        modality="pathology",
        block_cfg={
            "checkpoint_path": str(checkpoint_path),
            "trainable": True,
        },
        hidden_size=4,
    )
    assert metadata["embedding_dim"] == 3
    assert metadata["projector_mlp_ratio"] == 2.0
    assert all(parameter.requires_grad for parameter in reloaded.parameters())
    for key, value in pathology_projector.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key])
