#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import torch

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.pmc_oa_caption_dataset import PMCOACaptionDataset
from kidney_vlm.modeling.pmc_oa_caption import PMCOACaptionProjectorModel
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.training.collator import ProjectorCaptionCollator
from kidney_vlm.training.freeze_policy import apply_training_stage, count_trainable_parameters
from kidney_vlm.training.projector_trainer import train_projector_caption_model

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg(overrides: list[str] | None = None) -> DictConfig:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        base_cfg = compose(config_name="config", overrides=overrides or [])
    OmegaConf.set_struct(base_cfg, False)
    base_cfg.project.root_dir = str(ROOT)
    return base_cfg


def _optional_int(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _optional_path(value) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def main() -> None:
    cfg = load_cfg(overrides=sys.argv[1:])
    stage_cfg = cfg.projector_train
    registry_path = Path(str(stage_cfg.registry_path)).expanduser()
    if not registry_path.is_absolute():
        registry_path = (ROOT / registry_path).resolve()
    else:
        registry_path = registry_path.resolve()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Projector-training registry not found at '{registry_path}'. "
            "Build the source registry and extract the configured visual features first."
        )

    train_dataset = PMCOACaptionDataset(
        registry_path,
        split=str(stage_cfg.train_split),
        dataset_name=str(stage_cfg.dataset_name),
        feature_field=str(stage_cfg.feature_field),
        caption_field=str(stage_cfg.caption_field),
        fallback_caption_fields=list(stage_cfg.fallback_caption_fields),
        root_dir=ROOT,
        max_rows=_optional_int(stage_cfg.max_train_rows),
    )
    eval_dataset = None
    val_split = str(stage_cfg.get("val_split", "") or "").strip()
    if val_split:
        try:
            eval_dataset = PMCOACaptionDataset(
                registry_path,
                split=val_split,
                dataset_name=str(stage_cfg.dataset_name),
                feature_field=str(stage_cfg.feature_field),
                caption_field=str(stage_cfg.caption_field),
                fallback_caption_fields=list(stage_cfg.fallback_caption_fields),
                root_dir=ROOT,
                max_rows=_optional_int(stage_cfg.max_val_rows),
            )
        except ValueError:
            eval_dataset = None

    if train_dataset.feature_token_count != 1:
        raise ValueError(
            "This projector training path expects pooled image features with one token per sample, "
            f"but the dataset reported feature_token_count={train_dataset.feature_token_count}."
        )

    visual_input_dim = int(train_dataset.feature_dim)
    configured_input_dim = _optional_int(stage_cfg.model.get("input_dim"))
    if configured_input_dim is not None and configured_input_dim != visual_input_dim:
        raise ValueError(
            f"Configured projector input_dim={configured_input_dim} does not match "
            f"dataset feature_dim={visual_input_dim}."
        )

    stage_name = str(stage_cfg.get("name", "projector_train")).strip() or "projector_train"
    print(f"Projector training stage: {stage_name}")
    print(f"Train rows: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Val rows: {len(eval_dataset)}")
    print(f"Registry: {registry_path}")
    print(f"Feature dataset: {stage_cfg.dataset_name}")
    print(f"Feature field: {stage_cfg.feature_field}")
    print(f"Caption field: {stage_cfg.caption_field}")
    print(f"Inferred visual feature shape: ({train_dataset.feature_token_count}, {visual_input_dim})")
    print(f"Output dir: {stage_cfg.output_dir}")
    print(f"Always frozen prefixes: {list(stage_cfg.always_frozen_prefixes)}")
    print(f"Projector prefixes: {list(stage_cfg.projector_prefixes)}")
    if bool(stage_cfg.wandb.enabled):
        print(f"W&B project: {stage_cfg.wandb.project}")

    if not bool(stage_cfg.instantiate_model):
        print("Model instantiation disabled by config. Set projector_train.instantiate_model=true to run training.")
        return

    model = PMCOACaptionProjectorModel.from_pretrained(
        llm_model_name_or_path=str(stage_cfg.model.llm_model_name_or_path),
        visual_input_dim=visual_input_dim,
        # trust_remote_code=bool(stage_cfg.model.trust_remote_code),
        trust_remote_code=bool(stage_cfg.model.get("trust_remote_code", True)),
        llm_dtype=str(stage_cfg.model.llm_dtype),
        freeze_llm=bool(stage_cfg.model.freeze_llm),
        gradient_checkpointing=bool(stage_cfg.model.get("gradient_checkpointing", True)),
        use_cache=bool(stage_cfg.model.get("use_cache", False)),
    )
    resume_projector_path = _optional_path(stage_cfg.get("resume_projector_path"))
    if resume_projector_path is not None:
        if not resume_projector_path.exists():
            raise FileNotFoundError(f"Projector checkpoint not found at '{resume_projector_path}'.")
        checkpoint = torch.load(resume_projector_path, map_location="cpu")
        projector_state_dict = checkpoint.get("projector_state_dict")
        if projector_state_dict is None:
            raise KeyError(
                f"Checkpoint '{resume_projector_path}' does not contain 'projector_state_dict'."
            )
        model.projector.load_state_dict(projector_state_dict)
        checkpoint_metadata = checkpoint.get("metadata") or {}
        print(f"Loaded projector checkpoint: {resume_projector_path}")
        if checkpoint_metadata:
            print(f"Checkpoint metadata: {checkpoint_metadata}")
    apply_training_stage(
        model,
        stage="projectors",
        always_frozen_prefixes=list(stage_cfg.always_frozen_prefixes),
        projector_prefixes=list(stage_cfg.projector_prefixes),
    )

    '''
    collator = ProjectorCaptionCollator(
        tokenizer=model.tokenizer,
        prompt_text=str(stage_cfg.prompt_text),
        prompt_max_length=int(stage_cfg.prompt_max_length),
        target_max_length=int(stage_cfg.target_max_length),
        prepend_bos_token=bool(stage_cfg.prepend_bos_token),
        append_eos_token=bool(stage_cfg.append_eos_token),
    )
    '''
    
    collator = ProjectorCaptionCollator(
        tokenizer=model.tokenizer,
        prompt_text=str(stage_cfg.prompt_text),
        system_text=str(stage_cfg.get("system_text", "You are a medical image analysis assistant.")),
        prompt_max_length=int(stage_cfg.prompt_max_length),
        target_max_length=int(stage_cfg.target_max_length),
        prepend_bos_token=bool(stage_cfg.prepend_bos_token),
        append_eos_token=bool(stage_cfg.append_eos_token),
        use_chatml=bool(stage_cfg.get("use_chatml", True)),
    )

    output_dir = Path(str(stage_cfg.output_dir)).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Total parameters: {model.count_total_parameters():,}")

    summary = train_projector_caption_model(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
        output_dir=output_dir,
        batch_size=int(stage_cfg.batch_size),
        eval_batch_size=int(stage_cfg.eval_batch_size),
        num_epochs=int(stage_cfg.num_epochs),
        learning_rate=float(stage_cfg.learning_rate),
        weight_decay=float(stage_cfg.weight_decay),
        grad_accumulation_steps=int(stage_cfg.gradient_accumulation_steps),
        max_grad_norm=float(stage_cfg.max_grad_norm),
        warmup_ratio=float(stage_cfg.warmup_ratio),
        warmup_steps=int(stage_cfg.warmup_steps),
        num_workers=int(stage_cfg.num_workers),
        seed=int(stage_cfg.seed),
        device=str(stage_cfg.device),
        precision=str(stage_cfg.precision),
        log_every_steps=int(stage_cfg.log_every_steps),
        max_train_batches_per_epoch=_optional_int(stage_cfg.max_train_batches_per_epoch),
        max_eval_batches=_optional_int(stage_cfg.max_eval_batches),
        wandb_config=OmegaConf.to_container(stage_cfg.wandb, resolve=True),
        run_config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"Training complete. Summary: {output_dir / 'training_summary.json'}")
    if summary.get("best_val_loss") is not None:
        print(f"Best val loss: {summary['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
