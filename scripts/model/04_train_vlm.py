#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.hf_dataset import load_hf_dataset_from_registry
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.training.trainer_hf import build_training_arguments

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg(overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        return compose(config_name="config", overrides=overrides or [])


def main() -> None:
    cfg = load_cfg(overrides=sys.argv[1:])
    stage_cfg = cfg.vlm_train

    registry_path = Path(str(stage_cfg.get("registry_path", cfg.data.unified_registry_path))).expanduser()
    if not registry_path.is_absolute():
        registry_path = (ROOT / registry_path).resolve()
    else:
        registry_path = registry_path.resolve()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"VLM training registry not found at '{registry_path}'. "
            "Generate the stage-specific supervision parquet first."
        )

    registry_df = pd.read_parquet(registry_path)
    required_columns = [
        str(stage_cfg.get("question_field", "question")),
        str(stage_cfg.get("answer_field", "answer")),
        str(stage_cfg.get("feature_field", "pathology_slide_embedding_paths")),
    ]
    missing_columns = [column for column in required_columns if column not in registry_df.columns]
    if missing_columns:
        raise ValueError(
            f"VLM training registry is missing required columns: {missing_columns}. "
            "Build the derived instruct dataset and feature columns before running stage 3."
        )

    train_dataset = load_hf_dataset_from_registry(registry_path, split_filter="train")
    args = build_training_arguments(stage_cfg)

    print(f"VLM training stage: {stage_cfg.name}")
    print(f"Train rows: {len(train_dataset)}")
    print(f"Registry: {registry_path}")
    print(f"Model family: {stage_cfg.get('model_family', 'generic')}")
    print(f"Model checkpoint: {stage_cfg.model_name_or_path}")
    print(f"Feature field: {stage_cfg.get('feature_field', 'pathology_slide_embedding_paths')}")
    print(f"Question field: {stage_cfg.get('question_field', 'question')}")
    print(f"Answer field: {stage_cfg.get('answer_field', 'answer')}")
    print(f"Output dir: {args.output_dir}")
    print(f"Freeze projectors in stage 3: {bool(stage_cfg.freeze_projectors)}")
    if bool(stage_cfg.get("lora", {}).get("enabled", False)):
        print(
            "LoRA enabled: "
            f"r={stage_cfg.lora.r}, alpha={stage_cfg.lora.alpha}, dropout={stage_cfg.lora.dropout}"
        )
    if not bool(cfg.vlm_train.instantiate_model):
        print(
            "Model instantiation is disabled by config (vlm_train.instantiate_model=false). "
            "Enable it once the multimodal Qwen adapter and LoRA wiring are finalized. "
            "This script assumes projector pretraining and TCGA caption-stage tuning have already completed."
        )
        return

    raise NotImplementedError(
        "Qwen LoRA multimodal training is not wired yet. The remaining implementation step is to "
        "load the trained projector checkpoint, fuse projected pathology embeddings into Qwen inputs, "
        "and then wrap the language backbone with PEFT LoRA adapters."
    )


if __name__ == "__main__":
    main()
