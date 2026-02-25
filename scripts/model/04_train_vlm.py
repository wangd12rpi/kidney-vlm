#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.hf_dataset import load_hf_dataset_from_registry
from kidney_vlm.training.trainer_hf import build_training_arguments


def load_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        return compose(config_name="config")


def main() -> None:
    cfg = load_cfg()
    registry_path = Path(str(cfg.data.unified_registry_path))
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Unified registry not found at '{registry_path}'. Build at least one source first."
        )

    train_dataset = load_hf_dataset_from_registry(registry_path, split_filter="train")
    stage_cfg = cfg.train.stages.vlm
    args = build_training_arguments(stage_cfg)

    print("Stage 2 (VLM) training scaffold ready.")
    print(f"Train rows: {len(train_dataset)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Freeze projectors in stage 2: {bool(stage_cfg.freeze_projectors)}")
    if not bool(cfg.train.instantiate_model):
        print(
            "Model instantiation is disabled by config (train.instantiate_model=false). "
            "Enable it and wire model-specific I/O once MedGemma checkpoint/API are finalized. "
            "This script assumes stage-1 projector training has already completed."
        )
        return

    raise NotImplementedError(
        "Model-specific MedGemma HF training integration is intentionally left incomplete in scaffold."
    )


if __name__ == "__main__":
    main()
