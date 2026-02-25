#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        return compose(config_name="config")


def main() -> None:
    cfg = load_cfg()
    radiology_cfg = OmegaConf.load(ROOT / "conf" / "model" / "segmentation" / "radiology_placeholder.yaml")
    print("Segmentation scaffold ready.")
    print(f"Pathology segmentation config: {cfg.model.segmentation.pathology.name}")
    print(f"Radiology segmentation config: {radiology_cfg.radiology.name}")
    print(
        "Concrete segmentation model wiring is intentionally deferred. "
        "Populate adapters after selecting production checkpoints/APIs."
    )


if __name__ == "__main__":
    main()
