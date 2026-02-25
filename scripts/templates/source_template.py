#!/usr/bin/env python3
"""Template source script.

Copy this file to scripts/data/01_build_<source>_source.py and update source-specific logic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.manifest import write_run_manifest
from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.registry_schema import empty_registry_frame
from kidney_vlm.data.unified_registry import replace_source_slice
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg(source_name: str) -> DictConfig:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config")
    source_cfg_path = conf_dir / "data" / "sources" / f"{source_name}.yaml"
    source_cfg = OmegaConf.load(source_cfg_path)
    merged = OmegaConf.merge(cfg, source_cfg)
    OmegaConf.set_struct(merged, False)
    merged.project.root_dir = str(ROOT)
    return merged


def build_rows(_cfg: DictConfig) -> pd.DataFrame:
    # TODO: implement pull/index/harmonize for this source.
    return empty_registry_frame()


def main() -> None:
    raise RuntimeError(
        "This is a template and should not be run directly. "
        "Copy it to scripts/data/01_build_<source>_source.py first."
    )


if __name__ == "__main__":
    main()
