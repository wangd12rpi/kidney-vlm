from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_radiology_script_cfg(
    *,
    repo_root: Path,
    config_name: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    conf_dir = repo_root / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        base_cfg = compose(config_name="config")
    OmegaConf.set_struct(base_cfg, False)

    script_cfg_path = conf_dir / "radiology" / config_name
    if not script_cfg_path.exists():
        raise FileNotFoundError(f"Missing radiology script config: {script_cfg_path}")

    script_cfg = OmegaConf.load(script_cfg_path)
    merged = OmegaConf.merge(base_cfg, script_cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    OmegaConf.set_struct(merged, False)
    merged.project.root_dir = str(repo_root)
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)
    return merged


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)
