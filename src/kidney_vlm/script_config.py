from __future__ import annotations

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


_SCRIPT_CONFIG_PACKAGE_BY_DIR = {
    "01_pathology_features": "pathology_features",
    "01_pathology_proj": "pathology_proj",
    "01_pathology_segmentation": "pathology_segmentation",
    "02_radiology_proj": "radiology_proj",
    "03_dnam_proj": "dnam_proj",
}


def _maybe_wrap_script_cfg(*, config_relative_path: str, script_cfg: DictConfig) -> DictConfig:
    config_dir_name = Path(config_relative_path).parts[0]
    package_name = _SCRIPT_CONFIG_PACKAGE_BY_DIR.get(config_dir_name)
    if not package_name:
        return script_cfg
    if package_name in script_cfg:
        return script_cfg
    return OmegaConf.create({package_name: OmegaConf.to_container(script_cfg, resolve=False)})


def load_script_cfg(
    *,
    repo_root: Path,
    config_relative_path: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    conf_dir = repo_root / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        base_cfg = compose(config_name="config")
    OmegaConf.set_struct(base_cfg, False)

    script_cfg_path = conf_dir / config_relative_path
    if not script_cfg_path.exists():
        raise FileNotFoundError(f"Missing script config: {script_cfg_path}")

    script_cfg = OmegaConf.load(script_cfg_path)
    script_cfg = _maybe_wrap_script_cfg(config_relative_path=config_relative_path, script_cfg=script_cfg)
    merged = OmegaConf.merge(base_cfg, script_cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    OmegaConf.set_struct(merged, False)
    merged.project.root_dir = str(repo_root)
    os.environ["KIDNEY_VLM_ROOT"] = str(repo_root)
    return merged
