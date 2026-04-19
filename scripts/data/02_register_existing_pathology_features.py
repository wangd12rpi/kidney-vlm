#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.pathology.feature_registry import register_existing_pathology_features
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    return cfg


def main() -> None:
    cfg = load_cfg()
    pathology_cfg = cfg.embeding_extraction.pathology

    registry_path = Path(str(cfg.data.unified_registry_path)).expanduser().resolve()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Unified registry not found at '{registry_path}'. "
            "Build a source first before registering existing pathology features."
        )

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        print(f"Registry is empty at '{registry_path}'. Nothing to register.")
        return

    patch_encoder_name = str(pathology_cfg.patch_encoder)
    save_format = str(pathology_cfg.get("save_format", "h5")).lower()
    patch_size = int(pathology_cfg.get("patch_size", 512))
    target_mag = int(pathology_cfg.get("target_magnification", 20))

    features_root = Path(str(pathology_cfg.get("features_root", ROOT / "data" / "features"))).expanduser()
    if not features_root.is_absolute():
        features_root = (ROOT / features_root).resolve()
    else:
        features_root = features_root.resolve()

    patch_features_dir = features_root / f"features_{patch_encoder_name}"
    if not patch_features_dir.exists():
        raise FileNotFoundError(
            f"Patch features directory not found at '{patch_features_dir}'. "
            "Copy existing feature files there before running this script."
        )

    coords_root = Path(
        str(
            pathology_cfg.get(
                "coords_root",
                features_root / f"coords_{target_mag}x_{patch_size}px_{int(pathology_cfg.get('patch_overlap', 0))}px_overlap",
            )
        )
    ).expanduser()
    if not coords_root.is_absolute():
        coords_root = (ROOT / coords_root).resolve()
    else:
        coords_root = coords_root.resolve()

    print(f"Registry: {registry_path}")
    print(f"Patch features dir: {patch_features_dir}")
    print(f"Coords root: {coords_root}")
    print(f"Patch encoder: {patch_encoder_name}")
    print(f"Save format: {save_format}")

    updated_df, stats = register_existing_pathology_features(
        registry_df,
        patch_features_dir=patch_features_dir,
        coords_root=coords_root,
        save_format=save_format,
        patch_size=patch_size,
        target_mag=target_mag,
        root_dir=ROOT,
        progress=True,
    )
    write_registry_parquet(updated_df, registry_path, validate=True)

    print("Existing pathology feature registration complete.")
    print(f"Cases scanned: {stats.cases_scanned}")
    print(f"Cases with slide paths: {stats.cases_with_slide_paths}")
    print(f"Cases with matched features: {stats.cases_with_matches}")
    print(f"Matched feature paths written: {stats.matched_feature_paths}")
    print(f"Feature files indexed: {stats.feature_files_indexed}")
    print(f"Invalid feature files skipped: {stats.invalid_feature_files}")
    print(f"Unified registry written: {registry_path}")


if __name__ == "__main__":
    main()
