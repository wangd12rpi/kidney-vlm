#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.manifest import write_run_manifest
from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.sources.kits23 import (
    build_kits23_registry_rows,
    extract_kits23_medsiglip_features,
    extract_kits23_slice_mask_pairs,
)
from kidney_vlm.data.unified_registry import replace_source_slice
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg(source_name: str = "kits23", overrides: list[str] | None = None) -> DictConfig:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        base_cfg = compose(config_name="config")
    OmegaConf.set_struct(base_cfg, False)

    source_cfg_path = conf_dir / "data" / "sources" / f"{source_name}.yaml"
    if not source_cfg_path.exists():
        raise FileNotFoundError(f"Missing source config: {source_cfg_path}")

    source_cfg = OmegaConf.load(source_cfg_path)
    merged = OmegaConf.merge(base_cfg, source_cfg)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, override_cfg)

    OmegaConf.set_struct(merged, False)
    merged.project.root_dir = str(ROOT)
    return merged


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _resolve_path(path_value: str) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _split_ratios(kits_cfg: DictConfig) -> dict[str, float]:
    split_cfg = OmegaConf.to_container(kits_cfg.split_ratios, resolve=True) or {}
    return {
        "train": float(split_cfg.get("train", 0.8)),
        "val": float(split_cfg.get("val", 0.1)),
        "test": float(split_cfg.get("test", 0.1)),
    }


def main() -> None:
    cfg = load_cfg("kits23", overrides=sys.argv[1:])
    source_name = str(cfg.data.source.name)
    kits_cfg = cfg.data.source.kits23

    dataset_dir = _resolve_path(str(kits_cfg.dataset_dir))
    pairs_root = _resolve_path(str(kits_cfg.pairs_root))
    image_glob = str(kits_cfg.image_glob)
    target_axis = int(kits_cfg.extract.target_axis)
    split_ratios = _split_ratios(kits_cfg)

    extraction_stats: dict[str, int] = {}
    extraction_enabled = bool(kits_cfg.extract.enabled)

    if extraction_enabled:
        extraction_stats = extract_kits23_slice_mask_pairs(
            dataset_dir=dataset_dir,
            pairs_root=pairs_root,
            target_axis=target_axis,
            skip_existing=bool(kits_cfg.extract.skip_existing),
            max_cases=_optional_int(kits_cfg.extract.max_cases),
            show_progress=True,
        )
    

    feature_stats: dict[str, Any] = {}
    features_cfg = kits_cfg.get("features")
    features_enabled = bool(features_cfg.enabled) if features_cfg is not None else False
    attach_existing_features = (
        bool(features_cfg.get("attach_existing", True)) if features_cfg is not None else False
    )
    embeddings_root: Path | None = None

    if features_cfg is not None:
        embeddings_root = _resolve_path(str(features_cfg.output_root))
    if features_enabled:
        if features_cfg is None or embeddings_root is None:
            raise ValueError("KITS23 feature extraction is enabled but configuration is incomplete.")

        hf_token_env_var = _optional_str(features_cfg.get("hf_token_env_var")) or "HF_TOKEN"
        hf_token = _optional_str(os.environ.get(hf_token_env_var))

        feature_stats = extract_kits23_medsiglip_features(
            images_root=pairs_root / "images",
            output_root=embeddings_root,
            model_id=str(features_cfg.get("model_id", "google/medsiglip-448")),
            image_glob=image_glob,
            batch_size=int(features_cfg.get("batch_size", 8)),
            device=_optional_str(features_cfg.get("device")),
            overwrite_existing=bool(features_cfg.get("overwrite_existing", False)),
            max_images=_optional_int(features_cfg.get("max_images")),
            skip_errors=bool(features_cfg.get("skip_errors", True)),
            show_progress=True,
            use_tensorflow_resize=bool(features_cfg.get("use_tensorflow_resize", True)),
            hf_token=hf_token,
        )

    source_df = build_kits23_registry_rows(
        pairs_root=pairs_root,
        source_name=source_name,
        split_ratios=split_ratios,
        image_glob=image_glob,
        slice_axis=target_axis,
        embeddings_root=embeddings_root if attach_existing_features else None,
        show_progress=True,
    )

    staging_root = _resolve_path(str(cfg.data.staging_root))
    staging_path = staging_root / f"{source_name}.parquet"
    write_registry_parquet(source_df, staging_path, validate=False)

    unified_path = _resolve_path(str(cfg.data.unified_registry_path))
    unified_df = read_parquet_or_empty(unified_path)
    merged_df = replace_source_slice(unified_df, source_df, source_name=source_name)
    write_registry_parquet(merged_df, unified_path, validate=False)

    manifest_path = write_run_manifest(
        manifests_root=_resolve_path(str(cfg.data.manifests_root)),
        repo_root=ROOT,
        source_name=source_name,
        source_row_count=len(source_df),
        staging_path=staging_path,
        unified_path=unified_path,
        extra={
            "dataset_dir": str(dataset_dir),
            "pairs_root": str(pairs_root),
            "image_glob": image_glob,
            "split_ratios": split_ratios,
            "extraction_enabled": extraction_enabled,
            "extraction_stats": extraction_stats,
            "features_enabled": features_enabled,
            "features_attach_existing": attach_existing_features,
            "features_output_root": str(embeddings_root) if embeddings_root is not None else "",
            "feature_stats": feature_stats,
        },
    )

    print(f"KITS23 source build complete: {source_name}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Pairs root: {pairs_root}")
    print(f"Rows written: {len(source_df)}")
    if extraction_stats:
        print(f"Extraction stats: {extraction_stats}")
    if feature_stats:
        print(f"Feature stats: {feature_stats}")
    print(f"Staging parquet: {staging_path}")
    print(f"Unified parquet: {unified_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
