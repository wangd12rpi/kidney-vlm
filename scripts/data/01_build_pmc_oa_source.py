#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.manifest import write_run_manifest
from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.data.sources.pmc_oa import (
    build_pmc_oa_caption_rows,
    build_pmc_oa_registry_rows,
    load_pmc_oa_caption_frame,
)
from kidney_vlm.data.unified_registry import replace_source_slice
from kidney_vlm.radiology.pmc_oa_feature_store import build_pmc_oa_lookup_by_image_name, read_or_build_pmc_oa_feature_index
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg(source_name: str = "pmc_oa", overrides: list[str] | None = None) -> DictConfig:
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
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    OmegaConf.set_struct(merged, False)
    merged.project.root_dir = str(ROOT)
    return merged


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def main() -> None:
    cfg = load_cfg("pmc_oa", overrides=sys.argv[1:])

    source_name = str(cfg.data.source.name).strip() or "pmc_oa"
    source_cfg = cfg.data.source.pmc_oa
    project_id = str(source_cfg.get("project_id", source_name)).strip() or source_name
    split_scheme_version = str(
        source_cfg.get("split_scheme_version", "pmc_oa_explicit_image_split_v1")
    ).strip() or "pmc_oa_explicit_image_split_v1"

    feature_store_path = _resolve_repo_path(source_cfg.feature_store_path)
    feature_index_path = _resolve_repo_path(source_cfg.feature_index_path)
    image_root_dir = _resolve_repo_path(source_cfg.image_root_dir)
    train_jsonl_path = _resolve_repo_path(source_cfg.train_jsonl_path)
    validation_jsonl_path = _resolve_repo_path(source_cfg.validation_jsonl_path)
    test_jsonl_path = _resolve_repo_path(source_cfg.test_jsonl_path)

    feature_index = read_or_build_pmc_oa_feature_index(
        root_dir=ROOT,
        store_path=feature_store_path,
        index_path=feature_index_path,
        rebuild=bool(source_cfg.get("rebuild_feature_index", False)),
    )
    if feature_index.empty:
        raise RuntimeError(f"PMC-OA feature index is empty: {feature_index_path}")
    feature_index_by_image_name = build_pmc_oa_lookup_by_image_name(feature_index)

    caption_frame = load_pmc_oa_caption_frame(
        train_jsonl_path=train_jsonl_path,
        validation_jsonl_path=validation_jsonl_path,
        test_jsonl_path=test_jsonl_path,
    )
    if caption_frame.empty:
        raise RuntimeError("PMC-OA caption splits are empty.")

    caption_rows, missing_feature_images, missing_image_files = build_pmc_oa_caption_rows(
        caption_frame,
        root_dir=ROOT,
        feature_index_by_image_name=feature_index_by_image_name,
        image_root_dir=image_root_dir,
        require_existing_image_files=bool(source_cfg.get("require_existing_image_files", False)),
        default_instruction=str(source_cfg.get("instruction", "Describe the radiology image.")).strip(),
        caption_model=str(source_cfg.get("caption_model", "pmc_oa_human")).strip(),
        source_name=source_name,
        project_id=project_id,
    )
    if not caption_rows:
        raise RuntimeError("No PMC-OA source rows were generated.")

    source_df = build_pmc_oa_registry_rows(
        caption_rows,
        source_name=source_name,
        project_id=project_id,
        split_scheme_version=split_scheme_version,
    )

    staging_root = _resolve_repo_path(cfg.data.staging_root)
    staging_path = staging_root / f"{source_name}.parquet"
    write_registry_parquet(source_df, staging_path, validate=False)

    unified_path = _resolve_repo_path(cfg.data.unified_registry_path)
    unified_df = read_parquet_or_empty(unified_path)
    merged_df = replace_source_slice(unified_df, source_df, source_name=source_name)
    write_registry_parquet(merged_df, unified_path, validate=False)

    manifest_path = write_run_manifest(
        manifests_root=_resolve_repo_path(cfg.data.manifests_root),
        repo_root=ROOT,
        source_name=source_name,
        source_row_count=len(source_df),
        staging_path=staging_path,
        unified_path=unified_path,
        extra={
            "project_id": project_id,
            "caption_rows_loaded": len(caption_frame),
            "caption_rows_matched": len(caption_rows),
            "feature_index_rows": len(feature_index),
            "feature_store_path": str(feature_store_path),
            "feature_index_path": str(feature_index_path),
            "missing_feature_matches": len(missing_feature_images),
            "missing_image_files": len(missing_image_files),
            "notes": "PMC-OA image-level radiology source normalized into the unified registry.",
        },
    )

    print(f"PMC-OA source registry upsert complete: {source_name}")
    print(f"Caption rows loaded: {len(caption_frame)}")
    print(f"Feature index rows: {len(feature_index)}")
    print(f"Rows written: {len(source_df)}")
    print(f"Staging parquet: {staging_path}")
    print(f"Unified parquet: {unified_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Missing feature matches: {len(missing_feature_images)}")
    print(f"Missing image files: {len(missing_image_files)}")

    if missing_feature_images:
        print("Missing feature image examples:")
        for image_name in missing_feature_images[:10]:
            print(f"  {image_name}")
        if bool(source_cfg.get("fail_on_missing_features", True)):
            raise RuntimeError(
                "Some PMC-OA caption rows did not have matching feature-store entries. "
                "Set data.source.pmc_oa.fail_on_missing_features=false to allow partial output."
            )

    if missing_image_files:
        print("Missing image file examples:")
        for image_name in missing_image_files[:10]:
            print(f"  {image_name}")


if __name__ == "__main__":
    main()
