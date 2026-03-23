#!/usr/bin/env python3
from __future__ import annotations

import json
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
from kidney_vlm.data.registry_io import write_registry_parquet
from kidney_vlm.data.sources.pmc_oa import build_pmc_oa_ct_registry_rows
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


def _to_string_dict(value: Any, field_name: str) -> dict[str, str]:
    mapping = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if not isinstance(mapping, dict):
        raise ValueError(f"Expected '{field_name}' to be a mapping, got {type(mapping).__name__}")
    normalized: dict[str, str] = {}
    for key, item in mapping.items():
        key_text = str(key).strip()
        item_text = str(item).strip()
        if not key_text or not item_text:
            continue
        normalized[key_text] = item_text
    return normalized


def _to_optional_split_limits(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    mapping = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if not isinstance(mapping, dict):
        return None
    normalized: dict[str, int] = {}
    for key, item in mapping.items():
        key_text = str(key).strip()
        limit = _optional_int(item)
        if not key_text or limit is None:
            continue
        normalized[key_text] = limit
    return normalized or None


def _to_string_list(value: Any) -> list[str]:
    items = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError(f"Expected a list, got {type(items).__name__}")
    normalized: list[str] = []
    for item in items:
        if isinstance(item, (dict, list, tuple)):
            raise ValueError(
                f"Expected a list of strings, got nested {type(item).__name__}: {item!r}"
            )
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def main() -> None:
    overrides = sys.argv[1:]
    cfg = load_cfg("pmc_oa", overrides=overrides)

    source_name = str(cfg.data.source.name)
    if source_name != "pmc_oa":
        raise ValueError(f"Expected data.source.name='pmc_oa', got '{source_name}'")

    pmc_cfg = cfg.data.source.pmc_oa
    split_files = _to_string_dict(pmc_cfg.split_files, "data.source.pmc_oa.split_files")
    if not split_files:
        raise ValueError("No split files configured in data.source.pmc_oa.split_files.")

    registry_path = Path(str(pmc_cfg.output_registry_path)).expanduser()
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path

    source_df, stats = build_pmc_oa_ct_registry_rows(
        dataset_root=Path(str(pmc_cfg.dataset_root)).expanduser(),
        split_files=split_files,
        source_name=source_name,
        project_root=ROOT,
        image_dir=str(pmc_cfg.image_dir),
        selected_splits=_to_string_list(pmc_cfg.selected_splits),
        max_rows_total=_optional_int(pmc_cfg.max_rows_total),
        max_rows_per_split=_to_optional_split_limits(pmc_cfg.max_rows_per_split),
        verify_image_paths=bool(pmc_cfg.verify_image_paths),
        skip_rows_with_missing_images=bool(pmc_cfg.skip_rows_with_missing_images),
        skip_invalid_records=bool(pmc_cfg.skip_invalid_records),
        fail_on_missing_split_files=bool(pmc_cfg.fail_on_missing_split_files),
        show_progress=bool(pmc_cfg.show_progress),
        ct_include_patterns=_to_string_list(pmc_cfg.filters.include_patterns),
        ct_exclude_patterns=_to_string_list(pmc_cfg.filters.exclude_patterns),
        ambiguous_modality_patterns=_to_string_list(pmc_cfg.filters.ambiguous_modality_patterns),
    )

    write_registry_parquet(source_df, registry_path, validate=False)

    manifest_path = write_run_manifest(
        manifests_root=Path(str(cfg.data.manifests_root)),
        repo_root=ROOT,
        source_name=source_name,
        source_row_count=len(source_df),
        staging_path=registry_path,
        unified_path=registry_path,
        extra={
            "dataset_root": str(pmc_cfg.dataset_root),
            "registry_path": str(registry_path),
            "unified_updated": False,
            "notes": (
                "PMC-OA is handled as an exception dataset. "
                "Only CT-caption rows are saved, and they are written to a separate registry parquet."
            ),
            "stats": stats,
        },
    )

    print(f"PMC-OA CT registry build complete: {source_name}")
    print(f"Rows written: {len(source_df)}")
    print(f"Registry parquet: {registry_path}")
    print(f"Unified parquet unchanged: {cfg.data.unified_registry_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {json.dumps(stats, indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
