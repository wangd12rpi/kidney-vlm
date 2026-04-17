#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.radiology.feature_registry import (
    RadiologySeriesArtifactRecord,
    register_radiology_series_artifacts,
)
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.radiology.tcga_series_manifest import read_series_manifest


ROOT = find_repo_root(Path(__file__))


def main() -> None:
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="02_radiology_features/06_register_radiology_artifacts_into_registry.yaml",
        overrides=sys.argv[1:],
    )
    registry_path = Path(str(cfg.radiology.registry_path))
    manifest_path = Path(str(cfg.radiology.series_manifest_path))

    registry_df = read_parquet_or_empty(registry_path)
    manifest_df = read_series_manifest(manifest_path)
    if manifest_df.empty:
        print("Radiology series manifest is empty. Nothing to register.")
        return

    artifacts_by_series_dir: dict[str, RadiologySeriesArtifactRecord] = {}
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Indexing radiology artifacts", unit="series"):
        series_dir = str(row["selected_series_dir"]).strip()
        png_dir = str(row["png_dir"]).strip()
        source_zip_path = str(row["source_zip_path"]).strip()
        if not any([series_dir, png_dir, source_zip_path]):
            continue
        artifact = RadiologySeriesArtifactRecord(
            series_dir=series_dir,
            png_dir=png_dir,
            embedding_ref=str(row["embedding_ref"]).strip(),
            slice_count=int(row["slice_count"]),
            source_zip_path=source_zip_path,
            mask_paths=tuple(str(path).strip() for path in list(row["mask_paths"] or []) if str(path).strip()),
            mask_manifest_path=str(row["mask_manifest_path"]).strip(),
        )
        key = series_dir or png_dir or source_zip_path
        artifacts_by_series_dir[key] = artifact

    updated_df, stats = register_radiology_series_artifacts(
        registry_df,
        root_dir=ROOT,
        artifacts_by_series_dir=artifacts_by_series_dir,
    )
    write_registry_parquet(updated_df, registry_path, validate=False)

    print(f"Radiology artifacts indexed: {stats.series_artifacts_indexed}")
    print(f"Registry rows scanned: {stats.cases_scanned}")
    print(f"Registry rows with matches: {stats.cases_with_matches}")
    print(f"Matched embedding refs: {stats.matched_series_refs}")
    print(f"Registry updated: {registry_path}")


if __name__ == "__main__":
    main()
