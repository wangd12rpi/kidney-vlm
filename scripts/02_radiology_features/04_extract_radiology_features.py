#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.radiology.tcga_feature_extraction import TCGARadiologyFeatureExtractor
from kidney_vlm.radiology.tcga_series_manifest import read_series_manifest, write_series_manifest


ROOT = find_repo_root(Path(__file__))


def main() -> None:
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="02_radiology_features/04_extract_radiology_features.yaml",
        overrides=sys.argv[1:],
    )
    manifest_path = Path(str(cfg.radiology.series_manifest_path))
    manifest_df = read_series_manifest(manifest_path)
    if manifest_df.empty:
        print("Radiology series manifest is empty. Run 02_prepare_radiology_series_manifest.py first.")
        return

    extractor = TCGARadiologyFeatureExtractor(
        root_dir=ROOT,
        feature_store_path=Path(str(cfg.radiology.feature_store_path)),
        model_name=str(cfg.radiology.model_name),
        input_size=int(cfg.radiology.input_size),
        batch_size=int(cfg.radiology.batch_size),
        device=str(cfg.radiology.device),
        skip_existing_features=bool(cfg.radiology.skip_existing_features),
    )

    processed = 0
    for row_idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting radiology features", unit="series"):
        if not bool(row["accepted"]):
            continue
        png_paths = [Path(str(path)).resolve() for path in list(row["png_paths"] or []) if str(path).strip()]
        source_paths = [Path(str(path)).resolve() for path in list(row["usable_dicom_paths"] or []) if str(path).strip()]
        if not png_paths:
            continue
        if bool(row["processed_features"]) and str(row["embedding_ref"]).strip() and bool(cfg.radiology.skip_existing_features):
            continue

        result = extractor.extract_series(
            png_paths=png_paths,
            source_image_paths=source_paths,
            project_id=str(row["collection"]),
            patient_id=str(row["patient_id"]),
            study_instance_uid=str(row["study_instance_uid"]),
            series_instance_uid=str(row["series_instance_uid"]),
            modality=str(row["modality"]),
        )
        manifest_df.at[row_idx, "png_dir"] = result.png_dir
        manifest_df.at[row_idx, "png_paths"] = list(result.png_paths)
        manifest_df.at[row_idx, "embedding_ref"] = result.embedding_ref
        manifest_df.at[row_idx, "slice_count"] = int(result.slice_count)
        manifest_df.at[row_idx, "processed_features"] = True
        processed += 1

    write_series_manifest(manifest_df, manifest_path)
    print(f"Radiology feature series processed: {processed}")
    print(f"Series manifest: {manifest_path}")


if __name__ == "__main__":
    main()
