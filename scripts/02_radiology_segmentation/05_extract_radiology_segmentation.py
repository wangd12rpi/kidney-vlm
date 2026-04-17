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
from kidney_vlm.radiology.tcga_segmentation_extraction import TCGARadiologySegmentationExtractor
from kidney_vlm.radiology.tcga_series_manifest import read_series_manifest, write_series_manifest


ROOT = find_repo_root(Path(__file__))


def main() -> None:
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="02_radiology_segmentation/05_extract_radiology_segmentation.yaml",
        overrides=sys.argv[1:],
    )
    manifest_path = Path(str(cfg.radiology.series_manifest_path))
    manifest_df = read_series_manifest(manifest_path)
    if manifest_df.empty:
        print("Radiology series manifest is empty. Run 02_prepare_radiology_series_manifest.py first.")
        return

    extractor = TCGARadiologySegmentationExtractor(
        root_dir=ROOT,
        raw_root=Path(str(cfg.radiology.raw_root)),
        mask_root=Path(str(cfg.radiology.mask_root)),
        keyword_map_path=Path(str(cfg.radiology.keyword_map_path)),
        checkpoint_path=Path(str(cfg.radiology.checkpoint_path)),
        medical_sam3_root=Path(str(cfg.radiology.medical_sam3_root)),
        sam3_root=Path(str(cfg.radiology.sam3_root)),
        input_size=int(cfg.radiology.input_size),
        confidence_threshold=float(cfg.radiology.confidence_threshold),
        device=str(cfg.radiology.device),
        overwrite_masks=bool(cfg.radiology.overwrite_masks),
        skip_existing_masks=bool(cfg.radiology.skip_existing_masks),
        min_mask_pixels=int(cfg.radiology.min_mask_pixels),
    )

    processed = 0
    for row_idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting radiology masks", unit="series"):
        if not bool(row["accepted"]):
            continue
        png_paths = [Path(str(path)).resolve() for path in list(row["png_paths"] or []) if str(path).strip()]
        source_paths = [Path(str(path)).resolve() for path in list(row["usable_dicom_paths"] or []) if str(path).strip()]
        selected_series_dir = str(row["selected_series_dir"]).strip()
        if not png_paths or not selected_series_dir:
            continue
        if bool(row["processed_segmentation"]) and list(row["mask_paths"] or []) and not bool(cfg.radiology.overwrite_masks):
            continue

        result = extractor.extract_series(
            series_dir=Path(selected_series_dir),
            png_paths=png_paths,
            source_image_paths=source_paths,
            collection=str(row["collection"]),
            patient_id=str(row["patient_id"]),
            study_instance_uid=str(row["study_instance_uid"]),
            series_instance_uid=str(row["series_instance_uid"]),
            modality=str(row["modality"]),
        )
        manifest_df.at[row_idx, "mask_dir"] = result.mask_dir
        manifest_df.at[row_idx, "mask_paths"] = list(result.mask_paths)
        manifest_df.at[row_idx, "mask_manifest_path"] = result.manifest_path
        manifest_df.at[row_idx, "segmentation_keywords"] = list(result.keywords)
        manifest_df.at[row_idx, "processed_segmentation"] = True
        processed += 1

    write_series_manifest(manifest_df, manifest_path)
    print(f"Radiology segmentation series processed: {processed}")
    print(f"Series manifest: {manifest_path}")


if __name__ == "__main__":
    main()
