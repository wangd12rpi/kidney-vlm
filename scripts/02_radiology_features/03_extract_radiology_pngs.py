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
from kidney_vlm.radiology.script_config import optional_int
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.radiology.tcga_png_extraction import TCGARadiologyPngExtractor
from kidney_vlm.radiology.tcga_series_manifest import read_series_manifest, write_series_manifest


ROOT = find_repo_root(Path(__file__))


def main() -> None:
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="02_radiology_features/03_extract_radiology_pngs.yaml",
        overrides=sys.argv[1:],
    )
    manifest_path = Path(str(cfg.radiology.series_manifest_path))
    manifest_df = read_series_manifest(manifest_path)
    if manifest_df.empty:
        print("Radiology series manifest is empty. Run 02_prepare_radiology_series_manifest.py first.")
        return

    extractor = TCGARadiologyPngExtractor(
        root_dir=ROOT,
        raw_root=Path(str(cfg.radiology.raw_root)),
        png_root=Path(str(cfg.radiology.png_root)),
        overwrite_pngs=bool(cfg.radiology.overwrite_pngs),
        png_render_mode=str(cfg.radiology.png_render_mode),
        prefer_dicom_voi=bool(cfg.radiology.prefer_dicom_voi),
        apply_padding_mask=bool(cfg.radiology.apply_padding_mask),
        png_resize=optional_int(cfg.radiology.png_resize),
    )

    processed = 0
    for row_idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting radiology PNGs", unit="series"):
        if not bool(row["accepted"]):
            continue
        usable_dicom_paths = [Path(str(path)).resolve() for path in list(row["usable_dicom_paths"] or []) if str(path).strip()]
        selected_series_dir = str(row["selected_series_dir"]).strip()
        if not usable_dicom_paths or not selected_series_dir:
            continue
        if bool(row["processed_png"]) and list(row["png_paths"] or []) and not bool(cfg.radiology.overwrite_pngs):
            continue

        result = extractor.extract_series(
            series_dir=Path(selected_series_dir),
            usable_image_paths=usable_dicom_paths,
        )
        manifest_df.at[row_idx, "png_dir"] = result.png_dir
        manifest_df.at[row_idx, "png_paths"] = list(result.png_paths)
        manifest_df.at[row_idx, "slice_count"] = int(result.slice_count)
        manifest_df.at[row_idx, "processed_png"] = True
        processed += 1

    write_series_manifest(manifest_df, manifest_path)
    print(f"Radiology PNG series processed: {processed}")
    print(f"Series manifest: {manifest_path}")


if __name__ == "__main__":
    main()
