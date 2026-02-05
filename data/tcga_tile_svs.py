"""Tile downloaded TCGA SVS into tissue-rich patches for MedGemma.

This script reads data/tcga_path_index.jsonl and, for each downloaded SVS:
- creates <case_folder>/tiles/
- writes tiles_manifest.jsonl

Resumable behavior:
- If tiles_manifest.jsonl already has >= MAX_TILES lines, that case is skipped.

Run:
  python data/tcga_tile_svs.py
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from kidney_vlm.jsonl import read_jsonl
from kidney_vlm.tiling import tile_svs_to_dir


# -----------------
# Config (edit me)
# -----------------
INDEX_PATH = Path("data/tcga_path_index.jsonl")
MAX_TILES = 64
TILE_SIZE = 896  # MedGemma images are normalized to 896x896 by the processor
TARGET_MPP = 1.0  # about 10x if base is 0.25 mpp
TISSUE_MIN_FRAC = 0.30
THUMB_MAX_SIZE = 2048
BG_GRAY_THRESHOLD = 220


def main() -> None:
    rows = read_jsonl(INDEX_PATH)
    print(f"Index rows: {len(rows)}")

    for obj in tqdm(rows, desc="tiling"):
        svs_path = Path(obj["raw_svs_path"])
        tiles_dir = Path(obj["tiles_dir"])
        if not svs_path.exists():
            continue

        tile_svs_to_dir(
            svs_path=svs_path,
            tiles_dir=tiles_dir,
            tile_size=TILE_SIZE,
            max_tiles=MAX_TILES,
            target_mpp=TARGET_MPP,
            tissue_min_fraction=TISSUE_MIN_FRAC,
            thumb_max_size=THUMB_MAX_SIZE,
            bg_gray_threshold=BG_GRAY_THRESHOLD,
        )

    print("Done")


if __name__ == "__main__":
    main()
