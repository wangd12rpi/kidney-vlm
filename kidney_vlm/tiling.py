from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import openslide


@dataclass(frozen=True)
class TileCandidate:
    x_level: int
    y_level: int
    tissue_frac: float


def _to_gray(arr_rgb: np.ndarray) -> np.ndarray:
    # arr_rgb: (H, W, 3) uint8
    r = arr_rgb[..., 0].astype(np.float32)
    g = arr_rgb[..., 1].astype(np.float32)
    b = arr_rgb[..., 2].astype(np.float32)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def _integral_image(mask: np.ndarray) -> np.ndarray:
    # mask: (H, W) bool or 0/1
    # returns (H+1, W+1) integral image
    m = mask.astype(np.int32)
    ii = np.zeros((m.shape[0] + 1, m.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(m, axis=0), axis=1)
    return ii


def _rect_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    # ii is (H+1, W+1); rectangle is [x0,x1) x [y0,y1)
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def pick_level_for_target_mpp(slide: openslide.OpenSlide, target_mpp: float) -> int:
    """Pick the slide level whose effective MPP is closest to target_mpp.

    If openslide mpp metadata is missing, fall back to the coarsest level.
    """
    mpp_x = slide.properties.get("openslide.mpp-x")
    if mpp_x is None:
        return slide.level_count - 1

    base_mpp = float(mpp_x)
    best_level = 0
    best_err = float("inf")

    for lvl, ds in enumerate(slide.level_downsamples):
        eff_mpp = base_mpp * float(ds)
        err = abs(eff_mpp - target_mpp)
        if err < best_err:
            best_err = err
            best_level = lvl

    return best_level


def generate_tissue_mask_thumbnail(
    slide: openslide.OpenSlide,
    thumb_max_size: int = 2048,
    bg_gray_threshold: int = 220,
) -> Tuple[np.ndarray, Image.Image]:
    """Return (mask, thumbnail_image).

    mask is True where tissue is likely present.
    """
    thumb = slide.get_thumbnail((thumb_max_size, thumb_max_size)).convert("RGB")
    arr = np.array(thumb)
    gray = _to_gray(arr)
    tissue = gray < float(bg_gray_threshold)
    return tissue, thumb


def select_top_tissue_tiles(
    slide: openslide.OpenSlide,
    level: int,
    tile_size: int,
    max_tiles: int,
    tissue_min_fraction: float,
    thumb_max_size: int,
    bg_gray_threshold: int,
) -> List[TileCandidate]:
    """Select up to max_tiles non-overlapping tile locations with most tissue."""
    tissue_mask, thumb = generate_tissue_mask_thumbnail(
        slide,
        thumb_max_size=thumb_max_size,
        bg_gray_threshold=bg_gray_threshold,
    )
    ii = _integral_image(tissue_mask)

    level_w, level_h = slide.level_dimensions[level]
    base_w, base_h = slide.level_dimensions[0]

    # Map level0 coordinates -> thumbnail coordinates
    sx = thumb.size[0] / float(base_w)
    sy = thumb.size[1] / float(base_h)

    ds = float(slide.level_downsamples[level])
    tile_size0 = tile_size * ds

    candidates: List[TileCandidate] = []

    # Grid in level coordinates
    for y_level in range(0, max(1, level_h - tile_size + 1), tile_size):
        for x_level in range(0, max(1, level_w - tile_size + 1), tile_size):
            x0 = int(round(x_level * ds))
            y0 = int(round(y_level * ds))
            x1 = int(round(x0 + tile_size0))
            y1 = int(round(y0 + tile_size0))

            tx0 = int(round(x0 * sx))
            ty0 = int(round(y0 * sy))
            tx1 = int(round(x1 * sx))
            ty1 = int(round(y1 * sy))

            # Clamp to thumbnail
            tx0 = max(0, min(tx0, thumb.size[0]))
            tx1 = max(0, min(tx1, thumb.size[0]))
            ty0 = max(0, min(ty0, thumb.size[1]))
            ty1 = max(0, min(ty1, thumb.size[1]))

            area = (tx1 - tx0) * (ty1 - ty0)
            if area <= 0:
                continue

            tissue_sum = _rect_sum(ii, tx0, ty0, tx1, ty1)
            frac = tissue_sum / float(area)

            if frac >= tissue_min_fraction:
                candidates.append(TileCandidate(x_level=x_level, y_level=y_level, tissue_frac=frac))

    # Sort high tissue first
    candidates.sort(key=lambda c: c.tissue_frac, reverse=True)
    return candidates[:max_tiles]


def tile_svs_to_dir(
    svs_path: str | Path,
    tiles_dir: str | Path,
    tile_size: int = 896,
    max_tiles: int = 64,
    target_mpp: float = 1.0,
    tissue_min_fraction: float = 0.30,
    thumb_max_size: int = 2048,
    bg_gray_threshold: int = 220,
) -> None:
    """Extract a small set of tissue-rich tiles from an SVS.

    Resumable:
    - If tiles_manifest.jsonl exists and has >= max_tiles lines, skip.
    - Otherwise, (re)create missing tiles.
    """
    svs_path = Path(svs_path)
    tiles_dir = Path(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = tiles_dir / "tiles_manifest.jsonl"
    if manifest_path.exists():
        existing_lines = sum(1 for _ in manifest_path.open("r", encoding="utf-8"))
        if existing_lines >= max_tiles:
            return

    slide = openslide.OpenSlide(str(svs_path))
    level = pick_level_for_target_mpp(slide, target_mpp=target_mpp)

    candidates = select_top_tissue_tiles(
        slide=slide,
        level=level,
        tile_size=tile_size,
        max_tiles=max_tiles,
        tissue_min_fraction=tissue_min_fraction,
        thumb_max_size=thumb_max_size,
        bg_gray_threshold=bg_gray_threshold,
    )

    # Append-mode so reruns can continue.
    done = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                done.add(obj["tile_relpath"])

    ds = float(slide.level_downsamples[level])

    with manifest_path.open("a", encoding="utf-8") as mf:
        for i, c in enumerate(candidates):
            tile_name = f"tile_{i:04d}_x{c.x_level}_y{c.y_level}.png"
            tile_path = tiles_dir / tile_name
            rel = str(tile_path.relative_to(tiles_dir))
            if rel in done and tile_path.exists():
                continue

            x0 = int(round(c.x_level * ds))
            y0 = int(round(c.y_level * ds))
            region = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")
            region.save(tile_path)

            mf.write(
                json.dumps(
                    {
                        "tile_relpath": rel,
                        "tile_name": tile_name,
                        "x_level": c.x_level,
                        "y_level": c.y_level,
                        "level": level,
                        "downsample": ds,
                        "tissue_frac": round(float(c.tissue_frac), 4),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    slide.close()
