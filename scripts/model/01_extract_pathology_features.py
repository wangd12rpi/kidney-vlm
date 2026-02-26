#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.pathology.trident_adapter import TridentAdapter
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    return cfg


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_local_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def _collect_unique_wsi_paths(registry_df: pd.DataFrame) -> list[str]:
    unique: set[str] = set()
    for value in registry_df.get("pathology_wsi_paths", []):
        for raw_path in _as_list(value):
            if "://" in raw_path:
                continue
            unique.add(_normalize_local_path(raw_path))
    return sorted(unique)


def _is_valid_h5(path: Path, required_keys: tuple[str, ...]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            return all(key in handle for key in required_keys)
    except Exception:
        return False


def _is_valid_patch_features(path: Path) -> bool:
    return _is_valid_h5(path, ("features", "coords"))


def _is_valid_slide_features(path: Path) -> bool:
    return _is_valid_h5(path, ("features",))


def _is_valid_coords(path: Path) -> bool:
    return _is_valid_h5(path, ("coords",))


def _resolve_device(device: str) -> str:
    requested = str(device).strip() or "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested '{requested}' but CUDA is unavailable; falling back to 'cpu'.")
        return "cpu"
    return requested


def _resolve_trident_root(configured_root: str) -> Path:
    root = Path(configured_root).expanduser()
    if root.exists():
        return root

    fallbacks = [
        ROOT / "external" / "TRIDENT",
        ROOT / "external" / "trident",
    ]
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    return root


def _build_embedding_columns(
    registry_df: pd.DataFrame,
    *,
    patch_map: dict[str, str],
    slide_map: dict[str, str],
) -> tuple[list[list[str]], list[list[str]]]:
    tile_embeddings: list[list[str]] = []
    slide_embeddings: list[list[str]] = []

    for value in registry_df.get("pathology_wsi_paths", []):
        tile_paths: list[str] = []
        slide_paths: list[str] = []
        seen_tile: set[str] = set()
        seen_slide: set[str] = set()

        for raw_path in _as_list(value):
            if "://" in raw_path:
                continue
            normalized = _normalize_local_path(raw_path)

            patch_path = patch_map.get(normalized)
            if patch_path and patch_path not in seen_tile:
                tile_paths.append(patch_path)
                seen_tile.add(patch_path)

            slide_path = slide_map.get(normalized)
            if slide_path and slide_path not in seen_slide:
                slide_paths.append(slide_path)
                seen_slide.add(slide_path)

        tile_embeddings.append(tile_paths)
        slide_embeddings.append(slide_paths)

    return tile_embeddings, slide_embeddings


def main() -> None:
    cfg = load_cfg()

    pathology_cfg = cfg.embeding_extraction.pathology
    registry_path = Path(str(cfg.data.unified_registry_path)).expanduser().resolve()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Unified registry not found at '{registry_path}'. "
            "Build a source first before extracting pathology features."
        )

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        print(f"Registry is empty at '{registry_path}'. Nothing to extract.")
        return

    trident_root = _resolve_trident_root(str(pathology_cfg.trident_root))
    adapter = TridentAdapter(trident_root=trident_root)
    adapter.ensure_on_path()
    adapter.import_core()

    from trident import OpenSlideWSI
    from trident.patch_encoder_models import encoder_factory as patch_encoder_factory
    from trident.slide_encoder_models import encoder_factory as slide_encoder_factory

    patch_encoder_name = str(pathology_cfg.patch_encoder)
    slide_encoder_name = str(pathology_cfg.slide_encoder)

    patch_encoder_kwargs = OmegaConf.to_container(pathology_cfg.get("patch_encoder_kwargs", {}), resolve=True) or {}
    slide_encoder_kwargs = OmegaConf.to_container(pathology_cfg.get("slide_encoder_kwargs", {}), resolve=True) or {}

    overwrite_existing = bool(pathology_cfg.get("overwrite_existing", False))
    skip_errors = bool(pathology_cfg.get("skip_errors", True))
    device = _resolve_device(str(pathology_cfg.get("device", "cpu")))
    save_format = str(pathology_cfg.get("save_format", "h5"))
    if save_format != "h5":
        raise ValueError("TRIDENT slide extraction requires h5 patch features. Set save_format='h5'.")

    target_mag = int(pathology_cfg.get("target_magnification", 20))
    patch_size = int(pathology_cfg.get("patch_size", 512))
    patch_overlap = int(pathology_cfg.get("patch_overlap", 0))
    min_tissue_proportion = float(pathology_cfg.get("min_tissue_proportion", 0.0))
    batch_limit = int(pathology_cfg.get("batch_limit", 512))
    verbose_inner = bool(pathology_cfg.get("verbose", False))

    features_root = Path(str(pathology_cfg.get("features_root", ROOT / "data" / "features"))).expanduser()
    if not features_root.is_absolute():
        features_root = (ROOT / features_root).resolve()
    else:
        features_root = features_root.resolve()
    features_root.mkdir(parents=True, exist_ok=True)
    patch_features_dir = features_root / f"features_{patch_encoder_name}"
    slide_features_dir = features_root / f"slide_features_{slide_encoder_name}"
    patch_features_dir.mkdir(parents=True, exist_ok=True)
    slide_features_dir.mkdir(parents=True, exist_ok=True)

    coords_root = Path(
        str(
            pathology_cfg.get(
                "coords_root",
                features_root / f"coords_{target_mag}x_{patch_size}px_{patch_overlap}px_overlap",
            )
        )
    ).expanduser()
    if not coords_root.is_absolute():
        coords_root = (ROOT / coords_root).resolve()
    else:
        coords_root = coords_root.resolve()

    slide_paths = _collect_unique_wsi_paths(registry_df)
    if not slide_paths:
        print("No local pathology_wsi_paths found in registry. Nothing to extract.")
        return

    print(f"TRIDENT root: {trident_root}")
    print(f"Registry: {registry_path}")
    print(f"Slides to process: {len(slide_paths)}")
    print(f"Patch encoder: {patch_encoder_name}")
    print(f"Slide encoder: {slide_encoder_name}")
    print(f"Device: {device}")
    print(f"Patch features dir: {patch_features_dir}")
    print(f"Slide features dir: {slide_features_dir}")
    print(f"Coords root: {coords_root}")
    print(f"Overwrite existing: {overwrite_existing}")
    print(f"Skip errors: {skip_errors}")

    patch_encoder = patch_encoder_factory(patch_encoder_name, **patch_encoder_kwargs)
    patch_encoder.eval()
    patch_encoder.to(device)

    slide_encoder = slide_encoder_factory(slide_encoder_name, **slide_encoder_kwargs)
    slide_encoder.eval()
    slide_encoder.to(device)

    patch_by_wsi: dict[str, str] = {}
    slide_by_wsi: dict[str, str] = {}

    extracted_patch = 0
    extracted_slide = 0
    skipped_existing = 0
    missing_wsi = 0
    failed = 0

    loop = tqdm(slide_paths, total=len(slide_paths), desc="Extracting pathology embeddings")
    for normalized_wsi_path in loop:
        slide_path = Path(normalized_wsi_path)
        slide_stem = slide_path.stem

        final_patch_path = patch_features_dir / f"{slide_stem}.h5"
        final_slide_path = slide_features_dir / f"{slide_stem}.h5"
        coords_path = coords_root / "patches" / f"{slide_stem}_patches.h5"

        patch_ready = _is_valid_patch_features(final_patch_path)
        slide_ready = _is_valid_slide_features(final_slide_path)
        if patch_ready:
            patch_by_wsi[normalized_wsi_path] = str(final_patch_path)
        if slide_ready:
            slide_by_wsi[normalized_wsi_path] = str(final_slide_path)

        if not overwrite_existing and patch_ready and slide_ready:
            skipped_existing += 1
            continue

        if not slide_path.exists():
            missing_wsi += 1
            loop.write(f"[missing] WSI path not found: {slide_path}")
            continue

        if overwrite_existing:
            for stale_path in (coords_path, final_patch_path, final_slide_path):
                if stale_path.exists():
                    stale_path.unlink()
            patch_ready = False
            slide_ready = False

        needs_patch = not patch_ready
        needs_slide = not slide_ready
        slide = None
        try:
            slide = OpenSlideWSI(slide_path=str(slide_path), lazy_init=False)

            if needs_patch:
                coords_ok = _is_valid_coords(coords_path)
                if not coords_ok:
                    coords_path = Path(
                        slide.extract_tissue_coords(
                            target_mag=target_mag,
                            patch_size=patch_size,
                            save_coords=str(coords_root),
                            overlap=patch_overlap,
                            min_tissue_proportion=min_tissue_proportion,
                        )
                    )

                generated_patch_path = Path(
                    slide.extract_patch_features(
                        patch_encoder=patch_encoder,
                        coords_path=str(coords_path),
                        save_features=str(patch_features_dir),
                        device=device,
                        saveas=save_format,
                        batch_limit=batch_limit,
                        verbose=verbose_inner,
                    )
                )
                final_patch_path = generated_patch_path
                if not _is_valid_patch_features(final_patch_path):
                    raise RuntimeError(f"Generated patch features are invalid: {final_patch_path}")
                patch_by_wsi[normalized_wsi_path] = str(final_patch_path)
                extracted_patch += 1

            if needs_slide:
                if not _is_valid_patch_features(final_patch_path):
                    raise FileNotFoundError(
                        f"Patch features are missing or invalid, cannot run slide encoder: {final_patch_path}"
                    )
                generated_slide_path = Path(
                    slide.extract_slide_features(
                        patch_features_path=str(final_patch_path),
                        slide_encoder=slide_encoder,
                        save_features=str(slide_features_dir),
                        device=device,
                    )
                )
                final_slide_path = generated_slide_path
                if not _is_valid_slide_features(final_slide_path):
                    raise RuntimeError(f"Generated slide features are invalid: {final_slide_path}")
                slide_by_wsi[normalized_wsi_path] = str(final_slide_path)
                extracted_slide += 1

            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as exc:
            failed += 1
            loop.write(f"[error] {slide_path.name}: {exc}")
            if not skip_errors:
                raise
        finally:
            if slide is not None:
                try:
                    slide.release()
                except Exception:
                    pass

    tile_embeddings, slide_embeddings = _build_embedding_columns(
        registry_df,
        patch_map=patch_by_wsi,
        slide_map=slide_by_wsi,
    )
    old_tile = registry_df["pathology_tile_embedding_paths"].tolist()
    old_slide = registry_df["pathology_slide_embedding_paths"].tolist()

    updated_df = registry_df.copy()
    updated_df["pathology_tile_embedding_paths"] = tile_embeddings
    updated_df["pathology_slide_embedding_paths"] = slide_embeddings
    write_registry_parquet(updated_df, registry_path)

    changed_rows = sum(
        1
        for idx in range(len(updated_df))
        if old_tile[idx] != tile_embeddings[idx] or old_slide[idx] != slide_embeddings[idx]
    )

    print("Extraction complete.")
    print(f"Patch embeddings extracted this run: {extracted_patch}")
    print(f"Slide embeddings extracted this run: {extracted_slide}")
    print(f"Slides skipped (existing outputs): {skipped_existing}")
    print(f"Slides missing on disk: {missing_wsi}")
    print(f"Slides failed: {failed}")
    print(f"Registry rows updated: {changed_rows}")
    print(f"Unified registry written: {registry_path}")


if __name__ == "__main__":
    main()
