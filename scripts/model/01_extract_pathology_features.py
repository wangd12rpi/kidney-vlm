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
    text = str(path_value).strip().strip("'").strip('"')
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def _to_registry_relative_path(path_value: str | Path) -> str:
    text = str(path_value).strip()
    if not text:
        return ""
    if "://" in text:
        return text

    path_obj = Path(text).expanduser()
    if not path_obj.is_absolute():
        return path_obj.as_posix().lstrip("/")

    resolved = path_obj.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix().lstrip("/")


def _local_wsi_paths(value: Any) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in _as_list(value):
        if "://" in raw_path:
            continue
        normalized = _normalize_local_path(raw_path)
        if normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return paths


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


def _is_valid_patch_tensor(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_patch_count(coords_path: Path, patch_features_path: Path, save_format: str) -> int:
    if _is_valid_coords(coords_path):
        try:
            import h5py

            with h5py.File(coords_path, "r") as handle:
                if "coords" in handle:
                    return int(handle["coords"].shape[0])
        except Exception:
            pass

    if save_format == "h5" and _is_valid_patch_features(patch_features_path):
        try:
            import h5py

            with h5py.File(patch_features_path, "r") as handle:
                if "features" in handle:
                    return int(handle["features"].shape[0])
        except Exception:
            return 0

    if save_format == "pt" and _is_valid_patch_tensor(patch_features_path):
        try:
            tensor = torch.load(patch_features_path, map_location="cpu")
            if hasattr(tensor, "shape") and len(tensor.shape) > 0:
                return int(tensor.shape[0])
            if isinstance(tensor, list):
                return len(tensor)
        except Exception:
            return 0

    return 0


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


def main() -> None:
    cfg = load_cfg()

    pathology_cfg = cfg.embedding_extraction.pathology
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

    patch_encoder_name = str(pathology_cfg.patch_encoder)
    slide_encoder_name = str(pathology_cfg.slide_encoder)
    extract_patch_only = bool(pathology_cfg.get("extract_patch_only", False))

    patch_encoder_kwargs = OmegaConf.to_container(pathology_cfg.get("patch_encoder_kwargs", {}), resolve=True) or {}
    slide_encoder_kwargs = OmegaConf.to_container(pathology_cfg.get("slide_encoder_kwargs", {}), resolve=True) or {}

    overwrite_existing = bool(pathology_cfg.get("overwrite_existing", False))
    skip_errors = bool(pathology_cfg.get("skip_errors", True))
    device = _resolve_device(str(pathology_cfg.get("device", "cpu")))
    save_format = str(pathology_cfg.get("save_format", "h5")).lower()
    if save_format not in {"h5", "pt"}:
        raise ValueError("save_format must be one of: h5, pt")
    if not extract_patch_only and save_format != "h5":
        raise ValueError("Slide feature extraction requires save_format='h5' for patch features.")

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
    if not extract_patch_only:
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

    total_wsi_refs = sum(len(_local_wsi_paths(value)) for value in registry_df.get("pathology_wsi_paths", []))
    if total_wsi_refs == 0:
        print("No local pathology_wsi_paths found in registry. Nothing to extract.")
        return

    print(f"TRIDENT root: {trident_root}")
    print(f"Registry: {registry_path}")
    print(f"Cases to process: {len(registry_df)}")
    print(f"WSI references to process: {total_wsi_refs}")
    print(f"Patch encoder: {patch_encoder_name}")
    print(f"Slide encoder: {slide_encoder_name}")
    print(f"Extract patch only: {extract_patch_only}")
    print(f"Device: {device}")
    print(f"Patch features dir: {patch_features_dir}")
    if not extract_patch_only:
        print(f"Slide features dir: {slide_features_dir}")
    print(f"Coords root: {coords_root}")
    print(f"Overwrite existing: {overwrite_existing}")
    print(f"Skip errors: {skip_errors}")

    patch_encoder = patch_encoder_factory(patch_encoder_name, **patch_encoder_kwargs)
    patch_encoder.eval()
    patch_encoder.to(device)

    slide_encoder = None
    if not extract_patch_only:
        from trident.slide_encoder_models import encoder_factory as slide_encoder_factory

        slide_encoder = slide_encoder_factory(slide_encoder_name, **slide_encoder_kwargs)
        slide_encoder.eval()
        slide_encoder.to(device)

    patch_by_wsi: dict[str, str] = {}
    slide_by_wsi: dict[str, str] = {}
    patch_count_by_wsi: dict[str, int] = {}

    extracted_patch = 0
    extracted_slide = 0
    skipped_existing = 0
    missing_wsi = 0
    failed = 0
    cases_written = 0

    case_loop = tqdm(registry_df.index.tolist(), total=len(registry_df), desc="Extracting pathology embeddings")
    for row_idx in case_loop:
        row = registry_df.loc[row_idx]
        case_wsi_paths = _local_wsi_paths(row.get("pathology_wsi_paths"))
        case_tile_paths: list[str] = []
        case_patch_counts: list[int] = []
        case_slide_paths: list[str] = []
        seen_tile: set[str] = set()
        seen_slide: set[str] = set()

        for normalized_wsi_path in case_wsi_paths:
            slide_path = Path(normalized_wsi_path)
            slide_stem = slide_path.stem

            final_patch_path = patch_features_dir / f"{slide_stem}.{save_format}"
            final_slide_path = slide_features_dir / f"{slide_stem}.h5"
            coords_path = coords_root / "patches" / f"{slide_stem}_patches.h5"

            patch_ready = _is_valid_patch_features(final_patch_path) if save_format == "h5" else _is_valid_patch_tensor(final_patch_path)
            slide_ready = True if extract_patch_only else _is_valid_slide_features(final_slide_path)

            if patch_ready:
                patch_by_wsi[normalized_wsi_path] = str(final_patch_path)
                patch_count_by_wsi[normalized_wsi_path] = _read_patch_count(coords_path, final_patch_path, save_format)
            if not extract_patch_only and slide_ready:
                slide_by_wsi[normalized_wsi_path] = str(final_slide_path)

            if overwrite_existing:
                stale_paths = [coords_path, final_patch_path]
                if not extract_patch_only:
                    stale_paths.append(final_slide_path)
                for stale_path in stale_paths:
                    if stale_path.exists():
                        stale_path.unlink()
                patch_ready = False
                slide_ready = False

            needs_patch = not patch_ready
            needs_slide = (not extract_patch_only) and (not slide_ready)

            if not overwrite_existing and patch_ready and slide_ready:
                skipped_existing += 1
            elif needs_patch or needs_slide:
                if not slide_path.exists():
                    missing_wsi += 1
                    case_loop.write(f"[missing] WSI path not found: {slide_path}")
                    continue

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
                        valid_patch = _is_valid_patch_features(final_patch_path) if save_format == "h5" else _is_valid_patch_tensor(final_patch_path)
                        if not valid_patch:
                            raise RuntimeError(f"Generated patch features are invalid: {final_patch_path}")
                        patch_by_wsi[normalized_wsi_path] = str(final_patch_path)
                        patch_count_by_wsi[normalized_wsi_path] = _read_patch_count(coords_path, final_patch_path, save_format)
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
                    case_loop.write(f"[error] {slide_path.name}: {exc}")
                    if not skip_errors:
                        raise
                finally:
                    if slide is not None:
                        try:
                            slide.release()
                        except Exception:
                            pass

            patch_emb_path = patch_by_wsi.get(normalized_wsi_path)
            if patch_emb_path and patch_emb_path not in seen_tile:
                case_tile_paths.append(_to_registry_relative_path(patch_emb_path))
                case_patch_counts.append(int(patch_count_by_wsi.get(normalized_wsi_path, 0)))
                seen_tile.add(patch_emb_path)

            if not extract_patch_only:
                slide_emb_path = slide_by_wsi.get(normalized_wsi_path)
                if slide_emb_path and slide_emb_path not in seen_slide:
                    case_slide_paths.append(_to_registry_relative_path(slide_emb_path))
                    seen_slide.add(slide_emb_path)

        if extract_patch_only:
            final_case_slide_paths = [
                _to_registry_relative_path(path_value)
                for path_value in _as_list(registry_df.at[row_idx, "pathology_slide_embedding_paths"])
            ]
        else:
            final_case_slide_paths = case_slide_paths

        registry_df.at[row_idx, "pathology_tile_embedding_paths"] = case_tile_paths
        registry_df.at[row_idx, "pathology_slide_embedding_paths"] = final_case_slide_paths
        registry_df.at[row_idx, "pathology_embedding_patch_size"] = patch_size if len(case_tile_paths) > 0 else None
        registry_df.at[row_idx, "pathology_embedding_magnification"] = target_mag if len(case_tile_paths) > 0 else None
        registry_df.at[row_idx, "pathology_tile_embedding_patch_counts"] = case_patch_counts
        write_registry_parquet(registry_df, registry_path, validate=False)
        cases_written += 1

    print("Extraction complete.")
    print(f"Patch embeddings extracted this run: {extracted_patch}")
    print(f"Slide embeddings extracted this run: {extracted_slide}")
    print(f"Slides skipped (existing outputs): {skipped_existing}")
    print(f"Slides missing on disk: {missing_wsi}")
    print(f"Slides failed: {failed}")
    print(f"Cases written to registry: {cases_written}")
    if patch_count_by_wsi:
        avg_patch_count = sum(patch_count_by_wsi.values()) / float(len(patch_count_by_wsi))
    else:
        avg_patch_count = 0.0
    print(f"Average patch count per embedded slide: {avg_patch_count:.2f}")
    print(f"Unified registry written: {registry_path}")


if __name__ == "__main__":
    main()
