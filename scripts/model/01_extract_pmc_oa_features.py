#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


@dataclass
class PendingImageTask:
    row_idx: int
    image_idx: int
    image_path: str
    output_path: Path
    output_rel_path: str


@dataclass
class RowState:
    feature_paths: list[str]
    pending_count: int = 0
    collection_complete: bool = False


def load_cfg(overrides: list[str] | None = None) -> DictConfig:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        base_cfg = compose(config_name="config")
    OmegaConf.set_struct(base_cfg, False)

    source_cfg_path = conf_dir / "data" / "sources" / "pmc_oa.yaml"
    radiology_cfg_path = conf_dir / "embedding_extraction" / "radiology" / "medsiglip_pmc_oa.yaml"
    if not source_cfg_path.exists():
        raise FileNotFoundError(f"Missing source config: {source_cfg_path}")
    if not radiology_cfg_path.exists():
        raise FileNotFoundError(f"Missing radiology extraction config: {radiology_cfg_path}")

    source_cfg = OmegaConf.load(source_cfg_path)
    radiology_cfg = OmegaConf.create(
        {"embedding_extraction": {"radiology": OmegaConf.load(radiology_cfg_path)}}
    )
    merged = OmegaConf.merge(base_cfg, source_cfg, radiology_cfg)

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    OmegaConf.set_struct(merged, False)
    merged.project.root_dir = str(ROOT)
    return merged


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


def _to_string_list(value: Any) -> list[str]:
    items = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError(f"Expected a list of strings, got {type(items).__name__}")
    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _normalize_local_path(path_value: str | Path) -> str:
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


def _local_image_paths(value: Any) -> list[str]:
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


def _resolve_device(device: str) -> str:
    requested = str(device).strip() or "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested '{requested}' but CUDA is unavailable; falling back to 'cpu'.")
        return "cpu"
    return requested


def _resolve_input_size(value: Any) -> tuple[int, int]:
    values = _to_string_list(value)
    if len(values) != 2:
        raise ValueError("embedding_extraction.radiology.input_size must contain exactly two integers.")
    height, width = (int(item) for item in values)
    if height <= 0 or width <= 0:
        raise ValueError("embedding_extraction.radiology.input_size values must be positive.")
    return height, width


def _normalize_selected_splits(selected_splits: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for split in selected_splits:
        text = str(split).strip().lower()
        if text:
            normalized.add(text)
    return normalized


def _resolve_feature_output_path(
    *,
    image_path: str,
    dataset_root: Path,
    image_root: Path,
    features_root: Path,
    extension: str,
) -> Path:
    resolved_image_path = Path(image_path).expanduser().resolve()
    resolved_dataset_root = dataset_root.expanduser().resolve()
    resolved_image_root = image_root.expanduser().resolve()

    relative_image_path: Path | None = None
    try:
        relative_image_path = resolved_image_path.relative_to(resolved_image_root)
    except ValueError:
        try:
            dataset_relative = resolved_image_path.relative_to(resolved_dataset_root)
            if dataset_relative.parts and dataset_relative.parts[0] == resolved_image_root.name:
                remaining_parts = dataset_relative.parts[1:]
                relative_image_path = Path(*remaining_parts) if remaining_parts else Path(resolved_image_path.name)
            else:
                relative_image_path = dataset_relative
        except ValueError:
            relative_image_path = Path(resolved_image_path.name)

    if str(relative_image_path) in {"", "."}:
        relative_image_path = Path(resolved_image_path.name)

    return features_root / relative_image_path.parent / f"{relative_image_path.stem}.{extension}"


def _is_valid_feature_file(path: Path, dataset_name: str) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with h5py.File(path, "r") as handle:
            if dataset_name not in handle:
                return False
            dataset = handle[dataset_name]
            return dataset.shape is not None and len(dataset.shape) >= 1 and int(dataset.shape[0]) > 0
    except Exception:
        return False


def _safe_delete_file(path: Path, allowed_root: Path) -> None:
    if not path.exists():
        return
    try:
        path.resolve().relative_to(allowed_root.resolve())
    except ValueError:
        return
    path.unlink()


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if image.mode == "L":
        return Image.merge("RGB", [image, image, image])
    if image.mode == "RGBA":
        return image.convert("RGB")
    if image.mode == "P":
        return image.convert("RGBA").convert("RGB")

    array = np.asarray(image)
    array = array.astype(np.float32)
    array = array - float(array.min())
    scale = float(array.max())
    if scale > 0:
        array = array / scale
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return Image.fromarray(array)


def medsiglip_resize(image: Image.Image, input_size: tuple[int, int]) -> Image.Image:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)

    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(
        tensor,
        size=input_size,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    output = resized.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(output)


def preprocess_pmc_image(path: str, input_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        normalized = to_rgb(image)
    return medsiglip_resize(normalized, input_size)


def _encode_batch(
    *,
    images: list[Image.Image],
    image_processor: Any,
    model: Any,
    device: str,
) -> np.ndarray:
    inputs = image_processor(images=images, do_resize=False, return_tensors="pt")
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value for key, value in dict(inputs).items()
    }

    with torch.inference_mode():
        outputs = model(**model_inputs)
        
    pooled = outputs.pooler_output
    normalized = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
    return normalized.detach().to(dtype=torch.float32, device="cpu").numpy()


def _save_feature_file(
    *,
    output_path: Path,
    dataset_name: str,
    features: np.ndarray,
    model_name: str,
    image_path: str,
    input_size: tuple[int, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(dataset_name, data=features, compression="gzip")
        # handle.attrs["feature_type"] = "patch_features"
        handle.attrs["feature_type"] = "pooled_embedding"
        handle.attrs["model_name"] = model_name
        handle.attrs["image_path"] = image_path
        handle.attrs["input_height"] = int(input_size[0])
        handle.attrs["input_width"] = int(input_size[1])


def _run_extraction_batch(
    *,
    tasks: list[PendingImageTask],
    image_processor: Any,
    model: Any,
    device: str,
    input_size: tuple[int, int],
    dataset_name: str,
    model_name: str,
    skip_errors: bool,
) -> list[tuple[PendingImageTask, bool, str | None]]:
    results: list[tuple[PendingImageTask, bool, str | None]] = []
    valid_tasks: list[PendingImageTask] = []
    valid_images: list[Image.Image] = []

    for task in tasks:
        try:
            valid_images.append(preprocess_pmc_image(task.image_path, input_size))
            valid_tasks.append(task)
        except Exception as exc:
            results.append((task, False, str(exc)))

    if not valid_tasks:
        return results

    try:
        features_batch = _encode_batch(
            images=valid_images,
            image_processor=image_processor,
            model=model,
            device=device,
        )
        for task, features in zip(valid_tasks, features_batch, strict=True):
            try:
                _save_feature_file(
                    output_path=task.output_path,
                    dataset_name=dataset_name,
                    features=features,
                    model_name=model_name,
                    image_path=task.image_path,
                    input_size=input_size,
                )
                results.append((task, True, None))
            except Exception as exc:
                results.append((task, False, str(exc)))
        return results
    except Exception as exc:
        if not skip_errors or len(valid_tasks) == 1:
            for task in valid_tasks:
                results.append((task, False, str(exc)))
            return results

    for task, image in zip(valid_tasks, valid_images, strict=True):
        try:
            features_batch = _encode_batch(
                images=[image],
                image_processor=image_processor,
                model=model,
                device=device,
            )
            _save_feature_file(
                output_path=task.output_path,
                dataset_name=dataset_name,
                features=features_batch[0],
                model_name=model_name,
                image_path=task.image_path,
                input_size=input_size,
            )
            results.append((task, True, None))
        except Exception as single_exc:
            results.append((task, False, str(single_exc)))
    return results


def main() -> None:
    overrides = sys.argv[1:]
    cfg = load_cfg(overrides=overrides)

    source_name = str(cfg.data.source.name)
    if source_name != "pmc_oa":
        raise ValueError(f"Expected data.source.name='pmc_oa', got '{source_name}'")

    pmc_cfg = cfg.data.source.pmc_oa
    radiology_cfg = cfg.embedding_extraction.radiology

    registry_path = Path(str(radiology_cfg.registry_path)).expanduser()
    if not registry_path.is_absolute():
        registry_path = (ROOT / registry_path).resolve()
    else:
        registry_path = registry_path.resolve()

    if not registry_path.exists():
        raise FileNotFoundError(
            f"PMC-OA registry not found at '{registry_path}'. "
            "Build the PMC-OA source first before extracting features."
        )

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        print(f"PMC-OA registry is empty at '{registry_path}'. Nothing to extract.")
        return

    source_mask = registry_df["source"].fillna("").map(str).eq("pmc_oa")
    selected_splits = _normalize_selected_splits(_to_string_list(radiology_cfg.get("selected_splits", [])))
    if selected_splits:
        split_mask = registry_df["split"].fillna("").map(str).str.lower().isin(selected_splits)
        source_mask = source_mask & split_mask

    candidate_indices = registry_df.index[source_mask].tolist()
    if not candidate_indices:
        print(f"No PMC-OA rows matched in '{registry_path}'. Nothing to extract.")
        return

    dataset_root = Path(str(pmc_cfg.dataset_root)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    image_root = dataset_root / str(pmc_cfg.image_dir).strip()
    features_root = Path(str(radiology_cfg.features_root)).expanduser()
    if not features_root.is_absolute():
        features_root = (ROOT / features_root).resolve()
    else:
        features_root = features_root.resolve()
    features_root.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(radiology_cfg.get("device", "cpu")))
    model_name = str(radiology_cfg.model_name).strip()
    processor_name = str(radiology_cfg.get("processor_name", model_name)).strip() or model_name
    dataset_name = str(radiology_cfg.get("dataset_name", "patch_features")).strip() or "patch_features"
    file_extension = str(radiology_cfg.get("file_extension", "h5")).strip().lstrip(".") or "h5"
    batch_size = int(radiology_cfg.get("batch_size", 8))
    if batch_size <= 0:
        raise ValueError("embedding_extraction.radiology.batch_size must be >= 1")

    input_size = _resolve_input_size(radiology_cfg.get("input_size", [448, 448]))
    max_images = _optional_int(radiology_cfg.get("max_images"))
    if max_images is not None and max_images < 0:
        raise ValueError("embedding_extraction.radiology.max_images must be >= 0")

    registry_flush_every = int(radiology_cfg.get("registry_flush_every", 256))
    if registry_flush_every <= 0:
        raise ValueError("embedding_extraction.radiology.registry_flush_every must be >= 1")

    skip_existing = bool(radiology_cfg.get("skip_existing", True))
    overwrite_existing = bool(radiology_cfg.get("overwrite_existing", False))
    skip_errors = bool(radiology_cfg.get("skip_errors", True))
    show_progress = bool(radiology_cfg.get("show_progress", True))

    print(f"PMC-OA registry: {registry_path}")
    print(f"Rows selected: {len(candidate_indices)}")
    print(f"Features root: {features_root}")
    print(f"Image root: {image_root}")
    print(f"Model: {model_name}")
    print(f"Processor: {processor_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Input size: {input_size[0]}x{input_size[1]}")
    print(f"Skip existing: {skip_existing}")
    print(f"Overwrite existing: {overwrite_existing}")
    print(f"Skip errors: {skip_errors}")
    print(f"Max images to extract: {max_images if max_images is not None else 'all'}")

    from transformers import AutoProcessor, SiglipVisionModel

    processor = AutoProcessor.from_pretrained(processor_name)
    image_processor = getattr(processor, "image_processor", processor)
    model = SiglipVisionModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    row_states: dict[int, RowState] = {}
    pending_tasks: list[PendingImageTask] = []
    pending_registry_write = False
    rows_updated_since_write = 0
    rows_updated_total = 0
    rows_finalized_total = 0
    registry_write_count = 0
    extracted_images = 0
    scheduled_images = 0
    skipped_existing_images = 0
    missing_images = 0
    failed_images = 0
    rows_without_images = 0
    limit_reached = False

    row_iterable = candidate_indices
    if show_progress:
        row_iterable = tqdm(
            candidate_indices,
            total=len(candidate_indices),
            desc="Extracting PMC-OA CT features",
            dynamic_ncols=True,
        )

    def log(message: str) -> None:
        if hasattr(row_iterable, "write"):
            row_iterable.write(message)
        else:
            print(message)

    def flush_registry(force: bool = False) -> None:
        nonlocal pending_registry_write, rows_updated_since_write, registry_write_count
        if not pending_registry_write:
            return
        if not force and rows_updated_since_write < registry_flush_every:
            return
        write_registry_parquet(registry_df, registry_path, validate=False)
        pending_registry_write = False
        rows_updated_since_write = 0
        registry_write_count += 1

    def finalize_row(row_idx: int) -> None:
        nonlocal pending_registry_write, rows_updated_since_write, rows_updated_total, rows_finalized_total
        state = row_states.pop(row_idx, None)
        if state is None:
            return

        final_paths = [path for path in state.feature_paths if path]
        current_paths = _as_list(registry_df.at[row_idx, "radiology_embedding_paths"])
        if final_paths != current_paths:
            registry_df.at[row_idx, "radiology_embedding_paths"] = final_paths
            pending_registry_write = True
            rows_updated_since_write += 1
            rows_updated_total += 1
        rows_finalized_total += 1
        flush_registry(force=False)

    def flush_pending_tasks() -> None:
        nonlocal pending_tasks, extracted_images, failed_images
        if not pending_tasks:
            return

        batch_tasks = pending_tasks
        pending_tasks = []
        results = _run_extraction_batch(
            tasks=batch_tasks,
            image_processor=image_processor,
            model=model,
            device=device,
            input_size=input_size,
            dataset_name=dataset_name,
            model_name=model_name,
            skip_errors=skip_errors,
        )

        for task, success, error_message in results:
            state = row_states.get(task.row_idx)
            if state is None:
                continue
            if success:
                state.feature_paths[task.image_idx] = task.output_rel_path
                extracted_images += 1
            else:
                failed_images += 1
                log(f"[error] {Path(task.image_path).name}: {error_message}")

            state.pending_count -= 1
            if state.collection_complete and state.pending_count == 0:
                finalize_row(task.row_idx)

        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()

    for row_idx in row_iterable:
        row = registry_df.loc[row_idx]
        image_paths = _local_image_paths(row.get("radiology_image_paths"))
        if not image_paths:
            rows_without_images += 1
            continue

        current_feature_paths = _as_list(row.get("radiology_embedding_paths"))
        state = RowState(feature_paths=[""] * len(image_paths))
        row_states[row_idx] = state
        stop_before_finalizing_row = False

        for image_idx, image_path in enumerate(image_paths):
            image_path_obj = Path(image_path)
            output_path = _resolve_feature_output_path(
                image_path=image_path,
                dataset_root=dataset_root,
                image_root=image_root,
                features_root=features_root,
                extension=file_extension,
            )
            output_rel_path = _to_registry_relative_path(output_path)

            existing_rel_path = current_feature_paths[image_idx] if image_idx < len(current_feature_paths) else ""
            existing_abs_path = Path(_normalize_local_path(existing_rel_path)) if existing_rel_path else None

            if overwrite_existing:
                _safe_delete_file(output_path, features_root)
                if existing_abs_path is not None:
                    _safe_delete_file(existing_abs_path, features_root)

            if skip_existing and _is_valid_feature_file(output_path, dataset_name):
                state.feature_paths[image_idx] = output_rel_path
                skipped_existing_images += 1
                continue

            if max_images is not None and scheduled_images >= max_images:
                limit_reached = True
                stop_before_finalizing_row = True
                break

            if not image_path_obj.exists():
                missing_images += 1
                log(f"[missing] image not found: {image_path_obj}")
                continue

            state.pending_count += 1
            pending_tasks.append(
                PendingImageTask(
                    row_idx=row_idx,
                    image_idx=image_idx,
                    image_path=str(image_path_obj),
                    output_path=output_path,
                    output_rel_path=output_rel_path,
                )
            )
            scheduled_images += 1

            if len(pending_tasks) >= batch_size:
                flush_pending_tasks()

        if stop_before_finalizing_row and state.pending_count == 0:
            row_states.pop(row_idx, None)
            break

        state.collection_complete = True
        if state.pending_count == 0:
            finalize_row(row_idx)

        if limit_reached:
            break

    flush_pending_tasks()
    flush_registry(force=True)

    print("PMC-OA feature extraction complete.")
    print(f"Images scheduled for extraction: {scheduled_images}")
    print(f"Images extracted this run: {extracted_images}")
    print(f"Images skipped (existing outputs): {skipped_existing_images}")
    print(f"Images missing on disk: {missing_images}")
    print(f"Images failed: {failed_images}")
    print(f"Rows finalized: {rows_finalized_total}")
    print(f"Rows updated in registry: {rows_updated_total}")
    print(f"Rows without radiology images: {rows_without_images}")
    print(f"Registry writes this run: {registry_write_count}")
    if limit_reached:
        print("Stopped early because the configured max_images limit was reached.")
    print(f"Registry written: {registry_path}")


if __name__ == "__main__":
    main()
