from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from kidney_vlm.data.id_factory import make_sample_id
from kidney_vlm.data.registry_schema import CORE_COLUMNS, empty_registry_frame, normalize_registry_df

_SLICE_STEM_PATTERN = re.compile(r"^(?P<patient>.+)_slice_(?P<slice_index>\d+)$")
_CASE_DIR_PATTERN = re.compile(r"^case_\d+$")


def assign_split(patient_id: str, split_ratios: dict[str, float]) -> str:
    train_ratio = float(split_ratios.get("train", 0.8))
    val_ratio = float(split_ratios.get("val", 0.1))
    test_ratio = float(split_ratios.get("test", 0.1))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        return "train"

    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    digest = hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()
    stable_bucket = (int(digest[:8], 16) % 10_000) / 10_000.0
    if stable_bucket < train_ratio:
        return "train"
    if stable_bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def _parse_patient_and_slice(image_path: Path) -> tuple[str, int | None]:
    match = _SLICE_STEM_PATTERN.match(image_path.stem)
    if not match:
        return "", None
    patient_id = match.group("patient").strip()
    slice_text = match.group("slice_index").strip()
    try:
        slice_index = int(slice_text)
    except ValueError:
        slice_index = None
    return patient_id, slice_index


def _normalize_to_uint16(volume: Any) -> Any:
    v_min = float(volume.min())
    v_max = float(volume.max())
    if v_max == v_min:
        return (volume * 0).astype("uint16")
    normalized = (volume - v_min) / (v_max - v_min) * 65535.0
    return normalized.astype("uint16")


def _discover_case_dirs(dataset_dir: Path) -> list[Path]:
    direct_children = sorted([path for path in dataset_dir.iterdir() if path.is_dir()])
    direct_case_dirs = [path for path in direct_children if _CASE_DIR_PATTERN.match(path.name)]
    if direct_case_dirs:
        return direct_case_dirs

    nested_case_dirs: list[Path] = []
    for child in direct_children:
        try:
            grandchildren = sorted([path for path in child.iterdir() if path.is_dir()])
        except PermissionError:
            continue
        nested_case_dirs.extend([path for path in grandchildren if _CASE_DIR_PATTERN.match(path.name)])
    if nested_case_dirs:
        return sorted(nested_case_dirs)

    # Fallback for non-standard naming conventions.
    return direct_children


def _resolve_device(device: str | None) -> str:
    try:
        import torch
    except ImportError:
        return device or "cpu"

    requested = str(device).strip() if device is not None else ""
    if not requested:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def extract_kits23_medsiglip_features(
    *,
    images_root: Path,
    output_root: Path,
    model_id: str = "google/medsiglip-448",
    image_glob: str = "**/*.png",
    batch_size: int = 8,
    device: str | None = None,
    overwrite_existing: bool = False,
    max_images: int | None = None,
    skip_errors: bool = True,
    show_progress: bool = True,
    use_tensorflow_resize: bool = True,
    hf_token: str | None = None,
) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from PIL import Image
        from tqdm.auto import tqdm
        from transformers import AutoProcessor, SiglipVisionModel
    except ImportError as exc:
        raise RuntimeError(
            "MedSigLIP extraction requires numpy, pillow, torch, transformers, and tqdm."
        ) from exc

    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1. Received: {batch_size}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root does not exist: {images_root}")

    image_paths = sorted([path for path in images_root.glob(image_glob) if path.is_file()])

    if max_images is not None:
        image_paths = image_paths[: max(max_images, 0)]

    output_root.mkdir(parents=True, exist_ok=True)
    if not image_paths:
        return {
            "model_id": model_id,
            "images_total": 0,
            "images_saved": 0,
            "images_skipped_existing": 0,
            "images_failed": 0,
            "batches_total": 0,
            "batches_processed": 0,
            "device": _resolve_device(device),
            "output_root": str(output_root.resolve()),
        }

    device_name = _resolve_device(device)
    pairs: list[tuple[Path, Path]] = []
    skipped_existing = 0
    for image_path in image_paths:
        rel_path = image_path.relative_to(images_root)
        embedding_path = output_root / rel_path.with_suffix(".pt")
        if not overwrite_existing and embedding_path.exists():
            skipped_existing += 1
            continue
        pairs.append((image_path, embedding_path))

    batches_total = (len(pairs) + batch_size - 1) // batch_size
    if not pairs:
        return {
            "model_id": model_id,
            "images_total": len(image_paths),
            "images_saved": 0,
            "images_skipped_existing": skipped_existing,
            "images_failed": 0,
            "batches_total": batches_total,
            "batches_processed": 0,
            "device": device_name,
            "output_root": str(output_root.resolve()),
        }

    model_kwargs = {"token": hf_token} if hf_token else {}
    model = SiglipVisionModel.from_pretrained(model_id, **model_kwargs).to(device_name)
    processor = AutoProcessor.from_pretrained(model_id, **model_kwargs)
    model.eval()

    vision_cfg = getattr(model.config, "vision_config", None)
    if vision_cfg is None:
        vision_cfg = model.config
    image_size = int(getattr(vision_cfg, "image_size", 448))
    patch_size = int(getattr(vision_cfg, "patch_size", 14))
    hidden_dim = int(getattr(vision_cfg, "hidden_size", 0))
    patch_grid = image_size // patch_size if patch_size > 0 else 0
    n_patches = patch_grid * patch_grid

    tf_resize = None
    resize_method = "pil_bilinear"
    if use_tensorflow_resize:
        try:
            from tensorflow.image import resize as tf_resize  # type: ignore

            resize_method = "tf_bilinear_antialias_false"
        except Exception as exc:
            tf_resize = None
            warnings.warn(
                (
                    "Falling back to PIL resize because TensorFlow resize could not be "
                    f"initialized: {exc}"
                ),
                RuntimeWarning,
            )

    def _load_image(path: Path) -> Any:
        image = Image.open(path).convert("RGB")
        if tf_resize is not None:
            array = np.array(image)
            resized = tf_resize(
                images=array,
                size=[image_size, image_size],
                method="bilinear",
                antialias=False,
            ).numpy()
            return Image.fromarray(resized.astype(np.uint8))
        return image.resize((image_size, image_size), resample=Image.BILINEAR)

    failed = 0
    saved = 0
    batches_processed = 0
    batch_starts = range(0, len(pairs), batch_size)
    if show_progress:
        batch_starts = tqdm(
            batch_starts,
            total=batches_total,
            desc="Extracting MedSigLIP features",
            unit="batch",
        )

    with torch.no_grad():
        for start in batch_starts:
            batch_pairs = pairs[start : start + batch_size]
            loaded_images: list[Any] = []
            loaded_pairs: list[tuple[Path, Path]] = []

            for image_path, embedding_path in batch_pairs:
                try:
                    loaded_images.append(_load_image(image_path))
                    loaded_pairs.append((image_path, embedding_path))
                except Exception:
                    failed += 1
                    if not skip_errors:
                        raise

            if not loaded_images:
                continue

            try:
                inputs = processor(images=loaded_images, padding="max_length", return_tensors="pt")
                inputs = {
                    key: value.to(device_name) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                outputs = model(**inputs)
                pooled = outputs["pooler_output"]
                pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
                embeddings = pooled.detach().cpu().float()

                for (_, embedding_path), features in zip(loaded_pairs, embeddings):
                    embedding_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(features, embedding_path)
                    saved += 1
                batches_processed += 1
            except Exception:
                failed += len(loaded_pairs)
                if not skip_errors:
                    raise

            if device_name.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    meta = {
        "model": model_id,
        "embedding_type": "pooler_output_l2_normalized",
        "resize_method": resize_method,
        "image_size": image_size,
        "patch_size": patch_size,
        "hidden_dim": hidden_dim,
        "n_patches": n_patches,
        "patch_grid": [patch_grid, patch_grid],
        "dtype": "float32",
    }
    meta_path = output_root / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    return {
        "model_id": model_id,
        "images_total": len(image_paths),
        "images_saved": saved,
        "images_skipped_existing": skipped_existing,
        "images_failed": failed,
        "batches_total": batches_total,
        "batches_processed": batches_processed,
        "device": device_name,
        "output_root": str(output_root.resolve()),
        "meta_path": str(meta_path.resolve()),
    }


def extract_kits23_slice_mask_pairs(
    *,
    dataset_dir: Path,
    pairs_root: Path,
    target_axis: int = 0,
    skip_existing: bool = True,
    max_cases: int | None = None,
    show_progress: bool = True,
) -> dict[str, int]:
    try:
        import nibabel as nib
        import numpy as np
        from PIL import Image
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "KITS23 extraction requires nibabel, numpy, pillow, and tqdm. "
            "Install them before running extraction."
        ) from exc

    if target_axis not in (0, 1, 2):
        raise ValueError(f"target_axis must be 0, 1, or 2. Received: {target_axis}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"KITS23 dataset directory does not exist: {dataset_dir}")

    case_dirs = _discover_case_dirs(dataset_dir)
    if max_cases is not None:
        case_dirs = case_dirs[: max(max_cases, 0)]

    totals = {
        "cases_seen": len(case_dirs),
        "cases_processed": 0,
        "cases_missing_files": 0,
        "slices_total": 0,
        "slices_saved": 0,
        "slices_skipped_background": 0,
        "slices_skipped_existing": 0,
    }

    for case_dir in case_dirs:
        image_in = case_dir / "imaging.nii.gz"
        seg_in = case_dir / "segmentation.nii.gz"
        if not image_in.exists() or not seg_in.exists():
            totals["cases_missing_files"] += 1
            continue

        image_vol = nib.load(str(image_in)).get_fdata()
        seg_vol = nib.load(str(seg_in)).get_fdata()
        if image_vol.shape != seg_vol.shape:
            raise ValueError(
                f"Shape mismatch in {case_dir.name}: image {image_vol.shape} vs mask {seg_vol.shape}"
            )

        image_norm = _normalize_to_uint16(image_vol)
        n_slices = int(image_vol.shape[target_axis])
        totals["slices_total"] += n_slices

        image_out = pairs_root / "images" / case_dir.name
        mask_out = pairs_root / "masks" / case_dir.name
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        slice_iterable = range(n_slices)
        if show_progress:
            slice_iterable = tqdm(
                slice_iterable,
                total=n_slices,
                desc=f"{case_dir.name} slices",
                unit="slice",
                leave=False,
            )

        for slice_index in slice_iterable:
            seg_slice = np.take(seg_vol, slice_index, axis=target_axis)
            if not np.any(seg_slice > 0):
                totals["slices_skipped_background"] += 1
                continue

            filename = f"{case_dir.name}_slice_{slice_index:04d}.png"
            image_path = image_out / filename
            mask_path = mask_out / filename

            if skip_existing and image_path.exists() and mask_path.exists():
                totals["slices_skipped_existing"] += 1
                continue

            img_slice = np.take(image_norm, slice_index, axis=target_axis)
            Image.fromarray(img_slice).save(image_path)

            seg_max = float(seg_slice.max())
            seg_uint = seg_slice.astype(np.uint8) if seg_max <= 255 else seg_slice.astype(np.uint16)
            Image.fromarray(seg_uint).save(mask_path)
            totals["slices_saved"] += 1

        totals["cases_processed"] += 1

    return totals


def build_kits23_registry_rows(
    *,
    pairs_root: Path,
    source_name: str,
    split_ratios: dict[str, float],
    image_glob: str = "**/*.png",
    slice_axis: int | None = None,
    embeddings_root: Path | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    images_root = pairs_root / "images"
    masks_root = pairs_root / "masks"
    if not images_root.exists():
        return empty_registry_frame()

    image_paths = sorted([path for path in images_root.glob(image_glob) if path.is_file()])
    if not image_paths:
        return empty_registry_frame()

    rows: list[dict[str, Any]] = []
    iterable = image_paths
    if show_progress and tqdm is not None:
        iterable = tqdm(image_paths, total=len(image_paths), desc="Building kits23 rows", unit="slice")

    for image_path in iterable:
        rel_path = image_path.relative_to(images_root)
        mask_path = masks_root / rel_path
        if not mask_path.exists():
            continue

        parsed_patient_id, slice_index = _parse_patient_and_slice(image_path)
        parent_name = image_path.parent.name
        patient_id = parsed_patient_id or (parent_name if parent_name != "images" else "")
        if not patient_id:
            patient_id = "unknown_patient"

        split = assign_split(patient_id, split_ratios)
        sample_id = make_sample_id(
            source_name,
            patient_id,
            rel_path.as_posix(),
            modality_scope="radiology_slice_pair",
        )
        radiology_embedding_paths: list[str] = []
        if embeddings_root is not None:
            embedding_path = embeddings_root / rel_path.with_suffix(".pt")
            if embedding_path.exists():
                radiology_embedding_paths = [str(embedding_path.resolve())]

        biomarkers_parts = ["dataset: kits23", "modality: ct"]
        if slice_axis is not None:
            biomarkers_parts.append(f"slice_axis: {slice_axis}")
        if slice_index is not None:
            biomarkers_parts.append(f"slice_index: {slice_index}")

        rows.append(
            {
                "sample_id": sample_id,
                "source": source_name,
                "patient_id": patient_id,
                "study_id": patient_id,
                "split": split,
                "pathology_wsi_paths": [],
                "radiology_image_paths": [str(image_path.resolve())],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [str(mask_path.resolve())],
                "pathology_tile_embedding_paths": [],
                "pathology_slide_embedding_paths": [],
                "radiology_embedding_paths": radiology_embedding_paths,
                "biomarkers_text": "; ".join(biomarkers_parts),
                "question": "",
                "answer": "",
                "slice_index": slice_index,
                "slice_axis": slice_axis if slice_axis is not None else "",
                "pair_relative_path": rel_path.as_posix(),
                "has_radiology": True,
                "has_radiology_mask": True,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return empty_registry_frame()

    frame = frame.drop_duplicates(subset=["sample_id"], keep="last").reset_index(drop=True)
    frame = normalize_registry_df(frame)
    return frame[CORE_COLUMNS + [column for column in frame.columns if column not in CORE_COLUMNS]]
