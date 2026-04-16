from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from kidney_vlm.radiology.feature_registry import build_mask_series_dir
from kidney_vlm.radiology.medical_sam3_runtime import (
    load_sam3_module,
    patch_sam3_module_for_device,
    resolve_device,
)


@dataclass(frozen=True)
class SliceSegmentationJob:
    dicom_path: Path
    png_path: Path


@dataclass(frozen=True)
class SeriesSegmentationExtractionResult:
    png_dir: str
    png_paths: tuple[str, ...]
    mask_dir: str
    manifest_path: str
    mask_paths: tuple[str, ...]
    slice_count: int
    pngs_created: int
    mask_files_created: int
    existing_mask_files_reused: int
    keywords: tuple[str, ...]


class TCGARadiologySegmentationExtractor:
    def __init__(
        self,
        *,
        root_dir: Path,
        raw_root: Path,
        mask_root: Path,
        keyword_map_path: Path,
        checkpoint_path: Path,
        medical_sam3_root: Path,
        sam3_root: Path,
        input_size: int,
        confidence_threshold: float,
        device: str,
        overwrite_masks: bool,
        skip_existing_masks: bool,
        min_mask_pixels: int,
    ) -> None:
        self.root_dir = root_dir.expanduser().resolve()
        self.raw_root = raw_root.expanduser().resolve()
        self.mask_root = mask_root.expanduser().resolve()
        self.keyword_map_path = keyword_map_path.expanduser().resolve()
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.medical_sam3_root = medical_sam3_root.expanduser().resolve()
        self.sam3_root = sam3_root.expanduser().resolve()
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.overwrite_masks = bool(overwrite_masks)
        self.skip_existing_masks = bool(skip_existing_masks)
        self.min_mask_pixels = max(int(min_mask_pixels), 1)

        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("Radiology segmentation requires 'torch'.") from exc

        self.torch = torch
        self.device = resolve_device(str(device).strip() or "cpu", torch)

        sam3_module = load_sam3_module(
            module_path=self.medical_sam3_root / "inference" / "sam3_inference.py",
            sam3_root=self.sam3_root,
            torch_module=self.torch,
        )
        patch_sam3_module_for_device(
            sam3_module=sam3_module,
            device=self.device,
            torch_module=self.torch,
        )
        self.model = sam3_module.SAM3Model(
            confidence_threshold=self.confidence_threshold,
            device=self.device,
            checkpoint_path=str(self.checkpoint_path),
        )
        self.keyword_map = self._load_keyword_map(self.keyword_map_path)

    def _load_keyword_map(self, path: Path) -> dict[str, dict[str, list[str]]]:
        if not path.is_file():
            raise FileNotFoundError(f"TCGA Medical-SAM3 keyword map not found: {path}")
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        mapping = payload.get("tcga_medsam3_keywords", {}) if isinstance(payload, dict) else {}
        out: dict[str, dict[str, list[str]]] = {}
        for collection, per_modality in dict(mapping).items():
            collection_key = str(collection).strip().upper()
            modality_map: dict[str, list[str]] = {}
            for modality, keywords in dict(per_modality or {}).items():
                normalized_modality = self._normalize_modality(modality)
                ordered_keywords: list[str] = []
                for keyword in list(keywords or []):
                    text = str(keyword).strip()
                    if text and text not in ordered_keywords:
                        ordered_keywords.append(text)
                if normalized_modality and ordered_keywords:
                    modality_map[normalized_modality] = ordered_keywords
            if modality_map:
                out[collection_key] = modality_map
        return out

    def _normalize_modality(self, modality: Any) -> str:
        value = str(modality).strip().upper()
        aliases = {
            "MRI": "MR",
            "MAMMOGRAPHY": "MG",
            "PET": "PT",
        }
        return aliases.get(value, value)

    def _to_registry_relative_path(self, path_value: str | Path) -> str:
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            return path.as_posix().lstrip("/")
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.root_dir).as_posix()
        except ValueError:
            return resolved.as_posix().lstrip("/")

    def _slugify_keyword(self, keyword: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", keyword.strip().lower()).strip("_")
        return slug or "keyword"

    def _keyword_list(self, *, collection: str, modality: str) -> list[str]:
        return list(self.keyword_map.get(str(collection).strip().upper(), {}).get(self._normalize_modality(modality), []))

    def _mask_manifest_path(self, mask_dir: Path) -> Path:
        return mask_dir / "series_manifest.json"

    def _mask_is_usable(self, mask: np.ndarray | None) -> tuple[bool, np.ndarray | None]:
        if mask is None:
            return False, None
        mask_array = np.asarray(mask)
        mask_array = np.squeeze(mask_array)
        if mask_array.ndim > 2:
            mask_array = mask_array[0]
        if mask_array.ndim != 2:
            return False, None
        binary_mask = (mask_array > 0).astype(np.uint8)
        if int(binary_mask.sum()) < self.min_mask_pixels:
            return False, None
        return True, binary_mask

    def _save_mask(self, mask_array: np.ndarray, output_path: Path) -> None:
        mask_uint8 = (mask_array > 0).astype(np.uint8) * 255
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask_uint8, mode="L").save(output_path)

    def _load_png_array(self, png_path: Path) -> np.ndarray:
        with Image.open(png_path) as image:
            return np.asarray(image.convert("RGB"))

    def _load_existing_manifest(
        self,
        *,
        manifest_path: Path,
        expected_keywords: Sequence[str],
        expected_png_paths: Sequence[Path],
    ) -> SeriesSegmentationExtractionResult | None:
        if self.overwrite_masks or not self.skip_existing_masks or not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        manifest_keywords = [str(keyword).strip() for keyword in list(payload.get("keywords", []) or []) if str(keyword).strip()]
        if manifest_keywords != list(expected_keywords):
            return None

        slice_records = list(payload.get("slices", []) or [])
        if len(slice_records) != len(expected_png_paths):
            return None

        expected_png_relpaths = [self._to_registry_relative_path(path) for path in expected_png_paths]
        mask_paths: list[str] = []
        for expected_png_relpath, slice_record in zip(expected_png_relpaths, slice_records):
            png_relpath = str(slice_record.get("png_path", "")).strip()
            if png_relpath != expected_png_relpath:
                return None
            for keyword in expected_keywords:
                mask_relpath = str((slice_record.get("masks_by_keyword") or {}).get(keyword, "")).strip()
                if not mask_relpath:
                    continue
                mask_path = self.root_dir / mask_relpath
                if not mask_path.is_file():
                    return None
                mask_paths.append(str(mask_path.resolve()))

        return SeriesSegmentationExtractionResult(
            png_dir=str(payload.get("png_dir", "")),
            png_paths=tuple(str(path.resolve()) for path in expected_png_paths),
            mask_dir=str(payload.get("mask_dir", "")),
            manifest_path=str(manifest_path.resolve()),
            mask_paths=tuple(mask_paths),
            slice_count=int(payload.get("slice_count", len(expected_png_paths))),
            pngs_created=0,
            mask_files_created=0,
            existing_mask_files_reused=len(mask_paths),
            keywords=tuple(expected_keywords),
        )

    def extract_series(
        self,
        *,
        series_dir: Path,
        png_paths: Sequence[Path],
        source_image_paths: Sequence[Path] | None = None,
        collection: str,
        patient_id: str,
        study_instance_uid: str,
        series_instance_uid: str,
        modality: str,
    ) -> SeriesSegmentationExtractionResult:
        resolved_series_dir = series_dir.expanduser().resolve()
        resolved_png_inputs = [Path(path).expanduser().resolve() for path in list(png_paths or [])]
        if not resolved_png_inputs:
            raise ValueError("Radiology segmentation now requires persisted PNG paths.")
        source_paths = [Path(path).expanduser().resolve() for path in list(source_image_paths or [])]
        if source_paths and len(resolved_png_inputs) != len(source_paths):
            raise ValueError("PNG paths and source image paths must have the same length when both are provided.")
        png_dir = resolved_png_inputs[0].parent
        if not png_dir.exists():
            raise FileNotFoundError(f"Radiology PNG directory not found for segmentation: {png_dir}")
        mask_dir = build_mask_series_dir(
            root_dir=self.root_dir,
            raw_root=self.raw_root,
            mask_root=self.mask_root,
            series_dir=resolved_series_dir,
        )
        png_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        keywords = self._keyword_list(collection=collection, modality=modality)
        jobs: list[SliceSegmentationJob] = []
        pngs_created = 0
        resolved_png_paths: list[str] = []
        effective_source_paths = source_paths if source_paths else resolved_png_inputs

        for source_path, png_path in zip(effective_source_paths, resolved_png_inputs):
            resolved_png_paths.append(str(png_path.resolve()))
            if not png_path.is_file():
                raise FileNotFoundError(f"Radiology PNG not found for segmentation: {png_path}")
            jobs.append(SliceSegmentationJob(dicom_path=source_path, png_path=png_path))

        manifest_path = self._mask_manifest_path(mask_dir)
        cached_result = self._load_existing_manifest(
            manifest_path=manifest_path,
            expected_keywords=keywords,
            expected_png_paths=[job.png_path for job in jobs],
        )
        if cached_result is not None:
            return cached_result

        mask_paths: list[str] = []
        mask_files_created = 0
        existing_mask_files_reused = 0
        slice_records: list[dict[str, Any]] = []

        for job in jobs:
            image_array = self._load_png_array(job.png_path)
            state = self.model.encode_image(image_array)
            masks_by_keyword: dict[str, str] = {}
            missing_keywords: list[str] = []
            for keyword in keywords:
                keyword_slug = self._slugify_keyword(keyword)
                mask_path = mask_dir / f"{job.png_path.stem}__{keyword_slug}.mask.png"
                if mask_path.exists() and not self.overwrite_masks:
                    masks_by_keyword[keyword] = self._to_registry_relative_path(mask_path)
                    mask_paths.append(str(mask_path.resolve()))
                    existing_mask_files_reused += 1
                    continue

                predicted_mask = self.model.predict_text(state, keyword)
                is_usable, binary_mask = self._mask_is_usable(predicted_mask)
                if not is_usable or binary_mask is None:
                    missing_keywords.append(keyword)
                    continue

                self._save_mask(binary_mask, mask_path)
                masks_by_keyword[keyword] = self._to_registry_relative_path(mask_path)
                mask_paths.append(str(mask_path.resolve()))
                mask_files_created += 1

            slice_records.append(
                {
                    "source_dicom_relpath": self._to_registry_relative_path(job.dicom_path),
                    "source_file_name": job.dicom_path.name,
                    "png_path": self._to_registry_relative_path(job.png_path),
                    "masks_by_keyword": masks_by_keyword,
                    "missing_keywords": missing_keywords,
                }
            )

        manifest_payload = {
            "collection": str(collection).strip(),
            "patient_id": str(patient_id).strip(),
            "study_instance_uid": str(study_instance_uid).strip(),
            "series_instance_uid": str(series_instance_uid).strip(),
            "modality": self._normalize_modality(modality),
            "keywords": list(keywords),
            "png_dir": str(png_dir),
            "mask_dir": str(mask_dir),
            "slice_count": len(jobs),
            "slices": slice_records,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

        return SeriesSegmentationExtractionResult(
            png_dir=str(png_dir),
            png_paths=tuple(resolved_png_paths),
            mask_dir=str(mask_dir),
            manifest_path=str(manifest_path),
            mask_paths=tuple(mask_paths),
            slice_count=len(jobs),
            pngs_created=pngs_created,
            mask_files_created=mask_files_created,
            existing_mask_files_reused=existing_mask_files_reused,
            keywords=tuple(keywords),
        )
