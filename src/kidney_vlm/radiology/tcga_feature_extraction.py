from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from PIL import Image

from kidney_vlm.radiology.feature_registry import format_series_embedding_ref


STRING_FIELDS = (
    "keys",
    "project_ids",
    "patient_ids",
    "study_instance_uids",
    "series_instance_uids",
    "dicom_relpaths",
    "png_relpaths",
    "modalities",
)


@dataclass(frozen=True)
class SliceFeatureJob:
    key: str
    dicom_path: Path
    png_path: Path


@dataclass(frozen=True)
class SeriesFeatureExtractionResult:
    png_dir: str
    embedding_ref: str
    slice_count: int
    png_paths: tuple[str, ...]
    pngs_created: int
    existing_features_reused: int
    new_features_written: int


class TCGARadiologyFeatureExtractor:
    def __init__(
        self,
        *,
        root_dir: Path,
        feature_store_path: Path,
        model_name: str,
        input_size: int,
        batch_size: int,
        device: str,
        skip_existing_features: bool,
    ) -> None:
        self.root_dir = root_dir.expanduser().resolve()
        self.feature_store_path = feature_store_path.expanduser().resolve()
        self.model_name = str(model_name).strip()
        self.input_size = int(input_size)
        self.batch_size = int(batch_size)
        self.skip_existing_features = bool(skip_existing_features)

        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("Radiology feature extraction requires 'torch'.") from exc

        try:
            from transformers import AutoProcessor, SiglipVisionModel
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Radiology feature extraction requires 'transformers' for MedSigLIP encoding."
            ) from exc

        self.torch = torch
        self.device = self._resolve_device(str(device).strip() or "cpu")

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SiglipVisionModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

        self._existing_feature_keys = (
            self._load_existing_feature_keys(self.feature_store_path)
            if self.skip_existing_features
            else set()
        )

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device.startswith("cuda") and not self.torch.cuda.is_available():
            return "cpu"
        return requested_device

    def _to_registry_relative_path(self, path_value: str | Path) -> str:
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            return path.as_posix().lstrip("/")
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.root_dir).as_posix()
        except ValueError:
            return resolved.as_posix().lstrip("/")

    def _encode_png_batch(self, png_paths: Sequence[Path]) -> np.ndarray:
        images: list[Image.Image] = []
        try:
            for png_path in png_paths:
                images.append(Image.open(png_path).convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs["pooler_output"]
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
            return embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
        finally:
            for image in images:
                image.close()

    def _load_existing_feature_keys(self, store_path: Path) -> set[str]:
        if not store_path.exists():
            return set()
        with h5py.File(store_path, "r") as handle:
            if "features" not in handle or "keys" not in handle:
                raise RuntimeError(f"Radiology feature store is missing required datasets: {store_path}")
            if handle["features"].shape[0] != handle["keys"].shape[0]:
                raise RuntimeError(f"Radiology feature store datasets are misaligned: {store_path}")
            values = handle["keys"][:]
            return {
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in values
            }

    def _ensure_store(self, handle: h5py.File, *, feature_dim: int) -> None:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        if "features" not in handle:
            handle.create_dataset(
                "features",
                shape=(0, feature_dim),
                maxshape=(None, feature_dim),
                chunks=(1, feature_dim),
                dtype=np.float32,
            )
            for field in STRING_FIELDS:
                handle.create_dataset(field, shape=(0,), maxshape=(None,), dtype=string_dtype)
            handle.attrs["model_name"] = self.model_name
            handle.attrs["input_size"] = self.input_size
            handle.attrs["resize_backend"] = "transformers.image_processor"
            return

        if handle["features"].ndim != 2 or handle["features"].shape[1] != feature_dim:
            raise RuntimeError("Radiology feature store has incompatible feature dimensions.")
        expected_rows = handle["features"].shape[0]
        for field in STRING_FIELDS:
            if field not in handle:
                raise RuntimeError(f"Radiology feature store is missing metadata dataset '{field}'.")
            if handle[field].shape[0] != expected_rows:
                raise RuntimeError(f"Radiology feature store dataset '{field}' is misaligned with 'features'.")

    def _append_feature_batch(
        self,
        *,
        batch_features: np.ndarray,
        batch_jobs: Sequence[SliceFeatureJob],
        project_id: str,
        patient_id: str,
        study_instance_uid: str,
        series_instance_uid: str,
        modality: str,
    ) -> None:
        if batch_features.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {tuple(batch_features.shape)}")
        if len(batch_jobs) != batch_features.shape[0]:
            raise ValueError("Feature batch size does not match metadata batch size.")
        if not batch_jobs:
            return

        self.feature_store_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.feature_store_path, "a") as handle:
            self._ensure_store(handle, feature_dim=int(batch_features.shape[1]))
            start = int(handle["features"].shape[0])
            end = start + len(batch_jobs)
            handle["features"].resize((end, batch_features.shape[1]))
            handle["features"][start:end] = batch_features.astype(np.float32, copy=False)

            values_by_field = {
                "keys": [job.key for job in batch_jobs],
                "project_ids": [project_id for _ in batch_jobs],
                "patient_ids": [patient_id for _ in batch_jobs],
                "study_instance_uids": [study_instance_uid for _ in batch_jobs],
                "series_instance_uids": [series_instance_uid for _ in batch_jobs],
                "dicom_relpaths": [self._to_registry_relative_path(job.dicom_path) for job in batch_jobs],
                "png_relpaths": [self._to_registry_relative_path(job.png_path) for job in batch_jobs],
                "modalities": [modality for _ in batch_jobs],
            }
            for field, values in values_by_field.items():
                handle[field].resize((end,))
                handle[field][start:end] = list(values)

    def extract_series(
        self,
        *,
        png_paths: Sequence[Path],
        source_image_paths: Sequence[Path] | None = None,
        project_id: str,
        patient_id: str,
        study_instance_uid: str,
        series_instance_uid: str,
        modality: str,
    ) -> SeriesFeatureExtractionResult:
        resolved_png_inputs = [Path(path).expanduser().resolve() for path in list(png_paths or [])]
        if not resolved_png_inputs:
            raise ValueError("Radiology feature extraction now requires persisted PNG paths.")
        source_paths = [Path(path).expanduser().resolve() for path in list(source_image_paths or [])]
        if source_paths and len(resolved_png_inputs) != len(source_paths):
            raise ValueError("PNG paths and source image paths must have the same length when both are provided.")
        png_dir = resolved_png_inputs[0].parent
        if not png_dir.exists():
            raise FileNotFoundError(f"Radiology PNG directory not found for feature extraction: {png_dir}")

        pngs_created = 0
        existing_features_reused = 0
        new_features_written = 0
        resolved_png_paths: list[str] = []
        pending_jobs: list[SliceFeatureJob] = []
        effective_source_paths = source_paths if source_paths else resolved_png_inputs

        for resolved_source_path, png_path in zip(effective_source_paths, resolved_png_inputs):
            job = SliceFeatureJob(
                key=self._to_registry_relative_path(resolved_source_path),
                dicom_path=resolved_source_path,
                png_path=png_path,
            )
            resolved_png_paths.append(str(png_path.resolve()))
            if not png_path.is_file():
                raise FileNotFoundError(f"Radiology PNG not found for feature extraction: {png_path}")
            if self.skip_existing_features and job.key in self._existing_feature_keys:
                existing_features_reused += 1
                continue
            pending_jobs.append(job)

        for start in range(0, len(pending_jobs), self.batch_size):
            batch_jobs = pending_jobs[start : start + self.batch_size]
            batch_features = self._encode_png_batch([job.png_path for job in batch_jobs])
            self._append_feature_batch(
                batch_features=batch_features,
                batch_jobs=batch_jobs,
                project_id=project_id,
                patient_id=patient_id,
                study_instance_uid=study_instance_uid,
                series_instance_uid=series_instance_uid,
                modality=modality,
            )
            new_features_written += len(batch_jobs)
            self._existing_feature_keys.update(job.key for job in batch_jobs)

        return SeriesFeatureExtractionResult(
            png_dir=str(png_dir),
            embedding_ref=format_series_embedding_ref(
                root_dir=self.root_dir,
                store_path=self.feature_store_path,
                series_dir=png_dir,
            ),
            slice_count=len(resolved_png_inputs),
            png_paths=tuple(resolved_png_paths),
            pngs_created=pngs_created,
            existing_features_reused=existing_features_reused,
            new_features_written=new_features_written,
        )
