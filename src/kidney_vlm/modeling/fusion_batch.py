from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FusionBatch:
    input_ids: Any
    attention_mask: Any | None = None
    labels: Any | None = None
    pathology_features: Any | None = None  # [batch, n_pathology_images, d] or None
    radiology_features: Any | None = None  # [batch, n_radiology_images, d] or None
    pathology_feature_mask: Any | None = None  # [batch, n_pathology_images] or None
    radiology_feature_mask: Any | None = None  # [batch, n_radiology_images] or None
    pathology_mask_paths: list[list[str]] | None = None
    radiology_mask_paths: list[list[str]] | None = None
    pathology_image_paths: list[list[str]] | None = None
    radiology_image_paths: list[list[str]] | None = None
    biomarkers_text: list[str] | None = None
