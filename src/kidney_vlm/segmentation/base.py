from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class SegmentationAdapterBase(ABC):
    @abstractmethod
    def predict(self, image_paths: Sequence[str]) -> Any:
        """Return segmentation outputs for provided image paths."""


class PlaceholderPathologySegmentation(SegmentationAdapterBase):
    def predict(self, image_paths: Sequence[str]) -> Any:
        raise NotImplementedError("Pathology segmentation adapter is not implemented in scaffold.")


class PlaceholderRadiologySegmentation(SegmentationAdapterBase):
    def predict(self, image_paths: Sequence[str]) -> Any:
        raise NotImplementedError("Radiology segmentation adapter is not implemented in scaffold.")
