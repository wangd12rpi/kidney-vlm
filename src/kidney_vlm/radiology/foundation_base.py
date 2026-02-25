from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class RadiologyFoundationAdapterBase(ABC):
    @abstractmethod
    def encode(self, image_paths: Sequence[str]) -> Any:
        """Return radiology foundation embeddings for a batch of images."""


class PlaceholderRadiologyFoundationAdapter(RadiologyFoundationAdapterBase):
    def encode(self, image_paths: Sequence[str]) -> Any:
        raise NotImplementedError(
            "No concrete radiology foundation model has been wired yet. "
            "Implement this adapter once model/checkpoint/API are selected."
        )
