from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
    return [str(value)]


@dataclass
class QACollator:
    tokenizer: Any | None = None
    max_length: int = 512

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if self.tokenizer is None:
            return {key: [feature.get(key) for feature in features] for key in features[0]}

        questions = [str(feature.get("question", "")) for feature in features]
        answers = [str(feature.get("answer", "")) for feature in features]

        encoded_inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded_labels = self.tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded_inputs["labels"] = encoded_labels["input_ids"]
        pathology_tile_embedding_paths = [
            _normalize_list(feature.get("pathology_tile_embedding_paths", []))
            for feature in features
        ]
        pathology_slide_embedding_paths = [
            _normalize_list(feature.get("pathology_slide_embedding_paths", []))
            for feature in features
        ]
        radiology_embedding_paths = [
            _normalize_list(feature.get("radiology_embedding_paths", []))
            for feature in features
        ]
        encoded_inputs["pathology_tile_embedding_paths"] = pathology_tile_embedding_paths
        encoded_inputs["pathology_slide_embedding_paths"] = pathology_slide_embedding_paths
        encoded_inputs["radiology_embedding_paths"] = radiology_embedding_paths
        encoded_inputs["pathology_image_paths"] = [
            _normalize_list(feature.get("pathology_wsi_paths", [])) for feature in features
        ]
        encoded_inputs["radiology_image_paths"] = [
            _normalize_list(feature.get("radiology_image_paths", [])) for feature in features
        ]
        encoded_inputs["pathology_mask_paths"] = [
            _normalize_list(feature.get("pathology_mask_paths", [])) for feature in features
        ]
        encoded_inputs["radiology_mask_paths"] = [
            _normalize_list(feature.get("radiology_mask_paths", [])) for feature in features
        ]
        return encoded_inputs
