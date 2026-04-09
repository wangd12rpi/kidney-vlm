from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


DEFAULT_PATHOLOGY_PROJECTOR_PROMPT_TEXTS = (
    "Describe the pathology image.\nCaption:",
    "Summarize the pathology image.\nCaption:",
    "Provide a pathology caption for this image.\nCaption:",
    "Write a detailed pathology description for this image.\nCaption:",
    "Explain the key pathology findings visible in this image.\nCaption:",
    "Generate a pathology caption based on this image.\nCaption:",
    "Describe the histopathology shown in this image.\nCaption:",
    "Write a pathology report-style caption for this image.\nCaption:",
    "Give a concise pathology description of this image.\nCaption:",
    "State the main morphologic features seen in this pathology image.\nCaption:",
)


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


def _resolve_existing_path(root_dir: Path, values: Any) -> Path:
    for raw_value in _normalize_list(values):
        candidate = Path(str(raw_value)).expanduser()
        if not candidate.is_absolute():
            candidate = root_dir / candidate
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No existing path found in values: {_normalize_list(values)}")


def _sample_patch_features(features: np.ndarray, max_patch_tokens: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D pathology features, got shape {tuple(features.shape)}")
    if max_patch_tokens <= 0 or features.shape[0] <= max_patch_tokens:
        return features

    indices = np.linspace(0, features.shape[0] - 1, num=max_patch_tokens, dtype=np.int64)
    return features[indices]


def _load_h5_patch_features(path: Path, max_patch_tokens: int) -> torch.Tensor:
    import h5py

    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"Missing 'features' dataset in {path}")
        features = np.asarray(handle["features"])

    features = _sample_patch_features(features, max_patch_tokens=max_patch_tokens)
    return torch.from_numpy(features).to(dtype=torch.float32)


def _apply_patch_token_dropout(patch_tensor: torch.Tensor, dropout_prob: float) -> torch.Tensor:
    if patch_tensor.ndim != 2:
        raise ValueError(f"Expected 2D pathology patch tensor, got shape {tuple(patch_tensor.shape)}")

    keep_prob = float(dropout_prob)
    if keep_prob <= 0.0 or patch_tensor.shape[0] <= 1:
        return patch_tensor
    if keep_prob > 1.0:
        raise ValueError(f"patch_token_dropout_prob must be in [0, 1], got {dropout_prob}")

    scores = torch.rand(patch_tensor.shape[0], device=patch_tensor.device)
    keep_mask = scores >= keep_prob
    if not torch.any(keep_mask):
        keep_mask[torch.argmax(scores)] = True
    return patch_tensor[keep_mask]


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


@dataclass
class PathologyProjectorQACollator:
    tokenizer: Any
    root_dir: str | Path
    max_text_length: int = 512
    max_patch_tokens: int = 128
    patch_token_dropout_prob: float = 0.0
    instruction_field: str = "instruction"
    answer_field: str = "answer"
    pathology_embedding_field: str = "pathology_tile_embedding_paths"
    prompt_texts: tuple[str, ...] = DEFAULT_PATHOLOGY_PROJECTOR_PROMPT_TEXTS

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).expanduser().resolve()
        self.patch_token_dropout_prob = float(self.patch_token_dropout_prob)
        if not 0.0 <= self.patch_token_dropout_prob <= 1.0:
            raise ValueError("PathologyProjectorQACollator.patch_token_dropout_prob must be in [0, 1].")
        self.prompt_texts = tuple(str(prompt).strip() for prompt in self.prompt_texts if str(prompt).strip())
        if not self.prompt_texts:
            raise ValueError("PathologyProjectorQACollator requires at least one prompt text.")

    def _select_prompt_text(self) -> str:
        return random.choice(self.prompt_texts)

    def _build_text_pair(self, feature: dict[str, Any]) -> tuple[list[int], list[int]]:
        answer = str(feature.get(self.answer_field, "")).strip()
        if not answer:
            raise ValueError("Empty answer/caption encountered in projector QA batch.")

        prompt_text = self._select_prompt_text()
        if not prompt_text:
            raise ValueError("Prompt text is empty.")

        eos_text = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(f" {answer}{eos_text}", add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[: self.max_text_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_text_length]
        return input_ids, labels

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("PathologyProjectorQACollator received an empty batch.")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id before batching projector data.")

        text_input_ids: list[list[int]] = []
        text_labels: list[list[int]] = []
        pathology_tensors: list[torch.Tensor] = []
        metadata_keys = ("sample_id", "project_id", "source")
        metadata: dict[str, list[Any]] = {key: [] for key in metadata_keys}

        for feature in features:
            input_ids, labels = self._build_text_pair(feature)
            text_input_ids.append(input_ids)
            text_labels.append(labels)
            patch_path = _resolve_existing_path(self.root_dir, feature.get(self.pathology_embedding_field, []))
            patch_tensor = _load_h5_patch_features(patch_path, max_patch_tokens=self.max_patch_tokens)
            patch_tensor = _apply_patch_token_dropout(patch_tensor, self.patch_token_dropout_prob)
            pathology_tensors.append(patch_tensor)
            for key in metadata_keys:
                metadata[key].append(feature.get(key))

        batch_size = len(features)
        max_text_tokens = max(len(item) for item in text_input_ids)
        max_patch_tokens = max(tensor.shape[0] for tensor in pathology_tensors)
        pathology_dim = pathology_tensors[0].shape[1]

        input_ids = torch.full((batch_size, max_text_tokens), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_text_tokens), dtype=torch.long)
        labels = torch.full((batch_size, max_text_tokens), -100, dtype=torch.long)
        pathology_features = torch.zeros((batch_size, max_patch_tokens, pathology_dim), dtype=torch.float32)
        pathology_feature_mask = torch.zeros((batch_size, max_patch_tokens), dtype=torch.long)

        for row_idx, (token_ids, token_labels, patch_tensor) in enumerate(
            zip(text_input_ids, text_labels, pathology_tensors, strict=True)
        ):
            text_len = len(token_ids)
            patch_len = patch_tensor.shape[0]

            input_ids[row_idx, :text_len] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[row_idx, :text_len] = 1
            labels[row_idx, :text_len] = torch.tensor(token_labels, dtype=torch.long)
            pathology_features[row_idx, :patch_len] = patch_tensor
            pathology_feature_mask[row_idx, :patch_len] = 1

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pathology_features": pathology_features,
            "pathology_feature_mask": pathology_feature_mask,
        }
        batch.update(metadata)
        return batch
