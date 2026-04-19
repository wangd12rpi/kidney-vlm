from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


DEFAULT_PATHOLOGY_PROJECTOR_PROMPT_TEXTS = (
    "Describe the pathology image.",
    "Summarize the pathology image.",
    "Write a pathology caption.",
    "State the main pathology findings.",
    "Give a concise pathology description.",
)

DEFAULT_DNAM_PROJECTOR_PROMPT_TEXTS = (
    "Describe the DNA methylation profile.",
    "Summarize the DNAm case.",
    "Write a DNAm caption.",
    "State the main DNAm findings.",
    "Give a concise DNAm description.",
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


def _coerce_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError("Token payload mapping does not contain 'input_ids'.")
        return _coerce_token_ids(value["input_ids"])
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [int(value.item())]
        if value.ndim == 1:
            return [int(item) for item in value.tolist()]
        if value.ndim == 2 and value.shape[0] == 1:
            return [int(item) for item in value.squeeze(0).tolist()]
        raise TypeError(f"Unsupported token tensor shape: {tuple(value.shape)}")
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            if converted and isinstance(converted[0], list):
                if len(converted) != 1:
                    raise TypeError(f"Unsupported batched token id payload with {len(converted)} rows.")
                converted = converted[0]
            return [int(item) for item in converted]
    raise TypeError(f"Unsupported token id payload type: {type(value).__name__}")


def _shared_prefix_length(left: list[int], right: list[int]) -> int:
    prefix_len = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        prefix_len += 1
    return prefix_len


def _apply_chat_template_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise AttributeError("Tokenizer does not provide apply_chat_template.")

    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            chat_template_kwargs={"enable_thinking": False},
            **kwargs,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    return _coerce_token_ids(token_ids)


def _build_chat_text_pair(
    *,
    tokenizer: Any,
    prompt_text: str,
    answer_text: str,
    max_text_length: int,
) -> tuple[list[int], list[int]]:
    prompt_messages = [{"role": "user", "content": prompt_text}]
    full_messages = prompt_messages + [{"role": "assistant", "content": answer_text}]

    prompt_ids = _apply_chat_template_tokens(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    full_ids = _apply_chat_template_tokens(
        tokenizer,
        full_messages,
        add_generation_prompt=False,
    )
    prefix_len = _shared_prefix_length(prompt_ids, full_ids)
    if prefix_len == 0:
        raise ValueError("Chat template prompt prefix did not align with the full assistant conversation.")

    input_ids = full_ids[:max_text_length]
    labels = (([-100] * prefix_len) + full_ids[prefix_len:])[:max_text_length]
    return input_ids, labels


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


def _contiguous_mean_pool(features: np.ndarray, group_size: int) -> np.ndarray:
    if group_size <= 1 or features.shape[0] <= 1:
        return features
    pooled: list[np.ndarray] = []
    for start in range(0, features.shape[0], group_size):
        pooled.append(features[start : start + group_size].mean(axis=0, dtype=np.float32))
    return np.stack(pooled, axis=0)


def _infer_global_coord_step(coords: np.ndarray) -> int | None:
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] == 0:
        return None

    positive_diffs: list[np.ndarray] = []
    for axis in range(2):
        unique_axis = np.unique(coords[:, axis].astype(np.int64, copy=False))
        if unique_axis.size <= 1:
            continue
        axis_diffs = np.diff(unique_axis)
        axis_diffs = axis_diffs[axis_diffs > 0]
        if axis_diffs.size > 0:
            positive_diffs.append(axis_diffs)

    if not positive_diffs:
        return None

    step = int(np.gcd.reduce(np.concatenate(positive_diffs, axis=0)))
    if step <= 0:
        return None
    return step


def _spatial_bucket_keys(coords: np.ndarray, kernel_size: int) -> np.ndarray | None:
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] == 0:
        return None
    step = _infer_global_coord_step(coords)
    if step is None:
        return None

    x = coords[:, 0].astype(np.int64, copy=False)
    y = coords[:, 1].astype(np.int64, copy=False)
    grid_x = (x - int(x.min())) // step
    grid_y = (y - int(y.min())) // step
    bucket_x = grid_x // kernel_size
    bucket_y = grid_y // kernel_size
    return np.stack([bucket_x, bucket_y], axis=1)


def _spatial_mean_pool(features: np.ndarray, coords: np.ndarray | None, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or features.shape[0] <= 1:
        return features
    group_size = kernel_size * kernel_size
    if coords is None:
        return _contiguous_mean_pool(features, group_size)
    bucket_keys = _spatial_bucket_keys(coords, kernel_size)
    if bucket_keys is None:
        return _contiguous_mean_pool(features, group_size)

    _, inverse = np.unique(bucket_keys, axis=0, return_inverse=True)
    pooled = np.zeros((int(inverse.max()) + 1, features.shape[1]), dtype=np.float32)
    np.add.at(pooled, inverse, features.astype(np.float32, copy=False))
    counts = np.bincount(inverse).astype(np.float32, copy=False)
    pooled /= counts[:, None]
    return pooled


def _spatial_stride_subsample(features: np.ndarray, coords: np.ndarray | None, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or features.shape[0] <= 1:
        return features
    group_size = kernel_size * kernel_size
    if coords is None:
        return features[::group_size]
    bucket_keys = _spatial_bucket_keys(coords, kernel_size)
    if bucket_keys is None:
        return features[::group_size]

    _, first_indices = np.unique(bucket_keys, axis=0, return_index=True)
    ordered_indices = np.sort(first_indices)
    return features[ordered_indices]


def _compress_patch_features(
    features: np.ndarray,
    coords: np.ndarray | None,
    *,
    compression_method: str,
    compression_kernel_size: int,
) -> np.ndarray:
    method = str(compression_method).strip().lower()
    kernel_size = int(compression_kernel_size)
    if method in {"", "none"} or kernel_size <= 1:
        return features
    if method == "mean_pool":
        return _spatial_mean_pool(features, coords, kernel_size)
    if method == "stride":
        return _spatial_stride_subsample(features, coords, kernel_size)
    raise ValueError(f"Unsupported pathology patch compression method: {compression_method}")


def _sample_sequence_features(features: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor, got shape {tuple(features.shape)}")
    if max_tokens <= 0 or features.shape[0] <= max_tokens:
        return features
    indices = torch.linspace(0, features.shape[0] - 1, steps=max_tokens, device=features.device)
    return features[indices.round().to(dtype=torch.long)]


def _load_h5_patch_features(
    path: Path,
    max_patch_tokens: int,
    *,
    compression_method: str = "none",
    compression_kernel_size: int = 1,
) -> torch.Tensor:
    import h5py

    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"Missing 'features' dataset in {path}")
        features = np.asarray(handle["features"])
        coords = np.asarray(handle["coords"]) if "coords" in handle else None

    features = _compress_patch_features(
        features,
        coords,
        compression_method=compression_method,
        compression_kernel_size=compression_kernel_size,
    )
    features = _sample_patch_features(features, max_patch_tokens=max_patch_tokens)
    return torch.from_numpy(features).to(dtype=torch.float32)


def _coerce_loaded_tensor(obj: Any, path: Path) -> torch.Tensor:
    if torch.is_tensor(obj):
        tensor = obj
    elif isinstance(obj, dict):
        if "embedding" in obj and torch.is_tensor(obj["embedding"]):
            tensor = obj["embedding"]
        elif "features" in obj and torch.is_tensor(obj["features"]):
            tensor = obj["features"]
        else:
            tensor_candidates = [value for value in obj.values() if torch.is_tensor(value)]
            if len(tensor_candidates) != 1:
                raise ValueError(f"Unable to resolve DNAm tensor from {path}; found {len(tensor_candidates)} tensor candidates.")
            tensor = tensor_candidates[0]
    else:
        raise TypeError(f"Unsupported DNAm feature payload type from {path}: {type(obj).__name__}")

    tensor = tensor.detach().cpu().to(dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        pass
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    else:
        raise ValueError(f"Unsupported DNAm tensor shape from {path}: {tuple(tensor.shape)}")
    return tensor


def _load_pt_feature_tensor(path: Path, max_tokens: int) -> torch.Tensor:
    tensor = _coerce_loaded_tensor(torch.load(path, map_location="cpu", weights_only=True), path)
    return _sample_sequence_features(tensor, max_tokens=max_tokens)


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
    patch_compression_method: str = "none"
    patch_compression_kernel_size: int = 1
    patch_token_dropout_prob: float = 0.0
    instruction_field: str = "instruction"
    answer_field: str = "answer"
    pathology_embedding_field: str = "pathology_tile_embedding_paths"
    prompt_texts: tuple[str, ...] = DEFAULT_PATHOLOGY_PROJECTOR_PROMPT_TEXTS

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).expanduser().resolve()
        self.patch_compression_method = str(self.patch_compression_method).strip().lower() or "none"
        self.patch_compression_kernel_size = int(self.patch_compression_kernel_size)
        if self.patch_compression_kernel_size < 1:
            raise ValueError("PathologyProjectorQACollator.patch_compression_kernel_size must be >= 1.")
        if self.patch_compression_method not in {"none", "mean_pool", "stride"}:
            raise ValueError(
                "PathologyProjectorQACollator.patch_compression_method must be one of: none, mean_pool, stride."
            )
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

        if hasattr(self.tokenizer, "apply_chat_template"):
            return _build_chat_text_pair(
                tokenizer=self.tokenizer,
                prompt_text=prompt_text,
                answer_text=answer,
                max_text_length=self.max_text_length,
            )

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
            patch_tensor = _load_h5_patch_features(
                patch_path,
                max_patch_tokens=self.max_patch_tokens,
                compression_method=self.patch_compression_method,
                compression_kernel_size=self.patch_compression_kernel_size,
            )
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


@dataclass
class DNAMProjectorQACollator:
    tokenizer: Any
    root_dir: str | Path
    max_text_length: int = 512
    max_dnam_tokens: int = 8
    instruction_field: str = "instruction"
    answer_field: str = "answer"
    dnam_embedding_field: str = "genomics_dna_methylation_feature_path"
    prompt_texts: tuple[str, ...] = DEFAULT_DNAM_PROJECTOR_PROMPT_TEXTS

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).expanduser().resolve()
        self.prompt_texts = tuple(str(prompt).strip() for prompt in self.prompt_texts if str(prompt).strip())
        if not self.prompt_texts:
            raise ValueError("DNAMProjectorQACollator requires at least one prompt text.")

    def _select_prompt_text(self) -> str:
        return random.choice(self.prompt_texts)

    def _build_text_pair(self, feature: dict[str, Any]) -> tuple[list[int], list[int]]:
        answer = str(feature.get(self.answer_field, "")).strip()
        if not answer:
            raise ValueError("Empty answer/caption encountered in DNAm projector batch.")

        prompt_text = self._select_prompt_text()
        if hasattr(self.tokenizer, "apply_chat_template"):
            return _build_chat_text_pair(
                tokenizer=self.tokenizer,
                prompt_text=prompt_text,
                answer_text=answer,
                max_text_length=self.max_text_length,
            )

        eos_text = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(f" {answer}{eos_text}", add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[: self.max_text_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_text_length]
        return input_ids, labels

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("DNAMProjectorQACollator received an empty batch.")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id before batching projector data.")

        text_input_ids: list[list[int]] = []
        text_labels: list[list[int]] = []
        dnam_tensors: list[torch.Tensor] = []
        metadata_keys = ("sample_id", "project_id", "source", "patient_id")
        metadata: dict[str, list[Any]] = {key: [] for key in metadata_keys}

        for feature in features:
            input_ids, labels = self._build_text_pair(feature)
            text_input_ids.append(input_ids)
            text_labels.append(labels)
            feature_path = _resolve_existing_path(self.root_dir, feature.get(self.dnam_embedding_field, []))
            dnam_tensor = _load_pt_feature_tensor(feature_path, max_tokens=self.max_dnam_tokens)
            dnam_tensors.append(dnam_tensor)
            for key in metadata_keys:
                metadata[key].append(feature.get(key))

        batch_size = len(features)
        max_text_tokens = max(len(item) for item in text_input_ids)
        max_dnam_tokens = max(tensor.shape[0] for tensor in dnam_tensors)
        dnam_dim = dnam_tensors[0].shape[1]

        input_ids = torch.full((batch_size, max_text_tokens), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_text_tokens), dtype=torch.long)
        labels = torch.full((batch_size, max_text_tokens), -100, dtype=torch.long)
        dnam_features = torch.zeros((batch_size, max_dnam_tokens, dnam_dim), dtype=torch.float32)
        dnam_feature_mask = torch.zeros((batch_size, max_dnam_tokens), dtype=torch.long)

        for row_idx, (token_ids, token_labels, dnam_tensor) in enumerate(
            zip(text_input_ids, text_labels, dnam_tensors, strict=True)
        ):
            text_len = len(token_ids)
            token_count = dnam_tensor.shape[0]

            input_ids[row_idx, :text_len] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[row_idx, :text_len] = 1
            labels[row_idx, :text_len] = torch.tensor(token_labels, dtype=torch.long)
            dnam_features[row_idx, :token_count] = dnam_tensor
            dnam_feature_mask[row_idx, :token_count] = 1

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "dnam_features": dnam_features,
            "dnam_feature_mask": dnam_feature_mask,
        }
        batch.update(metadata)
        return batch
