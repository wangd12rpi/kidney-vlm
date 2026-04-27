from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from kidney_vlm.training.collator import (
    _build_chat_text_pair,
    _load_h5_patch_features,
    _load_pt_feature_tensor,
    _load_radiology_feature_tensor,
    _normalize_list,
    _resolve_existing_path,
    _sample_sequence_features,
)
from kidney_vlm.vqa.constants import MODALITIES, MODALITY_FEATURE_COLUMNS, MODALITY_FLAG_COLUMNS
from kidney_vlm.vqa.prompts import (
    build_vqa_prompt,
    is_open_ended_question_type,
    option_values,
    prefix_placeholder_for_modality,
    row_modalities,
    row_uses_modality,
)
from kidney_vlm.vqa.stage_config import cfg_get, cfg_list, clean_text, enabled_modality_names


class VQADataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.records = frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _coerce_token_ids(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if torch.is_tensor(value):
        return [int(item) for item in value.flatten().tolist()]
    if hasattr(value, "tolist"):
        return _coerce_token_ids(value.tolist())
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError("Token payload mapping does not contain 'input_ids'.")
        return _coerce_token_ids(value["input_ids"])
    raise TypeError(f"Unsupported token id payload type: {type(value).__name__}")


def _tokenize_plain(tokenizer: Any, text: str) -> list[int]:
    return _coerce_token_ids(tokenizer(text, add_special_tokens=False)["input_ids"])


def _find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        return None
    last_start = len(haystack) - len(needle)
    for start in range(last_start + 1):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def _placeholder_token_candidates(tokenizer: Any, placeholder: str) -> list[list[int]]:
    candidates: list[list[int]] = []
    for text in (placeholder, f"\n{placeholder}", f"{placeholder}\n", f"\n{placeholder}\n"):
        token_ids = _tokenize_plain(tokenizer, text)
        if token_ids and token_ids not in candidates:
            candidates.append(token_ids)
    return candidates


def _find_placeholder_span(tokenizer: Any, input_ids: list[int], placeholder: str) -> tuple[int, int]:
    for candidate_ids in _placeholder_token_candidates(tokenizer, placeholder):
        span = _find_subsequence(input_ids, candidate_ids)
        if span is not None:
            return span
    raise ValueError(f"Could not locate prefix placeholder tokens in VQA prompt: {placeholder}")


def build_text_pair(
    tokenizer: Any,
    *,
    prompt_text: str,
    answer_text: str,
    max_text_length: int,
) -> tuple[list[int], list[int]]:
    if hasattr(tokenizer, "apply_chat_template"):
        return _build_chat_text_pair(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            answer_text=answer_text,
            max_text_length=max_text_length,
        )

    eos_text = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(f" {answer_text}{eos_text}", add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_text_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_text_length]
    return input_ids, labels


def build_text_pair_with_prefix_spans(
    tokenizer: Any,
    *,
    row: Mapping[str, Any],
    prompt_cfg: Any,
    answer_text: str,
    max_text_length: int,
) -> tuple[list[int], list[int], list[dict[str, int | str]], str]:
    prompt_text = build_vqa_prompt(row, prompt_cfg)
    input_ids, labels = build_text_pair(
        tokenizer,
        prompt_text=prompt_text,
        answer_text=answer_text,
        max_text_length=max_text_length,
    )

    spans: list[dict[str, int | str]] = []
    for modality in row_modalities(row):
        placeholder = prefix_placeholder_for_modality(modality)
        start, end = _find_placeholder_span(tokenizer, input_ids, placeholder)
        if any(labels[index] != -100 for index in range(start, end)):
            raise ValueError(
                f"Prefix placeholder for {modality} landed in the supervised answer span; "
                "check VQA chat-template masking."
            )
        spans.append({"modality": modality, "start": start, "end": end})

    spans = sorted(spans, key=lambda item: int(item["start"]))
    for left, right in zip(spans, spans[1:], strict=False):
        if int(left["end"]) > int(right["start"]):
            raise ValueError(f"Overlapping VQA prefix placeholders: {spans}")
    return input_ids, labels, spans, prompt_text


def apply_token_dropout(feature_tensor: torch.Tensor, dropout_prob: float) -> torch.Tensor:
    if feature_tensor.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor for token dropout, got shape {tuple(feature_tensor.shape)}")
    probability = float(dropout_prob)
    if probability <= 0.0 or feature_tensor.shape[0] <= 1:
        return feature_tensor
    if probability > 1.0:
        raise ValueError(f"token_dropout_prob must be in [0, 1], got {dropout_prob}")
    scores = torch.rand(feature_tensor.shape[0], device=feature_tensor.device)
    keep_mask = scores >= probability
    if not torch.any(keep_mask):
        keep_mask[torch.argmax(scores)] = True
    return feature_tensor[keep_mask]


def load_pathology_feature_tensor(root_dir: Path, row: Mapping[str, Any], block_cfg: Any) -> torch.Tensor:
    values = _normalize_list(row.get("pathology_feature_paths", []))
    if not values:
        raise FileNotFoundError(f"Question {row.get('question_id', '<unknown>')} requires pathology features, but paths are empty.")

    tensors: list[torch.Tensor] = []
    for raw_value in values:
        path = _resolve_existing_path(root_dir, raw_value)
        tensors.append(
            _load_h5_patch_features(
                path,
                max_patch_tokens=int(cfg_get(block_cfg, "max_tokens", 4096)),
                compression_method=str(cfg_get(block_cfg, "patch_compression_method", "none")),
                compression_kernel_size=int(cfg_get(block_cfg, "patch_compression_kernel_size", 1)),
            )
        )
    tensor = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
    tensor = _sample_sequence_features(tensor, max_tokens=int(cfg_get(block_cfg, "max_tokens", 4096)))
    return apply_token_dropout(tensor, float(cfg_get(block_cfg, "token_dropout_prob", 0.0)))


def load_dnam_feature_tensor(root_dir: Path, row: Mapping[str, Any], block_cfg: Any) -> torch.Tensor:
    value = clean_text(row.get("dnam_feature_path", ""))
    if not value:
        raise FileNotFoundError(f"Question {row.get('question_id', '<unknown>')} requires DNAm features, but dnam_feature_path is empty.")
    path = _resolve_existing_path(root_dir, value)
    return _load_pt_feature_tensor(path, max_tokens=int(cfg_get(block_cfg, "max_tokens", 8)))


def load_rna_feature_tensor(root_dir: Path, row: Mapping[str, Any], block_cfg: Any) -> torch.Tensor:
    value = clean_text(row.get("rna_feature_path", ""))
    if not value:
        raise FileNotFoundError(f"Question {row.get('question_id', '<unknown>')} requires RNA features, but rna_feature_path is empty.")
    path = _resolve_existing_path(root_dir, value)
    return _load_pt_feature_tensor(path, max_tokens=int(cfg_get(block_cfg, "max_tokens", 1)))


def load_modality_feature_tensor(root_dir: Path, row: Mapping[str, Any], modality: str, block_cfg: Any) -> torch.Tensor:
    if modality == "pathology":
        return load_pathology_feature_tensor(root_dir, row, block_cfg)
    if modality == "radiology":
        tensor = _load_radiology_feature_tensor(
            root_dir,
            row.get("radiology_feature_paths", []),
            max_slice_tokens=int(cfg_get(block_cfg, "max_tokens", 32)),
        )
        return apply_token_dropout(tensor, float(cfg_get(block_cfg, "token_dropout_prob", 0.0)))
    if modality == "dnam":
        return load_dnam_feature_tensor(root_dir, row, block_cfg)
    if modality == "rna":
        return load_rna_feature_tensor(root_dir, row, block_cfg)
    raise ValueError(f"Unsupported modality: {modality}")


def pad_optional_feature_tensors(modality: str, tensors: list[torch.Tensor | None]) -> dict[str, torch.Tensor]:
    present_tensors = [tensor for tensor in tensors if tensor is not None]
    if not present_tensors:
        return {}
    feature_dim = int(present_tensors[0].shape[1])
    for tensor in present_tensors:
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D {modality} feature tensor, got shape {tuple(tensor.shape)}")
        if int(tensor.shape[1]) != feature_dim:
            raise ValueError(
                f"{modality} feature dimension mismatch in batch: expected {feature_dim}, got {int(tensor.shape[1])}"
            )

    batch_size = len(tensors)
    max_tokens = max(int(tensor.shape[0]) for tensor in present_tensors)
    features = torch.zeros((batch_size, max_tokens, feature_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    for row_index, tensor in enumerate(tensors):
        if tensor is None:
            continue
        token_count = int(tensor.shape[0])
        features[row_index, :token_count] = tensor
        mask[row_index, :token_count] = 1
    return {
        f"{modality}_features": features,
        f"{modality}_feature_mask": mask,
    }


@dataclass
class VQATrainingCollator:
    tokenizer: Any
    root_dir: str | Path
    stage_cfg: Any

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).expanduser().resolve()
        self.max_text_length = int(cfg_get(self.stage_cfg, "max_text_length", 1024))
        self.prompt_cfg = cfg_get(self.stage_cfg, "prompt", {})
        self.projectors_cfg = cfg_get(self.stage_cfg, "projectors", {})

    def _build_text_pair(self, row: Mapping[str, Any]) -> tuple[list[int], list[int], list[dict[str, int | str]], str]:
        answer = clean_text(row.get("answer", ""))
        if not answer:
            raise ValueError(f"Question {row.get('question_id', '<unknown>')} has empty answer text.")
        return build_text_pair_with_prefix_spans(
            self.tokenizer,
            row=row,
            prompt_cfg=self.prompt_cfg,
            answer_text=answer,
            max_text_length=self.max_text_length,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("VQATrainingCollator received an empty batch.")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id before batching VQA data.")

        text_input_ids: list[list[int]] = []
        text_labels: list[list[int]] = []
        prefix_spans: list[list[dict[str, int | str]]] = []
        prompt_texts: list[str] = []
        modality_tensors: dict[str, list[torch.Tensor | None]] = {modality: [] for modality in MODALITIES}
        metadata_keys = ("question_id", "case_id", "project_id", "task_id", "question_type", "generation_type")
        metadata: dict[str, list[Any]] = {key: [] for key in metadata_keys}

        for row in features:
            input_ids, labels, row_prefix_spans, prompt_text = self._build_text_pair(row)
            text_input_ids.append(input_ids)
            text_labels.append(labels)
            prefix_spans.append(row_prefix_spans)
            prompt_texts.append(prompt_text)
            for modality in MODALITIES:
                if row_uses_modality(row, modality):
                    block_cfg = cfg_get(self.projectors_cfg, modality, {})
                    modality_tensors[modality].append(load_modality_feature_tensor(self.root_dir, row, modality, block_cfg))
                else:
                    modality_tensors[modality].append(None)
            for key in metadata_keys:
                metadata[key].append(row.get(key))

        batch_size = len(features)
        max_text_tokens = max(len(item) for item in text_input_ids)
        input_ids_tensor = torch.full((batch_size, max_text_tokens), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_text_tokens), dtype=torch.long)
        labels_tensor = torch.full((batch_size, max_text_tokens), -100, dtype=torch.long)
        for row_index, (token_ids, token_labels) in enumerate(zip(text_input_ids, text_labels, strict=True)):
            token_count = len(token_ids)
            input_ids_tensor[row_index, :token_count] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[row_index, :token_count] = 1
            labels_tensor[row_index, :token_count] = torch.tensor(token_labels, dtype=torch.long)

        batch: dict[str, Any] = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
            "prefix_spans": prefix_spans,
            "prompt_text": prompt_texts,
        }
        for modality, tensors in modality_tensors.items():
            batch.update(pad_optional_feature_tensors(modality, tensors))
        batch.update(metadata)
        return batch


def apply_row_limit(frame: pd.DataFrame, *, max_rows: Any, sample: bool, seed: int) -> pd.DataFrame:
    if max_rows in (None, "", "null"):
        return frame.reset_index(drop=True)
    max_rows_int = int(max_rows)
    if max_rows_int < 0:
        raise ValueError(f"max rows must be non-negative, got {max_rows_int}")
    if len(frame) <= max_rows_int:
        return frame.reset_index(drop=True)
    if sample:
        return frame.sample(n=max_rows_int, random_state=seed).sort_values(
            ["project_id", "case_id", "task_id", "question_id"]
        ).reset_index(drop=True)
    return frame.head(max_rows_int).reset_index(drop=True)


def select_vqa_rows(frame: pd.DataFrame, stage_cfg: Any, *, split: str, max_samples_key: str, sample_key: str) -> pd.DataFrame:
    dataset_cfg = cfg_get(stage_cfg, "dataset", {})
    out = frame[frame["split"].astype(str).str.lower().eq(str(split).strip().lower())].copy()

    for column, values_key in [
        ("question_type", "question_types"),
        ("generation_type", "generation_types"),
        ("project_id", "project_ids"),
        ("task_category", "task_categories"),
        ("task_id", "task_ids"),
    ]:
        values = cfg_list(cfg_get(dataset_cfg, values_key, []))
        if values:
            out = out[out[column].astype(str).isin(values)]

    enabled_modalities = set(enabled_modality_names(stage_cfg))
    for modality in MODALITIES:
        if modality not in enabled_modalities:
            out = out[~out[MODALITY_FLAG_COLUMNS[modality]].astype(bool)]

    modality_mask = pd.Series(False, index=out.index)
    for modality in enabled_modalities:
        modality_mask |= out[MODALITY_FLAG_COLUMNS[modality]].astype(bool)
    out = out[modality_mask]

    out = out.sort_values(["project_id", "case_id", "task_id", "question_id"]).reset_index(drop=True)
    return apply_row_limit(
        out,
        max_rows=cfg_get(dataset_cfg, max_samples_key, None),
        sample=bool(cfg_get(dataset_cfg, sample_key, False)),
        seed=int(cfg_get(dataset_cfg, "sample_seed", 42)),
    )


def assert_vqa_rows_are_trainable(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RuntimeError("Selected VQA training frame is empty.")
    for row_index, row in frame.iterrows():
        question_id = row.get("question_id", row_index)
        if not clean_text(row.get("question", "")):
            raise ValueError(f"Question {question_id} has empty question text.")
        if not clean_text(row.get("answer", "")):
            raise ValueError(f"Question {question_id} has empty answer text.")
        question_type = clean_text(row.get("question_type", "")).lower()
        if question_type == "mcq":
            options = option_values(row)
            if len(options) < 2:
                raise ValueError(f"MCQ question {question_id} has fewer than two choices.")
            if clean_text(row.get("answer", "")) not in options:
                raise ValueError(f"MCQ question {question_id} answer does not exactly match a provided choice.")
        elif not is_open_ended_question_type(question_type):
            raise ValueError(f"Question {question_id} has unsupported question_type={question_type!r}.")

        for modality in MODALITIES:
            if not row_uses_modality(row, modality):
                continue
            values = _normalize_list(row.get(MODALITY_FEATURE_COLUMNS[modality], ""))
            if not values:
                raise FileNotFoundError(f"Question {question_id} uses {modality}, but {MODALITY_FEATURE_COLUMNS[modality]} is empty.")
