from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


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


def _encode_text(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    if isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    else:
        input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        raise ValueError("Tokenizer output must expose 'input_ids'.")
    if input_ids and isinstance(input_ids[0], list):
        return [int(token) for token in input_ids[0]]
    return [int(token) for token in input_ids]


def _pad_sequences(
    sequences: list[list[int]],
    *,
    pad_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(sequence) for sequence in sequences), default=0)
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        if not sequence:
            continue
        length = len(sequence)
        padded[index, :length] = torch.tensor(sequence, dtype=torch.long)
        mask[index, :length] = 1
    return padded, mask


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
class ProjectorCaptionCollator:
    tokenizer: Any
    prompt_text: str = "Generate a concise caption for this CT image."
    system_text: str = "You are a medical image analysis assistant."
    prompt_max_length: int = 96          # bumped from 64 — ChatML framing adds ~20 tokens
    target_max_length: int = 256         # bumped from 128 per Issue 4
    max_feature_tokens: int | None = None
    prepend_bos_token: bool = True
    append_eos_token: bool = True
    use_chatml: bool = True

    def _truncate_visual_features(self, visual_features: torch.Tensor) -> torch.Tensor:
        if self.max_feature_tokens is None or visual_features.size(0) <= self.max_feature_tokens:
            return visual_features
        return visual_features[: self.max_feature_tokens]
    
    def __post_init__(self) -> None:
        if self.use_chatml:
            # Build the ChatML-wrapped prompt once at init time.
            # Visual tokens will be prepended to this during _build_inputs.
            # The assistant turn is left open — the target caption completes it.
            self._formatted_prompt = (
                f"<|im_start|>system\n{self.system_text}<|im_end|>\n"
                f"<|im_start|>user\n{self.prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            self._formatted_prompt = self.prompt_text

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("ProjectorCaptionCollator received an empty batch.")

        sample_ids = [str(feature.get("sample_id", "")) for feature in features]
        captions = [str(feature.get("caption_text", "")) for feature in features]
        image_paths = [str(feature.get("image_path", "")) for feature in features]
        feature_paths = [str(feature.get("feature_path", "")) for feature in features]

        visual_sequences = [
            self._truncate_visual_features(feature["visual_features"].to(dtype=torch.float32))
            for feature in features
        ]
        feature_dim = int(visual_sequences[0].shape[-1])
        max_visual_tokens = max(sequence.size(0) for sequence in visual_sequences)
        visual_batch = torch.zeros((len(features), max_visual_tokens, feature_dim), dtype=torch.float32)
        visual_mask = torch.zeros((len(features), max_visual_tokens), dtype=torch.long)
        for index, sequence in enumerate(visual_sequences):
            length = sequence.size(0)
            visual_batch[index, :length] = sequence
            visual_mask[index, :length] = 1

        prompt_ids = _encode_text(
            self.tokenizer,
            self._formatted_prompt,
            max_length=self.prompt_max_length,
        )
        if self.prepend_bos_token and getattr(self.tokenizer, "bos_token_id", None) is not None:
            prompt_ids = [int(self.tokenizer.bos_token_id)] + prompt_ids
        prompt_batch, prompt_mask = _pad_sequences(
            [list(prompt_ids) for _ in features],
            pad_value=int(getattr(self.tokenizer, "pad_token_id", 0) or 0),
        )

        '''
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        target_sequences: list[list[int]] = []
        for caption in captions:
            max_caption_length = self.target_max_length
            if self.append_eos_token and eos_token_id is not None:
                max_caption_length = max(1, self.target_max_length - 1)
            target_ids = _encode_text(
                self.tokenizer,
                caption,
                max_length=max_caption_length,
            )
            if self.append_eos_token and eos_token_id is not None:
                target_ids = target_ids + [int(eos_token_id)]
            target_sequences.append(target_ids)
        '''
        
        # Resolve end-of-turn suffix: <|im_end|> for ChatML, generic EOS otherwise.
        if self.use_chatml:
            im_end_token_id = getattr(self.tokenizer, "im_end_id", None)
            if im_end_token_id is None:
                added = getattr(self.tokenizer, "added_tokens_encoder", {})
                im_end_token_id = added.get("<|im_end|>", None)
            if im_end_token_id is not None:
                eos_suffix = [int(im_end_token_id)]
            else:
                eos_suffix = _encode_text(self.tokenizer, "<|im_end|>", max_length=4)
                if not eos_suffix:
                    eos_suffix = [int(getattr(self.tokenizer, "eos_token_id", 0))]
        elif self.append_eos_token and getattr(self.tokenizer, "eos_token_id", None) is not None:
            eos_suffix = [int(self.tokenizer.eos_token_id)]
        else:
            eos_suffix = []

        target_sequences: list[list[int]] = []
        for caption in captions:
            reserved = len(eos_suffix)
            max_caption_length = max(1, self.target_max_length - reserved)
            target_ids = _encode_text(
                self.tokenizer,
                caption,
                max_length=max_caption_length,
            )
            target_ids = target_ids + eos_suffix
            target_sequences.append(target_ids)

        target_pad_value = int(
            getattr(self.tokenizer, "pad_token_id", None)
            or getattr(self.tokenizer, "eos_token_id", 0)
            or 0
        )
        target_batch, target_mask = _pad_sequences(
            target_sequences,
            pad_value=target_pad_value,
        )

        return {
            "sample_ids": sample_ids,
            "captions": captions,
            "image_paths": image_paths,
            "feature_paths": feature_paths,
            "visual_features": visual_batch,
            "visual_attention_mask": visual_mask,
            "prompt_input_ids": prompt_batch,
            "prompt_attention_mask": prompt_mask,
            "target_input_ids": target_batch,
            "target_attention_mask": target_mask,
        }
