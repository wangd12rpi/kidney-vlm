from __future__ import annotations

import re
from typing import Any, Sequence

import torch

from kidney_vlm.vqa.modeling import OncoVLMVQASFTModel
from kidney_vlm.vqa.stage_config import as_bool, cfg_get, clean_text


def normalize_answer_text(text: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip().lower()


def extract_tag(text: str, tag: str) -> str:
    matches = re.findall(
        rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL
    )
    return matches[0].strip() if len(matches) == 1 else ""


def _tag_matches(text: str, tag: str) -> list[re.Match[str]]:
    return list(
        re.finditer(
            rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL
        )
    )


def _thinking_span(text: str) -> tuple[str, int, str]:
    think_matches = _tag_matches(text, "think")
    if len(think_matches) == 1:
        match = think_matches[0]
        return match.group(1).strip(), match.end(), "explicit"
    close_matches = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))
    if len(think_matches) == 0 and len(close_matches) == 1:
        close_match = close_matches[0]
        return text[: close_match.start()].strip(), close_match.end(), "implicit_open"
    return "", -1, ""


def _reasoning_span(text: str) -> tuple[str, int, str]:
    return _thinking_span(text)


def has_clean_cot_format(text: str) -> bool:
    answer_matches = _tag_matches(text, "answer")
    if len(answer_matches) != 1 or not answer_matches[0].group(1).strip():
        return False
    reasoning_text, reasoning_end, _ = _reasoning_span(text)
    if (
        not reasoning_text
        or reasoning_end < 0
        or reasoning_end > answer_matches[0].start()
    ):
        return False
    if text[reasoning_end : answer_matches[0].start()].strip():
        return False
    think_matches = _tag_matches(text, "think")
    close_think_matches = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))
    if len(close_think_matches) != 1:
        return False
    if len(think_matches) == 1 and text[: think_matches[0].start()].strip():
        return False
    return not text[answer_matches[0].end() :].strip()


def parse_grpo_answer(completion: str) -> str:
    return extract_tag(completion, "answer")


STEP_1_HEADING = "Step 1 — Observation:"
STEP_2_HEADING = "Step 2 — Reasoning:"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def split_two_step_reasoning(reasoning_text: str) -> tuple[str, str]:
    if (
        reasoning_text.count(STEP_1_HEADING) != 1
        or reasoning_text.count(STEP_2_HEADING) != 1
    ):
        return "", ""
    if not reasoning_text.startswith(STEP_1_HEADING):
        return "", ""
    observation_start = len(STEP_1_HEADING)
    reasoning_start = reasoning_text.find(STEP_2_HEADING, observation_start)
    if reasoning_start < observation_start:
        return "", ""
    observation = reasoning_text[observation_start:reasoning_start].strip()
    reasoning = reasoning_text[reasoning_start + len(STEP_2_HEADING) :].strip()
    return observation, reasoning


def observation_is_choice_blind(observation: str) -> bool:
    answer_directed = re.compile(
        r"\b(?:choice|option|answer|distractor|alternative|favou?rs?|supports?|matches?|"
        r"fits?|diagnos(?:is|tic)|conclusion|suggests?|argues?\s+against|"
        r"rules?\s+out)\b|\b(?:consistent|compatible)\s+with\b|"
        r"\bindicative\s+of\b|\brather\s+than\b",
        flags=re.IGNORECASE,
    )
    return bool(observation) and answer_directed.search(observation) is None


def observation_is_presence_only(observation: str) -> bool:
    absence_or_contrast = re.compile(
        r"\b(?:no|not|without|absent|absence|lacks?|lacking|neither|nor)\b|"
        r"\b(?:free|negative)\s+(?:of|for)\b|"
        r"\b(?:fails?|unable)\s+to\s+(?:show|demonstrate|identify)\b|"
        r"\b(?:rather\s+than|instead\s+of|as\s+opposed\s+to)\b",
        flags=re.IGNORECASE,
    )
    return bool(observation) and absence_or_contrast.search(observation) is None


def has_choice_walkthrough(reasoning: str) -> bool:
    lower = reasoning.lower()
    if re.search(r"\b(?:first|second|third|fourth)\s+(?:choice|option)\b", lower):
        return True
    if re.search(r"\b(?:choice|option)\s*(?:a|b|c|d|1|2|3|4)\b", lower):
        return True
    if re.search(
        r"\b(?:all|each|every|other|remaining)\s+(?:choice|choices|option|options)\b",
        lower,
    ):
        return True
    return bool(
        re.search(
            r"(?<!\w)[A-D](?:[\).:]|\s)+(?:is|are|shows?|has|would|fits?|matches?|fails?|lacks?)\b",
            reasoning,
        )
    )


def copies_full_choice(reasoning_text: str, choices: list[str]) -> bool:
    normalized_reasoning = normalize_answer_text(reasoning_text)
    for choice in choices:
        normalized_choice = normalize_answer_text(choice)
        if (
            _word_count(normalized_choice) >= 6
            and normalized_choice in normalized_reasoning
        ):
            return True
    return False


def score_grpo_completion(
    *,
    completion: str,
    answer: str,
    choices: list[str],
    reward_cfg: Any,
) -> dict[str, float | str]:
    weights = cfg_get(reward_cfg, "reward_weights", {})
    parsed_answer = parse_grpo_answer(completion)
    cleaned_answer = clean_text(parsed_answer)
    cleaned_gold = clean_text(answer)
    cleaned_choices = {clean_text(choice) for choice in choices}

    reasoning_text, _, reasoning_format = _reasoning_span(completion)
    observation_text, step_2_text = split_two_step_reasoning(reasoning_text)
    observation_words = _word_count(observation_text)
    step_2_words = _word_count(step_2_text)
    think_words = _word_count(reasoning_text)
    min_observation_words = int(cfg_get(reward_cfg, "observation_min_words", 45) or 0)
    max_observation_words = int(cfg_get(reward_cfg, "observation_max_words", 95) or 0)
    min_reasoning_words = int(cfg_get(reward_cfg, "reasoning_min_words", 25) or 0)
    max_reasoning_words = int(cfg_get(reward_cfg, "reasoning_max_words", 90) or 0)
    min_think_words = int(cfg_get(reward_cfg, "think_min_words", 100) or 0)
    max_think_words = int(cfg_get(reward_cfg, "think_max_words", 160) or 0)
    has_two_step = bool(observation_text and step_2_text)
    observation_presence_only = observation_is_presence_only(observation_text)
    observation_focus = (
        has_two_step
        and observation_words >= min_observation_words
        and (max_observation_words <= 0 or observation_words <= max_observation_words)
        and observation_is_choice_blind(observation_text)
        and observation_presence_only
    )
    walkthrough = has_choice_walkthrough(step_2_text)
    choice_copy = copies_full_choice(reasoning_text, choices)
    reasoning_focus = (
        has_two_step
        and step_2_words >= min_reasoning_words
        and (max_reasoning_words <= 0 or step_2_words <= max_reasoning_words)
        and think_words >= min_think_words
        and (max_think_words <= 0 or think_words <= max_think_words)
        and not walkthrough
        and not choice_copy
    )

    is_correct = cleaned_answer == cleaned_gold and cleaned_answer != ""
    is_valid_choice = cleaned_answer in cleaned_choices and cleaned_answer != ""
    is_letter_only = cleaned_answer.lower() in {"a", "b", "c", "d"}
    format_reward_allowed = (
        not as_bool(cfg_get(reward_cfg, "format_reward_requires_correct", False))
        or is_correct
    )
    observation_reward_allowed = (
        not as_bool(cfg_get(reward_cfg, "observation_reward_requires_correct", False))
        or is_correct
    )
    reasoning_reward_allowed = (
        not as_bool(cfg_get(reward_cfg, "reasoning_reward_requires_correct", False))
        or is_correct
    )
    reward = 0.0
    reward += float(cfg_get(weights, "correctness", 1.0)) if is_correct else 0.0
    reward += float(cfg_get(weights, "valid_choice", 0.1)) if is_valid_choice else 0.0
    reward += (
        float(cfg_get(weights, "format", 0.1))
        if has_clean_cot_format(completion) and format_reward_allowed
        else 0.0
    )
    reward += (
        float(cfg_get(weights, "observation", 0.0))
        if observation_focus and observation_reward_allowed
        else 0.0
    )
    reward += (
        float(cfg_get(weights, "reasoning", 0.1))
        if reasoning_focus and reasoning_reward_allowed
        else 0.0
    )
    if is_letter_only:
        reward -= float(cfg_get(weights, "valid_choice", 0.1))

    return {
        "reward": float(reward),
        "correct": float(is_correct),
        "valid_choice": float(is_valid_choice),
        "format": float(has_clean_cot_format(completion)),
        "two_step": float(has_two_step),
        "observation": float(observation_focus),
        "observation_presence_only": float(observation_presence_only),
        "reasoning": float(reasoning_focus),
        "walkthrough": float(walkthrough),
        "choice_copy": float(choice_copy),
        "parsed_answer": parsed_answer,
        "observation_text": observation_text,
        "reasoning_text": step_2_text,
        "think_words": float(think_words),
        "observation_words": float(observation_words),
        "reasoning_words": float(step_2_words),
        "think_format": reasoning_format,
    }


def repeat_batch_for_generations(
    batch: dict[str, Any], num_generations: int
) -> dict[str, Any]:
    batch_size = int(batch["input_ids"].shape[0])
    repeated: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            repeated[key] = value.repeat_interleave(num_generations, dim=0)
        elif isinstance(value, list) and len(value) == batch_size:
            repeated[key] = [item for item in value for _ in range(num_generations)]
        else:
            repeated[key] = value
    return repeated


def append_completions_to_prompts(
    *,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_attention_mask: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows_input_ids: list[torch.Tensor] = []
    rows_attention: list[torch.Tensor] = []
    rows_labels: list[torch.Tensor] = []
    for row_idx in range(int(prompt_input_ids.shape[0])):
        prompt_len = int(prompt_attention_mask[row_idx].sum().item())
        completion_len = int(completion_attention_mask[row_idx].sum().item())
        prompt_tokens = prompt_input_ids[row_idx, :prompt_len]
        completion_tokens = completion_ids[row_idx, :completion_len]
        row_input_ids = torch.cat([prompt_tokens, completion_tokens], dim=0)
        row_attention = torch.ones_like(
            row_input_ids, dtype=prompt_attention_mask.dtype
        )
        row_labels = torch.cat(
            [
                torch.full(
                    (prompt_len,),
                    -100,
                    device=prompt_input_ids.device,
                    dtype=torch.long,
                ),
                completion_tokens.to(dtype=torch.long),
            ],
            dim=0,
        )
        rows_input_ids.append(row_input_ids)
        rows_attention.append(row_attention)
        rows_labels.append(row_labels)

    max_len = max(int(row.shape[0]) for row in rows_input_ids)
    full_input_ids = torch.full(
        (len(rows_input_ids), max_len),
        int(pad_token_id),
        device=prompt_input_ids.device,
        dtype=torch.long,
    )
    full_attention = torch.zeros(
        (len(rows_input_ids), max_len),
        device=prompt_input_ids.device,
        dtype=prompt_attention_mask.dtype,
    )
    full_labels = torch.full(
        (len(rows_input_ids), max_len),
        -100,
        device=prompt_input_ids.device,
        dtype=torch.long,
    )
    for row_idx, (row_input_ids, row_attention, row_labels) in enumerate(
        zip(rows_input_ids, rows_attention, rows_labels, strict=True)
    ):
        row_len = int(row_input_ids.shape[0])
        full_input_ids[row_idx, :row_len] = row_input_ids
        full_attention[row_idx, :row_len] = row_attention
        full_labels[row_idx, :row_len] = row_labels
    return full_input_ids, full_attention, full_labels


def _compact_masked_values(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [values[row_idx][mask[row_idx]] for row_idx in range(int(values.shape[0]))]
    max_len = max((int(row.shape[0]) for row in rows), default=0)
    compact = torch.zeros(
        (len(rows), max_len), device=values.device, dtype=values.dtype
    )
    compact_mask = torch.zeros(
        (len(rows), max_len), device=values.device, dtype=torch.bool
    )
    for row_idx, row in enumerate(rows):
        row_len = int(row.shape[0])
        if row_len:
            compact[row_idx, :row_len] = row
            compact_mask[row_idx, :row_len] = True
    return compact, compact_mask


def completion_logprobs(
    *,
    model: OncoVLMVQASFTModel,
    batch: dict[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = model.prepare_interleaved_forward_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pathology_features=batch.get("pathology_features"),
        pathology_feature_mask=batch.get("pathology_feature_mask"),
        radiology_features=batch.get("radiology_features"),
        radiology_feature_mask=batch.get("radiology_feature_mask"),
        dnam_features=batch.get("dnam_features"),
        dnam_feature_mask=batch.get("dnam_feature_mask"),
        rna_features=batch.get("rna_features"),
        rna_feature_mask=batch.get("rna_feature_mask"),
        pathology_prefix_embeddings=batch.get("pathology_prefix_embeddings"),
        pathology_prefix_mask=batch.get("pathology_prefix_mask"),
        radiology_prefix_embeddings=batch.get("radiology_prefix_embeddings"),
        radiology_prefix_mask=batch.get("radiology_prefix_mask"),
        dnam_prefix_embeddings=batch.get("dnam_prefix_embeddings"),
        dnam_prefix_mask=batch.get("dnam_prefix_mask"),
        rna_prefix_embeddings=batch.get("rna_prefix_embeddings"),
        rna_prefix_mask=batch.get("rna_prefix_mask"),
        prefix_spans=batch["prefix_spans"],
    )
    labels_after_prefix = prepared["labels"]
    completion_count = int(labels_after_prefix.ne(-100).sum(dim=1).max().item())
    forward_prepared = dict(prepared)
    forward_prepared["labels"] = None
    outputs = model.forward_prepared_interleaved_inputs(
        forward_prepared,
        logits_to_keep=completion_count + 1,
    )
    logits = outputs.logits
    labels_for_logits = labels_after_prefix[:, -int(logits.shape[1]) :]
    shifted_labels = labels_for_logits[:, 1:]
    label_mask = shifted_labels.ne(-100)
    gathered_labels = shifted_labels.clamp_min(0)
    token_logprobs = (
        torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        .gather(
            dim=-1,
            index=gathered_labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    return _compact_masked_values(token_logprobs, label_mask)


def grpo_advantages(rewards: torch.Tensor, *, num_generations: int) -> torch.Tensor:
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped - mean) / (std + 1e-6)
    return advantages.reshape(-1)


def centered_group_rewards(
    rewards: torch.Tensor, *, num_generations: int
) -> torch.Tensor:
    """Center auxiliary rewards without erasing their configured scale."""
    grouped = rewards.view(-1, num_generations)
    return (grouped - grouped.mean(dim=1, keepdim=True)).reshape(-1)


def completion_span_mask(
    *,
    completion_ids: torch.Tensor,
    completion_attention_mask: torch.Tensor,
    start_ids: list[int],
    end_ids: list[int],
) -> torch.Tensor:
    """Mask generated tokens strictly between two required marker sequences."""
    if not start_ids or not end_ids:
        raise ValueError("start_ids and end_ids must each contain at least one token.")
    span_mask = torch.zeros_like(completion_attention_mask, dtype=torch.bool)
    for row_index in range(int(completion_ids.shape[0])):
        completion_len = int(completion_attention_mask[row_index].sum().item())
        row_ids = completion_ids[row_index, :completion_len].tolist()
        start = next(
            (
                index
                for index in range(completion_len - len(start_ids) + 1)
                if row_ids[index : index + len(start_ids)] == start_ids
            ),
            -1,
        )
        if start < 0:
            continue
        content_start = start + len(start_ids)
        end = next(
            (
                index
                for index in range(content_start, completion_len - len(end_ids) + 1)
                if row_ids[index : index + len(end_ids)] == end_ids
            ),
            -1,
        )
        if end > content_start:
            span_mask[row_index, content_start:end] = True
    return span_mask


def grpo_loss(
    *,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    auxiliary_terms: Sequence[tuple[torch.Tensor, torch.Tensor]] = (),
) -> torch.Tensor:
    ratio = torch.exp(current_logprobs - old_logprobs)
    unclipped = ratio * advantages.unsqueeze(1)
    clipped = torch.clamp(
        ratio, 1.0 - clip_range, 1.0 + clip_range
    ) * advantages.unsqueeze(1)
    per_token_loss = -torch.minimum(unclipped, clipped)
    completion_token_counts = token_mask.sum(dim=1)
    completion_losses = (per_token_loss * token_mask).sum(
        dim=1
    ) / completion_token_counts.clamp_min(1)
    for auxiliary_advantages, auxiliary_token_mask in auxiliary_terms:
        auxiliary_mask = token_mask & auxiliary_token_mask
        auxiliary_unclipped = ratio * auxiliary_advantages.unsqueeze(1)
        auxiliary_clipped = torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        ) * auxiliary_advantages.unsqueeze(1)
        auxiliary_token_loss = -torch.minimum(auxiliary_unclipped, auxiliary_clipped)
        auxiliary_token_counts = auxiliary_mask.sum(dim=1)
        completion_losses = completion_losses + (
            (auxiliary_token_loss * auxiliary_mask).sum(dim=1)
            / auxiliary_token_counts.clamp_min(1)
        )
    valid_completions = completion_token_counts > 0
    return (
        completion_losses[valid_completions].mean()
        if valid_completions.any()
        else completion_losses.sum() * 0.0
    )
