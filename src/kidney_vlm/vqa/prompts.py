from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kidney_vlm.vqa.constants import MODALITIES, MODALITY_FLAG_COLUMNS, OPTION_COLUMNS
from kidney_vlm.vqa.stage_config import as_bool, cfg_get, clean_text

PREFIX_PLACEHOLDERS = {
    "pathology": "<|oncovlm_pathology_prefix|>",
    "radiology": "<|oncovlm_radiology_prefix|>",
    "dnam": "<|oncovlm_dnam_prefix|>",
    "rna": "<|oncovlm_rna_prefix|>",
}
PREFIX_TAGS = {
    "pathology": "pathology_features",
    "radiology": "radiology_features",
    "dnam": "dnam_features",
    "rna": "rna_features",
}


def option_values(row: Mapping[str, Any]) -> list[str]:
    return [clean_text(row.get(column, "")) for column in OPTION_COLUMNS if clean_text(row.get(column, ""))]


def row_uses_modality(row: Mapping[str, Any], modality: str) -> bool:
    return as_bool(row.get(MODALITY_FLAG_COLUMNS[modality], False))


def is_open_ended_question_type(question_type: str) -> bool:
    normalized = question_type.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"qa", "open", "open_ended", "openended", "free_text", "short_answer"}


def row_modalities(row: Mapping[str, Any]) -> list[str]:
    return [modality for modality in MODALITIES if row_uses_modality(row, modality)]


def prefix_placeholder_for_modality(modality: str) -> str:
    if modality not in PREFIX_PLACEHOLDERS:
        raise ValueError(f"Unsupported VQA modality for prefix placeholder: {modality}")
    return PREFIX_PLACEHOLDERS[modality]


def _modality_evidence_blocks(row: Mapping[str, Any], *, preview: bool = False) -> list[str]:
    blocks: list[str] = []
    for modality in row_modalities(row):
        tag = PREFIX_TAGS[modality]
        prefix_text = f"[PREFIX:{modality} soft tokens]" if preview else prefix_placeholder_for_modality(modality)
        blocks.append(f"<{tag}>\n{prefix_text}\n</{tag}>")
        if modality == "radiology":
            radiology_biomarker = clean_text(row.get("radiology_biomarker", ""))
            if radiology_biomarker:
                blocks.append(f"<radiology_biomarker>\n{radiology_biomarker}\n</radiology_biomarker>")
    if not blocks:
        raise ValueError(f"Question {row.get('question_id', '<unknown>')} has no enabled modalities.")
    return blocks


def build_vqa_prompt(row: Mapping[str, Any], prompt_cfg: Any) -> str:
    question = clean_text(row.get("question", ""))
    if not question:
        raise ValueError(f"Question {row.get('question_id', '<unknown>')} has empty question text.")

    system_prompt = clean_text(cfg_get(prompt_cfg, "system_prompt", ""))
    if not system_prompt:
        raise ValueError("vqa_train.prompt.system_prompt must be populated.")

    question_type = clean_text(row.get("question_type", "")).lower()
    options = option_values(row)
    if question_type == "mcq":
        if len(options) < 2:
            raise ValueError(f"MCQ question {row.get('question_id', '<unknown>')} has fewer than two choices.")
        response_instruction = clean_text(cfg_get(prompt_cfg, "mcq_response_instruction", ""))
        if not response_instruction:
            raise ValueError("vqa_train.prompt.mcq_response_instruction must be populated.")
        choice_text = "\n".join(f"- {option}" for option in options)
        return (
            f"{system_prompt}\n\n"
            f"{response_instruction}\n\n"
            "<modality_evidence>\n"
            f"{chr(10).join(_modality_evidence_blocks(row))}\n"
            "</modality_evidence>\n\n"
            "<question>\n"
            f"{question}\n"
            "</question>\n\n"
            "<choices>\n"
            f"{choice_text}\n"
            "</choices>"
        )

    if not is_open_ended_question_type(question_type):
        raise ValueError(f"Unsupported VQA question_type for training: {question_type!r}")
    response_instruction = clean_text(cfg_get(prompt_cfg, "open_response_instruction", ""))
    if not response_instruction:
        raise ValueError("vqa_train.prompt.open_response_instruction must be populated.")
    return (
        f"{system_prompt}\n\n"
        f"{response_instruction}\n\n"
        "<modality_evidence>\n"
        f"{chr(10).join(_modality_evidence_blocks(row))}\n"
        "</modality_evidence>\n\n"
        "<question>\n"
        f"{question}\n"
        "</question>"
    )


def build_vqa_prompt_preview(row: Mapping[str, Any], prompt_cfg: Any) -> str:
    prompt = build_vqa_prompt(row, prompt_cfg)
    for modality in row_modalities(row):
        prompt = prompt.replace(prefix_placeholder_for_modality(modality), f"[PREFIX:{modality} soft tokens]")
    return prompt
