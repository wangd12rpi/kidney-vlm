from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

OPTION_COLUMNS = ["option_a", "option_b", "option_c", "option_d"]
ANSWER_LABELS = ("A", "B", "C", "D")
MODALITY_COMBO_KEYS = {
    "path": "use_pathology",
    "rad": "use_radiology",
    "dnam": "use_dnam",
    "rna": "use_rna",
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def option_values(row: Mapping[str, Any]) -> list[str]:
    return [
        str(row.get(column, "")).strip()
        for column in OPTION_COLUMNS
        if str(row.get(column, "")).strip()
    ]


def answer_label_for_text(row: Mapping[str, Any], answer_text: str) -> str:
    answer = str(answer_text).strip()
    if not answer:
        return ""
    for label, column in zip(ANSWER_LABELS, OPTION_COLUMNS, strict=True):
        if str(row.get(column, "")).strip().casefold() == answer.casefold():
            return label
    return ""


def question_type_key(question_type: Any) -> str:
    normalized = str(question_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "mcq":
        return "mcq"
    if normalized in {"qa", "open", "open_ended", "openended", "free_text", "short_answer"}:
        return "qa"
    raise ValueError(f"Unsupported VQA question_type for evaluation: {question_type!r}")


def _required_text(row: Mapping[str, Any], column: str, modality_name: str) -> str:
    value = str(row.get(column, "")).strip()
    if not value:
        question_id = row.get("question_id", "<unknown>")
        raise ValueError(
            f"Question {question_id} requires {modality_name}, but '{column}' is empty."
        )
    return value


def _filter_block(filters: Mapping[str, Any], name: str) -> dict[str, Any]:
    raw_block = filters.get(name)
    if raw_block is None:
        return {"enabled": False}
    if not isinstance(raw_block, Mapping):
        raise TypeError(f"Filter '{name}' must be a mapping with an enabled switch.")
    return dict(raw_block)


def _enabled_filter(filters: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    block = _filter_block(filters, name)
    return block if bool(block.get("enabled", False)) else None


def _required_filter_values(block: Mapping[str, Any], name: str) -> list[str]:
    values = _as_list(block.get("values"))
    if not values:
        raise ValueError(f"Filter '{name}' is enabled but has no values.")
    return values


def _required_filter_sequence(block: Mapping[str, Any], name: str) -> list[Any]:
    values = block.get("values")
    if values is None:
        raise ValueError(f"Filter '{name}' is enabled but has no values.")
    if hasattr(values, "tolist") and not isinstance(values, (str, bytes)):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or isinstance(values, Mapping):
        raise TypeError(f"Filter '{name}' values must be a list.")
    if not isinstance(values, Sequence):
        raise TypeError(f"Filter '{name}' values must be a list.")
    items = list(values)
    if not items:
        raise ValueError(f"Filter '{name}' is enabled but has no values.")
    return items


def _as_bool(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{context} must be true or false.")


def _artifact_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return len(converted) > 0
    text = str(value).strip()
    return bool(text) and text.lower() not in {"none", "nan", "null", "[]"}


def _radiology_available_case_ids(frame: pd.DataFrame, sampling_cfg: Mapping[str, Any]) -> set[str]:
    split_name = str(sampling_cfg.get("protect_split", "test")).strip() or "test"
    required_columns = {"case_id", "split", "use_radiology", "radiology_feature_paths", "radiology_view_png_dir"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "sampling.protect_radiology_available_cases requires VQA columns: "
            f"{missing_columns}"
        )

    split_frame = frame[frame["split"].astype(str).eq(split_name)]
    if split_frame.empty:
        return set()
    radiology_mask = (
        split_frame["use_radiology"].astype(bool)
        | split_frame["radiology_feature_paths"].map(_artifact_populated)
        | split_frame["radiology_view_png_dir"].map(_artifact_populated)
    )
    return set(split_frame.loc[radiology_mask, "case_id"].astype(str))


def _allowed_modality_combo_mask(
    frame: pd.DataFrame, block: Mapping[str, Any]
) -> pd.Series:
    raw_combos = _required_filter_sequence(block, "allowed_modality_combo")
    combo_masks: list[pd.Series] = []
    expected_keys = set(MODALITY_COMBO_KEYS)

    for index, raw_combo in enumerate(raw_combos, start=1):
        if not isinstance(raw_combo, Mapping):
            raise TypeError(
                "Each allowed_modality_combo value must be a mapping with keys: "
                "path, rad, dnam, rna."
            )
        combo = dict(raw_combo)
        combo_keys = set(combo)
        missing_keys = expected_keys - combo_keys
        extra_keys = combo_keys - expected_keys
        if missing_keys or extra_keys:
            raise ValueError(
                f"allowed_modality_combo item {index} must use exactly "
                "path, rad, dnam, rna. "
                f"Missing: {sorted(missing_keys)}. Extra: {sorted(extra_keys)}."
            )

        combo_mask = pd.Series(True, index=frame.index)
        for short_key, column in MODALITY_COMBO_KEYS.items():
            combo_mask &= (
                frame[column]
                .astype(bool)
                .eq(
                    _as_bool(
                        combo[short_key],
                        context=f"allowed_modality_combo item {index}.{short_key}",
                    )
                )
            )
        combo_masks.append(combo_mask)

    mask = pd.Series(False, index=frame.index)
    for combo_mask in combo_masks:
        mask |= combo_mask
    return mask


def select_eval_rows(vqa_df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = vqa_df.copy()
    filters = dict(cfg.get("filters") or {})

    split_filter = _enabled_filter(filters, "split")
    if split_filter:
        split = str(split_filter.get("value", "")).strip()
        if not split:
            raise ValueError("Filter 'split' is enabled but has no value.")
        out = out[out["split"].astype(str).eq(split)]

    question_type_filter = _enabled_filter(filters, "question_types")
    if question_type_filter:
        question_types = _required_filter_values(question_type_filter, "question_types")
        out = out[out["question_type"].astype(str).isin(question_types)]

    generation_type_filter = _enabled_filter(filters, "generation_types")
    if generation_type_filter:
        generation_types = _required_filter_values(generation_type_filter, "generation_types")
        out = out[out["generation_type"].astype(str).isin(generation_types)]

    modality_name_filter = _enabled_filter(filters, "modality_combination_names")
    if modality_name_filter:
        modality_names = _required_filter_values(
            modality_name_filter, "modality_combination_names"
        )
        out = out[out["modality_combination_name"].astype(str).isin(modality_names)]

    project_filter = _enabled_filter(filters, "project_ids")
    if project_filter:
        project_ids = _required_filter_values(project_filter, "project_ids")
        out = out[out["project_id"].astype(str).isin(project_ids)]

    task_filter = _enabled_filter(filters, "task_ids")
    if task_filter:
        task_ids = _required_filter_values(task_filter, "task_ids")
        out = out[out["task_id"].astype(str).isin(task_ids)]

    task_category_filter = _enabled_filter(filters, "task_categories")
    if task_category_filter:
        task_categories = _required_filter_values(
            task_category_filter, "task_categories"
        )
        out = out[out["task_category"].astype(str).isin(task_categories)]

    modality_combo_filter = _enabled_filter(filters, "allowed_modality_combo")
    if modality_combo_filter:
        out = out[_allowed_modality_combo_mask(out, modality_combo_filter)]

    out = out.sort_values(
        ["project_id", "case_id", "task_id", "question_id"]
    ).reset_index(drop=True)

    row_limit_filter = _enabled_filter(filters, "row_limit")
    if row_limit_filter:
        max_rows = row_limit_filter.get("max_rows")
        if max_rows is None or not str(max_rows).strip():
            raise ValueError("Filter 'row_limit' is enabled but has no max_rows.")
        max_rows_int = int(max_rows)
        sample = bool(row_limit_filter.get("sample", False))
        if sample and len(out) > max_rows_int:
            out = out.sample(
                n=max_rows_int,
                random_state=int(row_limit_filter.get("sample_seed", 17)),
            ).sort_values(["project_id", "case_id", "task_id", "question_id"])
        else:
            out = out.head(max_rows_int)
        out = out.reset_index(drop=True)

    return out


def apply_group_sampling(frame: pd.DataFrame, sampling_cfg: Mapping[str, Any] | None) -> pd.DataFrame:
    cfg = dict(sampling_cfg or {})
    if frame.empty or not bool(cfg.get("enabled", False)):
        return frame.reset_index(drop=True)

    ratio = float(cfg.get("ratio", 1.0))
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"sampling.ratio must be in [0, 1], got {ratio}")
    min_per_group = int(cfg.get("min_per_group", 0) or 0)
    if min_per_group < 0:
        raise ValueError(f"sampling.min_per_group must be non-negative, got {min_per_group}")
    seed = int(cfg.get("seed", 42))
    group_by = _as_list(
        cfg.get(
            "group_by",
            [
                "question_type",
                "generation_type",
                "task_category",
                "task_id",
                "modality_combination_name",
            ],
        )
    )
    missing_columns = [column for column in group_by if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"sampling.group_by columns are missing from VQA frame: {missing_columns}")

    protected_case_ids: set[str] = set()
    if bool(cfg.get("protect_radiology_available_cases", False)):
        protected_case_ids = _radiology_available_case_ids(frame, cfg)

    sampled_groups: list[pd.DataFrame] = []
    grouped = frame.groupby(group_by, sort=False, dropna=False)
    for group_index, (_, group) in enumerate(grouped):
        group = group.reset_index(drop=True)
        target = max(min_per_group, int(math.ceil(len(group) * ratio)))
        target = min(len(group), target)
        protected = (
            group[group["case_id"].astype(str).isin(protected_case_ids)].reset_index(drop=True)
            if protected_case_ids
            else group.head(0)
        )
        candidates = (
            group[~group["case_id"].astype(str).isin(protected_case_ids)].reset_index(drop=True)
            if protected_case_ids
            else group
        )
        candidate_target = max(0, target - len(protected))
        if candidate_target >= len(candidates):
            sampled_groups.append(group)
            continue
        sampled = candidates.sample(n=candidate_target, random_state=seed + group_index)
        sampled_groups.append(pd.concat([protected, sampled], ignore_index=True))
    if not sampled_groups:
        return frame.head(0).reset_index(drop=True)
    return pd.concat(sampled_groups, ignore_index=True).sort_values(
        ["project_id", "case_id", "task_id", "question_id"]
    ).reset_index(drop=True)


def enabled_model_configs(models_cfg: Mapping[str, Any] | Sequence[Any]) -> list[dict[str, Any]]:
    if isinstance(models_cfg, Mapping):
        raw_items = list(models_cfg.items())
    elif isinstance(models_cfg, Sequence) and not isinstance(models_cfg, (str, bytes)):
        raw_items = [
            (
                str(raw_model_cfg.get("display_name", f"model_{index}"))
                if isinstance(raw_model_cfg, Mapping)
                else f"model_{index}",
                raw_model_cfg,
            )
            for index, raw_model_cfg in enumerate(models_cfg)
        ]
    else:
        raise TypeError("vqa_evaluation.models must be a mapping or list of model configs.")
    enabled: list[dict[str, Any]] = []
    for model_key, raw_model_cfg in raw_items:
        if not isinstance(raw_model_cfg, Mapping):
            raise TypeError(f"vqa_evaluation.models.{model_key} must be a mapping.")
        model_cfg = dict(raw_model_cfg)
        if not bool(model_cfg.get("enabled", False)):
            continue
        model_cfg["model_key"] = str(model_key)
        model_cfg["display_name"] = str(model_cfg.get("display_name") or model_key).strip()
        if not model_cfg["display_name"]:
            raise ValueError(f"vqa_evaluation.models.{model_key}.display_name is empty.")
        backend = str(model_cfg.get("backend", "")).strip()
        if not backend:
            raise ValueError(f"vqa_evaluation.models.{model_key}.backend is required.")
        model_cfg["backend"] = backend
        enabled.append(model_cfg)
    if not enabled:
        raise RuntimeError("No enabled VQA evaluation models found.")
    return enabled


def prompt_cfg_for_model(cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    prompts = cfg.get("prompts") or {}
    if not isinstance(prompts, Mapping):
        raise TypeError("vqa_evaluation.prompts must be a mapping.")
    profile_name = str(model_cfg.get("prompt_profile", "baseline")).strip() or "baseline"
    profile = prompts.get(profile_name)
    if not isinstance(profile, Mapping):
        raise ValueError(f"vqa_evaluation.prompts.{profile_name} must be defined.")
    return {
        "prompts": dict(profile),
        "image_inputs": cfg.get("image_inputs", {}),
        "cot": dict(model_cfg.get("cot") or {}),
    }


def _prompt_block_for_row(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    prompt_key = question_type_key(row.get("question_type", ""))
    prompts = cfg.get("prompts") or {}
    if not isinstance(prompts, Mapping):
        raise TypeError("vqa_evaluation.prompts must be a mapping.")
    block = prompts.get(prompt_key)
    if not isinstance(block, Mapping):
        raise ValueError(f"vqa_evaluation.prompts.{prompt_key} must be defined.")
    return dict(block)


def _ignore_radiology_biomarker(cfg: Mapping[str, Any]) -> bool:
    image_cfg = cfg.get("image_inputs") or {}
    if not isinstance(image_cfg, Mapping):
        raise TypeError("vqa_evaluation.image_inputs must be a mapping.")
    return bool(image_cfg.get("ignore_radiology_biomarker", False))


def _modality_evidence_text(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    modality_blocks: list[str] = []

    if bool(row.get("use_pathology")):
        modality_blocks.append("<pathology_images>attached</pathology_images>")
    if bool(row.get("use_dnam")):
        modality_blocks.append(
            "<dna_methylation_text_summary>\n"
            f"{_required_text(row, 'dnam_text_summary', 'DNAm text summary')}\n"
            "</dna_methylation_text_summary>"
        )
    if bool(row.get("use_rna")):
        modality_blocks.append(
            "<rna_text_summary>\n"
            f"{_required_text(row, 'rna_text_summary', 'RNA text summary')}\n"
            "</rna_text_summary>"
        )
    if bool(row.get("use_radiology")):
        modality_blocks.append("<radiology_images>attached</radiology_images>")
        if not _ignore_radiology_biomarker(cfg):
            modality_blocks.append(
                "<radiology_biomarker>\n"
                f"{_required_text(row, 'radiology_biomarker', 'radiology biomarker')}\n"
                "</radiology_biomarker>"
            )

    if not modality_blocks:
        question_id = row.get("question_id", "<unknown>")
        raise ValueError(f"Question {question_id} has no enabled modalities.")
    return "\n\n".join(modality_blocks).strip()


def build_eval_prompt(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[str, str]:
    prompt_block = _prompt_block_for_row(row, cfg)
    prompt_key = question_type_key(row.get("question_type", ""))
    system_prompt = str(prompt_block.get("system_prompt", "")).strip()
    response_instruction = str(prompt_block.get("response_instruction", "")).strip()
    if not system_prompt:
        raise ValueError(f"VQA evaluation prompt block for {prompt_key} must define system_prompt.")
    if not response_instruction:
        raise ValueError(f"VQA evaluation prompt block for {prompt_key} must define response_instruction.")

    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError(f"Question {row.get('question_id', '<unknown>')} has empty question text.")
    modality_text = _modality_evidence_text(row, cfg)
    user_prompt = (
        f"{response_instruction}\n\n"
        "<modality_evidence>\n"
        f"{modality_text}\n"
        "</modality_evidence>\n\n"
        "<question>\n"
        f"{question}\n"
        "</question>"
    )
    if prompt_key == "mcq":
        options = option_values(row)
        if len(options) < 2:
            raise ValueError(f"MCQ question {row.get('question_id', '<unknown>')} has fewer than two options.")
        option_lines = [f"- {option}" for option in options]
        user_prompt = (
            f"{user_prompt}\n\n"
            "<choices>\n"
            f"{chr(10).join(option_lines)}\n"
            "</choices>"
        )
    return system_prompt, user_prompt


def build_mcq_prompt(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[str, str]:
    return build_eval_prompt(row, cfg)


def _resolve_path(path_value: str, *, repo_root: Path) -> Path:
    path = Path(str(path_value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _collect_images_from_dir(
    dir_value: str,
    *,
    repo_root: Path,
    max_images: int,
    allowed_extensions: Sequence[str],
    field_name: str,
) -> list[Path]:
    if not str(dir_value).strip():
        raise FileNotFoundError(
            f"Required image directory field '{field_name}' is empty."
        )
    image_dir = _resolve_path(str(dir_value), repo_root=repo_root)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Required image directory does not exist for '{field_name}': {image_dir}"
        )
    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"Required image path is not a directory for '{field_name}': {image_dir}"
        )

    extensions = {str(ext).lower() for ext in allowed_extensions}
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No image files found in required image directory '{field_name}': {image_dir}"
        )
    return image_paths[:max_images]


def _collect_images_from_file_list(
    value: str,
    *,
    repo_root: Path,
    max_images: int,
    allowed_extensions: Sequence[str],
    field_name: str,
) -> list[Path]:
    if not str(value).strip():
        raise FileNotFoundError(f"Required image list field '{field_name}' is empty.")
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Image field '{field_name}' must be a JSON list.")

    extensions = {str(ext).lower() for ext in allowed_extensions}
    image_paths: list[Path] = []
    for item in parsed:
        path = _resolve_path(str(item), repo_root=repo_root)
        if not path.exists():
            raise FileNotFoundError(f"Required image file does not exist for '{field_name}': {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Required image path is not a file for '{field_name}': {path}")
        if path.suffix.lower() in extensions:
            image_paths.append(path)

    if not image_paths:
        raise FileNotFoundError(f"No image files found in required image list '{field_name}'.")
    return image_paths[:max_images]


def collect_required_image_paths(
    row: Mapping[str, Any], cfg: Mapping[str, Any], *, repo_root: Path
) -> list[Path]:
    image_cfg = dict(cfg.get("image_inputs") or {})
    if not bool(image_cfg.get("enabled", True)):
        if bool(row.get("use_pathology")) or bool(row.get("use_radiology")):
            raise RuntimeError(
                "This VQA row requires image modalities, but image_inputs.enabled is false."
            )
        return []

    allowed_extensions = _as_list(
        image_cfg.get("allowed_extensions", [".png", ".jpg", ".jpeg"])
    )
    image_paths: list[Path] = []
    if bool(row.get("use_pathology")):
        image_paths.extend(
            _collect_images_from_dir(
                str(row.get("pathology_roi_png_dir", "")),
                repo_root=repo_root,
                max_images=int(image_cfg.get("max_pathology_images", 2)),
                allowed_extensions=allowed_extensions,
                field_name="pathology_roi_png_dir",
            )
        )
    if bool(row.get("use_radiology")):
        image_paths.extend(
            _collect_images_from_file_list(
                str(row.get("radiology_view_png_dir", "")),
                repo_root=repo_root,
                max_images=int(image_cfg.get("max_radiology_images", 4)),
                allowed_extensions=allowed_extensions,
                field_name="radiology_view_png_dir",
            )
        )
    return image_paths


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_answer_tag(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", str(text), flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_mcq_response(response_text: str, options: Sequence[str]) -> dict[str, str]:
    options_list = [str(option).strip() for option in options if str(option).strip()]
    normalized_options = {option.casefold(): option for option in options_list}
    parsed = _extract_json_object(response_text)

    candidates: list[str] = []
    if parsed:
        raw_answer = str(parsed.get("answer", "")).strip()
        if raw_answer:
            candidates.append(raw_answer)
        else:
            return {"predicted_answer": "", "parse_status": "failed"}
    else:
        candidates.append(response_text.strip())

    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        if text.casefold() in normalized_options:
            answer = normalized_options[text.casefold()]
            return {"predicted_answer": answer, "parse_status": "exact"}

    if parsed:
        return {"predicted_answer": "", "parse_status": "failed"}

    response_folded = response_text.casefold()
    matches = [
        option for option in options_list if option.casefold() in response_folded
    ]
    if len(matches) == 1:
        answer = matches[0]
        return {"predicted_answer": answer, "parse_status": "substring"}

    return {"predicted_answer": "", "parse_status": "failed"}


def parse_model_response(row: Mapping[str, Any], response_text: str) -> dict[str, str]:
    prompt_key = question_type_key(row.get("question_type", ""))
    if prompt_key == "mcq":
        answer_tag = _extract_answer_tag(response_text)
        if answer_tag:
            parsed = parse_mcq_response(answer_tag, option_values(row))
            predicted_answer = parsed["predicted_answer"]
            parsed["predicted_answer_label"] = answer_label_for_text(row, predicted_answer)
            if parsed["parse_status"] != "failed":
                parsed["parse_status"] = f"answer_tag_{parsed['parse_status']}"
            return parsed
        parsed = parse_mcq_response(response_text, option_values(row))
        predicted_answer = parsed["predicted_answer"]
        parsed["predicted_answer_label"] = answer_label_for_text(row, predicted_answer)
        return parsed

    parsed_json = _extract_json_object(response_text)
    if parsed_json is not None:
        answer = str(parsed_json.get("answer", "")).strip()
        return {
            "predicted_answer": answer,
            "predicted_answer_label": "",
            "parse_status": "exact" if answer else "failed",
        }
    answer = str(response_text).strip()
    return {
        "predicted_answer": answer,
        "predicted_answer_label": "",
        "parse_status": "raw" if answer else "failed",
    }


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _classification_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "accuracy": None,
            "correct": 0,
            "f1_macro": None,
            "f1_weighted": None,
            "parse_failed": 0,
        }
    y_true = frame["answer_label"].fillna("").astype(str).str.strip().tolist()
    y_pred = frame["predicted_answer_label"].fillna("").astype(str).str.strip().tolist()
    correct = [bool(pred) and pred == true for true, pred in zip(y_true, y_pred, strict=True)]
    labels = sorted({label for label in y_true if label})
    f1_values: list[float] = []
    weighted_values: list[float] = []
    weights: list[int] = []
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred, strict=True) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred, strict=True) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred, strict=True) if true == label and pred != label)
        denom = (2 * tp) + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        support = sum(1 for true in y_true if true == label)
        f1_values.append(float(f1))
        weighted_values.append(float(f1) * support)
        weights.append(support)
    return {
        "accuracy": float(sum(correct) / len(correct)),
        "correct": int(sum(correct)),
        "f1_macro": _safe_mean(f1_values),
        "f1_weighted": float(sum(weighted_values) / sum(weights)) if weights else None,
        "parse_failed": int(frame["parse_status"].astype(str).eq("failed").sum()),
    }


def add_bertscore_columns(predictions_df: pd.DataFrame, bert_score_cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = predictions_df.copy()
    for column in ["bertscore_precision", "bertscore_recall", "bertscore_f1"]:
        if column not in out.columns:
            out[column] = None

    cfg = dict(bert_score_cfg or {})
    if not bool(cfg.get("enabled", True)):
        return out
    qa_mask = out["question_type"].astype(str).map(question_type_key).eq("qa")
    qa_frame = out[qa_mask]
    if qa_frame.empty:
        return out

    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError("Open-ended VQA evaluation requires bert-score. Install it with: uv add bert-score") from exc

    reference_series = qa_frame["answer"].fillna("").astype(str)
    empty_reference_mask = reference_series.str.strip().eq("")
    if bool(empty_reference_mask.any()):
        bad_ids = qa_frame.loc[empty_reference_mask, "question_id"].astype(str).head(10).tolist()
        raise ValueError(f"BERTScore references are empty for question_id(s): {bad_ids}")

    candidate_series = qa_frame["predicted_answer"].fillna("").astype(str)
    empty_candidate_mask = candidate_series.str.strip().eq("")
    if bool(empty_candidate_mask.any()):
        print(f"BERTScore: scoring {int(empty_candidate_mask.sum())} empty QA prediction(s) as [empty answer].")

    candidates = [text.strip() if text.strip() else "[empty answer]" for text in candidate_series.tolist()]
    references = [text.strip() for text in reference_series.tolist()]
    max_length = int(cfg.get("max_length", 512))
    if max_length <= 0:
        raise ValueError("metrics.bert_score.max_length must be positive.")
    restore_tokenizer = _patch_bert_score_tokenizer_max_length(max_length)
    try:
        precision, recall, f1 = bert_score(
            candidates,
            references,
            model_type=str(cfg.get("model_type", "roberta-large")),
            num_layers=int(cfg["num_layers"]) if cfg.get("num_layers") is not None else None,
            lang=str(cfg.get("lang", "en")),
            batch_size=int(cfg.get("batch_size", 8)),
            rescale_with_baseline=bool(cfg.get("rescale_with_baseline", True)),
            use_fast_tokenizer=bool(cfg.get("use_fast_tokenizer", True)),
        )
    finally:
        restore_tokenizer()
    indices = qa_frame.index.tolist()
    out.loc[indices, "bertscore_precision"] = [float(value) for value in precision.detach().cpu().tolist()]
    out.loc[indices, "bertscore_recall"] = [float(value) for value in recall.detach().cpu().tolist()]
    out.loc[indices, "bertscore_f1"] = [float(value) for value in f1.detach().cpu().tolist()]
    return out


def _rouge_l_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)?", text.casefold())


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def _rouge_l_scores(candidate: str, reference: str) -> dict[str, float]:
    candidate_tokens = _rouge_l_tokens(candidate)
    reference_tokens = _rouge_l_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}
    lcs = _lcs_length(candidate_tokens, reference_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "rouge_l_precision": float(precision),
        "rouge_l_recall": float(recall),
        "rouge_l_f1": float(f1),
    }


def add_rouge_l_columns(predictions_df: pd.DataFrame, rouge_l_cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = predictions_df.copy()
    for column in ["rouge_l_precision", "rouge_l_recall", "rouge_l_f1"]:
        if column not in out.columns:
            out[column] = None

    cfg = dict(rouge_l_cfg or {})
    if not bool(cfg.get("enabled", True)):
        return out
    qa_mask = out["question_type"].astype(str).map(question_type_key).eq("qa")
    qa_frame = out[qa_mask]
    if qa_frame.empty:
        return out

    reference_series = qa_frame["answer"].fillna("").astype(str)
    empty_reference_mask = reference_series.str.strip().eq("")
    if bool(empty_reference_mask.any()):
        bad_ids = qa_frame.loc[empty_reference_mask, "question_id"].astype(str).head(10).tolist()
        raise ValueError(f"ROUGE-L references are empty for question_id(s): {bad_ids}")

    for index, row in qa_frame.iterrows():
        scores = _rouge_l_scores(
            str(row.get("predicted_answer", "")).strip(),
            str(row.get("answer", "")).strip(),
        )
        for column, value in scores.items():
            out.at[index, column] = value
    return out


def _patch_bert_score_tokenizer_max_length(max_length: int):
    score_module = sys.modules.get("bert_score.score")
    if score_module is None or not hasattr(score_module, "get_tokenizer"):
        return lambda: None

    original_get_tokenizer = score_module.get_tokenizer

    def get_tokenizer_with_max_length(model_type: str, use_fast: bool = False):
        tokenizer = original_get_tokenizer(model_type, use_fast=use_fast)
        tokenizer.model_max_length = int(max_length)
        return tokenizer

    score_module.get_tokenizer = get_tokenizer_with_max_length

    def restore() -> None:
        score_module.get_tokenizer = original_get_tokenizer

    return restore


METRIC_DIMENSIONS = [
    "question_type",
    "generation_type",
    "task_category",
    "task_id",
    "modality_combination_name",
    "project_id",
]


def _base_metric_record(
    *,
    metric_group: str,
    model_display_name: str,
    backend: str,
    values: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "metric_group": metric_group,
        "model_display_name": model_display_name,
        "backend": backend,
    }
    for dimension in METRIC_DIMENSIONS:
        record[dimension] = "ALL"
    if values:
        record.update(values)
    return record


def _metric_values(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n": 0}
    prompt_keys = frame["question_type"].astype(str).map(question_type_key)
    if prompt_keys.nunique() > 1:
        return {"n": int(len(frame))}
    if prompt_keys.iloc[0] == "mcq":
        return {"n": int(len(frame)), **_classification_metrics(frame)}

    precision_values = [float(value) for value in frame["bertscore_precision"].dropna().tolist()]
    recall_values = [float(value) for value in frame["bertscore_recall"].dropna().tolist()]
    f1_values = [float(value) for value in frame["bertscore_f1"].dropna().tolist()]
    rouge_precision_values = [float(value) for value in frame["rouge_l_precision"].dropna().tolist()]
    rouge_recall_values = [float(value) for value in frame["rouge_l_recall"].dropna().tolist()]
    rouge_f1_values = [float(value) for value in frame["rouge_l_f1"].dropna().tolist()]
    return {
        "n": int(len(frame)),
        "parse_failed": int(frame["parse_status"].astype(str).eq("failed").sum()),
        "bertscore_precision_mean": _safe_mean(precision_values),
        "bertscore_recall_mean": _safe_mean(recall_values),
        "bertscore_f1_mean": _safe_mean(f1_values),
        "rouge_l_precision_mean": _safe_mean(rouge_precision_values),
        "rouge_l_recall_mean": _safe_mean(rouge_recall_values),
        "rouge_l_f1_mean": _safe_mean(rouge_f1_values),
    }


def build_flat_metric_records(
    predictions_df: pd.DataFrame,
    *,
    model_display_name: str,
    backend: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    group_specs: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("by_question_type", ["question_type"]),
        ("by_generation_type", ["generation_type"]),
        ("by_task_category", ["task_category"]),
        ("by_modality_combination_name", ["modality_combination_name"]),
        ("by_project_id", ["project_id"]),
        (
            "main_table",
            ["question_type", "generation_type", "task_category", "modality_combination_name"],
        ),
        (
            "cancer_table",
            ["question_type", "generation_type", "project_id", "modality_combination_name"],
        ),
        (
            "task_cancer_table",
            ["question_type", "generation_type", "task_category", "project_id", "modality_combination_name"],
        ),
    ]
    for metric_group, group_columns in group_specs:
        if not group_columns:
            record = _base_metric_record(
                metric_group=metric_group,
                model_display_name=model_display_name,
                backend=backend,
            )
            record.update(_metric_values(predictions_df))
            records.append(record)
            continue

        for group_values, group in predictions_df.groupby(group_columns, dropna=False, sort=True):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            dimension_values = {
                column: str(value)
                for column, value in zip(group_columns, group_values, strict=True)
            }
            record = _base_metric_record(
                metric_group=metric_group,
                model_display_name=model_display_name,
                backend=backend,
                values=dimension_values,
            )
            record.update(_metric_values(group))
            records.append(record)
    return records


def compute_mcq_metrics(predictions_df: pd.DataFrame) -> dict[str, Any]:
    if predictions_df.empty:
        return {
            "n": 0,
            "accuracy": None,
            "correct": 0,
            "parse_failed": 0,
            "by_task_id": {},
            "by_project_id": {},
        }

    correct = predictions_df["correct"].astype(bool)
    metrics: dict[str, Any] = {
        "n": int(len(predictions_df)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "parse_failed": int(
            predictions_df["parse_status"].astype(str).eq("failed").sum()
        ),
        "by_task_id": {},
        "by_project_id": {},
    }
    for key, out_key in [("task_id", "by_task_id"), ("project_id", "by_project_id")]:
        grouped = predictions_df.groupby(key)
        metrics[out_key] = {
            str(group_key): {
                "n": int(len(group)),
                "correct": int(group["correct"].astype(bool).sum()),
                "accuracy": float(group["correct"].astype(bool).mean()),
            }
            for group_key, group in grouped
        }
    return metrics
