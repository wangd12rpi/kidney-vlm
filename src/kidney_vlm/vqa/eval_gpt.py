from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

OPTION_COLUMNS = ["option_a", "option_b", "option_c", "option_d"]
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

    if _enabled_filter(filters, "nonempty_enabled_genomics_text"):
        if "use_dnam" in out.columns:
            out = out[
                (~out["use_dnam"])
                | out["dnam_text_summary"].fillna("").astype(str).str.strip().ne("")
            ]
        if "use_rna" in out.columns:
            out = out[
                (~out["use_rna"])
                | out["rna_text_summary"].fillna("").astype(str).str.strip().ne("")
            ]

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


def build_mcq_prompt(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[str, str]:
    options = option_values(row)
    option_lines = [f"- {option}" for option in options]
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
        modality_blocks.append(
            "<radiology_biomarker>\n"
            f"{_required_text(row, 'radiology_biomarker', 'radiology biomarker')}\n"
            "</radiology_biomarker>"
        )

    if not modality_blocks:
        question_id = row.get("question_id", "<unknown>")
        raise ValueError(f"Question {question_id} has no enabled modalities.")
    modality_text = "\n\n".join(modality_blocks).strip()
    system_prompt = str(cfg.get("system_prompt", "")).strip()
    response_instruction = str(cfg.get("response_instruction", "")).strip()
    if not system_prompt:
        raise ValueError("VQA GPT evaluation config must define system_prompt.")
    if not response_instruction:
        raise ValueError("VQA GPT evaluation config must define response_instruction.")
    user_prompt = (
        f"{response_instruction}\n\n"
        "<modality_evidence>\n"
        f"{modality_text}\n"
        "</modality_evidence>\n\n"
        "<question>\n"
        f"{str(row.get('question', '')).strip()}\n"
        "</question>\n\n"
        "<options>\n"
        f"{chr(10).join(option_lines)}\n"
        "</options>"
    )
    return system_prompt, user_prompt


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
            _collect_images_from_dir(
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
