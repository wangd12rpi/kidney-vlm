#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import base64
import mimetypes
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.vqa.eval_gpt import collect_required_image_paths, question_type_key
from kidney_vlm.vqa.genomics_text_summary import build_dnam_text_summary, build_rna_text_summary
from kidney_vlm.vqa.gt_mcq import _radiology_png_artifact_value
from kidney_vlm.vqa.schema import OPTION_COLUMNS, VQA_COLUMNS, normalize_vqa_df, validate_vqa_df
from kidney_vlm.vqa.stage_config import as_bool, cfg_list, clean_text

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

MODALITY_COMBO_KEYS = {
    "path": "use_pathology",
    "rad": "use_radiology",
    "dnam": "use_dnam",
    "rna": "use_rna",
}
FAILED_ATTEMPT_COLUMNS = [
    "question_id",
    "case_id",
    "project_id",
    "attempt",
    "error_type",
    "validation_error",
    "teacher_answer",
    "raw_response",
]


@dataclass(frozen=True)
class RationaleRequest:
    row_index: int
    question_id: int
    prompt_row: dict[str, Any]
    image_paths: list[Path]
    expected_answer: str


@dataclass(frozen=True)
class RationaleResult:
    row_index: int
    rationale: str
    failed_attempts: list[dict[str, Any]]


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="21_cot_rationale_gen/generate_vqa_rationales.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_repo_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env_value(name: str, repo_env: Mapping[str, str]) -> str:
    return os.environ.get(name, "").strip() or str(repo_env.get(name, "")).strip()


def _create_client(cfg: Mapping[str, Any]) -> tuple[Any, str]:
    try:
        from openai import AzureOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing Python package 'openai'. Run uv sync before using this script.") from exc

    repo_env = _read_repo_env()
    endpoint = clean_text(cfg.get("endpoint"))
    api_key_env = clean_text(cfg.get("api_key_env"))
    api_key = _env_value(api_key_env, repo_env)
    deployment = clean_text(cfg.get("deployment"))

    if not endpoint:
        raise ValueError("Missing azure.endpoint in config.")
    if not api_key_env:
        raise ValueError("Missing azure.api_key_env in config.")
    if not api_key:
        raise EnvironmentError(f"Missing Azure API key env var: {api_key_env}")
    if not deployment:
        raise ValueError("Missing azure.deployment in config.")

    client_kwargs: dict[str, Any] = {
        "azure_endpoint": endpoint,
        "api_key": api_key,
        "api_version": clean_text(cfg.get("api_version")),
    }
    request_timeout_seconds = float(cfg.get("request_timeout_seconds", 0) or 0)
    if request_timeout_seconds > 0:
        client_kwargs["timeout"] = request_timeout_seconds
    client = AzureOpenAI(**client_kwargs)
    return client, deployment


def _extract_text_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content.strip()
    if isinstance(raw_content, list):
        chunks: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    chunks.append(text)
                continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                chunks.append(text_attr.strip())
                continue
            if isinstance(item, Mapping):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    chunks.append(text_value.strip())
        return "\n".join(chunks).strip()
    return str(raw_content or "").strip()


def _image_data_url(image_path: Path, image_cfg: Mapping[str, Any]) -> str:
    max_image_side = int(image_cfg.get("max_image_side", 0) or 0)
    if max_image_side > 0:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("Pillow is required for image resizing before GPT calls.") from exc

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_image_side, max_image_side))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=int(image_cfg.get("jpeg_quality", 85) or 85))
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_user_content(
    user_prompt: str,
    image_paths: Sequence[Path],
    image_cfg: Mapping[str, Any],
) -> str | list[dict[str, Any]]:
    if not image_paths:
        return user_prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(image_path, image_cfg)}})
    return content


def _call_azure_gpt(
    *,
    client: Any,
    deployment: str,
    azure_cfg: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    image_paths: Sequence[Path],
    image_cfg: Mapping[str, Any],
) -> str:
    retries = int(azure_cfg.get("max_retries", 3) or 3)
    retry_sleep_seconds = float(azure_cfg.get("retry_sleep_seconds", 2.0) or 2.0)
    reasoning_effort = clean_text(azure_cfg.get("reasoning_effort"))
    verbosity = clean_text(azure_cfg.get("verbosity"))

    request_base: dict[str, Any] = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_content(user_prompt, image_paths, image_cfg)},
        ],
        "max_completion_tokens": int(azure_cfg.get("max_completion_tokens", 512) or 512),
    }
    if reasoning_effort:
        request_base["reasoning_effort"] = reasoning_effort
    if verbosity:
        request_base["verbosity"] = verbosity
    if azure_cfg.get("temperature") is not None:
        request_base["temperature"] = float(azure_cfg.get("temperature"))
    if azure_cfg.get("top_p") is not None:
        request_base["top_p"] = float(azure_cfg.get("top_p"))

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(**request_base)
            choice = response.choices[0]
            text = _extract_text_content(choice.message.content)
            if not text:
                raise RuntimeError(
                    "GPT returned an empty response. "
                    f"finish_reason={getattr(choice, 'finish_reason', None)!r}; "
                    f"usage={getattr(response, 'usage', None)!r}."
                )
            return text
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_seconds)
    raise RuntimeError(f"GPT rationale call failed after {retries} attempts: {last_error}")


def _extract_tag_text(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _validation_error(
    response_text: str,
    expected_answer: str,
    cfg: Mapping[str, Any],
    displayed_choices: Sequence[str] | None = None,
) -> str:
    if not bool(cfg.get("enabled", True)):
        return ""
    think_text = _extract_tag_text(response_text, "think")
    answer_text = _extract_tag_text(response_text, "answer")
    if not think_text:
        return "missing <think> rationale"
    if not answer_text:
        return "missing <answer> choice"
    if not re.fullmatch(
        r"\s*<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        return "response must contain only one <think> block followed by one <answer> block"

    choices = [clean_text(choice) for choice in displayed_choices or [] if clean_text(choice)]
    if choices and answer_text.strip() not in choices:
        return "<answer> must copy exactly one displayed choice"
    if answer_text.strip() != expected_answer.strip():
        return "answer_mismatch"
    if bool(cfg.get("reject_answer_text_in_think", False)):
        if expected_answer.strip().casefold() in think_text.casefold():
            return "<think> must not copy the exact target answer text"

    if bool(cfg.get("require_two_steps", True)):
        step_1 = re.findall(r"\bStep\s*1\s*[\-—]\s*Observation\s*:", think_text, flags=re.IGNORECASE)
        step_2 = re.findall(r"\bStep\s*2\s*[\-—]\s*Reasoning\s*:", think_text, flags=re.IGNORECASE)
        if len(step_1) != 1 or len(step_2) != 1:
            return "<think> must contain exactly Step 1 — Observation and Step 2 — Reasoning"
        if not re.match(r"\s*Step\s*1\s*[\-—]\s*Observation\s*:", think_text, flags=re.IGNORECASE):
            return "<think> must start with Step 1 — Observation"
        if re.search(r"\bStep\s*[3-9]\b|\bAction\s*\d+\b", think_text, flags=re.IGNORECASE):
            return "<think> must use only the two requested Step headings"

    word_count = len(re.findall(r"\b[\w.-]+\b", think_text))
    min_words = int(cfg.get("min_think_words", 100) or 100)
    max_words = int(cfg.get("max_think_words", 140) or 140)
    if word_count < min_words or word_count > max_words:
        return f"<think> word count must be between {min_words} and {max_words}; got {word_count}"

    meta_patterns = [
        r"\brationale should\b",
        r"\bshould mention\b",
        r"\bkeep concise\b",
        r"\bplanning notes?\b",
        r"\btarget answer\b",
        r"\bcorrect option\b",
        r"\bwrong option\b",
        r"\bfirst choice\b",
        r"\bsecond choice\b",
        r"\bthird choice\b",
        r"\bfourth choice\b",
    ]
    for pattern in meta_patterns:
        if re.search(pattern, think_text, flags=re.IGNORECASE):
            return "<think> contains planning/meta text instead of final rationale"
    return ""


def _read_vqa_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing VQA parquet: {path}")
    return normalize_vqa_df(pd.read_parquet(path))


def _read_registry_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing registry parquet: {path}")
    df = pd.read_parquet(path)
    required = {"patient_id", "pathology_png_roi_paths", "radiology_embedding_paths"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Registry parquet is missing columns: {missing}")
    if df["patient_id"].astype(str).duplicated().any():
        duplicated = df.loc[df["patient_id"].astype(str).duplicated(), "patient_id"].head(5).tolist()
        raise ValueError(f"Registry parquet has duplicated patient_id values: {duplicated}")
    return {
        clean_text(row["patient_id"]): dict(row)
        for row in df.to_dict(orient="records")
        if clean_text(row.get("patient_id"))
    }


def _first_existing_parent_dir(paths_value: Any) -> str:
    for path_text in cfg_list(paths_value):
        path = _resolve_path(path_text)
        if path.is_file():
            return path.parent.relative_to(ROOT).as_posix()
        if path.is_dir():
            return path.relative_to(ROOT).as_posix()
    return ""


def _build_registry_dnam_summary(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    if not cfg_list(row.get("genomics_dna_methylation_paths")):
        return ""
    return build_dnam_text_summary(
        row,
        max_beta_values=int(cfg.get("dnam_text_summary_max_beta_values", 50_000) or 50_000),
    )


def _build_registry_rna_summary(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    if not cfg_list(row.get("genomics_rna_bulk_paths")):
        return ""
    return build_rna_text_summary(
        row,
        max_top_genes=int(cfg.get("rna_text_summary_max_top_genes", 8) or 8),
    )


def _enrich_row_from_registry(
    row: Mapping[str, Any],
    registry_map: Mapping[str, dict[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    fallback_cfg = dict(cfg.get("artifact_fallbacks") or {})
    if not bool(fallback_cfg.get("enabled", True)):
        return out

    registry_row = registry_map.get(clean_text(row.get("case_id")))
    if not registry_row:
        return out

    if bool(out.get("use_pathology")) and not clean_text(out.get("pathology_roi_png_dir")):
        out["pathology_roi_png_dir"] = _first_existing_parent_dir(
            registry_row.get("pathology_png_roi_paths")
        )

    if bool(out.get("use_radiology")) and not clean_text(out.get("radiology_view_png_dir")):
        out["radiology_view_png_dir"] = _radiology_png_artifact_value(
            registry_row,
            {
                "radiology_png_root": fallback_cfg.get("radiology_png_root", "data/radiology_png"),
                "max_test_radiology_images": fallback_cfg.get("max_radiology_images", 6),
            },
        )

    fallback_task_ids = set(cfg_list(fallback_cfg.get("include_text_fallback_for_task_ids")))
    needs_text_fallback = clean_text(out.get("task_id")) in fallback_task_ids
    if bool(fallback_cfg.get("populate_genomics_text_summaries", True)) and needs_text_fallback:
        if bool(out.get("use_dnam")) and not clean_text(out.get("dnam_text_summary")):
            out["dnam_text_summary"] = _build_registry_dnam_summary(registry_row, fallback_cfg)
        if bool(out.get("use_rna")) and not clean_text(out.get("rna_text_summary")):
            out["rna_text_summary"] = _build_registry_rna_summary(registry_row, fallback_cfg)

    return out


def _row_for_task_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    task_id = clean_text(out.get("task_id"))
    if task_id == "pathology_findings":
        out["use_radiology"] = False
        out["use_dnam"] = False
        out["use_rna"] = False
    elif task_id == "radiology_findings":
        out["use_pathology"] = False
        out["use_dnam"] = False
        out["use_rna"] = False
    elif task_id == "genomic_findings":
        out["use_pathology"] = False
        out["use_radiology"] = False
    elif task_id == "integrated_interpretation":
        pass
    return out


def _filter_values(block: Mapping[str, Any], *, name: str) -> list[str]:
    values = cfg_list(block.get("values"))
    if not values:
        raise ValueError(f"Filter '{name}' is enabled but has no values.")
    return values


def _enabled_filter(filters: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    raw = filters.get(name)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"Filter '{name}' must be a mapping.")
    block = dict(raw)
    return block if bool(block.get("enabled", False)) else None


def _apply_column_values_filter(frame: pd.DataFrame, filters: Mapping[str, Any], name: str, column: str) -> pd.DataFrame:
    block = _enabled_filter(filters, name)
    if not block:
        return frame
    allowed = set(_filter_values(block, name=name))
    return frame[frame[column].astype(str).isin(allowed)].copy()


def _allowed_modality_combo_mask(frame: pd.DataFrame, block: Mapping[str, Any]) -> pd.Series:
    raw_values = block.get("values")
    if raw_values is None or isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
        raise TypeError("Filter 'allowed_modality_combo.values' must be a list of mappings.")
    values = list(raw_values)
    if not values:
        raise ValueError("Filter 'allowed_modality_combo' is enabled but has no values.")

    mask = pd.Series(False, index=frame.index)
    for index, raw_combo in enumerate(values, start=1):
        if not isinstance(raw_combo, Mapping):
            raise TypeError("Each allowed_modality_combo value must be a mapping with path, rad, dnam, rna.")
        combo = dict(raw_combo)
        if set(combo) != set(MODALITY_COMBO_KEYS):
            raise ValueError(
                f"allowed_modality_combo item {index} must use exactly path, rad, dnam, rna."
            )
        combo_mask = pd.Series(True, index=frame.index)
        for short_key, column in MODALITY_COMBO_KEYS.items():
            combo_mask &= frame[column].astype(bool).eq(as_bool(combo[short_key]))
        mask |= combo_mask
    return mask


def _select_rows(vqa_df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = vqa_df.copy()
    filters = dict(cfg.get("filters") or {})

    split_filter = _enabled_filter(filters, "split")
    if split_filter:
        values = cfg_list(split_filter.get("values"))
        value = clean_text(split_filter.get("value"))
        allowed = set(values or ([value] if value else []))
        if not allowed:
            raise ValueError("Filter 'split' is enabled but has no value or values.")
        out = out[out["split"].astype(str).isin(allowed)].copy()

    for filter_name, column in [
        ("question_types", "question_type"),
        ("generation_types", "generation_type"),
        ("modality_combination_names", "modality_combination_name"),
        ("project_ids", "project_id"),
        ("task_ids", "task_id"),
        ("task_categories", "task_category"),
        ("case_ids", "case_id"),
    ]:
        out = _apply_column_values_filter(out, filters, filter_name, column)

    combo_filter = _enabled_filter(filters, "allowed_modality_combo")
    if combo_filter:
        out = out[_allowed_modality_combo_mask(out, combo_filter)].copy()

    out = out.sort_values(["project_id", "case_id", "task_id", "question_id"]).reset_index(drop=True)

    row_limit_filter = _enabled_filter(filters, "row_limit")
    if row_limit_filter:
        max_rows = int(row_limit_filter.get("max_rows", 0) or 0)
        if max_rows <= 0:
            raise ValueError("Filter 'row_limit' is enabled but max_rows is not positive.")
        if bool(row_limit_filter.get("sample", False)) and len(out) > max_rows:
            out = out.sample(
                n=max_rows,
                random_state=int(row_limit_filter.get("sample_seed", 17) or 17),
            ).sort_values(["project_id", "case_id", "task_id", "question_id"])
        else:
            out = out.head(max_rows)
        out = out.reset_index(drop=True)

    if out.empty:
        raise ValueError("No VQA rows remain after filters.")
    return out


def _render_template(template: str, values: Mapping[str, Any]) -> str:
    pattern = re.compile(r"{{\s*([A-Za-z0-9_]+)\s*}}")
    placeholders = sorted(set(pattern.findall(template)))
    missing = [name for name in placeholders if name not in values]
    if missing:
        raise ValueError(f"Prompt template placeholders are missing values: {missing}")

    def replace(match: re.Match[str]) -> str:
        return str(values[match.group(1)])

    return pattern.sub(replace, template)


def _prompt_values(row: Mapping[str, Any]) -> dict[str, str]:
    values = {
        "case_id": clean_text(row.get("case_id")),
        "project_id": clean_text(row.get("project_id")),
        "question_id": clean_text(row.get("question_id")),
        "question_type": clean_text(row.get("question_type")),
        "generation_type": clean_text(row.get("generation_type")),
        "task_category": clean_text(row.get("task_category")),
        "task_id": clean_text(row.get("task_id")),
        "modality_combination_name": clean_text(row.get("modality_combination_name")),
        "use_pathology": str(bool(row.get("use_pathology"))).lower(),
        "use_radiology": str(bool(row.get("use_radiology"))).lower(),
        "use_dnam": str(bool(row.get("use_dnam"))).lower(),
        "use_rna": str(bool(row.get("use_rna"))).lower(),
        "question": clean_text(row.get("question")),
    }
    for column in OPTION_COLUMNS:
        values[column] = clean_text(row.get(column))
    return values


def _prompt_block(row: Mapping[str, Any], prompts_cfg: Mapping[str, Any]) -> dict[str, str]:
    prompt_key = question_type_key(row.get("question_type"))
    raw_block = prompts_cfg.get(prompt_key)
    if not isinstance(raw_block, Mapping):
        raise ValueError(f"prompts.{prompt_key} must be defined before generating rationales for this row.")
    system_prompt = clean_text(raw_block.get("system_prompt"))
    user_template = clean_text(raw_block.get("user_template"))
    if not system_prompt or not user_template:
        raise ValueError(f"prompts.{prompt_key} must define non-empty system_prompt and user_template.")
    return {"system_prompt": system_prompt, "user_template": user_template}


def _build_prompt(
    *,
    row: Mapping[str, Any],
    prompts_cfg: Mapping[str, Any],
) -> tuple[str, str]:
    values = _prompt_values(row)
    block = _prompt_block(row, prompts_cfg)
    return (
        _render_template(block["system_prompt"], values),
        _render_template(block["user_template"], values),
    )


def _displayed_choices(row: Mapping[str, Any]) -> list[str]:
    return [clean_text(row.get(column)) for column in OPTION_COLUMNS if clean_text(row.get(column))]


def _prompt_row_for_attempt(row: Mapping[str, Any], attempt: int) -> dict[str, Any]:
    out = dict(row)
    if attempt <= 1:
        return out
    choices = _displayed_choices(out)
    random.Random(f"{int(out['question_id'])}:{attempt}").shuffle(choices)
    for column, choice in zip(OPTION_COLUMNS, choices, strict=True):
        out[column] = choice
    return out


def _collect_images(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[Path]:
    image_cfg = dict(cfg.get("image_inputs") or {})
    if not bool(image_cfg.get("enabled", True)):
        return []
    max_pathology_images = int(image_cfg.get("max_pathology_images", 4) or 4)
    collection_cfg = dict(image_cfg)
    if bool(row.get("use_pathology")) and not bool(row.get("use_radiology")):
        collection_cfg["max_pathology_images"] = 100_000
    image_paths = collect_required_image_paths(
        row,
        {"image_inputs": collection_cfg},
        repo_root=ROOT,
    )
    if (
        bool(row.get("use_pathology"))
        and not bool(row.get("use_radiology"))
        and len(image_paths) > max_pathology_images
    ):
        if max_pathology_images == 1:
            return [image_paths[len(image_paths) // 2]]
        indices = [
            round(index * (len(image_paths) - 1) / (max_pathology_images - 1))
            for index in range(max_pathology_images)
        ]
        return [image_paths[index] for index in indices]
    return image_paths


def _modality_signature(row: Mapping[str, Any]) -> str:
    return (
        f"path={bool(row.get('use_pathology'))},"
        f"rad={bool(row.get('use_radiology'))},"
        f"dnam={bool(row.get('use_dnam'))},"
        f"rna={bool(row.get('use_rna'))}"
    )


def _should_print_prompt(
    *,
    row: Mapping[str, Any],
    cfg: Mapping[str, Any],
    printed_count: int,
    printed_signatures: set[str],
) -> bool:
    if not bool(cfg.get("enabled", False)):
        return False
    max_rows = int(cfg.get("max_rows", 0) or 0)
    if max_rows <= 0 or printed_count >= max_rows:
        return False
    if bool(cfg.get("group_by_modality_availability", True)):
        signature = _modality_signature(row)
        if signature in printed_signatures:
            return False
    return True


def _print_gpt_payload(
    *,
    source_row: Mapping[str, Any],
    prompt_row: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    image_paths: Sequence[Path],
    cfg: Mapping[str, Any],
) -> None:
    print("-" * 80)
    print("GPT PAYLOAD INSPECTION")
    print(f"question_id: {int(source_row['question_id'])}")
    print(f"case_id: {clean_text(source_row.get('case_id'))}")
    print(f"task_id: {clean_text(source_row.get('task_id'))}")
    print(f"source_row_modality_availability: {_modality_signature(source_row)}")
    print(f"gpt_payload_modality_availability: {_modality_signature(prompt_row)}")
    if bool(cfg.get("include_image_paths", True)):
        print(f"image_count: {len(image_paths)}")
        for image_path in image_paths:
            print(f"image_path: {image_path}")
    if bool(cfg.get("include_system_prompt", True)):
        print("\n[SYSTEM PROMPT]")
        print(system_prompt)
    if bool(cfg.get("include_user_prompt", True)):
        print("\n[USER PROMPT]")
        print(user_prompt)


def _existing_rationales(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    existing = pd.read_parquet(path)
    required = {"question_id", "rationale"}
    missing = sorted(required - set(existing.columns))
    if missing:
        raise ValueError(f"Existing rationale parquet is missing columns: {missing}")
    rationales: dict[int, str] = {}
    for row in existing[["question_id", "rationale"]].to_dict(orient="records"):
        rationale = clean_text(row.get("rationale"))
        if rationale:
            rationales[int(row["question_id"])] = rationale
    return rationales


def _write_output(rows: list[dict[str, Any]], output_path: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No rationale rows to write.")
    validate_vqa_df(frame[VQA_COLUMNS].copy(), required_columns=VQA_COLUMNS)
    frame = frame[VQA_COLUMNS + ["rationale"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def _error_attempts_path(output_path: Path, run_cfg: Mapping[str, Any]) -> Path:
    subdir = clean_text(run_cfg.get("error_attempts_subdir")) or "errors"
    return output_path.parent / subdir / f"{output_path.stem}_failed_attempts.parquet"


def _write_failed_attempts(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows, columns=FAILED_ATTEMPT_COLUMNS)
    frame = frame.sort_values(["question_id", "attempt"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def _rationale_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if clean_text(row.get("rationale"))]


def _generate_rationale_for_request(
    request: RationaleRequest,
    *,
    client: Any,
    deployment: str,
    azure_cfg: Mapping[str, Any],
    validation_cfg: Mapping[str, Any],
    image_cfg: Mapping[str, Any],
    prompts_cfg: Mapping[str, Any],
    teacher_attempts_cfg: Mapping[str, Any],
) -> RationaleResult:
    max_attempts = int(teacher_attempts_cfg.get("max_attempts", 3) or 3)
    if max_attempts < 1:
        raise ValueError("teacher_attempts.max_attempts must be at least 1.")

    failed_attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        attempt_row = _prompt_row_for_attempt(request.prompt_row, attempt)
        system_prompt, user_prompt = _build_prompt(row=attempt_row, prompts_cfg=prompts_cfg)
        response_text = ""
        try:
            response_text = _call_azure_gpt(
                client=client,
                deployment=deployment,
                azure_cfg=azure_cfg,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=request.image_paths,
                image_cfg=image_cfg,
            )
            validation_error = _validation_error(
                response_text,
                request.expected_answer,
                validation_cfg,
                displayed_choices=_displayed_choices(attempt_row),
            )
        except Exception as exc:
            validation_error = str(exc)
            error_type = "api_error"
        else:
            if not validation_error:
                return RationaleResult(
                    row_index=request.row_index,
                    rationale=response_text.strip(),
                    failed_attempts=failed_attempts,
                )
            error_type = "answer_mismatch" if validation_error == "answer_mismatch" else "validation_error"

        failed_attempts.append(
            {
                "question_id": request.question_id,
                "case_id": clean_text(request.prompt_row.get("case_id")),
                "project_id": clean_text(request.prompt_row.get("project_id")),
                "attempt": attempt,
                "error_type": error_type,
                "validation_error": validation_error,
                "teacher_answer": _extract_tag_text(response_text, "answer"),
                "raw_response": response_text,
            }
        )

    return RationaleResult(
        row_index=request.row_index,
        rationale="",
        failed_attempts=failed_attempts,
    )


def main() -> None:
    cfg = OmegaConf.to_container(load_cfg(), resolve=True)
    assert isinstance(cfg, Mapping)

    vqa_path = _resolve_path(cfg["data"]["vqa_parquet_path"])
    registry_path = _resolve_path(cfg["data"]["registry_path"])
    output_path = _resolve_path(cfg["run"]["output_parquet_path"])
    failed_attempts_path = _error_attempts_path(output_path, dict(cfg["run"]))

    vqa_df = _read_vqa_frame(vqa_path)
    selected_df = _select_rows(vqa_df, cfg)
    registry_map = _read_registry_map(registry_path)

    dry_run = bool(cfg["run"].get("dry_run", False))
    override_existing = bool(cfg["run"].get("override_existing", False))
    print_first_n_outputs = int(cfg["run"].get("print_first_n_outputs", 0) or 0)
    prompt_inspection_cfg = dict(cfg.get("prompt_inspection") or {})
    printed_prompt_count = 0
    printed_prompt_signatures: set[str] = set()
    skip_missing_images = bool((cfg.get("image_inputs") or {}).get("skip_rows_missing_required_images", True))
    save_every_n = int(cfg["run"].get("save_every_n", 0) or 0)
    max_concurrent_requests = int(dict(cfg.get("azure") or {}).get("max_concurrent_requests", 1) or 1)
    if max_concurrent_requests < 1:
        raise ValueError("azure.max_concurrent_requests must be at least 1.")

    if override_existing and output_path.exists() and not dry_run:
        output_path.unlink()
    if override_existing and failed_attempts_path.exists() and not dry_run:
        failed_attempts_path.unlink()
    existing_rationales = _existing_rationales(output_path) if not override_existing and not dry_run else {}
    client = None
    deployment = ""
    if not dry_run:
        client, deployment = _create_client(dict(cfg["azure"]))

    rows: list[dict[str, Any]] = []
    requests: list[RationaleRequest] = []
    failed_attempts: list[dict[str, Any]] = []
    stats = {
        "input_rows": int(len(vqa_df)),
        "selected_rows": int(len(selected_df)),
        "reused_existing": 0,
        "generated": 0,
        "failed_after_all_attempts": 0,
        "failed_attempts": 0,
        "skipped_missing_images": 0,
    }

    iterator = tqdm(selected_df.to_dict(orient="records"), desc="Preparing rationale requests")
    for row in iterator:
        row = _enrich_row_from_registry(
            row,
            registry_map,
            cfg,
        )
        row_out = dict(row)
        question_id = int(row["question_id"])
        prompt_row = _row_for_task_evidence(row)
        system_prompt, user_prompt = _build_prompt(
            row=prompt_row,
            prompts_cfg=dict(cfg["prompts"]),
        )

        try:
            image_paths = _collect_images(prompt_row, cfg)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            if not skip_missing_images:
                raise
            print(f"Skipping question_id={question_id} because required image input is missing: {exc}")
            stats["skipped_missing_images"] += 1
            continue

        if _should_print_prompt(
            row=row,
            cfg=prompt_inspection_cfg,
            printed_count=printed_prompt_count,
            printed_signatures=printed_prompt_signatures,
        ):
            _print_gpt_payload(
                source_row=row,
                prompt_row=prompt_row,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
                cfg=prompt_inspection_cfg,
            )
            printed_prompt_count += 1
            printed_prompt_signatures.add(_modality_signature(row))

        existing_rationale = existing_rationales.get(question_id, "")
        expected_answer = clean_text(prompt_row.get("answer"))
        if existing_rationale:
            existing_error = _validation_error(
                existing_rationale,
                expected_answer,
                dict(cfg.get("response_validation") or {}),
                displayed_choices=_displayed_choices(prompt_row),
            )
            if not existing_error:
                row_out["rationale"] = existing_rationale
                stats["reused_existing"] += 1
            else:
                print(
                    f"Regenerating question_id={question_id}; existing rationale is invalid: "
                    f"{existing_error}"
                )

        if not clean_text(row_out.get("rationale")) and dry_run:
            row_out["rationale"] = ""
        elif not clean_text(row_out.get("rationale")):
            requests.append(
                RationaleRequest(
                    row_index=len(rows),
                    question_id=question_id,
                    prompt_row=prompt_row,
                    image_paths=image_paths,
                    expected_answer=expected_answer,
                )
            )

        rows.append(row_out)

    if requests and not dry_run:
        assert client is not None
        azure_cfg = dict(cfg["azure"])
        validation_cfg = dict(cfg.get("response_validation") or {})
        image_cfg = dict(cfg.get("image_inputs") or {})
        prompts_cfg = dict(cfg.get("prompts") or {})
        teacher_attempts_cfg = dict(cfg.get("teacher_attempts") or {})
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            futures = [
                executor.submit(
                    _generate_rationale_for_request,
                    request,
                    client=client,
                    deployment=deployment,
                    azure_cfg=azure_cfg,
                    validation_cfg=validation_cfg,
                    image_cfg=image_cfg,
                    prompts_cfg=prompts_cfg,
                    teacher_attempts_cfg=teacher_attempts_cfg,
                )
                for request in requests
            ]
            future_to_request = dict(zip(futures, requests, strict=True))
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Generating rationales ({max_concurrent_requests} workers)",
            ):
                request = future_to_request[future]
                try:
                    result = future.result()
                except Exception as exc:
                    stats["failed_after_all_attempts"] += 1
                    print(
                        f"Skipping question_id={request.question_id} after an unexpected worker failure: {exc}"
                    )
                    continue
                failed_attempts.extend(result.failed_attempts)
                stats["failed_attempts"] += len(result.failed_attempts)
                if not result.rationale:
                    stats["failed_after_all_attempts"] += 1
                    print(
                        f"Skipping question_id={request.question_id}; no correct validated rationale "
                        f"after {len(result.failed_attempts)} attempts."
                    )
                    continue
                rows[result.row_index]["rationale"] = result.rationale
                stats["generated"] += 1
                if stats["generated"] <= print_first_n_outputs:
                    print("-" * 80)
                    print(f"question_id: {request.question_id}")
                    print(result.rationale)
                if save_every_n > 0 and stats["generated"] % save_every_n == 0:
                    _write_output(_rationale_output_rows(rows), output_path)
                    _write_failed_attempts(failed_attempts, failed_attempts_path)

    print(f"VQA path: {vqa_path}")
    print(f"Registry path: {registry_path}")
    print(f"Output path: {output_path}")
    print(f"Failed attempts path: {failed_attempts_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    output_rows = _rationale_output_rows(rows)
    print(f"output_rows: {len(output_rows)}")

    if dry_run:
        print("Dry run enabled; no output parquet was written.")
        return
    _write_failed_attempts(failed_attempts, failed_attempts_path)
    if not output_rows:
        print("No rationale rows available; output parquet was not written.")
        return
    _write_output(output_rows, output_path)


if __name__ == "__main__":
    main()
