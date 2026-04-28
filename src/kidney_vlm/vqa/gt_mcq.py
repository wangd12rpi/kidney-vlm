from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from kidney_vlm.vqa.genomics_text_summary import (
    build_dnam_text_summary,
    build_rna_text_summary,
)
from kidney_vlm.vqa.schema import (
    ANSWER_LABELS,
    OPTION_COLUMNS,
    empty_vqa_frame,
    normalize_vqa_df,
    validate_vqa_df,
)

DEFAULT_SKIP_VALUES = {
    "",
    "none",
    "nan",
    "na",
    "n/a",
    "null",
    "unknown",
    "not reported",
    "not available",
}

MODALITY_MUST_HAVE = "must_have"
MODALITY_USE_IF_AVAIL = "use_if_avail"
MODALITY_NOT_INCLUDE = "not_include"
MODALITY_REQUIREMENTS = {
    MODALITY_MUST_HAVE,
    MODALITY_USE_IF_AVAIL,
    MODALITY_NOT_INCLUDE,
}
MODALITY_KEYS = ("pathology", "radiology", "dnam", "rna")

TCGA_PATIENT_BARCODE_PATTERN = re.compile(
    r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$", re.IGNORECASE
)


def stable_int_id(*parts: object) -> int:
    text = "||".join(str(part).strip() for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)
    return value or 1


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return [text] if text else []


def _clean_text(value: Any) -> str:
    values = _as_list(value)
    return values[0] if values else ""


def _unique_text(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _config_list(
    cfg: Mapping[str, Any], key: str, default: Sequence[Any] | None = None
) -> list[Any]:
    value = cfg.get(key, default or [])
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _registry_records(
    registry_df: pd.DataFrame,
    *,
    global_cfg: Mapping[str, Any],
    desc: str,
) -> list[dict[str, Any]]:
    records = registry_df.to_dict(orient="records")
    if not bool(global_cfg.get("show_progress", False)):
        return records
    return list(tqdm(records, total=len(records), desc=desc, unit="case"))


def _skip_values(
    global_cfg: Mapping[str, Any], task_cfg: Mapping[str, Any]
) -> set[str]:
    values = set(DEFAULT_SKIP_VALUES)
    values.update(
        str(value).strip().casefold()
        for value in _config_list(global_cfg, "skip_values")
    )
    values.update(
        str(value).strip().casefold() for value in _config_list(task_cfg, "skip_values")
    )
    return values


def _is_skipped_value(value: str, skip_values: set[str]) -> bool:
    return str(value).strip().casefold() in skip_values


def _map_value(value: str, task_cfg: Mapping[str, Any]) -> str:
    text = str(value).strip()
    value_map = task_cfg.get("value_map") or {}
    if text in value_map:
        return str(value_map[text]).strip()
    lower_map = {
        str(key).strip().casefold(): str(mapped).strip()
        for key, mapped in dict(value_map).items()
    }
    return lower_map.get(text.casefold(), text)


def _choice_count(task_cfg: Mapping[str, Any], global_cfg: Mapping[str, Any]) -> int:
    value = task_cfg.get("choice_count", global_cfg.get("choice_count", 4))
    count = int(value)
    if count < 2 or count > 4:
        raise ValueError(f"choice_count must be between 2 and 4, got {count}")
    return count


def _minimum_semantic_questions(
    global_cfg: Mapping[str, Any], task_cfg: Mapping[str, Any]
) -> int:
    value = task_cfg.get(
        "min_semantic_questions_per_project_task",
        global_cfg.get("min_semantic_questions_per_project_task", 0),
    )
    if value is None or str(value).strip() == "":
        return 0
    count = int(value)
    if count < 0:
        raise ValueError(
            f"min_semantic_questions_per_project_task must be non-negative, got {count}"
        )
    return count


def _sampling_cfg(global_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    value = global_cfg.get("sampling") or {}
    if not isinstance(value, Mapping):
        raise ValueError("VQA GT MCQ sampling config must be a mapping.")
    return value


def _sampling_splits(sampling_cfg: Mapping[str, Any]) -> set[str]:
    values = {
        str(value).strip().casefold()
        for value in _config_list(sampling_cfg, "splits", ["train"])
        if str(value).strip()
    }
    return values or {"train"}


def _validate_keep_ratio(value: Any, *, label: str) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {ratio}")
    return ratio


def _task_keep_ratio(
    *,
    task_id: str,
    task_category: str,
    sampling_cfg: Mapping[str, Any],
) -> float:
    task_id = str(task_id).strip()
    task_category = str(task_category).strip()
    task_id_ratios = dict(sampling_cfg.get("task_id_keep_ratios") or {})
    if task_id in task_id_ratios:
        return _validate_keep_ratio(
            task_id_ratios[task_id],
            label=f"task_id_keep_ratios.{task_id}",
        )
    task_category_ratios = dict(sampling_cfg.get("task_category_keep_ratios") or {})
    if task_category in task_category_ratios:
        return _validate_keep_ratio(
            task_category_ratios[task_category],
            label=f"task_category_keep_ratios.{task_category}",
        )
    return _validate_keep_ratio(
        sampling_cfg.get("default_keep_ratio", 1.0),
        label="default_keep_ratio",
    )


def _sample_generated_frame(
    frame: pd.DataFrame,
    *,
    global_cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    stats: dict[str, int | bool] = {
        "enabled": False,
        "pre_sampling_rows": int(len(frame)),
        "pre_sampling_semantic_questions": (
            int(frame["base_question_id"].nunique()) if not frame.empty else 0
        ),
        "sampled_out_rows": 0,
        "sampled_out_semantic_questions": 0,
        "sampling_protected_radiology_questions": 0,
    }
    sampling_cfg = _sampling_cfg(global_cfg)
    if frame.empty or not bool(sampling_cfg.get("enabled", False)):
        return frame, stats

    stats["enabled"] = True
    splits = _sampling_splits(sampling_cfg)
    protect_radiology = bool(sampling_cfg.get("protect_radiology_questions", True))
    seed = int(sampling_cfg.get("seed", 42))
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    keep_base_question_ids: set[int] = set()
    row_counts_by_base_id: dict[int, int] = {}

    for base_question_id, group in frame.groupby("base_question_id", sort=False):
        base_id = int(base_question_id)
        row_counts_by_base_id[base_id] = int(len(group))
        first = group.iloc[0]
        split = _clean_text(first.get("split")).casefold()
        if split not in splits:
            keep_base_question_ids.add(base_id)
            continue
        if protect_radiology and bool(group["use_radiology"].astype(bool).any()):
            keep_base_question_ids.add(base_id)
            stats["sampling_protected_radiology_questions"] += 1
            continue
        task_category = _clean_text(first.get("task_category"))
        task_id = _clean_text(first.get("task_id"))
        ratio = _task_keep_ratio(
            task_id=task_id,
            task_category=task_category,
            sampling_cfg=sampling_cfg,
        )
        if ratio >= 1.0:
            keep_base_question_ids.add(base_id)
            continue
        grouped.setdefault(
            (
                split,
                task_category,
                task_id,
            ),
            [],
        ).append((base_id, row_counts_by_base_id[base_id]))

    for group_key, group_records in grouped.items():
        _, task_category, task_id = group_key
        ratio = _task_keep_ratio(
            task_id=task_id,
            task_category=task_category,
            sampling_cfg=sampling_cfg,
        )
        keep_count = int(math.ceil(len(group_records) * ratio)) if ratio > 0 else 0
        if keep_count >= len(group_records):
            keep_base_question_ids.update(base_id for base_id, _ in group_records)
            continue
        rng = random.Random(stable_int_id("semantic_record_sample", seed, *group_key))
        selected_indices = set(rng.sample(range(len(group_records)), keep_count))
        for index, (base_id, row_count) in enumerate(group_records):
            if index in selected_indices:
                keep_base_question_ids.add(base_id)
            else:
                stats["sampled_out_semantic_questions"] += 1
                stats["sampled_out_rows"] += row_count

    sampled = frame[frame["base_question_id"].isin(keep_base_question_ids)].reset_index(
        drop=True
    )
    return sampled, stats


def _format_template(template: str, context: Mapping[str, Any]) -> str:
    return str(template).format(**context).strip()


def _option_columns(choices: list[str]) -> dict[str, str]:
    out = {column: "" for column in OPTION_COLUMNS}
    for column, choice in zip(OPTION_COLUMNS, choices, strict=False):
        out[column] = str(choice).strip()
    return out


def _answer_label(answer: str, choices: list[str]) -> str:
    answer_text = str(answer).strip()
    for index, choice in enumerate(choices[:4]):
        if str(choice).strip() == answer_text:
            return ANSWER_LABELS[index]
    raise ValueError(f"Answer {answer_text!r} does not match any MCQ option.")


def _fixed_options(task_cfg: Mapping[str, Any], *, task_name: str) -> list[str]:
    options = _unique_text(_config_list(task_cfg, "options"))
    if len(options) < 2 or len(options) > 4:
        raise ValueError(
            f"{task_name} must define 2-4 fixed options, got {len(options)}"
        )
    return options


def _first_feature_path(
    row: Mapping[str, Any], *, use_column: str, fallback_column: str | None = None
) -> str:
    first = _clean_text(row.get(use_column))
    if first:
        return first
    if fallback_column:
        return _clean_text(row.get(fallback_column))
    return ""


def _feature_path_list(
    row: Mapping[str, Any], *, use_column: str, fallback_column: str | None = None
) -> list[str]:
    paths = _as_list(row.get(use_column))
    if paths:
        return paths
    if fallback_column:
        return _as_list(row.get(fallback_column))
    return []


def _first_parent_dir(row: Mapping[str, Any], column: str) -> str:
    path_value = _clean_text(row.get(column))
    if not path_value:
        return ""
    return Path(path_value).parent.as_posix()


def _normalize_modality_requirement(value: Any, *, key: str) -> str:
    text = str(value).strip()
    if text in MODALITY_REQUIREMENTS:
        return text
    raise ValueError(
        f"Unsupported modality requirement for {key}: {value!r}. "
        f"Expected one of: {sorted(MODALITY_REQUIREMENTS)}."
    )


def _resolve_modality_combinations(
    global_cfg: Mapping[str, Any], task_cfg: Mapping[str, Any]
) -> list[dict[str, str]]:
    raw_combinations = _config_list(
        task_cfg, "modality_combination_overrides"
    ) or _config_list(
        global_cfg,
        "default_modality_combinations",
    )
    combinations: list[dict[str, str]] = []
    for raw_combination in raw_combinations:
        combination = dict(raw_combination or {})
        name = _clean_text(combination.get("name"))
        if not name:
            raise ValueError("Each modality combination must define a non-empty name.")
        normalized_combination = {"name": name}
        normalized_combination.update(
            {
                key: _normalize_modality_requirement(
                    combination.get(f"use_{key}"), key=key
                )
                for key in MODALITY_KEYS
            }
        )
        combinations.append(normalized_combination)
    if not combinations:
        raise ValueError("At least one modality combination must be configured.")
    names = [combination["name"] for combination in combinations]
    if len(names) != len(set(names)):
        raise ValueError(f"Modality combination names must be unique: {names}")
    return combinations


def _candidate_variant_for_combination(combination: Mapping[str, str]) -> dict[str, bool]:
    return {
        f"use_{key}": str(combination.get(key, MODALITY_NOT_INCLUDE))
        != MODALITY_NOT_INCLUDE
        for key in MODALITY_KEYS
    }


def _paths_present(paths: Mapping[str, object], key: str) -> bool:
    value = (
        paths[f"{key}_feature_paths"]
        if key in {"pathology", "radiology"}
        else paths[f"{key}_feature_path"]
    )
    return bool(value)


def _effective_variant_for_combination(
    *,
    combination: Mapping[str, str],
    paths: Mapping[str, object],
) -> dict[str, bool] | None:
    variant: dict[str, bool] = {}
    for key in MODALITY_KEYS:
        requirement = str(combination.get(key, MODALITY_NOT_INCLUDE))
        present = _paths_present(paths, key)
        if requirement == MODALITY_NOT_INCLUDE:
            variant[f"use_{key}"] = False
            continue
        if requirement == MODALITY_MUST_HAVE and not present:
            return None
        variant[f"use_{key}"] = present

    if not any(variant.values()):
        return None
    return variant


def _feature_paths_for_variant(
    row: Mapping[str, Any], variant: Mapping[str, bool]
) -> dict[str, object]:
    return {
        "pathology_feature_paths": (
            _feature_path_list(
                row,
                use_column="pathology_tile_embedding_paths",
                fallback_column="pathology_slide_embedding_paths",
            )
            if variant["use_pathology"]
            else []
        ),
        "radiology_feature_paths": (
            _as_list(row.get("radiology_embedding_paths"))
            if variant["use_radiology"]
            else []
        ),
        "dnam_feature_path": (
            _first_feature_path(row, use_column="genomics_dna_methylation_feature_path")
            if variant["use_dnam"]
            else ""
        ),
        "rna_feature_path": (
            _first_feature_path(row, use_column="genomics_rna_bulk_feature_path")
            if variant["use_rna"]
            else ""
        ),
    }


def _artifact_paths_for_variant(
    row: Mapping[str, Any], variant: Mapping[str, bool], global_cfg: Mapping[str, Any]
) -> dict[str, str]:
    if str(row.get("split", "")).strip().lower() != "test":
        return {
            "pathology_roi_png_dir": "",
            "radiology_view_png_dir": "",
            "dnam_text_summary": "",
            "rna_text_summary": "",
        }
    populate_genomics_text = bool(
        global_cfg.get("populate_test_genomics_text_summaries", False)
    )
    mutation_panel = _mutation_panel_for_project(global_cfg, _project_id(row))
    dnam_summary = ""
    rna_summary = ""
    if populate_genomics_text and variant["use_dnam"]:
        dnam_summary = build_dnam_text_summary(
            row,
            max_beta_values=int(
                global_cfg.get("dnam_text_summary_max_beta_values", 50_000) or 50_000
            ),
            panel_genes=mutation_panel,
        )
    if populate_genomics_text and variant["use_rna"]:
        rna_summary = build_rna_text_summary(
            row,
            max_top_genes=int(global_cfg.get("rna_text_summary_max_top_genes", 8) or 8),
            panel_genes=mutation_panel,
        )
    return {
        "pathology_roi_png_dir": _first_parent_dir(row, "pathology_png_roi_paths")
        if variant["use_pathology"]
        else "",
        "radiology_view_png_dir": _clean_text(row.get("radiology_png_dirs"))
        if variant["use_radiology"]
        else "",
        "dnam_text_summary": dnam_summary,
        "rna_text_summary": rna_summary,
    }


def _required_test_artifacts_are_present(
    *,
    row: Mapping[str, Any],
    variant: Mapping[str, bool],
    artifacts: Mapping[str, str],
    global_cfg: Mapping[str, Any],
) -> bool:
    if str(row.get("split", "")).strip().lower() != "test":
        return True
    if bool(global_cfg.get("require_test_pathology_roi_png_dir", False)):
        if (
            variant["use_pathology"]
            and not str(artifacts.get("pathology_roi_png_dir", "")).strip()
        ):
            return False
    if bool(global_cfg.get("require_test_genomics_text_summaries", False)):
        if (
            variant["use_dnam"]
            and not str(artifacts.get("dnam_text_summary", "")).strip()
        ):
            return False
        if (
            variant["use_rna"]
            and not str(artifacts.get("rna_text_summary", "")).strip()
        ):
            return False
    return True


def _mutation_panel_for_project(
    global_cfg: Mapping[str, Any],
    project_id: str,
) -> list[str]:
    genes: list[str] = []
    for task_cfg in _config_list(global_cfg, "boolean_tasks"):
        task = dict(task_cfg or {})
        if not bool(task.get("enabled", True)):
            continue
        panel_by_project = dict(task.get("gene_panel_by_project") or {})
        for gene in _config_list({"genes": panel_by_project.get(project_id)}, "genes"):
            gene_text = str(gene).strip().upper()
            if gene_text and gene_text not in genes:
                genes.append(gene_text)
    return genes


def _case_id(row: Mapping[str, Any]) -> str:
    explicit_case_id = _clean_text(row.get("case_id"))
    if explicit_case_id:
        return explicit_case_id

    patient_id = _clean_text(row.get("patient_id"))
    sample_id = _clean_text(row.get("sample_id"))
    if TCGA_PATIENT_BARCODE_PATTERN.match(patient_id):
        return patient_id
    if TCGA_PATIENT_BARCODE_PATTERN.match(sample_id):
        return sample_id
    return patient_id or sample_id


def _project_id(row: Mapping[str, Any]) -> str:
    return _clean_text(row.get("project_id"))


def _base_context(
    row: Mapping[str, Any],
    task_cfg: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = {
        "case_id": _case_id(row),
        "project_id": _project_id(row),
        "task_id": str(task_cfg.get("task_id", "")).strip(),
        "task_category": str(task_cfg.get("task_category", "")).strip(),
    }
    if extra:
        context.update(extra)
    return context


def _build_vqa_rows_for_semantic_question(
    *,
    row: Mapping[str, Any],
    global_cfg: Mapping[str, Any],
    task_cfg: Mapping[str, Any],
    question: str,
    answer: str,
    choices: list[str],
    ground_truth_source: str,
    base_id_seed_parts: Sequence[object],
) -> list[dict[str, Any]]:
    case_id = _case_id(row)
    project_id = _project_id(row)
    if not case_id or not project_id:
        return []

    base_question_id = stable_int_id("base_question", *base_id_seed_parts)
    output_rows: list[dict[str, Any]] = []
    seen_question_ids: set[int] = set()
    for combination in _resolve_modality_combinations(global_cfg, task_cfg):
        combination_name = combination["name"]
        candidate_variant = _candidate_variant_for_combination(combination)
        candidate_feature_paths = _feature_paths_for_variant(row, candidate_variant)
        variant = _effective_variant_for_combination(
            combination=combination,
            paths=candidate_feature_paths,
        )
        if variant is None:
            continue
        feature_paths = _feature_paths_for_variant(row, variant)
        artifacts = _artifact_paths_for_variant(row, variant, global_cfg)
        if not _required_test_artifacts_are_present(
            row=row,
            variant=variant,
            artifacts=artifacts,
            global_cfg=global_cfg,
        ):
            continue

        question_id = stable_int_id(
            "question",
            base_question_id,
            combination_name,
            variant["use_pathology"],
            variant["use_radiology"],
            variant["use_dnam"],
            variant["use_rna"],
        )
        if question_id in seen_question_ids:
            continue
        seen_question_ids.add(question_id)
        output_rows.append(
            {
                "case_id": case_id,
                "project_id": project_id,
                "question_id": question_id,
                "base_question_id": base_question_id,
                "split": _clean_text(row.get("split")),
                "question_type": "mcq",
                "generation_type": "from_ground_truth",
                "task_category": str(task_cfg.get("task_category", "")).strip(),
                "task_id": str(task_cfg.get("task_id", "")).strip(),
                "modality_combination_name": combination_name,
                "use_pathology": variant["use_pathology"],
                "use_radiology": variant["use_radiology"],
                "use_dnam": variant["use_dnam"],
                "use_rna": variant["use_rna"],
                "question": question,
                **_option_columns(choices),
                "answer": str(answer).strip(),
                "answer_label": _answer_label(str(answer), choices),
                "caption_id": "",
                "ground_truth_source": str(ground_truth_source).strip(),
                "radiology_biomarker": _clean_text(row.get("radiology_biomarker"))
                if variant["use_radiology"]
                else "",
                **feature_paths,
                **artifacts,
            }
        )
    return output_rows


def _semantic_project_task_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _project_id(record["row"]),
        str(record["task_cfg"].get("task_id", "")).strip(),
    )


def _filter_semantic_records_by_minimum(
    records: list[dict[str, Any]],
    *,
    minimum_count: int,
) -> tuple[list[dict[str, Any]], int]:
    if minimum_count <= 1:
        return records, 0

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_semantic_project_task_key(record), []).append(record)

    kept: list[dict[str, Any]] = []
    skipped = 0
    for group_records in grouped.values():
        if len(group_records) < minimum_count:
            skipped += len(group_records)
            continue
        kept.extend(group_records)
    return kept, skipped


def _expand_semantic_records(
    records: list[dict[str, Any]], *, global_cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    generated_rows: list[dict[str, Any]] = []
    for record in records:
        generated_rows.extend(
            _build_vqa_rows_for_semantic_question(
                row=record["row"],
                global_cfg=global_cfg,
                task_cfg=record["task_cfg"],
                question=record["question"],
                answer=record["answer"],
                choices=record["choices"],
                ground_truth_source=record["ground_truth_source"],
                base_id_seed_parts=record["base_id_seed_parts"],
            )
        )
    return generated_rows


def _build_categorical_task_rows(
    *,
    registry_df: pd.DataFrame,
    task_cfg: Mapping[str, Any],
    global_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_column = str(task_cfg.get("source_column", "")).strip()
    if not source_column:
        raise ValueError("Categorical VQA task is missing source_column.")
    if source_column not in registry_df.columns:
        raise ValueError(
            f"Categorical VQA task source column not found in registry: {source_column}"
        )

    options = _fixed_options(
        task_cfg, task_name=f"Categorical VQA task {task_cfg.get('task_id')}"
    )
    option_set = set(options)
    question_template = str(task_cfg.get("question_template", "")).strip()
    if not question_template:
        raise ValueError(
            f"Categorical VQA task {task_cfg.get('task_id')} is missing question_template."
        )

    skip_values = _skip_values(global_cfg, task_cfg)
    semantic_records: list[dict[str, Any]] = []
    stats = {
        "candidate_rows": 0,
        "skipped_empty_answer": 0,
        "skipped_answer_not_in_options": 0,
        "skipped_minimum": 0,
        "generated_semantic_questions": 0,
        "generated_rows": 0,
    }
    ground_truth_source = str(
        task_cfg.get("ground_truth_source", source_column)
    ).strip()

    for row in _registry_records(
        registry_df,
        global_cfg=global_cfg,
        desc=f"Building {task_cfg.get('task_id', 'categorical')} MCQs",
    ):
        raw_answer = _clean_text(row.get(source_column))
        if not raw_answer or _is_skipped_value(raw_answer, skip_values):
            stats["skipped_empty_answer"] += 1
            continue
        answer = _map_value(raw_answer, task_cfg)
        if not answer or _is_skipped_value(answer, skip_values):
            stats["skipped_empty_answer"] += 1
            continue
        if answer not in option_set:
            stats["skipped_answer_not_in_options"] += 1
            continue

        stats["candidate_rows"] += 1
        context = _base_context(
            row,
            task_cfg,
            extra={
                "source_column": source_column,
                "answer": answer,
                "raw_answer": raw_answer,
            },
        )
        semantic_records.append(
            {
                "row": row,
                "task_cfg": dict(task_cfg),
                "question": _format_template(question_template, context),
                "answer": answer,
                "choices": options,
                "ground_truth_source": ground_truth_source,
                "base_id_seed_parts": [
                    context["case_id"],
                    task_cfg.get("task_id"),
                    source_column,
                ],
            }
        )

    semantic_records, skipped_minimum = _filter_semantic_records_by_minimum(
        semantic_records,
        minimum_count=_minimum_semantic_questions(global_cfg, task_cfg),
    )
    stats["skipped_minimum"] = skipped_minimum
    stats["generated_semantic_questions"] = len(semantic_records)
    generated_rows = _expand_semantic_records(semantic_records, global_cfg=global_cfg)
    stats["generated_rows"] = len(generated_rows)
    return generated_rows, stats


def _bool_value(value: Any, task_cfg: Mapping[str, Any]) -> bool | None:
    if isinstance(value, bool):
        return value
    if _clean_text(value).casefold() in DEFAULT_SKIP_VALUES:
        return None
    true_values = {
        str(v).strip().casefold()
        for v in _config_list(task_cfg, "true_values", ["true", "1", "yes"])
    }
    false_values = {
        str(v).strip().casefold()
        for v in _config_list(task_cfg, "false_values", ["false", "0", "no"])
    }
    text = _clean_text(value).casefold()
    if text in true_values:
        return True
    if text in false_values:
        return False
    return None


def _gene_source_column(gene: str) -> str:
    return f"mutation_{str(gene).strip().lower()}"


def _display_label_from_source_column(
    source_column: str, task_cfg: Mapping[str, Any]
) -> str:
    label_map = task_cfg.get("label_map") or {}
    if source_column in label_map:
        return str(label_map[source_column]).strip()
    if source_column.startswith("mutation_"):
        return source_column.removeprefix("mutation_").upper()
    return source_column


def _boolean_sources_by_project(
    *,
    registry_df: pd.DataFrame,
    task_cfg: Mapping[str, Any],
) -> dict[str, list[tuple[str, str]]]:
    gene_panel_by_project = task_cfg.get("gene_panel_by_project") or {}
    if gene_panel_by_project:
        out: dict[str, list[tuple[str, str]]] = {}
        missing_source_columns: set[str] = set()
        for project_id, genes in dict(gene_panel_by_project).items():
            project_sources: list[tuple[str, str]] = []
            for gene in _config_list({"genes": genes}, "genes"):
                label = str(gene).strip()
                if not label:
                    continue
                source_column = _gene_source_column(label)
                if source_column not in registry_df.columns:
                    missing_source_columns.add(source_column)
                    continue
                project_sources.append((source_column, label))
            out[str(project_id).strip()] = project_sources
        if missing_source_columns:
            raise ValueError(
                f"Boolean VQA task source columns not found in registry: {sorted(missing_source_columns)}"
            )
        if not any(out.values()):
            raise ValueError(
                "Boolean VQA task gene_panel_by_project did not define any genes."
            )
        return out

    source_columns = [
        str(column).strip()
        for column in _config_list(task_cfg, "source_columns")
        if str(column).strip()
    ]
    if not source_columns:
        raise ValueError(
            "Boolean VQA task is missing source_columns or gene_panel_by_project."
        )
    missing_source_columns = [
        column for column in source_columns if column not in registry_df.columns
    ]
    if missing_source_columns:
        raise ValueError(
            f"Boolean VQA task source columns not found in registry: {missing_source_columns}"
        )

    projects = sorted(
        {
            _project_id(row.to_dict())
            for _, row in registry_df.iterrows()
            if _project_id(row.to_dict())
        }
    )
    return {
        project_id: [
            (source_column, _display_label_from_source_column(source_column, task_cfg))
            for source_column in source_columns
        ]
        for project_id in projects
    }


def _downsample_boolean_records(
    records: list[tuple[bool, dict[str, Any]]],
    *,
    max_false_ratio: Any,
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[bool, dict[str, Any]]]] = {}
    for value, record in records:
        grouped.setdefault(_semantic_project_task_key(record), []).append(
            (value, record)
        )

    generated: list[dict[str, Any]] = []
    for group_key, group_records in grouped.items():
        train_positives = [
            record
            for value, record in group_records
            if value and _clean_text(record["row"].get("split")).casefold() == "train"
        ]
        train_negatives = [
            record
            for value, record in group_records
            if not value
            and _clean_text(record["row"].get("split")).casefold() == "train"
        ]
        non_train_records = [
            record
            for _, record in group_records
            if _clean_text(record["row"].get("split")).casefold() != "train"
        ]
        if max_false_ratio is not None and train_positives:
            max_false = int(math.ceil(len(train_positives) * float(max_false_ratio)))
            if len(train_negatives) > max_false:
                rng = random.Random(
                    stable_int_id("boolean_false_downsample_train", *group_key)
                )
                stats["downsampled_false"] += len(train_negatives) - max_false
                train_negatives = rng.sample(train_negatives, max_false)
        generated.extend(train_positives)
        generated.extend(train_negatives)
        generated.extend(non_train_records)
    return generated


def _build_boolean_task_rows(
    *,
    registry_df: pd.DataFrame,
    task_cfg: Mapping[str, Any],
    global_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sources_by_project = _boolean_sources_by_project(
        registry_df=registry_df, task_cfg=task_cfg
    )
    question_template = str(task_cfg.get("question_template", "")).strip()
    if not question_template:
        raise ValueError("Boolean VQA task is missing question_template.")
    true_answer_template = str(
        task_cfg.get("true_answer_template", "{label} mutation present")
    ).strip()
    false_answer_template = str(
        task_cfg.get("false_answer_template", "{label} mutation absent")
    ).strip()
    task_id_template = str(task_cfg.get("task_id_template", "{source_column}")).strip()
    choice_count = _choice_count(task_cfg, global_cfg)
    if choice_count != 2:
        raise ValueError("Boolean VQA tasks currently require choice_count: 2.")

    candidate_records: list[tuple[bool, dict[str, Any]]] = []
    stats = {
        "candidate_rows": 0,
        "skipped_empty_answer": 0,
        "downsampled_false": 0,
        "skipped_minimum": 0,
        "generated_semantic_questions": 0,
        "generated_rows": 0,
    }
    for row in _registry_records(
        registry_df,
        global_cfg=global_cfg,
        desc=f"Building {task_cfg.get('task_category', 'boolean')} MCQs",
    ):
        project_id = _project_id(row)
        for source_column, label in sources_by_project.get(project_id, []):
            value = _bool_value(row.get(source_column), task_cfg)
            if value is None:
                stats["skipped_empty_answer"] += 1
                continue

            answer_template = true_answer_template if value else false_answer_template
            context = _base_context(
                row,
                task_cfg,
                extra={
                    "source_column": source_column,
                    "label": label,
                    "gene": label,
                    "answer": _format_template(
                        answer_template, {"label": label, "gene": label}
                    ),
                },
            )
            task_id = _format_template(
                task_id_template,
                {"source_column": source_column, "label": label, "gene": label},
            )
            task_row_cfg = dict(task_cfg)
            task_row_cfg["task_id"] = task_id
            answer = str(context["answer"]).strip()
            choices = [
                _format_template(true_answer_template, {"label": label, "gene": label}),
                _format_template(
                    false_answer_template, {"label": label, "gene": label}
                ),
            ]
            stats["candidate_rows"] += 1
            candidate_records.append(
                (
                    value,
                    {
                        "row": row,
                        "task_cfg": task_row_cfg,
                        "question": _format_template(
                            question_template, {**context, "task_id": task_id}
                        ),
                        "answer": answer,
                        "choices": choices,
                        "ground_truth_source": source_column,
                        "base_id_seed_parts": [
                            context["case_id"],
                            task_id,
                            source_column,
                        ],
                    },
                )
            )

    max_false_ratio = task_cfg.get("max_false_per_true", None)
    semantic_records = _downsample_boolean_records(
        candidate_records, max_false_ratio=max_false_ratio, stats=stats
    )
    semantic_records, skipped_minimum = _filter_semantic_records_by_minimum(
        semantic_records,
        minimum_count=_minimum_semantic_questions(global_cfg, task_cfg),
    )
    stats["skipped_minimum"] = skipped_minimum
    stats["generated_semantic_questions"] = len(semantic_records)
    generated_rows = _expand_semantic_records(semantic_records, global_cfg=global_cfg)
    stats["generated_rows"] = len(generated_rows)
    return generated_rows, stats


def _selected_registry_frame(
    registry_df: pd.DataFrame, cfg: Mapping[str, Any]
) -> pd.DataFrame:
    out = registry_df.copy()
    allowed_project_ids = [
        str(value).strip()
        for value in _config_list(cfg, "allowed_project_ids")
        if str(value).strip()
    ]
    if allowed_project_ids and "project_id" in out.columns:
        out = out[out["project_id"].astype(str).isin(allowed_project_ids)]
    allowed_splits = [
        str(value).strip()
        for value in _config_list(cfg, "allowed_splits")
        if str(value).strip()
    ]
    if allowed_splits and "split" in out.columns:
        out = out[out["split"].astype(str).isin(allowed_splits)]
    first_n = cfg.get("first_n")
    if first_n is not None and str(first_n).strip():
        out = out.head(int(first_n))
    return out.reset_index(drop=True)


def build_ground_truth_mcq_frame(
    registry_df: pd.DataFrame, cfg: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = _selected_registry_frame(registry_df, cfg)
    generated_rows: list[dict[str, Any]] = []
    task_stats: dict[str, dict[str, int]] = {}

    for task_cfg in _config_list(cfg, "categorical_tasks"):
        task = dict(task_cfg or {})
        if not bool(task.get("enabled", True)):
            continue
        rows, stats = _build_categorical_task_rows(
            registry_df=selected, task_cfg=task, global_cfg=cfg
        )
        generated_rows.extend(rows)
        task_stats[
            str(task.get("task_id", task.get("source_column", "categorical")))
        ] = stats

    for task_cfg in _config_list(cfg, "boolean_tasks"):
        task = dict(task_cfg or {})
        if not bool(task.get("enabled", True)):
            continue
        rows, stats = _build_boolean_task_rows(
            registry_df=selected, task_cfg=task, global_cfg=cfg
        )
        generated_rows.extend(rows)
        task_stats[str(task.get("task_category", "boolean"))] = stats

    frame = normalize_vqa_df(
        pd.DataFrame(generated_rows) if generated_rows else empty_vqa_frame()
    )
    if not frame.empty:
        frame = frame.sort_values(
            ["split", "project_id", "case_id", "base_question_id", "question_id"]
        ).reset_index(drop=True)
    frame, sampling_stats = _sample_generated_frame(frame, global_cfg=cfg)
    if not frame.empty:
        frame = frame.sort_values(
            ["split", "project_id", "case_id", "base_question_id", "question_id"]
        ).reset_index(drop=True)
    validate_vqa_df(frame)
    return frame, {
        "registry_rows_selected": int(len(selected)),
        "generated_rows": int(len(frame)),
        "semantic_questions": int(frame["base_question_id"].nunique())
        if not frame.empty
        else 0,
        "sampling": sampling_stats,
        "task_stats": task_stats,
    }
