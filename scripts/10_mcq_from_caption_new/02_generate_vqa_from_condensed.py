#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import json
import os
import random
import sys
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
from kidney_vlm.vqa.eval_gpt import _patch_bert_score_tokenizer_max_length
from kidney_vlm.vqa.genomics_text_summary import build_dnam_text_summary, build_rna_text_summary
from kidney_vlm.vqa.gt_mcq import (
    _as_list,
    _clean_text,
    _feature_path_list,
    _first_feature_path,
    _first_parent_dir,
    _radiology_png_artifact_value,
    stable_int_id,
)
from kidney_vlm.vqa.schema import ANSWER_LABELS, OPTION_COLUMNS, VQA_COLUMNS, write_vqa_parquet

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

SECTION_COLUMNS = {
    "radiology_findings",
    "pathology_findings",
    "genomic_findings",
    "integrated_interpretation",
}


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="10_mcq_from_caption_new/generate_vqa_from_condensed.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_required_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def _validate_condensed_frame(df: pd.DataFrame) -> None:
    required = {"case_id", "project_id", *SECTION_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Condensed caption parquet is missing columns: {missing}")
    if df["case_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Condensed caption parquet has empty case_id values.")
    if df["case_id"].duplicated().any():
        duplicated = df.loc[df["case_id"].duplicated(), "case_id"].head(5).tolist()
        raise ValueError(f"Condensed caption parquet has duplicated case_id values: {duplicated}")


def _validate_registry_frame(df: pd.DataFrame) -> None:
    required = {
        "patient_id",
        "project_id",
        "split",
        "pathology_tile_embedding_paths",
        "pathology_slide_embedding_paths",
        "radiology_embedding_paths",
        "genomics_dna_methylation_feature_path",
        "genomics_rna_bulk_feature_path",
        "pathology_png_roi_paths",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Unified registry is missing columns: {missing}")
    if df["patient_id"].duplicated().any():
        duplicated = df.loc[df["patient_id"].duplicated(), "patient_id"].head(5).tolist()
        raise ValueError(f"Unified registry has duplicated patient_id values: {duplicated}")


def _enabled_tasks(cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for raw_task in list(cfg.get("tasks") or []):
        task = dict(raw_task or {})
        if not bool(task.get("enabled", True)):
            continue
        section_key = _clean_text(task.get("section_key"))
        if section_key not in SECTION_COLUMNS:
            raise ValueError(f"Unsupported condensed section_key: {section_key!r}")
        for key in ["task_category", "task_id", "question"]:
            if not _clean_text(task.get(key)):
                raise ValueError(f"Caption MCQ task for {section_key} is missing {key}.")
        tasks.append(
            {
                "section_key": section_key,
                "task_category": _clean_text(task.get("task_category")),
                "task_id": _clean_text(task.get("task_id")),
                "question": _clean_text(task.get("question")),
            }
        )
    if not tasks:
        raise ValueError("At least one enabled caption MCQ task is required.")
    return tasks


def _option_columns(choices: Sequence[str]) -> dict[str, str]:
    values = [str(choice).strip() for choice in choices]
    if len(values) != 4:
        raise ValueError(f"Caption MCQ rows require exactly 4 options, got {len(values)}.")
    return {column: values[index] for index, column in enumerate(OPTION_COLUMNS)}


def _answer_label(answer: str, choices: Sequence[str]) -> str:
    answer_text = str(answer).strip()
    for index, choice in enumerate(choices):
        if str(choice).strip() == answer_text:
            return ANSWER_LABELS[index]
    raise ValueError(f"Answer {answer_text!r} does not match any option.")


def _unique_nonempty_texts(values: Sequence[Any]) -> list[str]:
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


def _sample_candidate_distractors(
    *,
    condensed_df: pd.DataFrame,
    target_case_id: str,
    target_project_id: str,
    section_key: str,
    correct_answer: str,
    pool_mode: str,
    candidate_pull_count: int,
    seed: int,
) -> list[str]:
    if candidate_pull_count <= 0:
        raise ValueError("distractors.candidate_pull_count must be positive.")
    pool = condensed_df[condensed_df["case_id"].astype(str).ne(target_case_id)].copy()
    if pool_mode == "same_project":
        pool = pool[pool["project_id"].astype(str).eq(target_project_id)].copy()
    elif pool_mode != "global":
        raise ValueError(f"Unsupported distractors.pool_mode: {pool_mode!r}")

    candidates = [
        text
        for text in _unique_nonempty_texts(pool[section_key].tolist())
        if text.casefold() != correct_answer.casefold()
    ]
    rng = random.Random(seed)
    if len(candidates) > candidate_pull_count:
        return rng.sample(candidates, candidate_pull_count)
    rng.shuffle(candidates)
    return candidates


def _bertscore_f1_pairs(correct_answer: str, candidates: Sequence[str], cfg: Mapping[str, Any]) -> list[float]:
    if not candidates:
        return []
    try:
        from bert_score import score as bert_score
    except ModuleNotFoundError as exc:
        raise RuntimeError("Caption MCQ distractor selection requires bert-score.") from exc

    max_length = int(cfg.get("max_length", 512) or 512)
    if max_length <= 0:
        raise ValueError("bert_score.max_length must be positive.")
    restore_tokenizer = _patch_bert_score_tokenizer_max_length(max_length)
    try:
        _, _, f1 = bert_score(
            list(candidates),
            [correct_answer] * len(candidates),
            model_type=str(cfg.get("model_type", "roberta-large")),
            num_layers=int(cfg["num_layers"]) if cfg.get("num_layers") is not None else None,
            lang=str(cfg.get("lang", "en")),
            batch_size=int(cfg.get("batch_size", 8) or 8),
            rescale_with_baseline=bool(cfg.get("rescale_with_baseline", False)),
            use_fast_tokenizer=bool(cfg.get("use_fast_tokenizer", False)),
            verbose=False,
        )
    finally:
        restore_tokenizer()
    return [float(value) for value in f1.detach().cpu().tolist()]


def _select_wrong_options(
    *,
    correct_answer: str,
    candidates: Sequence[str],
    cfg: Mapping[str, Any],
    required_count: int,
    use_bertscore: bool,
) -> list[str]:
    if len(candidates) < required_count:
        return []
    if not use_bertscore:
        return list(candidates[:required_count])
    scores = _bertscore_f1_pairs(correct_answer, candidates, cfg)
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda item: item[1])
    return [candidate for candidate, _ in ranked[:required_count]]


def _shuffled_options(
    *,
    correct_answer: str,
    wrong_options: Sequence[str],
    seed: int,
) -> list[str]:
    choices = [correct_answer, *wrong_options]
    if len(_unique_nonempty_texts(choices)) != len(choices):
        raise ValueError(f"Duplicate option text for answer {correct_answer!r}: {choices}")
    rng = random.Random(seed)
    rng.shuffle(choices)
    return choices


def _feature_paths_for_case(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pathology_feature_paths": _feature_path_list(
            row,
            use_column="pathology_tile_embedding_paths",
            fallback_column="pathology_slide_embedding_paths",
        ),
        "radiology_feature_paths": _as_list(row.get("radiology_embedding_paths")),
        "dnam_feature_path": _first_feature_path(
            row,
            use_column="genomics_dna_methylation_feature_path",
        ),
        "rna_feature_path": _first_feature_path(
            row,
            use_column="genomics_rna_bulk_feature_path",
        ),
    }


def _modality_variants(feature_paths: Mapping[str, Any]) -> list[dict[str, bool | str]]:
    available = {
        "use_pathology": bool(feature_paths["pathology_feature_paths"]),
        "use_radiology": bool(feature_paths["radiology_feature_paths"]),
        "use_dnam": bool(feature_paths["dnam_feature_path"]),
        "use_rna": bool(feature_paths["rna_feature_path"]),
    }
    variants: list[dict[str, bool | str]] = []
    if any(available.values()):
        variants.append({"modality_combination_name": "all_available", **available})
    if available["use_pathology"]:
        variants.append(
            {
                "modality_combination_name": "path_only",
                "use_pathology": True,
                "use_radiology": False,
                "use_dnam": False,
                "use_rna": False,
            }
        )
    if available["use_radiology"]:
        variants.append(
            {
                "modality_combination_name": "radiology_only",
                "use_pathology": False,
                "use_radiology": True,
                "use_dnam": False,
                "use_rna": False,
            }
        )
    return variants


def _feature_paths_for_variant(
    feature_paths: Mapping[str, Any],
    variant: Mapping[str, bool | str],
) -> dict[str, Any]:
    return {
        "pathology_feature_paths": feature_paths["pathology_feature_paths"] if variant["use_pathology"] else [],
        "radiology_feature_paths": feature_paths["radiology_feature_paths"] if variant["use_radiology"] else [],
        "dnam_feature_path": feature_paths["dnam_feature_path"] if variant["use_dnam"] else "",
        "rna_feature_path": feature_paths["rna_feature_path"] if variant["use_rna"] else "",
    }


def _artifact_paths_for_variant(
    row: Mapping[str, Any],
    variant: Mapping[str, bool | str],
    cfg: Mapping[str, Any],
) -> dict[str, str]:
    if _clean_text(row.get("split")).lower() != "test":
        return {
            "pathology_roi_png_dir": "",
            "radiology_view_png_dir": "",
            "dnam_text_summary": "",
            "rna_text_summary": "",
        }

    dnam_summary = ""
    rna_summary = ""
    if bool(cfg.get("populate_test_genomics_text_summaries", True)) and variant["use_dnam"]:
        dnam_summary = build_dnam_text_summary(
            row,
            max_beta_values=int(cfg.get("dnam_text_summary_max_beta_values", 50_000) or 50_000),
        )
    if bool(cfg.get("populate_test_genomics_text_summaries", True)) and variant["use_rna"]:
        rna_summary = build_rna_text_summary(
            row,
            max_top_genes=int(cfg.get("rna_text_summary_max_top_genes", 8) or 8),
        )

    return {
        "pathology_roi_png_dir": _first_parent_dir(row, "pathology_png_roi_paths")
        if variant["use_pathology"]
        else "",
        "radiology_view_png_dir": _radiology_png_artifact_value(row, cfg)
        if variant["use_radiology"]
        else "",
        "dnam_text_summary": dnam_summary,
        "rna_text_summary": rna_summary,
    }


def _required_test_artifacts_present(
    *,
    row: Mapping[str, Any],
    variant: Mapping[str, bool | str],
    artifacts: Mapping[str, str],
    cfg: Mapping[str, Any],
) -> bool:
    if _clean_text(row.get("split")).lower() != "test":
        return True
    if bool(cfg.get("require_test_pathology_roi_png_dir", True)):
        if variant["use_pathology"] and not _clean_text(artifacts.get("pathology_roi_png_dir")):
            return False
    if bool(cfg.get("require_test_radiology_view_png_dir", True)):
        if variant["use_radiology"] and not _clean_text(artifacts.get("radiology_view_png_dir")):
            return False
    if bool(cfg.get("require_test_genomics_text_summaries", True)):
        if variant["use_dnam"] and not _clean_text(artifacts.get("dnam_text_summary")):
            return False
        if variant["use_rna"] and not _clean_text(artifacts.get("rna_text_summary")):
            return False
    return True


def _row_to_vqa_records(
    *,
    condensed_row: Mapping[str, Any],
    registry_row: Mapping[str, Any],
    task: Mapping[str, str],
    choices: Sequence[str],
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    case_id = _clean_text(condensed_row.get("case_id"))
    project_id = _clean_text(condensed_row.get("project_id"))
    section_key = task["section_key"]
    correct_answer = _clean_text(condensed_row.get(section_key))
    base_question_id = stable_int_id("caption_mcq_base", case_id, section_key)
    feature_paths = _feature_paths_for_case(registry_row)
    records: list[dict[str, Any]] = []

    for variant in _modality_variants(feature_paths):
        artifacts = _artifact_paths_for_variant(registry_row, variant, cfg)
        if not _required_test_artifacts_present(
            row=registry_row,
            variant=variant,
            artifacts=artifacts,
            cfg=cfg,
        ):
            continue
        modality_name = str(variant["modality_combination_name"])
        question_id = stable_int_id(
            "caption_mcq_question",
            base_question_id,
            modality_name,
            *choices,
        )
        records.append(
            {
                "case_id": case_id,
                "project_id": project_id,
                "question_id": question_id,
                "base_question_id": base_question_id,
                "split": _clean_text(registry_row.get("split")),
                "question_type": "mcq",
                "generation_type": "from_caption",
                "task_category": task["task_category"],
                "task_id": task["task_id"],
                "modality_combination_name": modality_name,
                "use_pathology": bool(variant["use_pathology"]),
                "use_radiology": bool(variant["use_radiology"]),
                "use_dnam": bool(variant["use_dnam"]),
                "use_rna": bool(variant["use_rna"]),
                "question": task["question"],
                **_option_columns(choices),
                "answer": correct_answer,
                "answer_label": _answer_label(correct_answer, choices),
                "caption_id": case_id,
                "ground_truth_source": f"condensed_caption:{section_key}",
                "radiology_biomarker": _clean_text(registry_row.get("radiology_biomarker"))
                if variant["use_radiology"]
                else "",
                **_feature_paths_for_variant(feature_paths, variant),
                **artifacts,
            }
        )
    return records


def build_caption_condensed_mcq_frame(
    *,
    condensed_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    _validate_condensed_frame(condensed_df)
    _validate_registry_frame(registry_df)
    tasks = _enabled_tasks(cfg)
    registry_by_patient = registry_df.set_index("patient_id", drop=False)
    seed = int(cfg.get("seed", 42) or 42)
    distractor_cfg = dict(cfg.get("distractors") or {})
    pool_mode = _clean_text(distractor_cfg.get("pool_mode")) or "same_project"
    use_bertscore = bool(distractor_cfg.get("use_bertscore", True))
    candidate_pull_count = int(distractor_cfg.get("candidate_pull_count", 6) or 6)
    required_wrong_options = int(distractor_cfg.get("required_wrong_options", 3) or 3)
    if required_wrong_options != 3:
        raise ValueError("Caption MCQ VQA currently requires exactly 3 wrong options.")

    rows: list[dict[str, Any]] = []
    stats = {
        "condensed_rows": int(len(condensed_df)),
        "skipped_missing_registry": 0,
        "skipped_empty_section": 0,
        "skipped_not_enough_distractors": 0,
        "skipped_no_modality_rows": 0,
        "generated_semantic_questions": 0,
        "generated_rows": 0,
    }
    iterator = condensed_df.to_dict(orient="records")
    if bool(cfg.get("show_progress", False)):
        iterator = tqdm(iterator, desc="Building caption MCQs")

    for condensed_row in iterator:
        case_id = _clean_text(condensed_row.get("case_id"))
        project_id = _clean_text(condensed_row.get("project_id"))
        if case_id not in registry_by_patient.index:
            stats["skipped_missing_registry"] += len(tasks)
            continue
        registry_row = registry_by_patient.loc[case_id].to_dict()
        for task in tasks:
            section_key = task["section_key"]
            correct_answer = _clean_text(condensed_row.get(section_key))
            if not correct_answer:
                stats["skipped_empty_section"] += 1
                continue
            candidate_seed = stable_int_id("caption_mcq_distractors", seed, case_id, section_key)
            candidates = _sample_candidate_distractors(
                condensed_df=condensed_df,
                target_case_id=case_id,
                target_project_id=project_id,
                section_key=section_key,
                correct_answer=correct_answer,
                pool_mode=pool_mode,
                candidate_pull_count=candidate_pull_count,
                seed=candidate_seed,
            )
            wrong_options = _select_wrong_options(
                correct_answer=correct_answer,
                candidates=candidates,
                cfg=dict(cfg.get("bert_score") or {}),
                required_count=required_wrong_options,
                use_bertscore=use_bertscore,
            )
            if len(wrong_options) < required_wrong_options:
                stats["skipped_not_enough_distractors"] += 1
                continue
            choices = _shuffled_options(
                correct_answer=correct_answer,
                wrong_options=wrong_options,
                seed=stable_int_id("caption_mcq_option_order", seed, case_id, section_key),
            )
            question_rows = _row_to_vqa_records(
                condensed_row=condensed_row,
                registry_row=registry_row,
                task=task,
                choices=choices,
                cfg=cfg,
            )
            if not question_rows:
                stats["skipped_no_modality_rows"] += 1
                continue
            rows.extend(question_rows)
            stats["generated_semantic_questions"] += 1

    frame = pd.DataFrame(rows, columns=VQA_COLUMNS)
    stats["generated_rows"] = int(len(frame))
    return frame, stats


def sample_train_rows_by_modality(
    frame: pd.DataFrame,
    *,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    sampling_cfg = dict(cfg.get("sampling") or {})
    if not bool(sampling_cfg.get("enabled", False)):
        return frame.copy(), {"enabled": 0, "sampled_out_rows": 0}

    seed = int(sampling_cfg.get("seed", cfg.get("seed", 42)) or 42)
    split = _clean_text(sampling_cfg.get("split")) or "train"
    ratios = dict(sampling_cfg.get("modality_keep_ratios") or {})
    out_parts: list[pd.DataFrame] = []
    sampled_out = 0
    train_mask = frame["split"].astype(str).str.lower().eq(split.lower())
    out_parts.append(frame[~train_mask].copy())

    train_df = frame[train_mask].copy()
    for modality_name, group in train_df.groupby("modality_combination_name", sort=True):
        ratio = float(ratios.get(modality_name, 1.0))
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"sampling.modality_keep_ratios.{modality_name} must be between 0 and 1.")
        if ratio >= 1.0:
            kept = group
        elif ratio <= 0.0:
            kept = group.iloc[0:0]
        else:
            random_state = stable_int_id("caption_condensed_mcq_sample", seed, modality_name) % (2**32 - 1)
            kept = group.sample(frac=ratio, random_state=random_state)
        sampled_out += len(group) - len(kept)
        out_parts.append(kept)

    sampled = pd.concat(out_parts, ignore_index=True) if out_parts else frame.iloc[0:0].copy()
    sampled = sampled.sort_values(["split", "case_id", "task_id", "modality_combination_name"]).reset_index(drop=True)
    return sampled[VQA_COLUMNS].copy(), {"enabled": 1, "sampled_out_rows": int(sampled_out)}


def main() -> None:
    cfg = load_cfg()
    generation_cfg = OmegaConf.to_container(cfg, resolve=True)

    condensed_path = _resolve_path(generation_cfg["condensed_caption_path"])
    registry_path = _resolve_path(generation_cfg["source_registry_path"])
    output_path = _resolve_path(generation_cfg["output_parquet_path"])
    full_output_path = _resolve_path(generation_cfg["full_output_parquet_path"])

    condensed_df = _read_required_parquet(condensed_path)
    registry_df = _read_required_parquet(registry_path)
    frame, stats = build_caption_condensed_mcq_frame(
        condensed_df=condensed_df,
        registry_df=registry_df,
        cfg=generation_cfg,
    )
    sampled_frame, sampling_stats = sample_train_rows_by_modality(frame, cfg=generation_cfg)

    write_vqa_parquet(frame, full_output_path)
    write_vqa_parquet(sampled_frame, output_path)

    print(f"Condensed caption path: {condensed_path}")
    print(f"Registry path: {registry_path}")
    print(f"Full output path: {full_output_path}")
    print(f"Output path: {output_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"sampling_enabled: {sampling_stats['enabled']}")
    print(f"sampling_sampled_out_rows: {sampling_stats['sampled_out_rows']}")
    print(f"sampled_rows: {len(sampled_frame)}")

    print_first_n = int(generation_cfg.get("print_first_n", 0) or 0)
    if print_first_n > 0 and not sampled_frame.empty:
        for row in sampled_frame.head(print_first_n).to_dict(orient="records"):
            print("-" * 80)
            print(f"question_id: {row['question_id']}")
            print(f"case_id: {row['case_id']}")
            print(f"task_id: {row['task_id']}")
            print(f"modality_combination_name: {row['modality_combination_name']}")
            print(f"question: {row['question']}")
            print(f"answer_label: {row['answer_label']}")
            print(f"answer: {row['answer']}")


if __name__ == "__main__":
    main()
