#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import os
import random
import sys
from collections.abc import Mapping
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
from kidney_vlm.vqa.schema import VQA_COLUMNS, write_vqa_parquet

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="05_vqa_question_generation/fix_caption_qa.yaml",
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


def _validate_legacy_vqa_frame(df: pd.DataFrame) -> None:
    missing = [column for column in VQA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Legacy VQA parquet is missing columns: {missing}")


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


def _row_to_fixed_qa_records(
    *,
    legacy_row: Mapping[str, Any],
    registry_row: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    case_id = _clean_text(legacy_row.get("case_id"))
    project_id = _clean_text(registry_row.get("project_id")) or _clean_text(legacy_row.get("project_id"))
    question = _clean_text(legacy_row.get("question"))
    answer = _clean_text(legacy_row.get("answer"))
    task_id = _clean_text(legacy_row.get("task_id"))
    task_category = _clean_text(legacy_row.get("task_category"))
    caption_id = _clean_text(legacy_row.get("caption_id")) or case_id
    ground_truth_source = _clean_text(legacy_row.get("ground_truth_source")) or "caption_qa"

    if not case_id or not project_id or not question or not answer or not task_id or not task_category:
        return []

    feature_paths = _feature_paths_for_case(registry_row)
    base_question_id = stable_int_id("fixed_caption_qa_base", case_id, task_id, question, answer)
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
            "fixed_caption_qa_question",
            base_question_id,
            modality_name,
            bool(variant["use_pathology"]),
            bool(variant["use_radiology"]),
            bool(variant["use_dnam"]),
            bool(variant["use_rna"]),
        )
        records.append(
            {
                "case_id": case_id,
                "project_id": project_id,
                "question_id": question_id,
                "base_question_id": base_question_id,
                "split": _clean_text(registry_row.get("split")),
                "question_type": "qa",
                "generation_type": "from_caption",
                "task_category": task_category,
                "task_id": task_id,
                "modality_combination_name": modality_name,
                "use_pathology": bool(variant["use_pathology"]),
                "use_radiology": bool(variant["use_radiology"]),
                "use_dnam": bool(variant["use_dnam"]),
                "use_rna": bool(variant["use_rna"]),
                "question": question,
                "option_a": "",
                "option_b": "",
                "option_c": "",
                "option_d": "",
                "answer": answer,
                "answer_label": "",
                "caption_id": caption_id,
                "ground_truth_source": ground_truth_source,
                "radiology_biomarker": _clean_text(registry_row.get("radiology_biomarker"))
                if variant["use_radiology"]
                else "",
                **_feature_paths_for_variant(feature_paths, variant),
                **artifacts,
            }
        )
    return records


def build_fixed_caption_qa_frame(
    *,
    legacy_vqa_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    _validate_legacy_vqa_frame(legacy_vqa_df)
    _validate_registry_frame(registry_df)
    qa_df = legacy_vqa_df[legacy_vqa_df["question_type"].astype(str).str.strip().eq("qa")].copy()
    registry_by_patient = registry_df.set_index("patient_id", drop=False)
    stats = {
        "legacy_rows": int(len(legacy_vqa_df)),
        "legacy_qa_rows": int(len(qa_df)),
        "skipped_missing_registry": 0,
        "skipped_empty_content": 0,
        "skipped_no_modality_rows": 0,
        "generated_rows": 0,
        "generated_base_questions": 0,
    }

    rows: list[dict[str, Any]] = []
    iterator = qa_df.to_dict(orient="records")
    if bool(cfg.get("show_progress", False)):
        iterator = tqdm(iterator, desc="Fixing caption QA")
    for legacy_row in iterator:
        case_id = _clean_text(legacy_row.get("case_id"))
        if case_id not in registry_by_patient.index:
            stats["skipped_missing_registry"] += 1
            continue
        if not _clean_text(legacy_row.get("question")) or not _clean_text(legacy_row.get("answer")):
            stats["skipped_empty_content"] += 1
            continue
        fixed_rows = _row_to_fixed_qa_records(
            legacy_row=legacy_row,
            registry_row=registry_by_patient.loc[case_id].to_dict(),
            cfg=cfg,
        )
        if not fixed_rows:
            stats["skipped_no_modality_rows"] += 1
            continue
        rows.extend(fixed_rows)
        stats["generated_base_questions"] += 1

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
    non_train = frame[~train_mask].copy()
    out_parts.append(non_train)

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
            random_state = stable_int_id("fixed_caption_qa_sample", seed, modality_name) % (2**32 - 1)
            kept = group.sample(frac=ratio, random_state=random_state)
        sampled_out += len(group) - len(kept)
        out_parts.append(kept)

    sampled = pd.concat(out_parts, ignore_index=True) if out_parts else frame.iloc[0:0].copy()
    sampled = sampled.sort_values(["split", "case_id", "task_id", "modality_combination_name"]).reset_index(drop=True)
    return sampled[VQA_COLUMNS].copy(), {"enabled": 1, "sampled_out_rows": int(sampled_out)}


def main() -> None:
    cfg = load_cfg()
    generation_cfg = OmegaConf.to_container(cfg.vqa_question_generation, resolve=True)

    legacy_path = _resolve_path(generation_cfg["source_legacy_vqa_path"])
    registry_path = _resolve_path(generation_cfg["source_registry_path"])
    output_path = _resolve_path(generation_cfg["output_parquet_path"])
    full_output_path = _resolve_path(generation_cfg["full_output_parquet_path"])

    legacy_df = _read_required_parquet(legacy_path)
    registry_df = _read_required_parquet(registry_path)
    full_frame, stats = build_fixed_caption_qa_frame(
        legacy_vqa_df=legacy_df,
        registry_df=registry_df,
        cfg=generation_cfg,
    )
    sampled_frame, sampling_stats = sample_train_rows_by_modality(full_frame, cfg=generation_cfg)

    write_vqa_parquet(full_frame, full_output_path)
    write_vqa_parquet(sampled_frame, output_path)

    print(f"Legacy VQA path: {legacy_path}")
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
            print(f"answer: {row['answer']}")


if __name__ == "__main__":
    main()
