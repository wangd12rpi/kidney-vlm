#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.vqa.gt_mcq import build_ground_truth_mcq_frames
from kidney_vlm.vqa.schema import write_vqa_parquet

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="05_vqa_question_generation/generate_gt_mcq.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _validate_required_test_genomics_text(generated_df, generation_cfg) -> None:
    if not bool(generation_cfg.get("require_test_genomics_text_summaries", False)):
        return
    test_df = generated_df[generated_df["split"].astype(str).str.lower().eq("test")]
    if test_df.empty:
        return

    genomics_test_df = test_df[test_df["use_dnam"] | test_df["use_rna"]]
    if genomics_test_df.empty:
        raise RuntimeError(
            "No test rows with DNAm/RNA modalities were generated even though "
            "require_test_genomics_text_summaries is enabled. Refusing to overwrite "
            "the output parquet with a path-only test set."
        )

    missing_dnam = genomics_test_df["use_dnam"] & genomics_test_df[
        "dnam_text_summary"
    ].astype(str).str.strip().eq("")
    missing_rna = genomics_test_df["use_rna"] & genomics_test_df[
        "rna_text_summary"
    ].astype(str).str.strip().eq("")
    missing_count = int((missing_dnam | missing_rna).sum())
    if missing_count:
        raise RuntimeError(
            f"{missing_count} test genomics rows are missing required DNAm/RNA text summaries. "
            "Refusing to overwrite the output parquet."
        )


def main() -> None:
    cfg = load_cfg()
    vqa_cfg = cfg.vqa_question_generation

    registry_path = _resolve_path(vqa_cfg.source_registry_path)
    output_path = _resolve_path(vqa_cfg.output_parquet_path)
    full_output_path = _resolve_path(vqa_cfg.full_output_parquet_path)

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        raise RuntimeError(f"Registry is empty: {registry_path}")

    generation_cfg = OmegaConf.to_container(vqa_cfg, resolve=True)
    generated_df, full_generated_df, stats = build_ground_truth_mcq_frames(
        registry_df, generation_cfg
    )
    _validate_required_test_genomics_text(generated_df, generation_cfg)

    write_vqa_parquet(full_generated_df, full_output_path)
    write_vqa_parquet(generated_df, output_path)

    print(f"Registry path: {registry_path}")
    print(f"Full output path: {full_output_path}")
    print(f"Output path: {output_path}")
    print(f"Full generated rows: {stats['full_generated_rows']}")
    print(f"Full generated semantic questions: {stats['full_semantic_questions']}")
    print(f"Generated rows: {stats['generated_rows']}")
    print(f"Generated semantic questions: {stats['semantic_questions']}")
    sampling_stats = dict(stats.get("sampling") or {})
    if sampling_stats.get("enabled"):
        print(
            "Sampling: "
            f"pre_rows={sampling_stats['pre_sampling_rows']} "
            f"pre_semantic_questions={sampling_stats['pre_sampling_semantic_questions']} "
            f"sampled_out_rows={sampling_stats['sampled_out_rows']} "
            f"sampled_out_semantic_questions={sampling_stats['sampled_out_semantic_questions']} "
            f"protected_radiology_questions={sampling_stats['sampling_protected_radiology_questions']}"
        )
    if generated_df.empty:
        print(
            "No ground-truth MCQ VQA rows generated; wrote an empty replacement parquet."
        )
        return

    for task_id, task_stats in dict(stats["task_stats"]).items():
        details = "\t".join(f"{key}={value}" for key, value in task_stats.items())
        print(f"Task {task_id}\t{details}")

    print_first_n = int(vqa_cfg.get("print_first_n", 0) or 0)
    if print_first_n > 0:
        for row in generated_df.head(print_first_n).to_dict(orient="records"):
            print("-" * 80)
            print(f"question_id: {row['question_id']}")
            print(f"base_question_id: {row['base_question_id']}")
            print(f"case_id: {row['case_id']}")
            print(f"task_id: {row['task_id']}")
            print(f"question: {row['question']}")
            print(f"answer: {row['answer']}")


if __name__ == "__main__":
    main()
