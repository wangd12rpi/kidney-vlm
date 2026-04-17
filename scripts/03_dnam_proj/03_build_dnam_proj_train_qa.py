#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="03_dnam_proj/03_build_dnam_proj_train_qa.yaml",
        overrides=sys.argv[1:],
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_local_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _case_join_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("source", "")).strip(),
        str(row.get("sample_id", "")).strip(),
    )


def _build_qa_row_id(sample_id: str, caption_variant_index: int) -> str:
    safe_sample_id = str(sample_id).strip() or "unknown-sample"
    return f"{safe_sample_id}::dnam-qa-{int(caption_variant_index) + 1}"


def _build_output_frame(existing_output: pd.DataFrame, generated_rows: list[dict[str, Any]], overwrite_output: bool) -> pd.DataFrame:
    generated_df = pd.DataFrame(generated_rows)
    if not existing_output.empty and not overwrite_output:
        final_df = pd.concat([existing_output, generated_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["qa_row_id"], keep="last").reset_index(drop=True)
        return final_df
    return generated_df


def main() -> None:
    cfg = load_cfg()
    qa_cfg = cfg.dnam_proj

    registry_path = Path(str(qa_cfg.source_registry_path)).expanduser()
    if not registry_path.is_absolute():
        registry_path = (ROOT / registry_path).resolve()
    else:
        registry_path = registry_path.resolve()

    caption_parquet_path = Path(str(qa_cfg.caption_parquet_path)).expanduser()
    if not caption_parquet_path.is_absolute():
        caption_parquet_path = (ROOT / caption_parquet_path).resolve()
    else:
        caption_parquet_path = caption_parquet_path.resolve()

    output_path = Path(str(qa_cfg.output_parquet_path)).expanduser()
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    else:
        output_path = output_path.resolve()

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        raise RuntimeError(f"Registry is empty: {registry_path}")

    if not caption_parquet_path.exists():
        raise FileNotFoundError(f"DNAm caption parquet not found: {caption_parquet_path}")
    caption_df = pd.read_parquet(caption_parquet_path)
    if caption_df.empty:
        raise RuntimeError(f"DNAm caption parquet is empty: {caption_parquet_path}")

    allowed_project_ids = [str(value).strip() for value in list(qa_cfg.allowed_project_ids or []) if str(value).strip()]
    if allowed_project_ids and "project_id" in registry_df.columns:
        registry_df = registry_df[registry_df["project_id"].astype(str).isin(allowed_project_ids)]
        if "project_id" in caption_df.columns:
            caption_df = caption_df[caption_df["project_id"].astype(str).isin(allowed_project_ids)]

    if bool(qa_cfg.get("require_dnam", True)):
        registry_df = registry_df[
            registry_df["genomics_dna_methylation_feature_path"].fillna("").astype(str).str.strip() != ""
        ]

    if bool(qa_cfg.get("require_existing_dnam_feature_file", True)):
        registry_df = registry_df[
            registry_df["genomics_dna_methylation_feature_path"].map(
                lambda value: _normalize_local_path(str(value)).exists() if str(value).strip() else False
            )
        ]

    if bool(qa_cfg.get("require_existing_beta_files", False)):
        registry_df = registry_df[
            registry_df["genomics_dna_methylation_paths"].map(
                lambda values: any(_normalize_local_path(path).exists() for path in _as_list(values))
            )
        ]

    required_feature_path_substrings = [
        str(value).strip()
        for value in list(qa_cfg.get("required_dnam_feature_path_substrings", []) or [])
        if str(value).strip()
    ]
    if required_feature_path_substrings:
        registry_df = registry_df[
            registry_df["genomics_dna_methylation_feature_path"].map(
                lambda value: any(token in str(value) for token in required_feature_path_substrings)
            )
        ]

    if registry_df.empty:
        print("No rows selected for DNAm projector QA building.")
        return

    first_n = qa_cfg.get("first_n")
    if first_n not in (None, "", "null"):
        registry_df = registry_df.head(int(first_n)).reset_index(drop=True)

    overwrite_output = bool(qa_cfg.overwrite_output)
    existing_output = pd.DataFrame()
    done_row_ids: set[str] = set()
    if output_path.exists() and not overwrite_output:
        existing_output = pd.read_parquet(output_path)
        done_row_ids = {
            str(row_id).strip()
            for row_id in existing_output.get("qa_row_id", pd.Series(dtype=str)).tolist()
            if str(row_id).strip()
        }

    captions_by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for _, row in caption_df.iterrows():
        row_dict = row.to_dict()
        captions_by_case.setdefault(_case_join_key(row_dict), []).append(row_dict)

    training_rows: list[dict[str, Any]] = []
    for _, row in registry_df.iterrows():
        row_dict = row.to_dict()
        matched_captions = captions_by_case.get(_case_join_key(row_dict), [])
        if not matched_captions:
            continue

        sample_id = str(row_dict.get("sample_id", "")).strip()
        for caption_row in matched_captions:
            caption_variant_index = int(caption_row.get("caption_variant_index", 0) or 0)
            qa_row_id = _build_qa_row_id(sample_id, caption_variant_index)
            if done_row_ids and qa_row_id in done_row_ids:
                continue

            training_rows.append(
                {
                    "qa_row_id": qa_row_id,
                    "dnam_caption_row_id": str(caption_row.get("dnam_caption_row_id", "")).strip(),
                    "sample_id": sample_id,
                    "source": str(row_dict.get("source", "")),
                    "project_id": str(row_dict.get("project_id", "")),
                    "patient_id": str(row_dict.get("patient_id", "")),
                    "study_id": str(row_dict.get("study_id", "")),
                    "split": str(row_dict.get("split", "")),
                    "caption_variant_index": caption_variant_index,
                    "genomics_dna_methylation_paths": _as_list(row_dict.get("genomics_dna_methylation_paths")),
                    "genomics_dna_methylation_feature_path": str(
                        row_dict.get("genomics_dna_methylation_feature_path", "")
                    ).strip(),
                    "instruction": str(caption_row.get("instruction", qa_cfg.instruction)).strip(),
                    "question": str(caption_row.get("question", qa_cfg.instruction)).strip(),
                    "caption": str(caption_row.get("caption", "")).strip(),
                    "answer": str(caption_row.get("answer", "")).strip(),
                    "caption_model": str(caption_row.get("caption_model", "")).strip(),
                    "selected_dnam_sample_id": str(caption_row.get("selected_dnam_sample_id", "")).strip(),
                    "selected_dnam_beta_path": str(caption_row.get("selected_dnam_beta_path", "")).strip(),
                    "selected_dnam_feature_path": str(caption_row.get("selected_dnam_feature_path", "")).strip(),
                }
            )

    final_df = _build_output_frame(existing_output=existing_output, generated_rows=training_rows, overwrite_output=overwrite_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    print(f"Selected registry rows: {len(registry_df)}")
    print(f"Selected DNAm caption rows: {len(caption_df)}")
    print(f"Training rows written: {len(final_df)}")
    print(f"Saved DNAm projector QA parquet: {output_path}")


if __name__ == "__main__":
    main()
