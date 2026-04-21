#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


REQUIRED_REGISTRY_COLUMNS = {
    "sample_id",
    "source",
    "project_id",
    "patient_id",
    "study_id",
    "split",
    "genomics_rna_bulk_paths",
    "genomics_rna_bulk_feature_path",
}

REQUIRED_CAPTION_COLUMNS = {
    "rna_caption_row_id",
    "sample_id",
    "source",
    "caption_variant_index",
    "instruction",
    "question",
    "caption",
    "answer",
    "caption_model",
    "selected_rna_sample_id",
    "selected_rna_tsv_path",
    "selected_rna_feature_path",
}


def load_cfg():
    try:
        from kidney_vlm.script_config import load_script_cfg
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "required dependency"
        raise RuntimeError(
            f"Missing Python dependency '{missing_name}' while loading the RNA QA config. "
            "Install the project dependencies first, then rerun this script."
        ) from exc

    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="04_rna_proj/03_build_rna_proj_train_qa.yaml",
        overrides=sys.argv[1:],
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "not_available", "[]"}:
        return ""
    return text


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, tuple):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [_clean_text(item) for item in converted if _clean_text(item)]
    text = _clean_text(value)
    return [text] if text else []


def _normalize_local_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _feature_path_lookup_key(path_value: str) -> str:
    text = _clean_text(path_value)
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_absolute():
        return Path(os.path.relpath(path, start=ROOT)).as_posix()
    return Path(os.path.normpath(text)).as_posix()


def _case_join_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        _clean_text(row.get("source")),
        _clean_text(row.get("sample_id")),
    )


def _build_qa_row_id(sample_id: str, caption_variant_index: int) -> str:
    safe_sample_id = _clean_text(sample_id) or "unknown-sample"
    return f"{safe_sample_id}::rna-qa-{int(caption_variant_index) + 1}"


def _existing_output_row_id(row: dict[str, Any]) -> str:
    qa_row_id = _clean_text(row.get("qa_row_id"))
    if qa_row_id:
        return qa_row_id
    return _build_qa_row_id(
        _clean_text(row.get("sample_id")),
        int(row.get("caption_variant_index", 0) or 0),
    )


def _validate_columns(frame: pd.DataFrame, required_columns: set[str], *, frame_name: str) -> None:
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def _build_output_frame(
    existing_output: pd.DataFrame,
    generated_rows: list[dict[str, Any]],
    overwrite_output: bool,
) -> pd.DataFrame:
    generated_df = pd.DataFrame(generated_rows)
    if not generated_df.empty:
        generated_df = generated_df.drop_duplicates(subset=["qa_row_id"], keep="last").reset_index(drop=True)
    if not existing_output.empty and not overwrite_output:
        existing = existing_output.copy()
        if "qa_row_id" not in existing.columns:
            existing["qa_row_id"] = existing.apply(lambda row: _existing_output_row_id(row.to_dict()), axis=1)
        final_df = pd.concat([existing, generated_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["qa_row_id"], keep="last").reset_index(drop=True)
        return final_df
    return generated_df


def _build_captions_by_case(caption_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], int]:
    captions_by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    skipped_blank_caption_count = 0
    for row in caption_rows:
        caption = _clean_text(row.get("caption"))
        answer = _clean_text(row.get("answer"))
        if not caption or not answer:
            skipped_blank_caption_count += 1
            continue
        key = _case_join_key(row)
        if not all(key):
            continue
        captions_by_case.setdefault(key, []).append(row)
    return captions_by_case, skipped_blank_caption_count


def _build_training_rows(
    registry_rows: list[dict[str, Any]],
    caption_rows: list[dict[str, Any]],
    *,
    default_instruction: str,
    require_matching_selected_rna_feature_path: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    captions_by_case, skipped_blank_caption_count = _build_captions_by_case(caption_rows)
    caption_keys = set(captions_by_case)
    registry_keys = {_case_join_key(row) for row in registry_rows if all(_case_join_key(row))}

    stats = {
        "skipped_blank_caption_rows": skipped_blank_caption_count,
        "registry_rows_without_caption": 0,
        "caption_cases_without_registry_rna_feature": len(caption_keys.difference(registry_keys)),
        "feature_path_mismatch_rows": 0,
    }

    training_rows: list[dict[str, Any]] = []
    for row_dict in registry_rows:
        key = _case_join_key(row_dict)
        matched_captions = captions_by_case.get(key, [])
        if not matched_captions:
            stats["registry_rows_without_caption"] += 1
            continue

        sample_id = _clean_text(row_dict.get("sample_id"))
        registry_feature_path = _feature_path_lookup_key(_clean_text(row_dict.get("genomics_rna_bulk_feature_path")))
        for caption_row in matched_captions:
            selected_feature_path = _feature_path_lookup_key(_clean_text(caption_row.get("selected_rna_feature_path")))
            if (
                require_matching_selected_rna_feature_path
                and selected_feature_path
                and registry_feature_path
                and selected_feature_path != registry_feature_path
            ):
                stats["feature_path_mismatch_rows"] += 1
                continue

            caption_variant_index = int(caption_row.get("caption_variant_index", 0) or 0)
            training_rows.append(
                {
                    "qa_row_id": _build_qa_row_id(sample_id, caption_variant_index),
                    "rna_caption_row_id": _clean_text(caption_row.get("rna_caption_row_id")),
                    "sample_id": sample_id,
                    "source": _clean_text(row_dict.get("source")),
                    "project_id": _clean_text(row_dict.get("project_id")),
                    "patient_id": _clean_text(row_dict.get("patient_id")),
                    "study_id": _clean_text(row_dict.get("study_id")),
                    "split": _clean_text(row_dict.get("split")),
                    "caption_variant_index": caption_variant_index,
                    "caption_prompt_variant": _clean_text(caption_row.get("caption_prompt_variant")),
                    "caption_length_instruction": _clean_text(caption_row.get("caption_length_instruction")),
                    "genomics_rna_bulk_paths": _as_list(row_dict.get("genomics_rna_bulk_paths")),
                    "genomics_rna_bulk_feature_path": registry_feature_path,
                    "instruction": _clean_text(caption_row.get("instruction")) or default_instruction,
                    "question": _clean_text(caption_row.get("question")) or default_instruction,
                    "caption": _clean_text(caption_row.get("caption")),
                    "answer": _clean_text(caption_row.get("answer")),
                    "caption_model": _clean_text(caption_row.get("caption_model")),
                    "caption_api_version": _clean_text(caption_row.get("caption_api_version")),
                    "selected_rna_sample_id": _clean_text(caption_row.get("selected_rna_sample_id")),
                    "selected_rna_sample_type": _clean_text(caption_row.get("selected_rna_sample_type")),
                    "selected_rna_tsv_path": _feature_path_lookup_key(_clean_text(caption_row.get("selected_rna_tsv_path"))),
                    "selected_rna_feature_path": selected_feature_path,
                }
            )

    return training_rows, stats


def _assert_output_sanity(final_df: pd.DataFrame) -> None:
    if final_df.empty:
        raise RuntimeError("RNA projector QA frame is empty.")
    required_output_columns = {
        "qa_row_id",
        "rna_caption_row_id",
        "sample_id",
        "source",
        "project_id",
        "patient_id",
        "study_id",
        "split",
        "caption_variant_index",
        "genomics_rna_bulk_paths",
        "genomics_rna_bulk_feature_path",
        "instruction",
        "question",
        "caption",
        "answer",
        "caption_model",
        "selected_rna_sample_id",
        "selected_rna_tsv_path",
        "selected_rna_feature_path",
    }
    _validate_columns(final_df, required_output_columns, frame_name="RNA projector QA output")

    duplicate_count = int(final_df["qa_row_id"].duplicated().sum())
    if duplicate_count:
        raise RuntimeError(f"RNA projector QA output contains duplicate qa_row_id values: {duplicate_count}")

    empty_caption_count = int(final_df["caption"].fillna("").astype(str).str.strip().eq("").sum())
    if empty_caption_count:
        raise RuntimeError(f"RNA projector QA output contains empty captions: {empty_caption_count}")

    empty_answer_count = int(final_df["answer"].fillna("").astype(str).str.strip().eq("").sum())
    if empty_answer_count:
        raise RuntimeError(f"RNA projector QA output contains empty answers: {empty_answer_count}")

    split_counts = final_df["split"].fillna("").astype(str).str.strip().value_counts().to_dict()
    missing_splits = [split for split in ("train", "val", "test") if int(split_counts.get(split, 0)) == 0]
    if missing_splits:
        raise RuntimeError(f"RNA projector QA output is missing split(s): {missing_splits}")

    missing_feature_paths = [
        value
        for value in final_df["genomics_rna_bulk_feature_path"].fillna("").astype(str).tolist()
        if not value.strip() or not _normalize_local_path(value).exists()
    ]
    if missing_feature_paths:
        preview = ", ".join(missing_feature_paths[:5])
        raise RuntimeError(
            f"RNA projector QA output references missing feature files: {len(missing_feature_paths)}. "
            f"First missing: {preview}"
        )


def main() -> None:
    cfg = load_cfg()
    qa_cfg = cfg.rna_proj

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
    _validate_columns(registry_df, REQUIRED_REGISTRY_COLUMNS, frame_name="Unified registry")

    if not caption_parquet_path.exists():
        raise FileNotFoundError(f"RNA caption parquet not found: {caption_parquet_path}")
    caption_df = pd.read_parquet(caption_parquet_path)
    if caption_df.empty:
        raise RuntimeError(f"RNA caption parquet is empty: {caption_parquet_path}")
    _validate_columns(caption_df, REQUIRED_CAPTION_COLUMNS, frame_name="RNA caption parquet")

    allowed_project_ids = [str(value).strip() for value in list(qa_cfg.allowed_project_ids or []) if str(value).strip()]
    if allowed_project_ids:
        registry_df = registry_df[registry_df["project_id"].astype(str).isin(allowed_project_ids)]
        caption_df = caption_df[caption_df["project_id"].astype(str).isin(allowed_project_ids)]

    if bool(qa_cfg.get("require_rna", True)):
        registry_df = registry_df[
            registry_df["genomics_rna_bulk_feature_path"].fillna("").astype(str).str.strip() != ""
        ]

    if bool(qa_cfg.get("require_existing_rna_feature_file", True)):
        registry_df = registry_df[
            registry_df["genomics_rna_bulk_feature_path"].map(
                lambda value: _normalize_local_path(_clean_text(value)).exists() if _clean_text(value) else False
            )
        ]

    if bool(qa_cfg.get("require_existing_rna_tsv_file", False)):
        registry_df = registry_df[
            registry_df["genomics_rna_bulk_paths"].map(
                lambda values: any(_normalize_local_path(path).exists() for path in _as_list(values))
            )
        ]

    required_feature_path_substrings = [
        str(value).strip()
        for value in list(qa_cfg.get("required_rna_feature_path_substrings", []) or [])
        if str(value).strip()
    ]
    if required_feature_path_substrings:
        registry_df = registry_df[
            registry_df["genomics_rna_bulk_feature_path"].map(
                lambda value: any(token in str(value) for token in required_feature_path_substrings)
            )
        ]

    if registry_df.empty:
        print("No rows selected for RNA projector QA building.")
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
            _existing_output_row_id(row.to_dict())
            for _, row in existing_output.iterrows()
            if _existing_output_row_id(row.to_dict())
        }

    training_rows, stats = _build_training_rows(
        registry_df.to_dict(orient="records"),
        caption_df.to_dict(orient="records"),
        default_instruction=_clean_text(qa_cfg.get("instruction")) or "Describe the bulk RNA-seq expression profile.",
        require_matching_selected_rna_feature_path=bool(
            qa_cfg.get("require_matching_selected_rna_feature_path", True)
        ),
    )
    if done_row_ids:
        training_rows = [row for row in training_rows if row["qa_row_id"] not in done_row_ids]

    if stats["feature_path_mismatch_rows"] and bool(qa_cfg.get("require_matching_selected_rna_feature_path", True)):
        raise RuntimeError(
            "RNA caption selected feature paths did not match registry feature paths for "
            f"{stats['feature_path_mismatch_rows']} joined row(s)."
        )

    final_df = _build_output_frame(
        existing_output=existing_output,
        generated_rows=training_rows,
        overwrite_output=overwrite_output,
    )
    _assert_output_sanity(final_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    print(f"Selected registry rows: {len(registry_df)}")
    print(f"Selected RNA caption rows: {len(caption_df)}")
    print(f"Training rows written: {len(final_df)}")
    print(f"Registry rows without matching caption: {stats['registry_rows_without_caption']}")
    print(f"Caption cases without selected registry RNA feature: {stats['caption_cases_without_registry_rna_feature']}")
    print(f"Skipped blank caption rows: {stats['skipped_blank_caption_rows']}")
    print(f"Feature path mismatch rows: {stats['feature_path_mismatch_rows']}")
    print(f"Split counts: {final_df['split'].value_counts().to_dict()}")
    print(f"Saved RNA projector QA parquet: {output_path}")

    print_first_n = int(qa_cfg.get("print_first_n", 0) or 0)
    if print_first_n > 0:
        print(final_df.head(print_first_n).to_string(index=False))


if __name__ == "__main__":
    main()
