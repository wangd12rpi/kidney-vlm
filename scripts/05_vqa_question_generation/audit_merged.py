#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.vqa.gt_mcq import _as_list, _clean_text, _feature_path_list, _first_feature_path
from kidney_vlm.vqa.schema import ARRAY_COLUMNS, VQA_COLUMNS, normalize_vqa_df, validate_vqa_df

ROOT = find_repo_root(Path(__file__))

MERGED_VQA_PATH = ROOT / "data/vqa/merged_vqa.parquet"
UNIFIED_PATH = ROOT / "data/registry/unified.parquet"

BOOL_COLUMNS = ("use_pathology", "use_radiology", "use_dnam", "use_rna")
FEATURE_COLUMNS = (
    "pathology_feature_paths",
    "radiology_feature_paths",
    "dnam_feature_path",
    "rna_feature_path",
)
FALLBACK_COLUMNS = (
    "pathology_roi_png_dir",
    "radiology_view_png_dir",
    "dnam_text_summary",
    "rna_text_summary",
)
STRING_LIST_COLUMNS = ("radiology_view_png_dir",)
STRING_PATH_COLUMNS = (
    "dnam_feature_path",
    "rna_feature_path",
    "pathology_roi_png_dir",
    "radiology_view_png_dir",
    "dnam_text_summary",
    "rna_text_summary",
)


def _fail(failures: list[str], title: str, detail: str) -> None:
    failures.append(f"{title}: {detail}")


def _examples(df: pd.DataFrame, columns: list[str], n: int = 5) -> str:
    return df[columns].head(n).to_dict(orient="records").__repr__()


def _repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _feature_file_path(ref: str) -> Path:
    return _repo_path(str(ref).split("::", 1)[0])


def _json_list(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed).__name__}: {value!r}")
    if any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise ValueError(f"Expected non-empty string items in JSON list: {value!r}")
    return parsed


def _registry_feature_paths(row: pd.Series) -> dict[str, Any]:
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


def _expected_use_flags(registry_features: dict[str, Any]) -> dict[str, bool]:
    return {
        "use_pathology": bool(registry_features["pathology_feature_paths"]),
        "use_radiology": bool(registry_features["radiology_feature_paths"]),
        "use_dnam": bool(registry_features["dnam_feature_path"]),
        "use_rna": bool(registry_features["rna_feature_path"]),
    }


def _expected_feature_paths(
    registry_features: dict[str, Any],
    row: pd.Series,
) -> dict[str, Any]:
    return {
        "pathology_feature_paths": registry_features["pathology_feature_paths"]
        if bool(row["use_pathology"])
        else [],
        "radiology_feature_paths": registry_features["radiology_feature_paths"]
        if bool(row["use_radiology"])
        else [],
        "dnam_feature_path": registry_features["dnam_feature_path"] if bool(row["use_dnam"]) else "",
        "rna_feature_path": registry_features["rna_feature_path"] if bool(row["use_rna"]) else "",
    }


def _variant_signature(row: pd.Series) -> tuple[Any, ...]:
    return (
        bool(row["use_pathology"]),
        bool(row["use_radiology"]),
        bool(row["use_dnam"]),
        bool(row["use_rna"]),
        tuple(row["pathology_feature_paths"]),
        tuple(row["radiology_feature_paths"]),
        str(row["dnam_feature_path"]),
        str(row["rna_feature_path"]),
        str(row["pathology_roi_png_dir"]),
        str(row["radiology_view_png_dir"]),
    )


def _audit_schema(raw_df: pd.DataFrame, df: pd.DataFrame, failures: list[str]) -> None:
    try:
        validate_vqa_df(df)
    except Exception as exc:
        _fail(failures, "canonical schema validation failed", str(exc))

    raw_columns = list(raw_df.columns)
    if raw_columns != VQA_COLUMNS:
        missing = [column for column in VQA_COLUMNS if column not in raw_columns]
        extra = [column for column in raw_columns if column not in VQA_COLUMNS]
        _fail(failures, "column order/schema mismatch", f"missing={missing}; extra={extra}")

    for column in ARRAY_COLUMNS:
        bad = raw_df[
            ~raw_df[column].map(lambda value: isinstance(value, (list, tuple, np.ndarray)))
        ]
        if not bad.empty:
            _fail(
                failures,
                f"{column} is not parquet list-like",
                _examples(bad, ["question_id", "case_id", column]),
            )

    for column in STRING_PATH_COLUMNS:
        bad = raw_df[raw_df[column].map(lambda value: isinstance(value, (list, tuple, np.ndarray, dict)))]
        if not bad.empty:
            _fail(
                failures,
                f"{column} should be a string field",
                _examples(bad, ["question_id", "case_id", column]),
            )

    for column in STRING_LIST_COLUMNS:
        bad_rows = []
        for idx, value in raw_df[column].items():
            text = "" if value is None else str(value).strip()
            if not text:
                continue
            try:
                _json_list(text)
            except Exception as exc:
                bad_rows.append({"idx": idx, "question_id": raw_df.at[idx, "question_id"], "error": str(exc)})
        if bad_rows:
            _fail(failures, f"{column} has invalid JSON-list strings", str(bad_rows[:5]))


def _audit_legacy_caption_mcq(df: pd.DataFrame, failures: list[str]) -> None:
    bad = df[
        df["question_type"].eq("mcq")
        & df["generation_type"].eq("from_caption")
        & ~df["ground_truth_source"].astype(str).str.startswith("condensed_caption:")
    ]
    if not bad.empty:
        _fail(
            failures,
            "legacy caption MCQ rows are present",
            _examples(bad, ["question_id", "case_id", "task_id", "ground_truth_source"]),
        )


def _audit_modality_against_unified(
    df: pd.DataFrame,
    unified_df: pd.DataFrame,
    failures: list[str],
) -> None:
    required = {
        "patient_id",
        "pathology_tile_embedding_paths",
        "pathology_slide_embedding_paths",
        "radiology_embedding_paths",
        "genomics_dna_methylation_feature_path",
        "genomics_rna_bulk_feature_path",
    }
    missing = sorted(required - set(unified_df.columns))
    if missing:
        _fail(failures, "unified is missing required columns", str(missing))
        return
    if unified_df["patient_id"].duplicated().any():
        dupes = unified_df.loc[unified_df["patient_id"].duplicated(), "patient_id"].head(5).tolist()
        _fail(failures, "unified has duplicated patient_id", str(dupes))
        return

    unified_by_case = unified_df.set_index("patient_id", drop=False)
    bad_missing_case = df[~df["case_id"].isin(unified_by_case.index)]
    if not bad_missing_case.empty:
        _fail(
            failures,
            "merged rows missing from unified registry",
            _examples(bad_missing_case, ["question_id", "case_id", "task_id"]),
        )
        return

    bad_combo_rows: list[dict[str, Any]] = []
    bad_feature_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        registry_features = _registry_feature_paths(unified_by_case.loc[row["case_id"]])
        combo = str(row["modality_combination_name"])
        if combo == "all_available":
            expected_flags = _expected_use_flags(registry_features)
        elif combo == "path_only":
            expected_flags = {
                "use_pathology": True,
                "use_radiology": False,
                "use_dnam": False,
                "use_rna": False,
            }
            if not registry_features["pathology_feature_paths"]:
                expected_flags["use_pathology"] = False
        elif combo == "radiology_only":
            expected_flags = {
                "use_pathology": False,
                "use_radiology": True,
                "use_dnam": False,
                "use_rna": False,
            }
            if not registry_features["radiology_feature_paths"]:
                expected_flags["use_radiology"] = False
        else:
            bad_combo_rows.append(
                {
                    "question_id": row["question_id"],
                    "case_id": row["case_id"],
                    "modality_combination_name": combo,
                    "reason": "unknown combo",
                }
            )
            continue

        actual_flags = {column: bool(row[column]) for column in BOOL_COLUMNS}
        if actual_flags != expected_flags or not any(actual_flags.values()):
            bad_combo_rows.append(
                {
                    "question_id": row["question_id"],
                    "case_id": row["case_id"],
                    "modality_combination_name": combo,
                    "actual": actual_flags,
                    "expected": expected_flags,
                }
            )

        expected_features = _expected_feature_paths(registry_features, row)
        for column, expected in expected_features.items():
            actual = row[column]
            if column in ARRAY_COLUMNS:
                actual = list(actual)
            if actual != expected:
                bad_feature_rows.append(
                    {
                        "question_id": row["question_id"],
                        "case_id": row["case_id"],
                        "column": column,
                        "actual": actual,
                        "expected": expected,
                    }
                )
                break

    if bad_combo_rows:
        _fail(failures, "modality flags do not match unified availability", str(bad_combo_rows[:5]))
    if bad_feature_rows:
        _fail(failures, "feature paths do not match unified + modality flags", str(bad_feature_rows[:5]))


def _audit_paths_exist(df: pd.DataFrame, failures: list[str]) -> None:
    missing_files: list[str] = []
    feature_refs: set[str] = set()
    for column in ("pathology_feature_paths", "radiology_feature_paths"):
        for values in df[column]:
            feature_refs.update(values)
    for column in ("dnam_feature_path", "rna_feature_path"):
        feature_refs.update(value for value in df[column].astype(str) if value.strip())
    for ref in sorted(feature_refs):
        if not _feature_file_path(ref).exists():
            missing_files.append(ref)

    if missing_files:
        _fail(failures, "feature files are missing", str(missing_files[:10]))

    missing_fallbacks: list[str] = []
    for value in sorted(set(df["pathology_roi_png_dir"].astype(str))):
        if value.strip() and not _repo_path(value).is_dir():
            missing_fallbacks.append(value)
    for value in sorted(set(df["radiology_view_png_dir"].astype(str))):
        for png_path in _json_list(value):
            if not _repo_path(png_path).is_file():
                missing_fallbacks.append(png_path)
    if missing_fallbacks:
        _fail(failures, "fallback image paths are missing", str(missing_fallbacks[:10]))


def _audit_row_consistency(df: pd.DataFrame, failures: list[str]) -> None:
    bad_rows: list[dict[str, Any]] = []
    for (case_id, combo), group in df.groupby(["case_id", "modality_combination_name"], dropna=False):
        signatures = {_variant_signature(row) for _, row in group.iterrows()}
        if len(signatures) > 1:
            bad_rows.append({"case_id": case_id, "modality_combination_name": combo, "variants": len(signatures)})
    if bad_rows:
        _fail(failures, "same case/modality rows have inconsistent paths or fallbacks", str(bad_rows[:10]))


def _audit_test_fallbacks(df: pd.DataFrame, failures: list[str]) -> None:
    test_df = df[df["split"].astype(str).str.lower().eq("test")]
    bad_rows: list[dict[str, Any]] = []
    for _, row in test_df.iterrows():
        if bool(row["use_pathology"]) and not str(row["pathology_roi_png_dir"]).strip():
            bad_rows.append({"question_id": row["question_id"], "case_id": row["case_id"], "missing": "pathology_roi_png_dir"})
        if bool(row["use_radiology"]) and not str(row["radiology_view_png_dir"]).strip():
            bad_rows.append({"question_id": row["question_id"], "case_id": row["case_id"], "missing": "radiology_view_png_dir"})
        if bool(row["use_dnam"]) and not str(row["dnam_text_summary"]).strip():
            bad_rows.append({"question_id": row["question_id"], "case_id": row["case_id"], "missing": "dnam_text_summary"})
        if bool(row["use_rna"]) and not str(row["rna_text_summary"]).strip():
            bad_rows.append({"question_id": row["question_id"], "case_id": row["case_id"], "missing": "rna_text_summary"})
    if bad_rows:
        _fail(failures, "test rows with modality input are missing fallback artifacts", str(bad_rows[:10]))

    non_test = df[~df["split"].astype(str).str.lower().eq("test")]
    bad_non_test = non_test[
        non_test[list(FALLBACK_COLUMNS)].map(lambda value: bool(str(value).strip())).any(axis=1)
    ]
    if not bad_non_test.empty:
        _fail(
            failures,
            "non-test rows contain fallback artifacts",
            _examples(bad_non_test, ["question_id", "case_id", "split", *FALLBACK_COLUMNS]),
        )


def _audit_question_type_fields(df: pd.DataFrame, failures: list[str]) -> None:
    qa_bad = df[
        df["question_type"].eq("qa")
        & df[["option_a", "option_b", "option_c", "option_d", "answer_label"]]
        .map(lambda value: bool(str(value).strip()))
        .any(axis=1)
    ]
    if not qa_bad.empty:
        _fail(
            failures,
            "QA rows should not have options or answer_label",
            _examples(qa_bad, ["question_id", "case_id", "task_id", "option_a", "answer_label"]),
        )


def main() -> None:
    if not MERGED_VQA_PATH.exists():
        raise FileNotFoundError(f"Missing merged VQA parquet: {MERGED_VQA_PATH}")
    if not UNIFIED_PATH.exists():
        raise FileNotFoundError(f"Missing unified parquet: {UNIFIED_PATH}")

    raw_df = pd.read_parquet(MERGED_VQA_PATH)
    df = normalize_vqa_df(raw_df)
    unified_df = pd.read_parquet(UNIFIED_PATH)

    failures: list[str] = []
    _audit_schema(raw_df, df, failures)
    _audit_legacy_caption_mcq(df, failures)
    _audit_modality_against_unified(df, unified_df, failures)
    _audit_paths_exist(df, failures)
    _audit_row_consistency(df, failures)
    _audit_test_fallbacks(df, failures)
    _audit_question_type_fields(df, failures)

    for failure in failures:
        print(failure)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
