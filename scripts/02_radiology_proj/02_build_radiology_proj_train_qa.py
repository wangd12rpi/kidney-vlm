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
from kidney_vlm.data.sources.pmc_oa import normalize_pmc_oa_split_name
from kidney_vlm.radiology.pmc_oa_feature_store import looks_like_pmc_oa_feature_ref, parse_pmc_oa_feature_ref
from kidney_vlm.radiology.series_feature_store import looks_like_series_feature_ref, parse_series_feature_ref
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    from kidney_vlm.script_config import load_script_cfg

    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="02_radiology_proj/02_build_radiology_proj_train_qa.yaml",
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


def _normalize_local_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
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
    return f"{safe_sample_id}::radiology-qa-{int(caption_variant_index) + 1}"


def _resolve_series_stem(registry_row: dict[str, Any], caption_row: dict[str, Any]) -> str:
    explicit = str(caption_row.get("series_stem", "")).strip()
    if explicit:
        return explicit
    image_name = str(caption_row.get("image_name", "")).strip()
    if image_name:
        stem = Path(image_name).stem
        if stem:
            return stem
    embedding_paths = _as_list(registry_row.get("radiology_embedding_paths"))
    if embedding_paths:
        return embedding_paths[0].split("::", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
    return str(registry_row.get("sample_id", "")).strip() or "unknown-series"


def _feature_ref_exists(values: Any) -> bool:
    for raw_value in _as_list(values):
        text = str(raw_value).strip()
        if not text:
            continue
        if looks_like_series_feature_ref(text):
            parsed = parse_series_feature_ref(text)
            if _normalize_local_path(parsed.store_path).exists():
                return True
            continue
        if looks_like_pmc_oa_feature_ref(text):
            parsed = parse_pmc_oa_feature_ref(text)
            if _normalize_local_path(parsed.store_path).exists():
                return True
            continue
        if _normalize_local_path(text).exists():
            return True
    return False


def _existing_image_paths(values: Any) -> list[str]:
    existing_paths: list[str] = []
    for raw_path in _as_list(values):
        local_path = _normalize_local_path(raw_path)
        if local_path.exists():
            existing_paths.append(str(raw_path).strip())
    return existing_paths


def _build_training_rows(
    registry_rows: list[dict[str, Any]],
    caption_rows: list[dict[str, Any]],
    *,
    default_instruction: str,
) -> tuple[list[dict[str, Any]], int]:
    captions_by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for caption_row in caption_rows:
        captions_by_case.setdefault(_case_join_key(caption_row), []).append(caption_row)

    training_rows: list[dict[str, Any]] = []
    split_mismatch_count = 0
    for registry_row in registry_rows:
        matched_captions = captions_by_case.get(_case_join_key(registry_row), [])
        if not matched_captions:
            continue

        sample_id = str(registry_row.get("sample_id", "")).strip()
        registry_split = normalize_pmc_oa_split_name(str(registry_row.get("split", "train")).strip() or "train")
        for caption_row in matched_captions:
            caption_variant_index = int(caption_row.get("caption_variant_index", 0) or 0)
            caption_split = normalize_pmc_oa_split_name(str(caption_row.get("split", "train")).strip() or "train")
            if caption_split != registry_split:
                split_mismatch_count += 1
            instruction = str(caption_row.get("instruction", "")).strip() or default_instruction
            training_rows.append(
                {
                    "qa_row_id": _build_qa_row_id(sample_id, caption_variant_index),
                    "radiology_caption_row_id": str(caption_row.get("radiology_caption_row_id", "")).strip(),
                    "sample_id": sample_id,
                    "source": str(registry_row.get("source", "")),
                    "project_id": str(registry_row.get("project_id", "")),
                    "patient_id": str(registry_row.get("patient_id", "")),
                    "study_id": str(registry_row.get("study_id", "")),
                    "split": registry_split,
                    "series_stem": _resolve_series_stem(registry_row, caption_row),
                    "caption_variant_index": caption_variant_index,
                    "image_name": str(caption_row.get("image_name", "")).strip(),
                    "image_key": str(caption_row.get("image_key", "")).strip(),
                    "radiology_image_paths": _as_list(registry_row.get("radiology_image_paths"))
                    or _as_list(caption_row.get("radiology_image_paths")),
                    "radiology_image_modalities": _as_list(registry_row.get("radiology_image_modalities"))
                    or _as_list(caption_row.get("radiology_image_modalities")),
                    "radiology_embedding_paths": _as_list(registry_row.get("radiology_embedding_paths"))
                    or _as_list(caption_row.get("radiology_embedding_paths")),
                    "instruction": instruction,
                    "question": str(caption_row.get("question", instruction)).strip() or instruction,
                    "caption": str(caption_row.get("caption", "")).strip(),
                    "answer": str(caption_row.get("answer", "")).strip(),
                    "caption_model": str(caption_row.get("caption_model", "")).strip(),
                    "pmcid": str(caption_row.get("pmcid", "")).strip(),
                    "url_name": str(caption_row.get("url_name", "")).strip(),
                }
            )

    return training_rows, split_mismatch_count


def _build_output_frame(
    existing_output: pd.DataFrame,
    generated_rows: list[dict[str, Any]],
    overwrite_output: bool,
) -> pd.DataFrame:
    generated_df = pd.DataFrame(generated_rows)
    if not existing_output.empty and not overwrite_output:
        final_df = pd.concat([existing_output, generated_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["qa_row_id"], keep="last").reset_index(drop=True)
        return final_df
    return generated_df


def main() -> None:
    cfg = load_cfg()
    stage_cfg = cfg.radiology_proj

    registry_path = _normalize_local_path(stage_cfg.source_registry_path)
    caption_parquet_path = _normalize_local_path(stage_cfg.caption_parquet_path)
    output_path = _normalize_local_path(stage_cfg.output_parquet_path)

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        raise RuntimeError(f"Registry is empty: {registry_path}")
    if "split" not in registry_df.columns:
        raise RuntimeError("Radiology projector QA building requires the unified registry split column.")

    if not caption_parquet_path.exists():
        raise FileNotFoundError(f"Radiology caption parquet not found: {caption_parquet_path}")
    caption_df = pd.read_parquet(caption_parquet_path)
    if caption_df.empty:
        raise RuntimeError(f"Radiology caption parquet is empty: {caption_parquet_path}")

    allowed_project_ids = [str(value).strip() for value in list(stage_cfg.allowed_project_ids or []) if str(value).strip()]
    if allowed_project_ids and "project_id" in registry_df.columns:
        registry_df = registry_df[registry_df["project_id"].astype(str).isin(allowed_project_ids)]
        if "project_id" in caption_df.columns:
            caption_df = caption_df[caption_df["project_id"].astype(str).isin(allowed_project_ids)]

    if bool(stage_cfg.get("require_radiology", True)):
        registry_df = registry_df[registry_df["radiology_embedding_paths"].map(lambda value: len(_as_list(value)) > 0)]

    if bool(stage_cfg.get("require_existing_radiology_image_files", False)):
        registry_df = registry_df[
            registry_df["radiology_image_paths"].map(lambda value: len(_existing_image_paths(value)) > 0)
        ]

    if bool(stage_cfg.get("require_existing_radiology_feature_files", False)):
        registry_df = registry_df[registry_df["radiology_embedding_paths"].map(_feature_ref_exists)]

    required_embedding_path_substrings = [
        str(value).strip()
        for value in list(stage_cfg.get("required_radiology_embedding_path_substrings", []) or [])
        if str(value).strip()
    ]
    if required_embedding_path_substrings:
        registry_df = registry_df[
            registry_df["radiology_embedding_paths"].map(
                lambda values: any(
                    substring in path
                    for path in _as_list(values)
                    for substring in required_embedding_path_substrings
                )
            )
        ]

    if registry_df.empty:
        print("No rows selected for radiology projector QA building.")
        return

    first_n = stage_cfg.get("first_n")
    if first_n not in (None, "", "null"):
        registry_df = registry_df.head(int(first_n)).reset_index(drop=True)

    overwrite_output = bool(stage_cfg.get("overwrite_output", False))
    existing_output = pd.DataFrame()
    done_row_ids: set[str] = set()
    if output_path.exists() and not overwrite_output:
        existing_output = pd.read_parquet(output_path)
        done_row_ids = {
            str(row_id).strip()
            for row_id in existing_output.get("qa_row_id", pd.Series(dtype=str)).tolist()
            if str(row_id).strip()
        }

    training_rows, split_mismatch_count = _build_training_rows(
        registry_df.to_dict(orient="records"),
        caption_df.to_dict(orient="records"),
        default_instruction=str(stage_cfg.get("instruction", "Describe the radiology image.")).strip(),
    )
    if done_row_ids:
        training_rows = [
            row for row in training_rows if str(row.get("qa_row_id", "")).strip() not in done_row_ids
        ]
    if not training_rows and existing_output.empty:
        raise RuntimeError("No radiology projector QA rows matched registry rows and caption rows.")

    final_df = _build_output_frame(
        existing_output=existing_output,
        generated_rows=training_rows,
        overwrite_output=overwrite_output,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    print(f"Selected registry rows: {len(registry_df)}")
    print(f"Selected radiology caption rows: {len(caption_df)}")
    print(f"Training rows written: {len(final_df)}")
    print(f"Registry/caption split mismatches resolved in favor of registry split: {split_mismatch_count}")
    print(f"Saved radiology projector QA parquet: {output_path}")

    print_first_n = int(stage_cfg.get("print_first_n", 0) or 0)
    for row in training_rows[:print_first_n]:
        print("-" * 80)
        print(f"qa_row_id: {row['qa_row_id']}")
        print(f"sample_id: {row['sample_id']}")
        print(f"series_stem: {row['series_stem']}")
        print(f"caption_variant_index: {row['caption_variant_index']}")
        print(f"caption: {row['caption']}")


if __name__ == "__main__":
    main()
