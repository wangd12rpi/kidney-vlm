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

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

_PATCH_COUNT_CACHE: dict[str, int] = {}


def load_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(
            config_name="config",
            overrides=["qa_genereation=path_proj_train_qa", *sys.argv[1:]],
        )
    OmegaConf.set_struct(cfg, False)
    return cfg


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


def _existing_local_relative_paths(value: Any) -> list[str]:
    existing_paths: list[str] = []
    for raw_path in _as_list(value):
        local_path = _normalize_local_path(raw_path)
        if not local_path.exists():
            continue
        try:
            relative_path = local_path.relative_to(ROOT).as_posix()
        except ValueError:
            relative_path = local_path.as_posix()
        existing_paths.append(relative_path)
    return existing_paths


def _slide_kind(slide_stem: str) -> str:
    upper_stem = str(slide_stem).upper()
    if "-DX" in upper_stem:
        return "DX"
    if "-TS" in upper_stem:
        return "TS"
    if "-BS" in upper_stem:
        return "BS"
    return ""


def _first_path_per_stem(paths: list[str]) -> dict[str, str]:
    matched: dict[str, str] = {}
    for path_value in paths:
        stem = Path(str(path_value)).stem
        if stem and stem not in matched:
            matched[stem] = str(path_value)
    return matched


def _build_slide_caption_row_id(sample_id: str, slide_stem: str, caption_variant_index: int) -> str:
    safe_sample_id = str(sample_id).strip() or "unknown-sample"
    safe_slide_stem = str(slide_stem).strip() or "unknown-slide"
    return f"{safe_sample_id}::{safe_slide_stem}::caption-{int(caption_variant_index) + 1}"


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    return int(text)


def _tcga_sample_type_code(slide_stem: str) -> str:
    import re

    match = re.search(r"-([0-9]{2}[A-Z])-[0-9]{2}-", str(slide_stem).upper())
    if match is None:
        return ""
    return str(match.group(1))


def _is_normal_tcga_slide(slide_stem: str) -> bool:
    return _tcga_sample_type_code(slide_stem).startswith("11")


def _resolve_patch_count(path_value: str) -> int:
    local_path = _normalize_local_path(path_value)
    cache_key = str(local_path)
    if cache_key in _PATCH_COUNT_CACHE:
        return _PATCH_COUNT_CACHE[cache_key]

    import h5py

    with h5py.File(local_path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"Missing 'features' dataset in {local_path}")
        patch_count = int(handle["features"].shape[0])
    _PATCH_COUNT_CACHE[cache_key] = patch_count
    return patch_count


def _attach_patch_counts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    patch_counts: list[int | None] = []
    for _, row in enriched.iterrows():
        existing_value = row.get("pathology_patch_count")
        if existing_value is not None and not (isinstance(existing_value, float) and pd.isna(existing_value)):
            text = str(existing_value).strip()
            if text:
                patch_counts.append(int(float(existing_value)))
                continue

        tile_paths = _as_list(row.get("pathology_tile_embedding_paths"))
        if not tile_paths:
            patch_counts.append(None)
            continue
        patch_counts.append(_resolve_patch_count(tile_paths[0]))

    enriched["pathology_patch_count"] = patch_counts
    return enriched


def _prepare_training_frame(
    frame: pd.DataFrame,
    *,
    exclude_normal_tcga_slides: bool,
    patch_count_lower_quantile: float | None,
    patch_count_upper_quantile: float | None,
) -> tuple[pd.DataFrame, dict[str, int | float | None]]:
    prepared = _attach_patch_counts(frame)
    stats: dict[str, int | float | None] = {
        "rows_before": int(len(prepared)),
        "rows_removed_normal_tcga": 0,
        "rows_removed_patch_count_outliers": 0,
        "patch_count_lower_bound": None,
        "patch_count_upper_bound": None,
    }

    if exclude_normal_tcga_slides and not prepared.empty and {"source", "slide_stem"}.issubset(prepared.columns):
        normal_mask = (
            prepared["source"].fillna("").astype(str).str.strip().str.lower().eq("tcga")
            & prepared["slide_stem"].fillna("").astype(str).map(_is_normal_tcga_slide)
        )
        stats["rows_removed_normal_tcga"] = int(normal_mask.sum())
        prepared = prepared.loc[~normal_mask].reset_index(drop=True)

    lower_q = None if patch_count_lower_quantile in (None, "", "null") else float(patch_count_lower_quantile)
    upper_q = None if patch_count_upper_quantile in (None, "", "null") else float(patch_count_upper_quantile)
    if lower_q is not None or upper_q is not None:
        valid_counts = prepared["pathology_patch_count"].dropna().astype(float)
        if not valid_counts.empty:
            lower_bound = float(valid_counts.quantile(lower_q)) if lower_q is not None else float(valid_counts.min())
            upper_bound = float(valid_counts.quantile(upper_q)) if upper_q is not None else float(valid_counts.max())
            stats["patch_count_lower_bound"] = lower_bound
            stats["patch_count_upper_bound"] = upper_bound
            keep_mask = prepared["pathology_patch_count"].astype(float).between(lower_bound, upper_bound, inclusive="both")
            stats["rows_removed_patch_count_outliers"] = int((~keep_mask).sum())
            prepared = prepared.loc[keep_mask].reset_index(drop=True)

    stats["rows_after"] = int(len(prepared))
    return prepared, stats


def _existing_output_row_id(row: dict[str, Any]) -> str:
    explicit_row_id = str(row.get("qa_row_id", "")).strip()
    if explicit_row_id:
        return explicit_row_id

    sample_id = str(row.get("sample_id", "")).strip()
    tile_paths = _as_list(row.get("pathology_tile_embedding_paths"))
    wsi_paths = _as_list(row.get("pathology_wsi_paths"))
    slide_stem = ""
    if tile_paths:
        slide_stem = Path(tile_paths[0]).stem
    elif wsi_paths:
        slide_stem = Path(wsi_paths[0]).stem

    if not sample_id or not slide_stem:
        return ""

    caption_variant_index = _coerce_int(row.get("caption_variant_index"), default=0)
    return _build_slide_caption_row_id(sample_id, slide_stem, caption_variant_index)


def _expand_case_row_to_slide_rows(
    row: dict[str, Any],
    *,
    require_existing_patch_embedding_files: bool,
    allowed_slide_kinds: set[str] | None = None,
    required_embedding_path_substrings: list[str] | None = None,
) -> list[dict[str, Any]]:
    tile_paths = (
        _existing_local_relative_paths(row.get("pathology_tile_embedding_paths"))
        if require_existing_patch_embedding_files
        else _as_list(row.get("pathology_tile_embedding_paths"))
    )
    if not tile_paths:
        return []

    normalized_allowed_slide_kinds = {
        str(kind).strip().upper() for kind in (allowed_slide_kinds or set()) if str(kind).strip()
    }
    normalized_required_substrings = [
        str(value).strip() for value in (required_embedding_path_substrings or []) if str(value).strip()
    ]

    filtered_tile_paths: list[str] = []
    for tile_path in tile_paths:
        tile_path_text = str(tile_path)
        slide_stem = Path(tile_path_text).stem
        if normalized_allowed_slide_kinds and _slide_kind(slide_stem) not in normalized_allowed_slide_kinds:
            continue
        if normalized_required_substrings and not any(
            substring in tile_path_text for substring in normalized_required_substrings
        ):
            continue
        filtered_tile_paths.append(tile_path_text)
    tile_paths = filtered_tile_paths

    if not tile_paths:
        return []

    wsi_by_stem = _first_path_per_stem(_as_list(row.get("pathology_wsi_paths")))
    slide_embedding_by_stem = _first_path_per_stem(_as_list(row.get("pathology_slide_embedding_paths")))

    slide_rows: list[dict[str, Any]] = []
    for slide_index, tile_path in enumerate(tile_paths):
        slide_stem = Path(str(tile_path)).stem
        slide_row = dict(row)
        slide_row["slide_stem"] = slide_stem
        slide_row["slide_index"] = slide_index
        slide_row["pathology_tile_embedding_paths"] = [str(tile_path)]
        slide_row["pathology_wsi_paths"] = [wsi_by_stem[slide_stem]] if slide_stem in wsi_by_stem else []
        slide_row["pathology_slide_embedding_paths"] = (
            [slide_embedding_by_stem[slide_stem]] if slide_stem in slide_embedding_by_stem else []
        )
        slide_rows.append(slide_row)
    return slide_rows


def _case_join_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("source", "")).strip(),
        str(row.get("sample_id", "")).strip(),
    )


def _expand_slide_rows_to_training_rows(
    slide_rows: list[dict[str, Any]],
    case_caption_rows: list[dict[str, Any]],
    *,
    default_instruction: str,
) -> list[dict[str, Any]]:
    captions_by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case_caption_row in case_caption_rows:
        captions_by_case.setdefault(_case_join_key(case_caption_row), []).append(case_caption_row)

    training_rows: list[dict[str, Any]] = []
    for slide_row in slide_rows:
        matched_case_captions = captions_by_case.get(_case_join_key(slide_row), [])
        if not matched_case_captions:
            continue

        sample_id = str(slide_row.get("sample_id", "")).strip()
        slide_stem = str(slide_row.get("slide_stem", "")).strip()
        for case_caption_row in matched_case_captions:
            caption_variant_index = _coerce_int(case_caption_row.get("caption_variant_index"), default=0)
            instruction = str(case_caption_row.get("instruction", "")).strip() or default_instruction
            training_rows.append(
                {
                    "qa_row_id": _build_slide_caption_row_id(sample_id, slide_stem, caption_variant_index),
                    "case_caption_row_id": str(case_caption_row.get("case_caption_row_id", "")).strip(),
                    "sample_id": sample_id,
                    "source": str(slide_row.get("source", "")),
                    "project_id": str(slide_row.get("project_id", "")),
                    "patient_id": str(slide_row.get("patient_id", "")),
                    "study_id": str(slide_row.get("study_id", "")),
                    "split": str(slide_row.get("split", "")),
                    "slide_stem": slide_stem,
                    "slide_index": _coerce_int(slide_row.get("slide_index"), default=0),
                    "caption_variant_index": caption_variant_index,
                    "caption_prompt_variant": str(case_caption_row.get("caption_prompt_variant", "")).strip(),
                    "caption_length_instruction": str(case_caption_row.get("caption_length_instruction", "")).strip(),
                    "pathology_wsi_paths": _as_list(slide_row.get("pathology_wsi_paths")),
                    "pathology_tile_embedding_paths": _as_list(slide_row.get("pathology_tile_embedding_paths")),
                    "pathology_slide_embedding_paths": _as_list(slide_row.get("pathology_slide_embedding_paths")),
                    "report_pdf_paths": _as_list(case_caption_row.get("report_pdf_paths")),
                    "instruction": instruction,
                    "question": instruction,
                    "caption": str(case_caption_row.get("caption", "")).strip(),
                    "answer": str(case_caption_row.get("answer", "")).strip(),
                    "caption_model": str(case_caption_row.get("caption_model", "")).strip(),
                }
            )
    return training_rows


def _build_output_frame(
    *,
    existing_output: pd.DataFrame,
    generated_rows: list[dict[str, Any]],
    overwrite_output: bool,
) -> pd.DataFrame:
    generated_df = pd.DataFrame(generated_rows)
    if not existing_output.empty and not overwrite_output:
        final_df = pd.concat([existing_output, generated_df], ignore_index=True)
        dedupe_column = "qa_row_id" if "qa_row_id" in final_df.columns else "sample_id"
        final_df = final_df.drop_duplicates(subset=[dedupe_column], keep="last").reset_index(drop=True)
        return final_df
    return generated_df


def _flush_output_parquet(
    *,
    output_path: Path,
    existing_output: pd.DataFrame,
    generated_rows: list[dict[str, Any]],
    overwrite_output: bool,
) -> pd.DataFrame:
    final_df = _build_output_frame(
        existing_output=existing_output,
        generated_rows=generated_rows,
        overwrite_output=overwrite_output,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    return final_df


def main() -> None:
    cfg = load_cfg()
    qa_cfg = cfg.qa_genereation

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
        raise FileNotFoundError(f"Case captions parquet not found: {caption_parquet_path}")
    case_caption_df = pd.read_parquet(caption_parquet_path)
    if case_caption_df.empty:
        raise RuntimeError(f"Case captions parquet is empty: {caption_parquet_path}")

    if bool(qa_cfg.require_pathology) and "pathology_wsi_paths" in registry_df.columns:
        registry_df = registry_df[registry_df["pathology_wsi_paths"].map(lambda v: len(_as_list(v)) > 0)]

    allowed_project_ids = [str(value).strip() for value in list(qa_cfg.allowed_project_ids or []) if str(value).strip()]
    if allowed_project_ids and "project_id" in registry_df.columns:
        registry_df = registry_df[registry_df["project_id"].isin(allowed_project_ids)]
        if "project_id" in case_caption_df.columns:
            case_caption_df = case_caption_df[case_caption_df["project_id"].astype(str).isin(allowed_project_ids)]

    if bool(qa_cfg.get("require_patch_embeddings", False)) and "pathology_tile_embedding_paths" in registry_df.columns:
        registry_df = registry_df[registry_df["pathology_tile_embedding_paths"].map(lambda v: len(_as_list(v)) > 0)]

    if bool(qa_cfg.get("require_existing_patch_embedding_files", False)) and "pathology_tile_embedding_paths" in registry_df.columns:
        registry_df = registry_df[
            registry_df["pathology_tile_embedding_paths"].map(lambda v: len(_existing_local_relative_paths(v)) > 0)
        ]

    if registry_df.empty:
        print("No rows selected for pathology projector QA building.")
        return

    existing_output = pd.DataFrame()
    done_row_ids: set[str] = set()
    existing_output_changed = False
    overwrite_output = bool(qa_cfg.overwrite_output)
    if output_path.exists() and not overwrite_output:
        existing_output = pd.read_parquet(output_path)
        existing_output, existing_filter_stats = _prepare_training_frame(
            existing_output,
            exclude_normal_tcga_slides=bool(qa_cfg.get("exclude_normal_tcga_slides", True)),
            patch_count_lower_quantile=qa_cfg.get("patch_count_lower_quantile"),
            patch_count_upper_quantile=qa_cfg.get("patch_count_upper_quantile"),
        )
        if existing_filter_stats["rows_before"] != existing_filter_stats["rows_after"]:
            existing_output_changed = True
            print(
                "Filtered existing pathology projector QA output\t"
                f"removed_normal={existing_filter_stats['rows_removed_normal_tcga']}\t"
                f"removed_patch_outliers={existing_filter_stats['rows_removed_patch_count_outliers']}\t"
                f"rows_after={existing_filter_stats['rows_after']}"
            )

    if not existing_output.empty:
        done_row_ids = {
            row_id
            for row_id in (
                _existing_output_row_id(row.to_dict())
                for _, row in existing_output.iterrows()
            )
            if row_id
        }

    require_existing_patch_embedding_files = bool(qa_cfg.get("require_existing_patch_embedding_files", False))
    allowed_slide_kinds = {
        str(value).strip().upper() for value in list(qa_cfg.get("allowed_slide_kinds", []) or []) if str(value).strip()
    }
    required_embedding_path_substrings = [
        str(value).strip()
        for value in list(qa_cfg.get("required_pathology_embedding_path_substrings", []) or [])
        if str(value).strip()
    ]
    slide_rows: list[dict[str, Any]] = []
    for _, row in registry_df.iterrows():
        slide_rows.extend(
            _expand_case_row_to_slide_rows(
                row.to_dict(),
                require_existing_patch_embedding_files=require_existing_patch_embedding_files,
                allowed_slide_kinds=allowed_slide_kinds,
                required_embedding_path_substrings=required_embedding_path_substrings,
            )
        )

    first_n = qa_cfg.get("first_n")
    if first_n is not None and str(first_n).strip():
        slide_rows = slide_rows[: int(first_n)]

    all_training_rows = _expand_slide_rows_to_training_rows(
        slide_rows,
        [row.to_dict() for _, row in case_caption_df.iterrows()],
        default_instruction=str(qa_cfg.get("instruction", "Describe the pathology image.")).strip(),
    )
    all_training_frame, filter_stats = _prepare_training_frame(
        pd.DataFrame(all_training_rows),
        exclude_normal_tcga_slides=bool(qa_cfg.get("exclude_normal_tcga_slides", True)),
        patch_count_lower_quantile=qa_cfg.get("patch_count_lower_quantile"),
        patch_count_upper_quantile=qa_cfg.get("patch_count_upper_quantile"),
    )
    all_training_rows = all_training_frame.to_dict(orient="records")
    rows_to_write = [
        row for row in all_training_rows if not done_row_ids or str(row.get("qa_row_id", "")).strip() not in done_row_ids
    ]

    print(f"Selected registry rows: {len(registry_df)}")
    print(f"Selected case caption rows: {len(case_caption_df)}")
    print(f"Selected slide rows: {len(slide_rows)}")
    print(f"Training rows requested: {len(all_training_rows)}")
    print(
        "Pathology projector QA filters\t"
        f"removed_normal={filter_stats['rows_removed_normal_tcga']}\t"
        f"removed_patch_outliers={filter_stats['rows_removed_patch_count_outliers']}\t"
        f"patch_lower={filter_stats['patch_count_lower_bound']}\t"
        f"patch_upper={filter_stats['patch_count_upper_bound']}"
    )

    if not rows_to_write:
        if existing_output_changed:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            existing_output.to_parquet(output_path, index=False)
            print(f"Rewrote filtered pathology projector QA parquet: {output_path}")
        print("All selected slide-caption rows already generated in output parquet.")
        return

    final_df = _flush_output_parquet(
        output_path=output_path,
        existing_output=existing_output,
        generated_rows=rows_to_write,
        overwrite_output=overwrite_output,
    )

    print(f"Saved pathology projector QA parquet: {output_path}")
    print(f"Rows written: {len(final_df)}")

    print_first_n = int(qa_cfg.get("print_first_n", 0) or 0)
    for row in rows_to_write[:print_first_n]:
        print("-" * 80)
        print(f"sample_id: {row['sample_id']}")
        print(f"slide_stem: {row['slide_stem']}")
        print(f"case_caption_row_id: {row['case_caption_row_id']}")
        print(f"qa_row_id: {row['qa_row_id']}")
        print(f"caption_variant_index: {row['caption_variant_index']}")
        print(f"caption: {row['caption']}")


if __name__ == "__main__":
    main()
