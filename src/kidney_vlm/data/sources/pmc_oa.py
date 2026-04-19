from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from kidney_vlm.data.registry_schema import normalize_registry_df


DEFAULT_PMC_OA_SOURCE_NAME = "pmc_oa"
DEFAULT_PMC_OA_PROJECT_ID = "pmc_oa"
DEFAULT_PMC_OA_SPLIT_SCHEME_VERSION = "pmc_oa_explicit_image_split_v1"


def normalize_pmc_oa_split_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"validation", "valid", "val", "dev"}:
        return "val"
    if normalized == "test":
        return "test"
    return "train"


def load_pmc_oa_caption_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PMC-OA caption split not found: {path}")
    frame = pd.read_json(path, lines=True)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["split"] = normalize_pmc_oa_split_name(split_name)
    return frame


def load_pmc_oa_caption_frame(
    *,
    train_jsonl_path: Path,
    validation_jsonl_path: Path,
    test_jsonl_path: Path,
) -> pd.DataFrame:
    train_frame = load_pmc_oa_caption_split(train_jsonl_path, "train")
    validation_frame = load_pmc_oa_caption_split(validation_jsonl_path, "val")
    test_frame = load_pmc_oa_caption_split(test_jsonl_path, "test")
    return pd.concat([train_frame, validation_frame, test_frame], ignore_index=True)


def _resolve_repo_path(path_value: str | Path, *, root_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


def _project_relative_or_absolute(path: Path, *, root_dir: Path) -> str:
    absolute_path = path if path.is_absolute() else (root_dir / path)
    try:
        return absolute_path.resolve().relative_to(root_dir).as_posix()
    except ValueError:
        return absolute_path.resolve().as_posix()


def build_pmc_oa_caption_rows(
    caption_frame: pd.DataFrame,
    *,
    root_dir: Path,
    feature_index_by_image_name: dict[str, dict[str, object]],
    image_root_dir: Path,
    require_existing_image_files: bool,
    default_instruction: str,
    caption_model: str = "pmc_oa_human",
    source_name: str = DEFAULT_PMC_OA_SOURCE_NAME,
    project_id: str = DEFAULT_PMC_OA_PROJECT_ID,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    rows: list[dict[str, object]] = []
    missing_feature_images: list[str] = []
    missing_image_files: list[str] = []
    caption_variant_index_by_sample_id: dict[str, int] = defaultdict(int)

    for _, row in caption_frame.iterrows():
        image_name = str(row.get("image", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if not image_name or not caption:
            continue

        feature_row = feature_index_by_image_name.get(image_name)
        if feature_row is None:
            missing_feature_images.append(image_name)
            continue

        image_path = image_root_dir / image_name
        if require_existing_image_files and not image_path.exists():
            missing_image_files.append(image_name)
            continue

        sample_id = str(feature_row["sample_id"]).strip()
        caption_variant_index = caption_variant_index_by_sample_id[sample_id]
        caption_variant_index_by_sample_id[sample_id] += 1
        instruction = default_instruction
        rows.append(
            {
                "radiology_caption_row_id": f"{sample_id}::caption-{caption_variant_index + 1}",
                "sample_id": sample_id,
                "source": source_name,
                "project_id": project_id,
                "patient_id": str(row.get("pmcid", "")).strip(),
                "study_id": str(row.get("url_name", "")).strip(),
                "split": normalize_pmc_oa_split_name(row.get("split", "train")),
                "caption_variant_index": caption_variant_index,
                "image_name": image_name,
                "image_key": str(feature_row.get("image_key", "")).strip(),
                "radiology_image_paths": [_project_relative_or_absolute(image_path, root_dir=root_dir)],
                "radiology_image_modalities": ["figure"],
                "radiology_embedding_paths": [str(feature_row["embedding_ref"]).strip()],
                "instruction": instruction,
                "question": instruction,
                "caption": caption,
                "answer": caption,
                "caption_model": str(caption_model).strip(),
                "pmcid": str(row.get("pmcid", "")).strip(),
                "url_name": str(row.get("url_name", "")).strip(),
            }
        )

    return rows, missing_feature_images, missing_image_files


def build_pmc_oa_registry_rows(
    caption_rows: list[dict[str, object]],
    *,
    source_name: str = DEFAULT_PMC_OA_SOURCE_NAME,
    project_id: str = DEFAULT_PMC_OA_PROJECT_ID,
    split_scheme_version: str = DEFAULT_PMC_OA_SPLIT_SCHEME_VERSION,
) -> pd.DataFrame:
    by_sample_id: dict[str, dict[str, object]] = {}
    for caption_row in caption_rows:
        sample_id = str(caption_row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if sample_id in by_sample_id:
            continue
        patient_id = str(caption_row.get("patient_id", "")).strip()
        by_sample_id[sample_id] = {
            "sample_id": sample_id,
            "source": source_name,
            "project_id": project_id,
            "patient_id": patient_id,
            "study_id": str(caption_row.get("study_id", "")).strip(),
            "split": str(caption_row.get("split", "train")).strip() or "train",
            "split_group_id": f"{source_name}:{project_id}:{sample_id}",
            "split_scheme_version": str(split_scheme_version).strip() or DEFAULT_PMC_OA_SPLIT_SCHEME_VERSION,
            "pathology_wsi_paths": [],
            "radiology_image_paths": list(caption_row.get("radiology_image_paths", []) or []),
            "radiology_image_modalities": list(caption_row.get("radiology_image_modalities", []) or []),
            "pathology_mask_paths": [],
            "radiology_mask_paths": [],
            "pathology_tile_embedding_paths": [],
            "pathology_slide_embedding_paths": [],
            "radiology_embedding_paths": list(caption_row.get("radiology_embedding_paths", []) or []),
            "biomarkers_text": "",
            "question": "",
            "answer": "",
            "pmcid": str(caption_row.get("pmcid", "")).strip(),
            "url_name": str(caption_row.get("url_name", "")).strip(),
            "image_name": str(caption_row.get("image_name", "")).strip(),
        }
    return normalize_registry_df(pd.DataFrame(by_sample_id.values()))
