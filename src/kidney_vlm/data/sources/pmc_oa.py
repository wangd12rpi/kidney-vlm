from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from kidney_vlm.data.id_factory import make_sample_id
from kidney_vlm.data.registry_schema import CORE_COLUMNS, empty_registry_frame, normalize_registry_df

_SPLIT_ALIAS = {
    "train": "train",
    "tr": "train",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "dev": "val",
    "test": "test",
    "te": "test",
}

DEFAULT_CT_INCLUDE_PATTERNS = (
    r"(?i)\b(?:computed|computerized)\s+tomograph(?:y|ic)\b",
    r"(?i)\bmicro[- ]?ct\b",
    r"(?i)\bhrct\b",
    r"(?i)\bcat scan\b",
    r"(?i)\b(?:axial|coronal|sagittal|contrast(?:-enhanced)?|plain|enhanced|helical|spiral|chest|abdominal?|pelvic|brain|head|neck|thoracic|lung|renal|kidney)\s+ct\b",
    r"(?i)\bct\s+(?:scan|scans|image|images|guided|guidance|angiograph(?:y|ic)?|urography|urogram|colonography|myelography|slice|slices|findings?|reveals|show(?:ing|s)?|demonstrat(?:es|ing)|before|after)\b",
)

DEFAULT_CT_EXCLUDE_PATTERNS = (
    r"(?i)\bconnective tissue\s*\(ct\)",
    r"(?i)\bcytotrophoblast\s*\(ct\)",
    r"(?i)\bcontrol\s*\(ct\)",
)

DEFAULT_AMBIGUOUS_MODALITY_PATTERNS = (
    r"(?i)\b(?:light|electron|transmission electron|scanning electron|confocal|fluorescence)\s+microscop(?:y|ic)\b",
    r"(?i)\b(?:histolog(?:y|ic)|histopatholog(?:y|ic)|immunohistochem(?:istry|ical))\b",
    r"(?i)\b(?:mri|mr image(?:s)?|mr imaging|magnetic resonance(?: imaging)?)\b",
    r"(?i)\b(?:pet|positron emission tomography)\b",
    r"(?i)\b(?:ultrasound|sonograph(?:y|ic)|echograph(?:y|ic))\b",
    r"(?i)\b(?:radiograph|x-?ray)\b",
    r"(?i)\b(?:conventional|clinical|gross)\s+photograph(?:y)?\b",
)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _coerce_split_name(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    return _SPLIT_ALIAS.get(normalized, normalized)


def _normalize_split_files(split_files: Mapping[str, Any]) -> list[tuple[str, Path]]:
    normalized: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw_split, raw_path in split_files.items():
        split = _coerce_split_name(raw_split)
        if split in seen:
            continue
        path_text = str(raw_path).strip()
        if not path_text:
            continue
        normalized.append((split, Path(path_text)))
        seen.add(split)
    return normalized


def _normalize_selected_splits(selected_splits: Iterable[str] | None) -> set[str]:
    if selected_splits is None:
        return set()
    normalized: set[str] = set()
    for split in selected_splits:
        text = str(split).strip()
        if not text:
            continue
        normalized.add(_coerce_split_name(text))
    return normalized


def _to_project_relative_path(path: str | Path, project_root: Path) -> str:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        try:
            return path_obj.relative_to(project_root).as_posix()
        except ValueError:
            return path_obj.as_posix()
    return path_obj.as_posix()


def _iter_jsonl_records(jsonl_path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {jsonl_path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object in {jsonl_path}:{line_number}, got {type(payload).__name__}"
                )
            yield line_number, payload


def _split_limit_for(max_rows_per_split: Mapping[str, Any] | None, split: str) -> int | None:
    if not isinstance(max_rows_per_split, Mapping):
        return None
    if split == "val":
        raw_value = max_rows_per_split.get("val", max_rows_per_split.get("valid"))
    else:
        raw_value = max_rows_per_split.get(split)
    return _optional_int(raw_value)


def _compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        text = str(pattern).strip()
        if not text:
            continue
        compiled.append(re.compile(text))
    return compiled


def _match_ct_caption(
    caption: str,
    *,
    include_patterns: list[re.Pattern[str]],
    exclude_patterns: list[re.Pattern[str]],
) -> str | None:
    for pattern in exclude_patterns:
        if pattern.search(caption):
            return None

    for pattern in include_patterns:
        if pattern.search(caption):
            return pattern.pattern
    return None


def build_pmc_oa_ct_registry_rows(
    *,
    dataset_root: Path,
    split_files: Mapping[str, Any],
    source_name: str,
    project_root: Path,
    image_dir: str = "images",
    selected_splits: Iterable[str] | None = None,
    max_rows_total: int | None = None,
    max_rows_per_split: Mapping[str, Any] | None = None,
    verify_image_paths: bool = False,
    skip_rows_with_missing_images: bool = False,
    skip_invalid_records: bool = True,
    fail_on_missing_split_files: bool = True,
    show_progress: bool = True,
    ct_include_patterns: Iterable[str] | None = None,
    ct_exclude_patterns: Iterable[str] | None = None,
    ambiguous_modality_patterns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    dataset_root = Path(dataset_root).expanduser()
    selected = _normalize_selected_splits(selected_splits)
    normalized_split_files = _normalize_split_files(split_files)
    if not normalized_split_files:
        raise ValueError("No PMC-OA split files were configured.")

    include_patterns = _compile_patterns(ct_include_patterns or DEFAULT_CT_INCLUDE_PATTERNS)
    exclude_patterns = _compile_patterns(ct_exclude_patterns or DEFAULT_CT_EXCLUDE_PATTERNS)
    ambiguous_patterns = _compile_patterns(ambiguous_modality_patterns or DEFAULT_AMBIGUOUS_MODALITY_PATTERNS)
    if not include_patterns:
        raise ValueError("PMC-OA CT filtering requires at least one include pattern.")

    image_dir_text = str(image_dir).strip() or "images"
    logical_image_root = dataset_root / image_dir_text

    resolved_split_files: list[tuple[str, Path]] = []
    missing_split_files: list[str] = []
    for split, split_path in normalized_split_files:
        if selected and split not in selected:
            continue
        candidate = split_path if split_path.is_absolute() else dataset_root / split_path
        if candidate.exists():
            resolved_split_files.append((split, candidate))
            continue
        missing_split_files.append(str(candidate))

    if missing_split_files and fail_on_missing_split_files:
        joined = ", ".join(sorted(missing_split_files))
        raise FileNotFoundError(f"Missing PMC-OA split file(s): {joined}")

    rows: list[dict[str, Any]] = []
    rows_scanned = 0
    ct_rows_kept = 0
    caption_filtered_rows = 0
    invalid_rows = 0
    missing_image_rows = 0
    ambiguous_modality_rows = 0
    per_split_scanned: dict[str, int] = {}
    per_split_kept: dict[str, int] = {}

    for split, jsonl_path in resolved_split_files:
        split_scanned = 0
        split_kept = 0
        split_limit = _split_limit_for(max_rows_per_split, split)
        iterator: Iterable[tuple[int, dict[str, Any]]] = _iter_jsonl_records(jsonl_path)

        if show_progress and tqdm is not None:
            iterator = tqdm(
                iterator,
                desc=f"Building {source_name}:{split}",
                unit="row",
                dynamic_ncols=True,
            )

        for line_number, record in iterator:
            if max_rows_total is not None and ct_rows_kept >= max_rows_total:
                break
            if split_limit is not None and split_kept >= split_limit:
                break

            rows_scanned += 1
            split_scanned += 1

            image_ref = str(record.get("image", "")).strip()
            caption = str(record.get("caption", "")).strip()
            pmcid = str(record.get("pmcid", "")).strip()
            url_name = str(record.get("url_name", "")).strip()

            if not image_ref or not caption or not pmcid:
                invalid_rows += 1
                if skip_invalid_records:
                    continue
                raise ValueError(
                    f"PMC-OA record missing required fields in {jsonl_path}:{line_number}"
                )

            matched_pattern = _match_ct_caption(
                caption,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            if matched_pattern is None:
                caption_filtered_rows += 1
                continue
            if any(pattern.search(caption) for pattern in ambiguous_patterns):
                ambiguous_modality_rows += 1
                continue

            logical_image_path = logical_image_root / image_ref
            image_path_exists: bool | None
            if verify_image_paths:
                image_path_exists = logical_image_path.exists()
                if not image_path_exists:
                    missing_image_rows += 1
                    if skip_rows_with_missing_images:
                        continue
            else:
                image_path_exists = None

            study_id = Path(url_name or image_ref).stem or Path(image_ref).stem
            image_rel_path = _to_project_relative_path(logical_image_path, project_root)
            sample_id = make_sample_id(
                source_name,
                pmcid,
                image_ref,
                modality_scope="pmc_oa_ct_image_caption",
            )

            rows.append(
                {
                    "sample_id": sample_id,
                    "source": source_name,
                    "patient_id": pmcid,
                    "study_id": study_id,
                    "split": split,
                    "pathology_wsi_paths": [],
                    "radiology_image_paths": [image_rel_path],
                    "pathology_mask_paths": [],
                    "radiology_mask_paths": [],
                    "pathology_tile_embedding_paths": [],
                    "pathology_slide_embedding_paths": [],
                    "radiology_embedding_paths": [],
                    "biomarkers_text": caption,
                    "question": "",
                    "answer": "",
                    "caption_text": caption,
                    "caption_match_pattern": matched_pattern,
                    "caption_modality": "ct",
                    "image_filename": Path(image_ref).name,
                    "image_rel_path": image_rel_path,
                    "image_path_exists": image_path_exists,
                    "pmcid": pmcid,
                    "url_name": url_name,
                    "source_jsonl": _to_project_relative_path(jsonl_path, project_root),
                }
            )
            ct_rows_kept += 1
            split_kept += 1

        per_split_scanned[split] = split_scanned
        per_split_kept[split] = split_kept
        if max_rows_total is not None and ct_rows_kept >= max_rows_total:
            break

    if not rows:
        stats = {
            "rows_scanned": rows_scanned,
            "rows_kept": 0,
            "caption_filtered_rows": caption_filtered_rows,
            "invalid_rows": invalid_rows,
            "missing_image_rows": missing_image_rows,
            "ambiguous_modality_rows": ambiguous_modality_rows,
            "missing_split_files": missing_split_files,
            "per_split_scanned": per_split_scanned,
            "per_split_kept": per_split_kept,
        }
        return empty_registry_frame(), stats

    frame = pd.DataFrame(rows)
    duplicates_dropped = 0
    if "sample_id" in frame.columns:
        before = len(frame)
        frame = frame.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
        duplicates_dropped = before - len(frame)

    frame = normalize_registry_df(frame)
    ordered = CORE_COLUMNS + [column for column in frame.columns if column not in CORE_COLUMNS]
    stats = {
        "rows_scanned": rows_scanned,
        "rows_kept": int(len(frame)),
        "caption_filtered_rows": caption_filtered_rows,
        "invalid_rows": invalid_rows,
        "missing_image_rows": missing_image_rows,
        "ambiguous_modality_rows": ambiguous_modality_rows,
        "duplicates_dropped": duplicates_dropped,
        "missing_split_files": missing_split_files,
        "per_split_scanned": per_split_scanned,
        "per_split_kept": per_split_kept,
        "verify_image_paths": bool(verify_image_paths),
        "logical_image_root": _to_project_relative_path(logical_image_root, project_root),
    }
    return frame[ordered], stats
