from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import zipfile

import pandas as pd

from kidney_vlm.radiology.dicom_qc import Config as DicomQCConfig
from kidney_vlm.radiology.dicom_qc import analyze_series


MANIFEST_LIST_COLUMNS = (
    "candidate_series_dirs",
    "usable_dicom_paths",
    "source_file_paths",
    "all_image_records",
    "accepted_image_records",
    "image_reject_records",
    "png_paths",
    "mask_paths",
    "segmentation_keywords",
)


MANIFEST_TEXT_COLUMNS = (
    "collection",
    "patient_id",
    "study_instance_uid",
    "series_instance_uid",
    "modality",
    "source_zip_path",
    "source_zip_relpath",
    "series_zip_path",
    "extracted_root",
    "selected_series_dir",
    "source_series_dir_relpath",
    "reject_reason",
    "reject_details",
    "qc_detail_path",
    "png_dir",
    "embedding_ref",
    "mask_dir",
    "mask_manifest_path",
)


MANIFEST_BOOL_COLUMNS = (
    "accepted",
    "processed_qc",
    "processed_png",
    "processed_features",
    "processed_segmentation",
)


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    text = str(value).strip()
    return [text] if text else []


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _to_abs_path(root_dir: Path, path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root_dir / path).resolve()


def to_registry_relative_path(root_dir: Path, path_value: str | Path) -> str:
    text = str(path_value).strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        return path.as_posix().lstrip("/")
    resolved = path.resolve()
    try:
        return resolved.relative_to(root_dir).as_posix()
    except ValueError:
        return resolved.as_posix().lstrip("/")


def empty_series_manifest_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        columns=[
            "collection",
            "patient_id",
            "study_instance_uid",
            "series_instance_uid",
            "modality",
            "source_zip_path",
            "source_zip_relpath",
            "series_zip_path",
            "extracted_root",
            "selected_series_dir",
            "source_series_dir_relpath",
            "accepted",
            "processed_qc",
            "processed_png",
            "processed_features",
            "processed_segmentation",
            "reject_reason",
            "reject_details",
            "candidate_series_dirs",
            "usable_dicom_paths",
            "source_file_paths",
            "all_image_records",
            "accepted_image_records",
            "image_reject_records",
            "qc_detail_path",
            "png_dir",
            "png_paths",
            "slice_count",
            "embedding_ref",
            "mask_dir",
            "mask_paths",
            "mask_manifest_path",
            "segmentation_keywords",
        ]
    )
    return ensure_series_manifest_df(frame)


def ensure_series_manifest_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in MANIFEST_LIST_COLUMNS:
        if column not in out.columns:
            out[column] = [[] for _ in range(len(out))]
        out[column] = out[column].map(_as_list)
    for column in MANIFEST_TEXT_COLUMNS:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").map(str)
    for column in MANIFEST_BOOL_COLUMNS:
        if column not in out.columns:
            out[column] = False
        out[column] = out[column].map(_as_bool)
    if "slice_count" not in out.columns:
        out["slice_count"] = 0
    out["slice_count"] = out["slice_count"].fillna(0).map(lambda value: int(float(value)) if str(value).strip() else 0)
    return out


def read_series_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return empty_series_manifest_frame()
    return ensure_series_manifest_df(pd.read_parquet(manifest_path))


def write_series_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = ensure_series_manifest_df(df)
    normalized.to_parquet(manifest_path, index=False)
    return manifest_path


def parse_tcia_series_zip_path(
    *,
    root_dir: Path,
    zip_path: str | Path,
) -> dict[str, str]:
    resolved = _to_abs_path(root_dir, zip_path)
    relative_parts = Path(to_registry_relative_path(root_dir, resolved)).parts
    try:
        radiology_index = relative_parts.index("radiology")
        suffix_parts = list(relative_parts[radiology_index + 1 :])
    except ValueError:
        suffix_parts = list(relative_parts[-4:])
    while len(suffix_parts) < 4:
        suffix_parts.insert(0, "")
    collection, patient_id, study_instance_uid, series_file_name = suffix_parts[-4:]
    series_instance_uid = series_file_name[:-4] if series_file_name.lower().endswith(".zip") else series_file_name
    return {
        "collection": collection,
        "patient_id": patient_id,
        "study_instance_uid": study_instance_uid,
        "series_instance_uid": series_instance_uid,
        "source_zip_path": str(resolved),
        "source_zip_relpath": to_registry_relative_path(root_dir, resolved),
    }


def series_entries_from_registry(
    registry_df: pd.DataFrame,
    *,
    root_dir: Path,
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in registry_df.iterrows():
        for raw_value in _as_list(row.get("radiology_download_paths")):
            text = str(raw_value).strip()
            if not text or "://" in text:
                continue
            abs_path = str(_to_abs_path(root_dir, text))
            if not abs_path.endswith(".zip") or abs_path in seen:
                continue
            seen.add(abs_path)
            entries.append(parse_tcia_series_zip_path(root_dir=root_dir, zip_path=abs_path))
    return entries


def is_generated_radiology_artifact(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered.endswith(".json") or lowered.endswith(".png")


def is_source_like_series_file(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and not is_generated_radiology_artifact(path)


def list_visible_files_recursive(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if is_source_like_series_file(path))


def has_visible_files(path: Path) -> bool:
    try:
        return any(is_source_like_series_file(child) for child in path.iterdir())
    except FileNotFoundError:
        return False


def discover_candidate_series_dirs(extracted_root: Path) -> list[Path]:
    if not extracted_root.exists():
        return []
    candidates: list[Path] = []
    if has_visible_files(extracted_root):
        candidates.append(extracted_root)
    for path in sorted(candidate for candidate in extracted_root.rglob("*") if candidate.is_dir()):
        if has_visible_files(path):
            candidates.append(path)
    return candidates


def delete_empty_parent_dirs(start_dir: Path, stop_dir: Path) -> None:
    current = start_dir.expanduser().resolve()
    stop = stop_dir.expanduser().resolve()
    while True:
        if current == stop:
            try:
                next(current.iterdir())
                return
            except (FileNotFoundError, StopIteration):
                try:
                    current.rmdir()
                except OSError:
                    return
                return
        if not current.exists():
            parent = current.parent
            if parent == current:
                return
            current = parent
            continue
        try:
            next(current.iterdir())
            return
        except (FileNotFoundError, StopIteration):
            try:
                current.rmdir()
            except OSError:
                return
            parent = current.parent
            if parent == current:
                return
            current = parent


def extract_tcia_series_zip(
    *,
    zip_path: Path,
    extracted_root: Path,
    skip_existing: bool,
) -> Path:
    existing_source_files = list_visible_files_recursive(extracted_root)
    if skip_existing and extracted_root.exists() and existing_source_files:
        return extracted_root

    extracted_root.mkdir(parents=True, exist_ok=True)
    for source_path in existing_source_files:
        try:
            source_path.unlink()
        except FileNotFoundError:
            continue
    if existing_source_files:
        delete_empty_parent_dirs(existing_source_files[0].parent, extracted_root)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extracted_root)
    return extracted_root


def run_tcia_series_qc(
    *,
    extracted_root: Path,
    qc_cfg: DicomQCConfig,
) -> dict[str, object]:
    candidate_dirs = discover_candidate_series_dirs(extracted_root)
    candidate_dir_paths = [str(path) for path in candidate_dirs]

    if not candidate_dirs:
        return {
            "accepted": False,
            "selected_series_dir": "",
            "candidate_series_dirs": candidate_dir_paths,
            "usable_image_paths": [],
            "source_file_paths": [],
            "all_image_records": [],
            "series_reject_record": {
                "series_dir": str(extracted_root),
                "reject_reason": "no_candidate_series_dirs",
                "reject_details": "No extracted directory containing visible files was found for the downloaded series zip.",
            },
            "image_reject_records": [],
        }

    if len(candidate_dirs) > 1:
        return {
            "accepted": False,
            "selected_series_dir": "",
            "candidate_series_dirs": candidate_dir_paths,
            "usable_image_paths": [],
            "source_file_paths": [],
            "all_image_records": [],
            "series_reject_record": {
                "series_dir": str(extracted_root),
                "reject_reason": "multiple_candidate_series_dirs",
                "reject_details": (
                    "The extracted zip expanded into multiple directories containing files, "
                    "so the series layout was treated as ambiguous."
                ),
            },
            "image_reject_records": [],
        }

    selected_series_dir = candidate_dirs[0]
    image_records, series_record = analyze_series(selected_series_dir, qc_cfg)
    all_image_records = [asdict(record) for record in image_records]
    image_reject_records = [asdict(record) for record in image_records if record.reject_reason is not None]
    usable_image_paths = [
        str(record.file_path)
        for record in image_records
        if record.readable_dicom and record.reject_reason is None
    ]
    return {
        "accepted": series_record is None,
        "selected_series_dir": str(selected_series_dir),
        "candidate_series_dirs": candidate_dir_paths,
        "usable_image_paths": usable_image_paths,
        "source_file_paths": [str(record.file_path) for record in image_records],
        "all_image_records": all_image_records,
        "series_reject_record": asdict(series_record) if series_record is not None else None,
        "image_reject_records": image_reject_records,
    }


def build_tcia_series_result_without_qc(
    *,
    extracted_root: Path,
) -> dict[str, object]:
    candidate_dirs = discover_candidate_series_dirs(extracted_root)
    candidate_dir_paths = [str(path) for path in candidate_dirs]
    if not candidate_dirs:
        return {
            "accepted": False,
            "selected_series_dir": "",
            "candidate_series_dirs": candidate_dir_paths,
            "usable_image_paths": [],
            "source_file_paths": [],
            "all_image_records": [],
            "series_reject_record": {
                "series_dir": str(extracted_root),
                "reject_reason": "no_candidate_series_dirs",
                "reject_details": "The extracted series zip did not contain a directory with visible files.",
            },
            "image_reject_records": [],
        }
    if len(candidate_dirs) > 1:
        return {
            "accepted": False,
            "selected_series_dir": "",
            "candidate_series_dirs": candidate_dir_paths,
            "usable_image_paths": [],
            "source_file_paths": [],
            "all_image_records": [],
            "series_reject_record": {
                "series_dir": str(extracted_root),
                "reject_reason": "multiple_candidate_series_dirs",
                "reject_details": "The extracted series zip expanded into multiple directories with visible files.",
            },
            "image_reject_records": [],
        }

    selected_series_dir = candidate_dirs[0]
    usable_image_paths = [str(path) for path in list_visible_files_recursive(selected_series_dir)]
    return {
        "accepted": bool(usable_image_paths),
        "selected_series_dir": str(selected_series_dir) if usable_image_paths else "",
        "candidate_series_dirs": candidate_dir_paths,
        "usable_image_paths": usable_image_paths,
        "source_file_paths": usable_image_paths,
        "all_image_records": [],
        "series_reject_record": None
        if usable_image_paths
        else {
            "series_dir": str(selected_series_dir),
            "reject_reason": "no_visible_files",
            "reject_details": "The extracted series directory did not contain any visible files.",
        },
        "image_reject_records": [],
    }


def infer_modality_from_qc_result(qc_result: dict[str, object]) -> str:
    all_records = _as_list(qc_result.get("all_image_records"))
    for record in all_records:
        if isinstance(record, dict):
            modality = str(record.get("modality", "")).strip()
            if modality:
                return modality
    return ""


def write_tcia_qc_detail(entry: dict[str, object], *, detail_root: Path) -> Path:
    detail_path = (
        detail_root
        / str(entry.get("collection", "unknown_collection")).strip()
        / str(entry.get("patient_id", "unknown_patient")).strip()
        / str(entry.get("study_instance_uid", "unknown_study")).strip()
        / f"{str(entry.get('series_instance_uid', 'unknown_series')).strip()}.json"
    )
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8")
    return detail_path


def write_tcia_qc_report(entries: list[dict[str, object]], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for entry in sorted(
            entries,
            key=lambda item: (
                str(item.get("patient_id", "")),
                str(item.get("study_instance_uid", "")),
                str(item.get("series_instance_uid", "")),
            ),
        ):
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return destination
