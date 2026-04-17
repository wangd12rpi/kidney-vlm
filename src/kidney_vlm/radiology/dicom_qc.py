from __future__ import annotations

import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


SECONDARY_CAPTURE_SOP_UIDS = {
    "1.2.840.10008.5.1.4.1.1.7",
    "1.2.840.10008.5.1.4.1.1.7.1",
    "1.2.840.10008.5.1.4.1.1.7.2",
    "1.2.840.10008.5.1.4.1.1.7.3",
    "1.2.840.10008.5.1.4.1.1.7.4",
}

LOCALIZER_KEYWORDS = {
    "LOCALIZER",
    "SCOUT",
    "TOPOGRAM",
    "SCANOGRAM",
    "SURVEY",
    "LOCATOR",
    "PILOT",
}

DERIVED_KEYWORDS = {
    "DERIVED",
    "SECONDARY",
    "REFORMATTED",
    "MPR",
    "MIP",
    "MINIP",
    "AVERAGE",
    "SUB",
}

HEADER_TAGS = [
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "ImageType",
    "SeriesDescription",
    "ProtocolName",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "NumberOfFrames",
    "InstanceNumber",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "SliceThickness",
    "AcquisitionNumber",
]

GENERATED_ARTIFACT_SUFFIXES = (
    ".json",
    ".png",
)


@dataclass
class Config:
    min_rows: int = 64
    min_cols: int = 64
    min_usable_images: int = 5
    max_rejected_fraction_for_series: float = 0.50
    reject_multiframe_series: bool = True
    reject_derived_series: bool = False
    decode_pixels: bool = False
    force_read: bool = True
    verbose: bool = False


@dataclass
class ImageRecord:
    file_path: str
    series_dir: str
    readable_dicom: bool
    reject_reason: str | None = None
    error: str | None = None
    modality: str | None = None
    sop_class_uid: str | None = None
    sop_instance_uid: str | None = None
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    instance_number: str | None = None
    image_type: str | None = None
    series_description: str | None = None
    protocol_name: str | None = None
    rows: int | None = None
    cols: int | None = None
    bits_allocated: int | None = None
    number_of_frames: int | None = None
    pixel_decode_checked: bool = False


@dataclass
class SeriesRejectRecord:
    series_dir: str
    reject_reason: str
    reject_details: str
    total_files: int
    readable_dicoms: int
    rejected_images: int
    usable_images: int
    modalities: str
    series_instance_uids: str
    series_descriptions: str
    protocol_names: str


@lru_cache(maxsize=1)
def _get_pydicom_runtime() -> tuple[Any, Any, Any | None]:
    try:
        from pydicom import dcmread
        from pydicom.errors import InvalidDicomError
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Radiology QC requires 'pydicom'. Install project dependencies first."
        ) from exc

    try:
        from pydicom.pixels import pixel_array as load_pixel_array
    except Exception:  # pragma: no cover - optional acceleration path
        load_pixel_array = None

    return dcmread, InvalidDicomError, load_pixel_array


def ensure_pydicom_available() -> None:
    _get_pydicom_runtime()


def is_generated_artifact_name(name: str) -> bool:
    lowered = str(name).strip().lower()
    return any(lowered.endswith(suffix) for suffix in GENERATED_ARTIFACT_SUFFIXES)


def is_source_candidate_file(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and not is_generated_artifact_name(path.name)


def log(msg: str, cfg: Config) -> None:
    if cfg.verbose:
        print(msg, file=sys.stderr)


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return "\\".join(str(v) for v in value)
    return str(value)


def to_upper_tokens(*values: str | None) -> list[str]:
    tokens: list[str] = []
    for value in values:
        if not value:
            continue
        clean = re.sub(r"[^A-Z0-9]+", " ", value.upper())
        tokens.extend(t for t in clean.split() if t)
    return tokens


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        return int(value)
    except Exception:
        try:
            return int(float(str(value)))
        except Exception:
            return None


def get_number_of_frames(ds: Any) -> int:
    n = parse_int(getattr(ds, "NumberOfFrames", None))
    return n if n and n > 0 else 1


def looks_like_localizer(
    image_type: str | None,
    series_description: str | None,
    protocol_name: str | None,
) -> bool:
    tokens = set(to_upper_tokens(image_type, series_description, protocol_name))
    return any(keyword in tokens for keyword in LOCALIZER_KEYWORDS)


def looks_like_derived(
    image_type: str | None,
    series_description: str | None,
    protocol_name: str | None,
) -> bool:
    tokens = set(to_upper_tokens(image_type, series_description, protocol_name))
    return any(keyword in tokens for keyword in DERIVED_KEYWORDS)


def is_secondary_capture_sop(sop_uid: str | None) -> bool:
    return bool(sop_uid and sop_uid in SECONDARY_CAPTURE_SOP_UIDS)


def has_required_image_tags(rec: ImageRecord) -> bool:
    return (
        rec.modality in {"CT", "MR"}
        and rec.rows is not None
        and rec.cols is not None
        and rec.bits_allocated is not None
    )


def maybe_decode_pixels(path: Path) -> tuple[bool, str | None]:
    dcmread, _invalid_dicom_error, load_pixel_array = _get_pydicom_runtime()
    try:
        if load_pixel_array is not None:
            arr = load_pixel_array(path)
        else:
            ds = dcmread(str(path), force=True)
            arr = ds.pixel_array
        if arr is None:
            return False, "pixel_array_is_none"
        if hasattr(arr, "size") and int(arr.size) <= 0:
            return False, "empty_pixel_array"
        return True, None
    except Exception as exc:
        return False, f"pixel_decode_failed: {type(exc).__name__}: {exc}"


def read_header(path: Path, cfg: Config) -> ImageRecord:
    dcmread, invalid_dicom_error, _load_pixel_array = _get_pydicom_runtime()
    rec = ImageRecord(
        file_path=str(path),
        series_dir=str(path.parent),
        readable_dicom=False,
    )
    try:
        ds = dcmread(
            str(path),
            force=cfg.force_read,
            specific_tags=HEADER_TAGS,
        )
        rec.readable_dicom = True
        rec.modality = safe_str(getattr(ds, "Modality", None))
        rec.sop_class_uid = safe_str(getattr(ds, "SOPClassUID", None))
        rec.sop_instance_uid = safe_str(getattr(ds, "SOPInstanceUID", None))
        rec.study_instance_uid = safe_str(getattr(ds, "StudyInstanceUID", None))
        rec.series_instance_uid = safe_str(getattr(ds, "SeriesInstanceUID", None))
        rec.instance_number = safe_str(getattr(ds, "InstanceNumber", None))
        rec.image_type = safe_str(getattr(ds, "ImageType", None))
        rec.series_description = safe_str(getattr(ds, "SeriesDescription", None))
        rec.protocol_name = safe_str(getattr(ds, "ProtocolName", None))
        rec.rows = parse_int(getattr(ds, "Rows", None))
        rec.cols = parse_int(getattr(ds, "Columns", None))
        rec.bits_allocated = parse_int(getattr(ds, "BitsAllocated", None))
        rec.number_of_frames = get_number_of_frames(ds)
        return rec
    except invalid_dicom_error as exc:
        rec.reject_reason = "invalid_dicom"
        rec.error = f"InvalidDicomError: {exc}"
        return rec
    except Exception as exc:
        rec.reject_reason = "unreadable_dicom_header"
        rec.error = f"{type(exc).__name__}: {exc}"
        return rec


def image_reject_reason(rec: ImageRecord, cfg: Config) -> str | None:
    if not rec.readable_dicom:
        return rec.reject_reason or "unreadable_dicom_header"

    if rec.modality not in {"CT", "MR"}:
        return "unsupported_modality"

    if is_secondary_capture_sop(rec.sop_class_uid):
        return "secondary_capture_sop"

    if looks_like_localizer(rec.image_type, rec.series_description, rec.protocol_name):
        return "localizer_or_scout"

    if cfg.reject_multiframe_series and (rec.number_of_frames or 1) > 1:
        return "multiframe_instance"

    if not has_required_image_tags(rec):
        return "missing_required_image_tags"

    if rec.rows is not None and rec.rows < cfg.min_rows:
        return "tiny_matrix"
    if rec.cols is not None and rec.cols < cfg.min_cols:
        return "tiny_matrix"

    if cfg.reject_derived_series and looks_like_derived(rec.image_type, rec.series_description, rec.protocol_name):
        return "derived_or_reformatted_image"

    return None


def analyze_series(series_dir: Path, cfg: Config) -> tuple[list[ImageRecord], SeriesRejectRecord | None]:
    files = sorted(path for path in series_dir.iterdir() if is_source_candidate_file(path))
    image_records: list[ImageRecord] = []

    for path in files:
        rec = read_header(path, cfg)
        rec.reject_reason = image_reject_reason(rec, cfg)

        if cfg.decode_pixels and rec.reject_reason is None:
            ok, err = maybe_decode_pixels(path)
            rec.pixel_decode_checked = True
            if not ok:
                rec.reject_reason = "pixel_decode_failed"
                rec.error = err

        image_records.append(rec)

    total_files = len(image_records)
    readable = [record for record in image_records if record.readable_dicom]
    rejected = [record for record in image_records if record.reject_reason is not None]
    usable = [record for record in image_records if record.readable_dicom and record.reject_reason is None]

    modalities = sorted({record.modality for record in readable if record.modality})
    series_uids = sorted({record.series_instance_uid for record in readable if record.series_instance_uid})
    series_descs = sorted({record.series_description for record in readable if record.series_description})
    protocol_names = sorted({record.protocol_name for record in readable if record.protocol_name})

    localizer_count = sum(
        1
        for record in readable
        if looks_like_localizer(record.image_type, record.series_description, record.protocol_name)
    )
    secondary_capture_count = sum(1 for record in readable if is_secondary_capture_sop(record.sop_class_uid))
    multiframe_count = sum(1 for record in readable if (record.number_of_frames or 1) > 1)
    derived_count = sum(
        1
        for record in readable
        if looks_like_derived(record.image_type, record.series_description, record.protocol_name)
    )

    reject_reason: str | None = None
    reject_details = ""

    if total_files == 0:
        reject_reason = "empty_directory"
        reject_details = "Directory contained no files."
    elif len(readable) == 0:
        reject_reason = "no_readable_dicom_files"
        reject_details = "No files in the directory could be read as DICOM."
    elif len(modalities) > 1:
        reject_reason = "mixed_modality_in_directory"
        reject_details = f"Readable files contain multiple modalities: {modalities}."
    elif len(series_uids) > 1:
        reject_reason = "mixed_series_instance_uid_in_directory"
        reject_details = f"Readable files contain multiple SeriesInstanceUID values: {series_uids}."
    elif localizer_count == len(readable):
        reject_reason = "series_is_localizer_or_scout"
        reject_details = "All readable images were marked as localizer/scout/topogram-like."
    elif secondary_capture_count == len(readable):
        reject_reason = "series_is_secondary_capture"
        reject_details = "All readable images use Secondary Capture SOP class."
    elif cfg.reject_multiframe_series and multiframe_count == len(readable):
        reject_reason = "series_is_multiframe"
        reject_details = "All readable images are multi-frame, but single-frame 2D slices are required."
    elif cfg.reject_derived_series and derived_count == len(readable):
        reject_reason = "series_is_derived_or_reformatted"
        reject_details = "All readable images look derived/reformatted and strict derived rejection is enabled."
    elif len(usable) == 0:
        reason_counts = Counter(record.reject_reason for record in rejected if record.reject_reason)
        top_reasons = ", ".join(f"{key}:{value}" for key, value in reason_counts.most_common())
        reject_reason = "all_images_rejected"
        reject_details = f"No usable images remain after image-level hard rejects. Reasons: {top_reasons}"
    elif len(usable) < cfg.min_usable_images:
        reject_reason = "too_few_usable_images"
        reject_details = (
            f"Only {len(usable)} usable images remain after filtering; "
            f"minimum required is {cfg.min_usable_images}."
        )
    else:
        rejected_fraction = len(rejected) / max(len(readable), 1)
        if rejected_fraction > cfg.max_rejected_fraction_for_series:
            reason_counts = Counter(record.reject_reason for record in rejected if record.reject_reason)
            top_reasons = ", ".join(f"{key}:{value}" for key, value in reason_counts.most_common())
            reject_reason = "rejected_fraction_too_high"
            reject_details = (
                f"Rejected {len(rejected)}/{len(readable)} readable files "
                f"({rejected_fraction:.3f}) which exceeds the threshold of "
                f"{cfg.max_rejected_fraction_for_series:.3f}. Reasons: {top_reasons}"
            )

    series_record = None
    if reject_reason is not None:
        series_record = SeriesRejectRecord(
            series_dir=str(series_dir),
            reject_reason=reject_reason,
            reject_details=reject_details,
            total_files=total_files,
            readable_dicoms=len(readable),
            rejected_images=len(rejected),
            usable_images=len(usable),
            modalities=";".join(modalities),
            series_instance_uids=";".join(series_uids),
            series_descriptions=";".join(series_descs),
            protocol_names=";".join(protocol_names),
        )

    return image_records, series_record


def find_candidate_series_dirs(root: Path) -> list[Path]:
    series_dirs: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        visible_files = [
            name
            for name in filenames
            if not name.startswith(".") and not is_generated_artifact_name(name)
        ]
        if visible_files:
            series_dirs.append(Path(dirpath))
    return sorted(set(series_dirs))


__all__ = [
    "Config",
    "ImageRecord",
    "SeriesRejectRecord",
    "analyze_series",
    "ensure_pydicom_available",
    "find_candidate_series_dirs",
    "image_reject_reason",
    "is_generated_artifact_name",
    "is_source_candidate_file",
    "looks_like_derived",
    "looks_like_localizer",
    "maybe_decode_pixels",
    "parse_int",
    "read_header",
]
