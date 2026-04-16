from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


COHORT_REGION_FALLBACK = {
    "TCGA-LUAD": "thorax",
    "TCGA-LUSC": "thorax",
    "TCGA-ESCA": "thorax_abdomen",
    "TCGA-BLCA": "abdomen_pelvis",
    "TCGA-COAD": "abdomen_pelvis",
    "TCGA-KICH": "abdomen_pelvis",
    "TCGA-KIRC": "abdomen_pelvis",
    "TCGA-KIRP": "abdomen_pelvis",
    "TCGA-LIHC": "abdomen_pelvis",
    "TCGA-OV": "abdomen_pelvis",
    "TCGA-PRAD": "abdomen_pelvis",
    "TCGA-READ": "abdomen_pelvis",
    "TCGA-STAD": "abdomen_pelvis",
    "TCGA-THCA": "head_neck",
    "TCGA-UCEC": "abdomen_pelvis",
    "TCGA-SARC": "mixed_sarc",
}

REGION_WINDOWS: dict[str, list[tuple[str, float, float]]] = {
    "thorax": [
        ("lung", -600.0, 1500.0),
        ("mediastinum", 40.0, 400.0),
        ("bone", 300.0, 1500.0),
    ],
    "thorax_abdomen": [
        ("lung", -600.0, 1500.0),
        ("soft_tissue", 50.0, 400.0),
        ("bone", 300.0, 1500.0),
    ],
    "abdomen_pelvis": [
        ("soft_tissue", 50.0, 400.0),
        ("abdomen_detail", 60.0, 180.0),
        ("bone", 300.0, 1500.0),
    ],
    "head_neck": [
        ("soft_tissue", 50.0, 350.0),
        ("neck_detail", 60.0, 180.0),
        ("bone", 300.0, 1500.0),
    ],
    "mixed_sarc": [
        ("soft_tissue", 50.0, 400.0),
        ("bone", 300.0, 1500.0),
        ("wide_detail", 50.0, 800.0),
    ],
    "generic": [
        ("soft_tissue", 50.0, 400.0),
        ("bone", 300.0, 1500.0),
        ("wide_detail", 50.0, 800.0),
    ],
}

DEFAULT_PRIMARY_WINDOW_BY_REGION: dict[str, str] = {
    "thorax": "lung",
    "thorax_abdomen": "soft_tissue",
    "abdomen_pelvis": "soft_tissue",
    "head_neck": "soft_tissue",
    "mixed_sarc": "soft_tissue",
    "generic": "soft_tissue",
}


@dataclass(frozen=True)
class RadiologyPngRenderConfig:
    render_mode: str = "ct_grayscale"
    prefer_dicom_voi: bool = True
    apply_padding_mask: bool = True
    resize: int | None = None
    force_rgb_output: bool = True
    primary_window_name: str | None = None


def normalize_to_uint8(pixels: np.ndarray) -> np.ndarray:
    pixels = pixels.astype(np.float32)
    finite = np.isfinite(pixels)
    if not finite.any():
        return np.zeros(pixels.shape, dtype=np.uint8)

    valid_pixels = pixels[finite]
    low = float(np.percentile(valid_pixels, 1))
    high = float(np.percentile(valid_pixels, 99))
    if high <= low:
        low = float(valid_pixels.min())
        high = float(valid_pixels.max())
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)

    clipped = np.clip(pixels, low, high)
    scaled = (clipped - low) / (high - low)
    return np.round(scaled * 255.0).astype(np.uint8)


def _is_multi_value(value: object) -> bool:
    return not isinstance(value, (str, bytes)) and isinstance(value, Iterable)


def _first_item(value: object) -> object | None:
    if value is None:
        return None
    if _is_multi_value(value):
        for item in value:
            return item
        return None
    return value


def first_number(value: object) -> float | None:
    item = _first_item(value)
    if item is None:
        return None
    try:
        return float(item)
    except Exception:
        return None


def first_text(value: object) -> str:
    if value is None:
        return ""
    if _is_multi_value(value):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


def detect_cohort_from_path(path: Path) -> str | None:
    upper_parts = [part.upper() for part in path.parts]
    for part in upper_parts:
        if part in COHORT_REGION_FALLBACK:
            return part
    return None


def get_text_blob(dataset: Any, dicom_path: Path) -> str:
    fields = [
        first_text(getattr(dataset, "BodyPartExamined", "")),
        first_text(getattr(dataset, "SeriesDescription", "")),
        first_text(getattr(dataset, "StudyDescription", "")),
        first_text(getattr(dataset, "ProtocolName", "")),
        first_text(getattr(dataset, "RequestedProcedureDescription", "")),
        first_text(getattr(dataset, "ImageComments", "")),
        first_text(getattr(dataset, "PerformedProcedureStepDescription", "")),
        dicom_path.as_posix(),
    ]
    return " | ".join(fields).lower()


def infer_region(dataset: Any, dicom_path: Path) -> str:
    text = get_text_blob(dataset, dicom_path)

    if any(keyword in text for keyword in ["thyroid", "neck", "c-spine", "cervical", "laryn", "pharyn"]):
        return "head_neck"

    if any(keyword in text for keyword in ["lung", "thorax", "chest", "mediast", "pulmo", "pleura"]):
        return "thorax"

    if any(keyword in text for keyword in ["esoph", "esophagus"]):
        return "thorax_abdomen"

    if any(
        keyword in text
        for keyword in [
            "abd",
            "abdomen",
            "pelvis",
            "liver",
            "hepatic",
            "kidney",
            "renal",
            "bladder",
            "prostate",
            "ovar",
            "uter",
            "endometr",
            "colon",
            "rect",
            "stomach",
            "gastric",
            "pancre",
            "adrenal",
            "testi",
        ]
    ):
        return "abdomen_pelvis"

    if any(
        keyword in text
        for keyword in [
            "sarcoma",
            "extrem",
            "arm",
            "leg",
            "femur",
            "tibia",
            "fibula",
            "humerus",
            "shoulder",
            "pelvic bone",
            "osseous",
            "bone",
            "soft tissue mass",
        ]
    ):
        return "mixed_sarc"

    cohort = detect_cohort_from_path(dicom_path)
    if cohort is not None:
        return COHORT_REGION_FALLBACK[cohort]
    return "generic"


def _squeeze_to_2d(pixels: np.ndarray, dicom_path: Path) -> np.ndarray:
    if pixels.ndim > 2:
        pixels = np.squeeze(pixels)
    if pixels.ndim > 2:
        pixels = pixels[0]
    if pixels.ndim != 2:
        raise ValueError(f"Unsupported pixel array shape for {dicom_path}: {pixels.shape}")
    return pixels


def get_hu_array(dataset: Any, dicom_path: Path, apply_modality_lut: Any) -> np.ndarray:
    pixels = apply_modality_lut(dataset.pixel_array, dataset)
    pixels = np.asarray(pixels, dtype=np.float32)
    return _squeeze_to_2d(pixels, dicom_path)


def get_padding_mask(dataset: Any, hu: np.ndarray) -> np.ndarray:
    mask = np.zeros(hu.shape, dtype=bool)
    raw_pad = getattr(dataset, "PixelPaddingValue", None)
    raw_pad_limit = getattr(dataset, "PixelPaddingRangeLimit", None)
    slope = first_number(getattr(dataset, "RescaleSlope", 1.0)) or 1.0
    intercept = first_number(getattr(dataset, "RescaleIntercept", 0.0)) or 0.0

    if raw_pad is not None:
        try:
            raw_pad_value = float(raw_pad)
            pad_hu = raw_pad_value * slope + intercept
            mask |= np.isclose(hu, pad_hu, atol=max(abs(slope), 1.0))
        except Exception:
            pass

    if raw_pad is not None and raw_pad_limit is not None:
        try:
            lo = min(float(raw_pad), float(raw_pad_limit)) * slope + intercept
            hi = max(float(raw_pad), float(raw_pad_limit)) * slope + intercept
            mask |= (hu >= lo) & (hu <= hi)
        except Exception:
            pass

    return mask


def window_to_uint8(hu: np.ndarray, center: float, width: float, padding_mask: np.ndarray | None = None) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    clipped = np.clip(hu, lo, hi)
    scaled = (clipped - lo) / max(hi - lo, 1e-6)
    out = np.round(scaled * 255.0).astype(np.uint8)
    if padding_mask is not None:
        out[padding_mask] = 0
    return out


def percentile_fallback_uint8(hu: np.ndarray, padding_mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(hu)
    if padding_mask is not None:
        valid &= ~padding_mask
    values = hu[valid]
    if values.size == 0:
        return np.zeros(hu.shape, dtype=np.uint8)
    lo = float(np.percentile(values, 0.5))
    hi = float(np.percentile(values, 99.5))
    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        return np.zeros(hu.shape, dtype=np.uint8)
    clipped = np.clip(hu, lo, hi)
    out = np.round((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    if padding_mask is not None:
        out[padding_mask] = 0
    return out


def choose_single_voi_window(dataset: Any, region: str) -> tuple[float, float] | None:
    window_centers = getattr(dataset, "WindowCenter", None)
    window_widths = getattr(dataset, "WindowWidth", None)
    if window_centers is None or window_widths is None:
        return None

    centers = list(window_centers) if _is_multi_value(window_centers) else [window_centers]
    widths = list(window_widths) if _is_multi_value(window_widths) else [window_widths]
    explanations_raw = getattr(dataset, "WindowCenterWidthExplanation", None)
    explanations = (
        list(explanations_raw)
        if _is_multi_value(explanations_raw)
        else [explanations_raw] * max(len(centers), len(widths))
    )

    pairs: list[tuple[float, float, str]] = []
    for center, width, explanation in zip(centers, widths, explanations):
        center_value = first_number(center)
        width_value = first_number(width)
        if center_value is None or width_value is None or width_value <= 0:
            continue
        pairs.append((center_value, width_value, first_text(explanation).lower()))

    if not pairs:
        return None

    preferred_terms = {
        "thorax": ["lung", "mediast", "soft", "chest"],
        "thorax_abdomen": ["soft", "lung", "abd", "mediast"],
        "abdomen_pelvis": ["soft", "abd", "liver", "renal", "pelvis"],
        "head_neck": ["soft", "neck", "thyroid"],
        "mixed_sarc": ["soft", "bone", "extrem"],
        "generic": ["soft", "standard", "abd", "body"],
    }
    terms = preferred_terms.get(region, preferred_terms["generic"])
    for term in terms:
        for center_value, width_value, explanation in pairs:
            if term and term in explanation:
                return center_value, width_value
    center_value, width_value, _explanation = pairs[0]
    return center_value, width_value


def _invert_if_needed(dataset: Any, image: np.ndarray) -> np.ndarray:
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        return 255 - image
    return image


def _normalize_render_mode(render_mode: str) -> str:
    normalized = str(render_mode).strip().lower()
    aliases = {
        "rgb_multiwindow": "ct_rgb_multiwindow",
        "grayscale": "ct_grayscale",
        "dicom_voi": "ct_dicom_voi",
        "gray": "ct_grayscale",
    }
    return aliases.get(normalized, normalized)


def _as_model_ready_output(gray: np.ndarray, force_rgb_output: bool) -> np.ndarray:
    if force_rgb_output:
        return np.stack([gray, gray, gray], axis=-1)
    return gray


def _named_window(region: str, preferred_name: str | None) -> tuple[float, float]:
    windows = REGION_WINDOWS.get(region, REGION_WINDOWS["generic"])
    if preferred_name:
        for name, center, width in windows:
            if name == preferred_name:
                return center, width
    default_name = DEFAULT_PRIMARY_WINDOW_BY_REGION.get(region, "soft_tissue")
    for name, center, width in windows:
        if name == default_name:
            return center, width
    _name, center, width = windows[0]
    return center, width


def _render_ct(
    *,
    dataset: Any,
    dicom_path: Path,
    hu: np.ndarray,
    render_config: RadiologyPngRenderConfig,
    apply_voi_lut: Any | None,
) -> np.ndarray:
    region = infer_region(dataset, dicom_path)
    padding_mask = get_padding_mask(dataset, hu) if render_config.apply_padding_mask else None
    render_mode = _normalize_render_mode(render_config.render_mode)

    if render_mode == "legacy_percentile":
        gray = percentile_fallback_uint8(hu, padding_mask)
        return _invert_if_needed(dataset, _as_model_ready_output(gray, render_config.force_rgb_output))

    if render_mode == "ct_rgb_multiwindow":
        windows = REGION_WINDOWS.get(region, REGION_WINDOWS["generic"])
        channels = [window_to_uint8(hu, center, width, padding_mask) for _name, center, width in windows[:3]]
        rgb = np.stack(channels, axis=-1)
        return _invert_if_needed(dataset, rgb)

    if render_mode == "ct_dicom_voi" and apply_voi_lut is not None:
        try:
            transformed = apply_voi_lut(hu, dataset)
            transformed = np.asarray(transformed, dtype=np.float32)
            transformed = _squeeze_to_2d(transformed, dicom_path)
            valid = np.isfinite(transformed)
            if padding_mask is not None:
                valid &= ~padding_mask
            values = transformed[valid]
            if values.size > 0:
                lo = float(values.min())
                hi = float(values.max())
                if hi > lo:
                    gray = np.round(np.clip((transformed - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
                    if padding_mask is not None:
                        gray[padding_mask] = 0
                    return _invert_if_needed(dataset, _as_model_ready_output(gray, render_config.force_rgb_output))
        except Exception:
            pass

    if render_config.prefer_dicom_voi:
        voi_window = choose_single_voi_window(dataset, region)
        if voi_window is not None:
            center_value, width_value = voi_window
            gray = window_to_uint8(hu, center_value, width_value, padding_mask)
            return _invert_if_needed(dataset, _as_model_ready_output(gray, render_config.force_rgb_output))

    center_value, width_value = _named_window(region, render_config.primary_window_name)
    gray = window_to_uint8(hu, center_value, width_value, padding_mask)
    return _invert_if_needed(dataset, _as_model_ready_output(gray, render_config.force_rgb_output))


def _render_legacy_pixels(dataset: Any, dicom_path: Path, apply_modality_lut: Any, force_rgb_output: bool) -> np.ndarray:
    pixels = apply_modality_lut(dataset.pixel_array, dataset)
    pixels = _squeeze_to_2d(np.asarray(pixels), dicom_path)

    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = pixels.max() - pixels

    pixels = pixels.astype(np.float32)
    finite = np.isfinite(pixels)
    if not finite.any():
        gray = np.zeros(pixels.shape, dtype=np.uint8)
    else:
        values = pixels[finite]
        lo = float(np.percentile(values, 1.0))
        hi = float(np.percentile(values, 99.0))
        if hi <= lo:
            lo = float(values.min())
            hi = float(values.max())
        if hi <= lo:
            gray = np.zeros(pixels.shape, dtype=np.uint8)
        else:
            gray = np.round(np.clip((pixels - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)

    return _as_model_ready_output(gray, force_rgb_output)


def dicom_to_array(
    dicom_path: Path,
    pydicom: Any,
    apply_modality_lut: Any,
    *,
    apply_voi_lut: Any | None = None,
    render_config: RadiologyPngRenderConfig | None = None,
) -> np.ndarray:
    dataset = pydicom.dcmread(str(dicom_path))
    effective_render_config = render_config or RadiologyPngRenderConfig()
    modality = str(getattr(dataset, "Modality", "")).upper()

    if modality == "CT":
        hu = get_hu_array(dataset, dicom_path, apply_modality_lut)
        return _render_ct(
            dataset=dataset,
            dicom_path=dicom_path,
            hu=hu,
            render_config=effective_render_config,
            apply_voi_lut=apply_voi_lut,
        )

    return _render_legacy_pixels(dataset, dicom_path, apply_modality_lut, effective_render_config.force_rgb_output)


def dicom_to_rgb_array(
    dicom_path: Path,
    pydicom: Any,
    apply_modality_lut: Any,
    *,
    apply_voi_lut: Any | None = None,
    render_config: RadiologyPngRenderConfig | None = None,
) -> np.ndarray:
    effective_render_config = render_config or RadiologyPngRenderConfig()
    rgb_render_config = (
        effective_render_config
        if effective_render_config.force_rgb_output
        else RadiologyPngRenderConfig(
            render_mode=effective_render_config.render_mode,
            prefer_dicom_voi=effective_render_config.prefer_dicom_voi,
            apply_padding_mask=effective_render_config.apply_padding_mask,
            resize=effective_render_config.resize,
            force_rgb_output=True,
            primary_window_name=effective_render_config.primary_window_name,
        )
    )
    return dicom_to_array(
        dicom_path,
        pydicom,
        apply_modality_lut,
        apply_voi_lut=apply_voi_lut,
        render_config=rgb_render_config,
    )


def ensure_resized_png(
    *,
    dicom_path: Path,
    png_path: Path,
    overwrite_pngs: bool,
    pydicom: Any,
    apply_modality_lut: Any,
    apply_voi_lut: Any | None = None,
    render_config: RadiologyPngRenderConfig | None = None,
) -> bool:
    effective_render_config = render_config or RadiologyPngRenderConfig()
    if png_path.exists() and not overwrite_pngs:
        try:
            with Image.open(png_path) as existing:
                if existing.size[0] > 0 and existing.size[1] > 0:
                    resize = effective_render_config.resize
                    if resize is None or existing.size == (resize, resize):
                        return False
        except Exception:
            pass

    image_array = dicom_to_array(
        dicom_path,
        pydicom,
        apply_modality_lut,
        apply_voi_lut=apply_voi_lut,
        render_config=effective_render_config,
    )
    image = Image.fromarray(image_array)
    resize = effective_render_config.resize
    if resize is not None and resize > 0:
        image = image.resize((resize, resize), resample=Image.Resampling.BILINEAR)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)
    return True
