from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kidney_vlm.radiology.png_preprocessing import (
    RadiologyPngRenderConfig,
    dicom_to_rgb_array,
    normalize_to_uint8,
    window_to_uint8,
)


def _fake_pydicom(dataset: object) -> SimpleNamespace:
    return SimpleNamespace(dcmread=lambda _path: dataset)


def test_ct_rgb_multiwindow_uses_region_windows_and_padding_mask() -> None:
    pixels = np.array(
        [
            [-2000.0, -600.0, 50.0],
            [60.0, 300.0, 1200.0],
        ],
        dtype=np.float32,
    )
    dataset = SimpleNamespace(
        Modality="CT",
        pixel_array=pixels,
        PhotometricInterpretation="MONOCHROME2",
        BodyPartExamined="abdomen",
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        PixelPaddingValue=-2000.0,
    )

    rgb = dicom_to_rgb_array(
        Path("data/raw/tcga/radiology/TCGA-KIRC/case/study/series/image-1.dcm"),
        _fake_pydicom(dataset),
        lambda values, _dataset: values,
        render_config=RadiologyPngRenderConfig(render_mode="ct_rgb_multiwindow", apply_padding_mask=True),
    )

    expected_soft = window_to_uint8(pixels, 50.0, 400.0, padding_mask=np.array([[True, False, False], [False, False, False]]))
    expected_detail = window_to_uint8(pixels, 60.0, 180.0, padding_mask=np.array([[True, False, False], [False, False, False]]))
    expected_bone = window_to_uint8(pixels, 300.0, 1500.0, padding_mask=np.array([[True, False, False], [False, False, False]]))

    assert rgb.shape == (2, 3, 3)
    assert np.array_equal(rgb[..., 0], expected_soft)
    assert np.array_equal(rgb[..., 1], expected_detail)
    assert np.array_equal(rgb[..., 2], expected_bone)
    assert np.array_equal(rgb[0, 0], np.array([0, 0, 0], dtype=np.uint8))
    assert not np.array_equal(rgb[..., 0], rgb[..., 1])


def test_ct_grayscale_prefers_dicom_voi_window() -> None:
    pixels = np.array(
        [
            [-100.0, 50.0],
            [100.0, 300.0],
        ],
        dtype=np.float32,
    )
    dataset = SimpleNamespace(
        Modality="CT",
        pixel_array=pixels,
        PhotometricInterpretation="MONOCHROME2",
        BodyPartExamined="abdomen",
        WindowCenter=[-600.0, 60.0],
        WindowWidth=[1500.0, 180.0],
        WindowCenterWidthExplanation=["lung", "abdomen detail"],
    )

    rgb = dicom_to_rgb_array(
        Path("data/raw/tcga/radiology/TCGA-KIRC/case/study/series/image-2.dcm"),
        _fake_pydicom(dataset),
        lambda values, _dataset: values,
        render_config=RadiologyPngRenderConfig(render_mode="ct_grayscale", prefer_dicom_voi=True),
    )

    expected = window_to_uint8(pixels, 60.0, 180.0)
    assert np.array_equal(rgb[..., 0], expected)
    assert np.array_equal(rgb[..., 1], expected)
    assert np.array_equal(rgb[..., 2], expected)


def test_non_ct_uses_legacy_percentile_normalization() -> None:
    pixels = np.array(
        [
            [0.0, 10.0],
            [20.0, 30.0],
        ],
        dtype=np.float32,
    )
    dataset = SimpleNamespace(
        Modality="MR",
        pixel_array=pixels,
        PhotometricInterpretation="MONOCHROME2",
    )

    rgb = dicom_to_rgb_array(
        Path("data/raw/tcga/radiology/TCGA-KIRC/case/study/series/image-3.dcm"),
        _fake_pydicom(dataset),
        lambda values, _dataset: values,
        render_config=RadiologyPngRenderConfig(render_mode="ct_rgb_multiwindow"),
    )

    expected = normalize_to_uint8(pixels)
    assert np.array_equal(rgb[..., 0], expected)
    assert np.array_equal(rgb[..., 1], expected)
    assert np.array_equal(rgb[..., 2], expected)


def test_ct_dicom_voi_uses_apply_voi_lut_when_available() -> None:
    pixels = np.array(
        [
            [-100.0, 0.0],
            [100.0, 200.0],
        ],
        dtype=np.float32,
    )
    transformed = np.array(
        [
            [0.0, 0.0],
            [1.0, 3.0],
        ],
        dtype=np.float32,
    )
    dataset = SimpleNamespace(
        Modality="CT",
        pixel_array=pixels,
        PhotometricInterpretation="MONOCHROME2",
        BodyPartExamined="abdomen",
    )

    rgb = dicom_to_rgb_array(
        Path("data/raw/tcga/radiology/TCGA-KIRC/case/study/series/image-4.dcm"),
        _fake_pydicom(dataset),
        lambda values, _dataset: values,
        apply_voi_lut=lambda _hu, _dataset: transformed,
        render_config=RadiologyPngRenderConfig(render_mode="ct_dicom_voi"),
    )

    expected = np.array(
        [
            [0, 0],
            [85, 255],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(rgb[..., 0], expected)
    assert np.array_equal(rgb[..., 1], expected)
    assert np.array_equal(rgb[..., 2], expected)
