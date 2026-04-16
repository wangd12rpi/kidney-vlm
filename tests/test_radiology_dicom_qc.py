from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kidney_vlm.radiology import dicom_qc


def _write_series_files(series_dir: Path, *names: str) -> None:
    series_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (series_dir / name).write_bytes(b"placeholder")


def _fake_runtime(entries_by_name: dict[str, object], *, load_pixel_array=None):
    class FakeInvalidDicomError(Exception):
        pass

    def fake_dcmread(path: str, force: bool = True, specific_tags=None):
        del force, specific_tags
        entry = entries_by_name[Path(path).name]
        if isinstance(entry, Exception):
            raise entry
        return SimpleNamespace(**entry)

    return fake_dcmread, FakeInvalidDicomError, load_pixel_array


def test_analyze_series_accepts_valid_ct_series(monkeypatch, tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    _write_series_files(series_dir, "image-1.dcm", "image-2.dcm")
    runtime = _fake_runtime(
        {
            "image-1.dcm": {
                "Modality": "CT",
                "SOPClassUID": "1.2.3",
                "SOPInstanceUID": "1.2.3.1",
                "StudyInstanceUID": "study-1",
                "SeriesInstanceUID": "series-1",
                "InstanceNumber": 1,
                "ImageType": ["ORIGINAL", "PRIMARY"],
                "SeriesDescription": "Axial Abdomen",
                "ProtocolName": "Abdomen",
                "Rows": 512,
                "Columns": 512,
                "BitsAllocated": 16,
                "NumberOfFrames": 1,
            },
            "image-2.dcm": {
                "Modality": "CT",
                "SOPClassUID": "1.2.3",
                "SOPInstanceUID": "1.2.3.2",
                "StudyInstanceUID": "study-1",
                "SeriesInstanceUID": "series-1",
                "InstanceNumber": 2,
                "ImageType": ["ORIGINAL", "PRIMARY"],
                "SeriesDescription": "Axial Abdomen",
                "ProtocolName": "Abdomen",
                "Rows": 512,
                "Columns": 512,
                "BitsAllocated": 16,
                "NumberOfFrames": 1,
            },
        }
    )
    monkeypatch.setattr(dicom_qc, "_get_pydicom_runtime", lambda: runtime)

    image_records, series_record = dicom_qc.analyze_series(
        series_dir,
        dicom_qc.Config(min_usable_images=2),
    )

    assert series_record is None
    assert len(image_records) == 2
    assert all(record.readable_dicom for record in image_records)
    assert all(record.reject_reason is None for record in image_records)


def test_analyze_series_rejects_mixed_series_uids(monkeypatch, tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    _write_series_files(series_dir, "image-1.dcm", "image-2.dcm")
    runtime = _fake_runtime(
        {
            "image-1.dcm": {
                "Modality": "CT",
                "SOPClassUID": "1.2.3",
                "StudyInstanceUID": "study-1",
                "SeriesInstanceUID": "series-1",
                "Rows": 512,
                "Columns": 512,
                "BitsAllocated": 16,
                "NumberOfFrames": 1,
            },
            "image-2.dcm": {
                "Modality": "CT",
                "SOPClassUID": "1.2.3",
                "StudyInstanceUID": "study-1",
                "SeriesInstanceUID": "series-2",
                "Rows": 512,
                "Columns": 512,
                "BitsAllocated": 16,
                "NumberOfFrames": 1,
            },
        }
    )
    monkeypatch.setattr(dicom_qc, "_get_pydicom_runtime", lambda: runtime)

    _image_records, series_record = dicom_qc.analyze_series(
        series_dir,
        dicom_qc.Config(min_usable_images=1),
    )

    assert series_record is not None
    assert series_record.reject_reason == "mixed_series_instance_uid_in_directory"


def test_analyze_series_rejects_localizer_only_series(monkeypatch, tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    _write_series_files(series_dir, "image-1.dcm", "image-2.dcm")
    runtime = _fake_runtime(
        {
            "image-1.dcm": {
                "Modality": "CT",
                "SOPClassUID": "1.2.3",
                "StudyInstanceUID": "study-1",
                "SeriesInstanceUID": "series-1",
                "ImageType": ["LOCALIZER"],
                "SeriesDescription": "Scout",
                "ProtocolName": "Topogram",
                "Rows": 512,
                "Columns": 512,
                "BitsAllocated": 16,
                "NumberOfFrames": 1,
            },
            "image-2.dcm": {
                "Modality": "CT",
                "SOPClassUID": "1.2.3",
                "StudyInstanceUID": "study-1",
                "SeriesInstanceUID": "series-1",
                "ImageType": ["LOCALIZER"],
                "SeriesDescription": "Scout",
                "ProtocolName": "Topogram",
                "Rows": 512,
                "Columns": 512,
                "BitsAllocated": 16,
                "NumberOfFrames": 1,
            },
        }
    )
    monkeypatch.setattr(dicom_qc, "_get_pydicom_runtime", lambda: runtime)

    image_records, series_record = dicom_qc.analyze_series(
        series_dir,
        dicom_qc.Config(min_usable_images=1),
    )

    assert series_record is not None
    assert series_record.reject_reason == "series_is_localizer_or_scout"
    assert [record.reject_reason for record in image_records] == ["localizer_or_scout", "localizer_or_scout"]


def test_analyze_series_marks_decode_failures_when_enabled(monkeypatch, tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    _write_series_files(series_dir, "image-1.dcm")
    runtime = _fake_runtime(
        {
            "image-1.dcm": {
                "Modality": "CT",
                "SOPClassUID": "1.2.3",
                "StudyInstanceUID": "study-1",
                "SeriesInstanceUID": "series-1",
                "Rows": 512,
                "Columns": 512,
                "BitsAllocated": 16,
                "NumberOfFrames": 1,
            },
        },
        load_pixel_array=lambda _path: (_ for _ in ()).throw(ValueError("decode failed")),
    )
    monkeypatch.setattr(dicom_qc, "_get_pydicom_runtime", lambda: runtime)

    image_records, series_record = dicom_qc.analyze_series(
        series_dir,
        dicom_qc.Config(min_usable_images=1, decode_pixels=True),
    )

    assert series_record is not None
    assert series_record.reject_reason == "all_images_rejected"
    assert image_records[0].reject_reason == "pixel_decode_failed"
    assert image_records[0].pixel_decode_checked is True


def test_find_candidate_series_dirs_ignores_generated_artifacts(tmp_path: Path) -> None:
    usable_series = tmp_path / "usable"
    ignored_series = tmp_path / "ignored"
    _write_series_files(usable_series, "image-1.dcm", "image-1.json", "image-1.png")
    _write_series_files(ignored_series, "preview.png", "metadata.json")

    series_dirs = dicom_qc.find_candidate_series_dirs(tmp_path)

    assert series_dirs == [usable_series]
