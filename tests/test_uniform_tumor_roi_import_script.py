from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "01_pathology_png" / "02_import_uniform_tumor_rois.py"
    spec = importlib.util.spec_from_file_location("uniform_tumor_roi_import_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_tcga_slide_key() -> None:
    module = _load_module()

    filename = "TCGA-02-0001-01Z-00-DX1_cont0_x23038_y18986"

    assert module._extract_slide_key(filename) == "TCGA-02-0001-01Z-00-DX1"


def test_build_registry_matches_uses_wsi_slide_key() -> None:
    module = _load_module()
    frame = pd.DataFrame(
        [
            {
                "sample_id": "tcga-a",
                "patient_id": "TCGA-02-0001",
                "pathology_wsi_paths": [
                    "data/raw/tcga/pathology/TCGA-GBM/TCGA-02-0001/TCGA-02-0001-01Z-00-DX1.uuid.svs"
                ],
            }
        ]
    )

    slide_matches = module._build_registry_matches(frame)

    assert slide_matches["TCGA-02-0001-01Z-00-DX1"] == 0


def test_iter_local_roi_paths_selects_uniform_tumor_images(tmp_path: Path) -> None:
    module = _load_module()
    roi_dir = tmp_path / "data" / "pathology_png" / "TCGA-02-0001"
    roi_dir.mkdir(parents=True)
    roi = roi_dir / "TCGA-02-0001-01Z-00-DX1__uniform_tumor_8k__roi.png"
    thumbnail = roi_dir / "TCGA-02-0001-01Z-00-DX1__thumbnail.png"
    roi.write_bytes(b"roi")
    thumbnail.write_bytes(b"thumbnail")

    assert module.iter_local_roi_paths(tmp_path / "data" / "pathology_png") == [roi]


def test_register_local_roi_paths_matches_by_slide_key(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    roi_path = (
        tmp_path
        / "data"
        / "pathology_png"
        / "TCGA-02-0001"
        / "TCGA-02-0001-01Z-00-DX1__uniform_tumor_8k__roi.png"
    )
    roi_path.parent.mkdir(parents=True)
    roi_path.write_bytes(b"roi")
    frame = pd.DataFrame(
        [
            {
                "sample_id": "tcga-a",
                "patient_id": "TCGA-02-0001",
                "pathology_wsi_paths": [
                    "data/raw/tcga/pathology/TCGA-GBM/TCGA-02-0001/TCGA-02-0001-01Z-00-DX1.uuid.svs"
                ],
            }
        ]
    )

    updated_df, stats = module.register_local_roi_paths(frame, [roi_path])

    assert stats["registered_roi_paths"] == 1
    assert updated_df.at[0, "pathology_png_roi_paths"] == [
        "data/pathology_png/TCGA-02-0001/TCGA-02-0001-01Z-00-DX1__uniform_tumor_8k__roi.png"
    ]


def test_register_local_roi_paths_reports_unmatched_roi(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    roi_path = (
        tmp_path
        / "data"
        / "pathology_png"
        / "TCGA-02-0001"
        / "TCGA-02-0001-01Z-00-DX1__uniform_tumor_8k__roi.png"
    )
    roi_path.parent.mkdir(parents=True)
    roi_path.write_bytes(b"roi")
    frame = pd.DataFrame(
        [
            {
                "sample_id": "tcga-a",
                "patient_id": "TCGA-02-0001",
                "pathology_wsi_paths": [
                    "data/raw/tcga/pathology/TCGA-GBM/TCGA-02-0001/TCGA-02-0001-01Z-00-DX2.uuid.svs"
                ],
            }
        ]
    )

    _updated_df, stats = module.register_local_roi_paths(frame, [roi_path])

    assert stats["registered_roi_paths"] == 0
    assert stats["skipped_missing_match"] == 1
    assert stats["missing_match_examples"] == [
        "data/pathology_png/TCGA-02-0001/TCGA-02-0001-01Z-00-DX1__uniform_tumor_8k__roi.png"
    ]


def test_uniform_tumor_config_is_local_registration_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "conf" / "01_pathology_png" / "02_import_uniform_tumor_rois.yaml").read_text()

    assert "roi_root:" in text
    assert "repo_id:" not in text
    assert "cache_root:" not in text
    assert "output_root:" not in text
