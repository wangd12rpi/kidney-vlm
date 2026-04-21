from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "01_pathology_png" / "01_extract_pathology_pngs.py"
    spec = importlib.util.spec_from_file_location("pathology_png_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_slide_kind_filter_accepts_dx_only() -> None:
    module = _load_module()

    assert module._slide_kind("TCGA-AA-3855-01Z-00-DX1.f305ce6c") == "DX"
    assert module._slide_kind("TCGA-AA-3855-01A-01-BS1.f305ce6c") == "BS"
    assert module._slide_kind_allowed("TCGA-AA-3855-01Z-00-DX1.f305ce6c", {"DX"}) is True
    assert module._slide_kind_allowed("TCGA-AA-3855-01A-01-BS1.f305ce6c", {"DX"}) is False


def test_register_slide_thumbnail_writes_portable_png_paths(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    output_dir = tmp_path / "data" / "pathology_png" / "TCGA-AA-0001"
    output_dir.mkdir(parents=True)
    thumbnail = output_dir / "slide-a__thumbnail.png"
    thumbnail.write_bytes(b"not-empty")
    frame = pd.DataFrame([{"sample_id": "tcga-1"}])

    module._register_slide_thumbnail(
        frame,
        row_idx=0,
        slide_stem="slide-a",
        thumbnail_paths=[thumbnail],
    )

    assert frame.at[0, "pathology_png_thumbnail_paths"] == [
        "data/pathology_png/TCGA-AA-0001/slide-a__thumbnail.png"
    ]
