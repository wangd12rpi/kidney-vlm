from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_print_registry_status_uses_field_per_line_sample_output(tmp_path: Path) -> None:
    registry_path = tmp_path / "unified.parquet"
    frame = pd.DataFrame(
        [
            {
                "sample_id": "tcga-1",
                "source": "tcga",
                "patient_id": "TCGA-AA-0001",
                "study_id": "case-1",
                "split": "train",
                "pathology_wsi_paths": ["/tmp/slide.svs"],
                "radiology_image_paths": ["tcia://TCGA-KIRC/TCGA-AA-0001/1.2.3"],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
                "pathology_tile_embedding_paths": ["/tmp/tile.npy"],
                "pathology_slide_embedding_paths": ["/tmp/slide.npy"],
                "radiology_embedding_paths": ["/tmp/rad.npy"],
                "biomarkers_text": "project: TCGA-KIRC",
                "question": "",
                "answer": "",
            }
        ]
    )
    frame.to_parquet(registry_path, index=False)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "data" / "02_print_registry_status.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--registry", str(registry_path)],
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "Sample row (tcga):" in stdout
    assert "sample_id: tcga-1" in stdout
    assert "pathology_tile_embedding_paths: ['/tmp/tile.npy']" in stdout
    assert "radiology_embedding_paths: ['/tmp/rad.npy']" in stdout
    assert "sample_id source patient_id" not in stdout
