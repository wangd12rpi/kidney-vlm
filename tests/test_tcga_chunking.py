from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from kidney_vlm.data.tcga_chunking import (
    build_patient_chunk_label,
    build_portable_radiology_chunk_layout,
    resolve_patient_chunk_descriptor,
)


def test_resolve_patient_chunk_descriptor_builds_human_chunk_label() -> None:
    cfg = OmegaConf.create({"patient_chunk": {"index": 2, "size": 64}})
    chunk = resolve_patient_chunk_descriptor(cfg)
    assert chunk is not None
    assert chunk.index == 2
    assert chunk.size == 64
    assert chunk.label == build_patient_chunk_label(2)


def test_build_portable_radiology_chunk_layout_uses_chunk_bundle_paths(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "radiology_process_root": str(tmp_path / "data" / "processes" / "radiology"),
            "patient_chunk": {"index": 0, "size": 32},
        }
    )

    layout = build_portable_radiology_chunk_layout(tcga_cfg=cfg, repo_root=tmp_path)

    assert layout is not None
    assert layout.chunk.label == "chunk1"
    assert layout.bundle_root == tmp_path / "data" / "processes" / "radiology" / "chunk1"
    assert layout.png_root == layout.bundle_root / "pngs"
    assert layout.mask_root == layout.bundle_root / "mask_medicalsam3"
    assert layout.feature_store_path == layout.bundle_root / "features_medsiglip448" / "chunk1.h5"
    assert layout.qc_detail_root == layout.bundle_root / "qc"
    assert layout.qc_report_path == layout.bundle_root / "qc_report.jsonl"
    assert layout.registry_path == layout.bundle_root / "registry" / "tcga.parquet"
    assert layout.manifest_path == layout.bundle_root / "chunk_manifest.json"
