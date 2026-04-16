from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RADIOLOGY_PROCESS_ROOT = "data/processes/radiology"


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


@dataclass(frozen=True)
class PatientChunkDescriptor:
    index: int
    size: int
    label: str


@dataclass(frozen=True)
class PortableRadiologyChunkLayout:
    chunk: PatientChunkDescriptor
    bundle_root: Path
    png_root: Path
    mask_root: Path
    feature_store_path: Path
    qc_detail_root: Path
    qc_report_path: Path
    registry_dir: Path
    registry_path: Path
    manifest_path: Path


def build_patient_chunk_label(index: int) -> str:
    return f"chunk{int(index) + 1}"


def resolve_patient_chunk_descriptor(tcga_cfg: Any) -> PatientChunkDescriptor | None:
    chunk_cfg = _config_get(tcga_cfg, "patient_chunk")
    if chunk_cfg is None:
        return None

    chunk_size = _optional_int(_config_get(chunk_cfg, "size"))
    if chunk_size is None:
        return None
    if chunk_size <= 0:
        raise ValueError("data.source.tcga.patient_chunk.size must be a positive integer when set.")

    chunk_index = int(_config_get(chunk_cfg, "index", 0))
    if chunk_index < 0:
        raise ValueError("data.source.tcga.patient_chunk.index must be >= 0.")

    return PatientChunkDescriptor(
        index=chunk_index,
        size=chunk_size,
        label=build_patient_chunk_label(chunk_index),
    )


def resolve_radiology_process_root(tcga_cfg: Any, *, repo_root: str | Path) -> Path:
    configured = str(
        _config_get(
            tcga_cfg,
            "radiology_process_root",
            DEFAULT_RADIOLOGY_PROCESS_ROOT,
        )
    ).strip() or DEFAULT_RADIOLOGY_PROCESS_ROOT
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = Path(repo_root).expanduser().resolve() / root
    return root.resolve()


def build_portable_radiology_chunk_layout(
    *,
    tcga_cfg: Any,
    repo_root: str | Path,
    feature_store_dirname: str = "features_medsiglip448",
    mask_root_dirname: str = "mask_medicalsam3",
    png_root_dirname: str = "pngs",
    qc_root_dirname: str = "qc",
    registry_dirname: str = "registry",
    manifest_filename: str = "chunk_manifest.json",
) -> PortableRadiologyChunkLayout | None:
    chunk = resolve_patient_chunk_descriptor(tcga_cfg)
    if chunk is None:
        return None

    process_root = resolve_radiology_process_root(tcga_cfg, repo_root=repo_root)
    bundle_root = (process_root / chunk.label).resolve()
    registry_dir = bundle_root / registry_dirname
    return PortableRadiologyChunkLayout(
        chunk=chunk,
        bundle_root=bundle_root,
        png_root=(bundle_root / png_root_dirname).resolve(),
        mask_root=(bundle_root / mask_root_dirname).resolve(),
        feature_store_path=(bundle_root / feature_store_dirname / f"{chunk.label}.h5").resolve(),
        qc_detail_root=(bundle_root / qc_root_dirname).resolve(),
        qc_report_path=(bundle_root / "qc_report.jsonl").resolve(),
        registry_dir=registry_dir.resolve(),
        registry_path=(registry_dir / "tcga.parquet").resolve(),
        manifest_path=(bundle_root / manifest_filename).resolve(),
    )
