#!/usr/bin/env python3
"""Build per-case genomics text blocks for caption generation and training.

This stage consumes the raw genomics artifacts registered by
`scripts/05_text_genomics/01_download_tcga_extra_genomics.py` plus the existing TCGA RNA
paths in the unified registry. For each case with at least one usable genomics
artifact it writes:

    <features_root>/genomics_text_blocks/<source>/<project_id>/<patient_id>/
        teacher.txt    # full teacher-facing genomics context
        student.txt    # restricted text channel for train/inference
        features.json  # structured feature/debug payload

The teacher/student split follows the project genomics guidelines:
DNAm/RNA-derived summaries are teacher-only because the student should learn
them from CpGPT/BulkFormer embeddings, while mutation/CNA/TMB/MSI/HRD remain
in the student text channel.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from omegaconf import DictConfig

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.genomics import (
    cohort_config,
    dnam_text_features as dnam_ext,
    mutation_cna_text_features as mut_ext,
    rna_text_features as rna_ext,
    text_block,
)
from kidney_vlm.genomics.case_filter import load_patient_ids_from_json
from kidney_vlm.genomics.registry_integration import (
    update_registry_with_genomics_json_manifest,
)
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg(source_name: str = "tcga", overrides: list[str] | None = None) -> "DictConfig":
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        missing_name = exc.name or "hydra-core"
        raise SystemExit(
            f"Missing Python dependency '{missing_name}' while starting "
            "scripts/05_text_genomics/02_build_genomics_text_blocks.py.\n"
            f"Active interpreter: {sys.executable}\n"
            "Install the project dependencies first, then rerun this script."
        ) from exc

    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        base_cfg = compose(config_name="config")
    OmegaConf.set_struct(base_cfg, False)
    source_cfg_path = conf_dir / "data" / "sources" / f"{source_name}.yaml"
    if source_cfg_path.exists():
        merged = OmegaConf.merge(base_cfg, OmegaConf.load(source_cfg_path))
    else:
        merged = base_cfg
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(merged, False)
    merged.project.root_dir = str(ROOT)
    return merged


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _optional_float(value: Any) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _optional_int(value: Any) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    return int(text)


def _resolve_existing_path(path_value: str | Path | None, *, root_dir: Path) -> str | None:
    text = _clean_text(path_value)
    if not text or "://" in text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return str(path.resolve()) if path.exists() else None


def _pick_first_existing(paths: list[str], *, root_dir: Path) -> str | None:
    for path_value in paths:
        resolved = _resolve_existing_path(path_value, root_dir=root_dir)
        if resolved:
            return resolved
    return None


def _pick_manifest_path(
    extra_paths: dict[str, list[str]],
    modality: str,
    *,
    root_dir: Path,
) -> str | None:
    return _pick_first_existing(extra_paths.get(modality, []), root_dir=root_dir)


def _pick_registry_path(
    case_row: dict[str, Any],
    columns: list[str],
    *,
    root_dir: Path,
) -> str | None:
    for column in columns:
        resolved = _pick_first_existing(_as_list(case_row.get(column)), root_dir=root_dir)
        if resolved:
            return resolved
    return None


def _select_rna_path(case_row: dict[str, Any], *, root_dir: Path) -> str | None:
    """Pick a case RNA STAR TSV, preferring tumor samples when metadata exists."""
    paths = _as_list(case_row.get("genomics_rna_bulk_paths"))
    sample_types = _as_list(case_row.get("genomics_rna_bulk_sample_types"))
    if paths and sample_types and len(paths) == len(sample_types):
        paired = list(zip(paths, sample_types, strict=False))
        tumor_paths = [
            path
            for path, sample_type in paired
            if "tumor" in sample_type.lower() or "metastatic" in sample_type.lower()
        ]
        resolved = _pick_first_existing(tumor_paths, root_dir=root_dir)
        if resolved:
            return resolved
    resolved = _pick_first_existing(paths, root_dir=root_dir)
    if resolved:
        return resolved
    return _pick_registry_path(
        case_row,
        ["rna_bulk_local_path", "rna_bulk_path"],
        root_dir=root_dir,
    )


def _load_latest_extra_genomics_manifest(
    manifests_root: Path,
    source_name: str,
    explicit_path: str | Path | None = None,
) -> pd.DataFrame:
    if explicit_path:
        manifest_path = Path(str(explicit_path)).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Extra-genomics manifest not found: {manifest_path}")
        return pd.read_parquet(manifest_path)

    candidates = sorted(
        manifests_root.glob(f"{source_name}_extra_genomics_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return pd.DataFrame()
    return pd.read_parquet(candidates[0])


def _index_manifest_by_patient(manifest_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, list[str]]]:
    index: dict[tuple[str, str], dict[str, list[str]]] = {}
    required = {"project_id", "patient_id", "modality", "output_path"}
    if manifest_df.empty or not required.issubset(manifest_df.columns):
        return index
    for row in manifest_df.itertuples(index=False):
        key = (str(row.project_id).strip(), str(row.patient_id).strip())
        if not key[0] or not key[1]:
            continue
        entry = index.setdefault(key, {})
        entry.setdefault(str(row.modality).strip(), []).append(str(row.output_path).strip())
    return index


def _available_modalities(
    *,
    beta_path: str | None,
    rna_path: str | None,
    maf_path: str | None,
    gene_cna_path: str | None,
    segment_cna_path: str | None,
) -> list[str]:
    modalities: list[str] = []
    if beta_path:
        modalities.append("dna_methylation_beta")
    if rna_path:
        modalities.append("rna_bulk")
    if maf_path:
        modalities.append("mutation_maf")
    if gene_cna_path:
        modalities.append("copy_number_gene")
    if segment_cna_path:
        modalities.append("copy_number_segment")
    return modalities


def process_case(
    *,
    project_id: str,
    patient_id: str,
    case_row: dict[str, Any],
    extra_genomics_by_patient: dict[tuple[str, str], dict[str, list[str]]],
    output_root: Path,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    extra_paths = extra_genomics_by_patient.get((project_id, patient_id), {})

    beta_path = _pick_manifest_path(extra_paths, "dna_methylation_beta", root_dir=root_dir)
    beta_path = beta_path or _pick_registry_path(
        case_row,
        ["genomics_dna_methylation_paths"],
        root_dir=root_dir,
    )

    gene_cna_path = _pick_manifest_path(extra_paths, "copy_number_gene", root_dir=root_dir)
    gene_cna_path = gene_cna_path or _pick_registry_path(
        case_row,
        ["genomics_cnv_gene_paths"],
        root_dir=root_dir,
    )

    segment_cna_path = _pick_manifest_path(extra_paths, "copy_number_segment", root_dir=root_dir)
    segment_cna_path = segment_cna_path or _pick_registry_path(
        case_row,
        ["genomics_cnv_segment_paths"],
        root_dir=root_dir,
    )

    maf_path = _pick_manifest_path(extra_paths, "mutation_maf", root_dir=root_dir)
    maf_path = maf_path or _pick_registry_path(
        case_row,
        ["genomics_mutation_paths"],
        root_dir=root_dir,
    )

    rna_path = _select_rna_path(case_row, root_dir=root_dir)
    fusion_path = _pick_registry_path(
        case_row,
        ["genomics_rna_fusion_paths", "rna_fusion_local_path"],
        root_dir=root_dir,
    )

    chronological_age = _optional_float(case_row.get("age_at_diagnosis_years"))
    methylation_subtype = (
        _clean_text(case_row.get("genomics_dna_methylation_subtype"))
        or _clean_text(case_row.get("tcga_methylation_subtype"))
        or None
    )
    msi_status = _clean_text(case_row.get("genomics_msi_status")) or _clean_text(
        case_row.get("tcga_msi_status")
    )
    hrd_score = _optional_float(case_row.get("genomics_hrd_score"))
    if hrd_score is None:
        hrd_score = _optional_float(case_row.get("tcga_hrd_score"))

    dnam_features_dict: dict[str, Any] | None = None
    rna_features_dict: dict[str, Any] | None = None
    mut_cna_features_dict: dict[str, Any] | None = None
    errors: list[str] = []

    if beta_path:
        try:
            dnam_features = dnam_ext.extract_dnam_text_features(
                beta_tsv_path=beta_path,
                project_id=project_id,
                chronological_age_years=chronological_age,
                methylation_subtype_label=methylation_subtype,
            )
            dnam_features_dict = dnam_features.to_dict()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"dnam:{exc.__class__.__name__}: {exc}")

    if rna_path:
        try:
            rna_features = rna_ext.extract_rna_text_features(
                star_tsv_path=rna_path,
                project_id=project_id,
                fusion_tsv_path=fusion_path,
            )
            rna_features_dict = rna_features.to_dict()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"rna:{exc.__class__.__name__}: {exc}")

    if maf_path or gene_cna_path or segment_cna_path:
        try:
            mut_cna_features = mut_ext.extract_mutation_cna_text_features(
                maf_path=maf_path,
                gene_cna_path=gene_cna_path,
                segment_cna_path=segment_cna_path,
                project_id=project_id,
                msi_status=msi_status or None,
                hrd_score=hrd_score,
            )
            mut_cna_features_dict = mut_cna_features.to_dict()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"mut_cna:{exc.__class__.__name__}: {exc}")

    integrated = text_block.derive_integrated_surrogates(
        dnam_features=dnam_features_dict,
        rna_features=rna_features_dict,
        mut_cna_features=mut_cna_features_dict,
        project_id=project_id,
    )

    teacher_text = text_block.assemble_teacher_text_block(
        dnam_features=dnam_features_dict,
        rna_features=rna_features_dict,
        mut_cna_features=mut_cna_features_dict,
        integrated_surrogates=integrated,
        project_id=project_id,
    )
    student_text = text_block.assemble_student_text_block(
        mut_cna_features=mut_cna_features_dict,
        project_id=project_id,
    )

    case_dir = output_root / project_id / patient_id
    case_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = case_dir / "teacher.txt"
    student_path = case_dir / "student.txt"
    features_json_path = case_dir / "features.json"
    teacher_path.write_text(teacher_text, encoding="utf-8")
    student_path.write_text(student_text, encoding="utf-8")

    available_modalities = _available_modalities(
        beta_path=beta_path,
        rna_path=rna_path,
        maf_path=maf_path,
        gene_cna_path=gene_cna_path,
        segment_cna_path=segment_cna_path,
    )
    features_json_path.write_text(
        json.dumps(
            {
                "project_id": project_id,
                "patient_id": patient_id,
                "inputs": {
                    "dna_methylation_beta": beta_path,
                    "rna_bulk": rna_path,
                    "mutation_maf": maf_path,
                    "copy_number_gene": gene_cna_path,
                    "copy_number_segment": segment_cna_path,
                    "rna_fusion": fusion_path,
                },
                "available_modalities": available_modalities,
                "dnam": dnam_features_dict,
                "rna": rna_features_dict,
                "mut_cna": mut_cna_features_dict,
                "integrated_surrogates": integrated,
                "errors": errors,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    return {
        "project_id": project_id,
        "patient_id": patient_id,
        "teacher_text_path": str(teacher_path),
        "student_text_path": str(student_path),
        "features_json_path": str(features_json_path),
        "genomics_json_path": str(features_json_path),
        "available_modalities": available_modalities,
        "dnam_available": dnam_features_dict is not None,
        "rna_available": rna_features_dict is not None,
        "mut_cna_available": mut_cna_features_dict is not None,
        "errors": "; ".join(errors) if errors else "",
    }


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _resolve_features_root(cfg: "DictConfig") -> Path:
    if "features_root" in cfg.data:
        return Path(str(cfg.data.features_root))
    return Path(str(cfg.data.raw_root)).parent / "features"


def _filter_source_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "source" in frame.columns:
        return frame[frame["source"].fillna("").astype(str) == source_name].copy()
    if "source_name" in frame.columns:
        return frame[frame["source_name"].fillna("").astype(str) == source_name].copy()
    return frame.copy()


def _case_subset_patient_filter(text_cfg: Any, *, root_dir: Path = ROOT) -> tuple[list[str], Path | None]:
    case_subset = str(text_cfg.get("case_subset", "") or "").strip().lower()
    case_cfg = text_cfg.get("pathology_cases", {})

    # Backward compatibility if callers copied the old extra_genomics shape.
    if not case_subset and bool(case_cfg.get("enabled", False)):
        case_subset = "pathology_cases"
    case_subset = case_subset or "all"

    if case_subset == "all":
        return [], None
    if case_subset != "pathology_cases":
        raise ValueError(
            "data.source.text_genomics.case_subset must be one of "
            "['all', 'pathology_cases']."
        )

    json_path = Path(str(case_cfg.get("path", "pathology_cases.json"))).expanduser()
    if not json_path.is_absolute():
        json_path = root_dir / json_path
    patient_ids = load_patient_ids_from_json(json_path)
    if not patient_ids:
        raise RuntimeError(f"No patient IDs found in case subset JSON: {json_path}")
    return patient_ids, json_path


def main() -> None:
    cfg = load_cfg("tcga", overrides=sys.argv[1:])
    source_name = str(cfg.data.source.name)
    text_cfg = cfg.data.source.get("text_genomics", {})
    skip_empty_cases = bool(text_cfg.get("skip_cases_without_any_modality", True))

    unified_path = Path(str(cfg.data.unified_registry_path))
    unified_df = read_parquet_or_empty(unified_path)
    if unified_df.empty:
        raise RuntimeError(
            f"Unified registry empty at {unified_path}. "
            "Run scripts/data/01_upsert_tcga_registry_rows.py first."
        )
    source_df = _filter_source_frame(unified_df, source_name)
    patient_subset_ids, patient_subset_path = _case_subset_patient_filter(text_cfg)
    if patient_subset_ids:
        source_df = source_df[
            source_df["patient_id"].fillna("").astype(str).isin(set(patient_subset_ids))
        ].copy()
        print(
            "[text-genomics] Case subset: "
            f"{patient_subset_path or 'pathology_cases'} ({len(patient_subset_ids)} patients)"
        )
    else:
        print("[text-genomics] Case subset: all")

    max_cases = _optional_int(text_cfg.get("max_cases")) if text_cfg else None
    if max_cases is not None:
        source_df = source_df.head(max_cases)
        print(f"[text-genomics] Capping processing to {max_cases} cases.")

    manifests_root = Path(str(cfg.data.manifests_root))
    manifest_path_override = text_cfg.get("extra_genomics_manifest_path") if text_cfg else None
    manifest_df = _load_latest_extra_genomics_manifest(
        manifests_root,
        source_name,
        explicit_path=manifest_path_override,
    )
    extra_by_patient = _index_manifest_by_patient(manifest_df)
    if manifest_df.empty:
        print("[text-genomics] No extra-genomics manifest found; using registry path columns only.")
    else:
        print(
            f"[text-genomics] Loaded extra-genomics manifest with "
            f"{len(manifest_df)} rows covering {len(extra_by_patient)} patients."
        )

    features_root = _resolve_features_root(cfg)
    output_root_cfg = text_cfg.get("output_root") if text_cfg else None
    output_root = (
        Path(str(output_root_cfg))
        if output_root_cfg
        else features_root / "genomics_text_blocks" / source_name
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[text-genomics] Writing text blocks to {output_root}")

    manifest_rows: list[dict[str, Any]] = []
    skipped_no_inputs = 0
    progress = tqdm(
        source_df.itertuples(index=False),
        total=len(source_df),
        desc="Building text blocks",
        unit="case",
    )
    for case in progress:
        case_dict = case._asdict() if hasattr(case, "_asdict") else dict(case._mapping)
        project_id = _clean_text(case_dict.get("project_id"))
        patient_id = _clean_text(case_dict.get("patient_id") or case_dict.get("submitter_id"))
        if not project_id or not patient_id:
            continue
        if project_id not in cohort_config.get_all_cohorts():
            continue

        try:
            row = process_case(
                project_id=project_id,
                patient_id=patient_id,
                case_row=case_dict,
                extra_genomics_by_patient=extra_by_patient,
                output_root=output_root,
                root_dir=ROOT,
            )
            if skip_empty_cases and not row["available_modalities"]:
                skipped_no_inputs += 1
                for path_key in ("teacher_text_path", "student_text_path", "features_json_path"):
                    path_value = _clean_text(row.get(path_key))
                    if path_value:
                        path = Path(path_value)
                        if path.is_file():
                            path.unlink()
                continue
            manifest_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            manifest_rows.append(
                {
                    "project_id": project_id,
                    "patient_id": patient_id,
                    "teacher_text_path": "",
                    "student_text_path": "",
                    "features_json_path": "",
                    "genomics_json_path": "",
                    "available_modalities": [],
                    "dnam_available": False,
                    "rna_available": False,
                    "mut_cna_available": False,
                    "errors": f"orchestrator_failure: {exc.__class__.__name__}: {exc}",
                }
            )
            traceback.print_exc()

    manifest_path = features_root / f"{source_name}_genomics_text_blocks_manifest.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df_out = pd.DataFrame(manifest_rows)
    manifest_df_out.to_parquet(manifest_path, index=False)
    print(f"[text-genomics] Manifest written: {manifest_path}")

    update_cfg = text_cfg.get("registry_update", {}) if text_cfg else {}
    if bool(update_cfg.get("enabled", True)) and not manifest_df_out.empty:
        updated_df, stats = update_registry_with_genomics_json_manifest(
            unified_df,
            manifest_df_out,
            repo_root=ROOT,
            source_name=source_name,
            allowed_patient_ids=set(patient_subset_ids) if patient_subset_ids else None,
            overwrite_existing=bool(update_cfg.get("overwrite_existing", True)),
        )
        write_registry_parquet(updated_df, unified_path, validate=True)
        print("[text-genomics] Registry updated.")
        print(f"[text-genomics] Matched registry rows: {stats.matched_registry_rows}")
        print(f"[text-genomics] Updated registry rows: {stats.updated_registry_rows}")
        print(f"[text-genomics] Unmatched text-block cases: {stats.unmatched_manifest_cases}")

    total = len(manifest_rows)
    dnam_ok = sum(1 for row in manifest_rows if row["dnam_available"])
    rna_ok = sum(1 for row in manifest_rows if row["rna_available"])
    mut_ok = sum(1 for row in manifest_rows if row["mut_cna_available"])
    errs = sum(1 for row in manifest_rows if row["errors"])
    print(
        f"[text-genomics] Cases written: {total} | DNAm: {dnam_ok} | "
        f"RNA: {rna_ok} | Mut/CNA: {mut_ok} | errors: {errs} | "
        f"skipped_no_inputs: {skipped_no_inputs}"
    )


if __name__ == "__main__":
    main()
