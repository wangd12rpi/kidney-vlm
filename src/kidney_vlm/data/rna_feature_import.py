from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from kidney_vlm.data.dnam_feature_import import sanitize_filename_component, tcga_sample_submitter_sort_key


@dataclass(frozen=True)
class RnaFeatureRecord:
    project_id: str
    case_submitter_id: str
    sample_submitter_id: str
    rna_file_id: str
    rna_file_name: str
    rna_tsv_path: str
    sample_type: str = ""
    workflow_type: str = ""
    source: str = "tcga"


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _as_string_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, tuple):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [_clean_text(item) for item in converted if _clean_text(item)]
    text = _clean_text(value)
    return [text] if text else []


def _value_for_index(values: list[str], index: int) -> str:
    if len(values) == 1:
        return values[0]
    if 0 <= index < len(values):
        return values[index]
    return ""


def portable_relative_path(path: str | Path, *, root: Path) -> str:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = root / path_obj
    return Path(os.path.relpath(path_obj.resolve(), start=root.resolve())).as_posix()


def resolve_local_path(path: str | Path, *, root: Path) -> Path:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = root / path_obj
    return path_obj.resolve()


def infer_rna_file_id_from_name(file_name: str) -> str:
    stem = Path(file_name).name
    for marker in (".rna_seq.", ".star_gene_counts", ".tsv"):
        if marker in stem:
            stem = stem.split(marker, 1)[0]
    return stem.strip()


def build_rna_records_from_registry(registry_df: pd.DataFrame, *, repo_root: Path) -> list[RnaFeatureRecord]:
    records: list[RnaFeatureRecord] = []
    if registry_df.empty or "genomics_rna_bulk_paths" not in registry_df.columns:
        return records

    for row in registry_df.itertuples(index=False):
        row_dict = row._asdict()
        source = _clean_text(row_dict.get("source")) or "tcga"
        project_id = _clean_text(row_dict.get("project_id")) or _clean_text(row_dict.get("study_id"))
        case_submitter_id = _clean_text(row_dict.get("patient_id")) or _clean_text(row_dict.get("sample_id"))
        paths = _as_string_list(row_dict.get("genomics_rna_bulk_paths"))
        if not paths or not project_id or not case_submitter_id:
            continue

        file_ids = _as_string_list(row_dict.get("genomics_rna_bulk_file_ids"))
        file_names = _as_string_list(row_dict.get("genomics_rna_bulk_file_names"))
        sample_types = _as_string_list(row_dict.get("genomics_rna_bulk_sample_types"))
        workflow_types = _as_string_list(row_dict.get("genomics_rna_bulk_workflow_types"))

        for idx, path_value in enumerate(paths):
            resolved_path = resolve_local_path(path_value, root=repo_root)
            rna_file_name = _value_for_index(file_names, idx) or resolved_path.name
            rna_file_id = _value_for_index(file_ids, idx) or infer_rna_file_id_from_name(rna_file_name)
            records.append(
                RnaFeatureRecord(
                    project_id=project_id,
                    case_submitter_id=case_submitter_id,
                    # Current registry rows retain case-level sample-type metadata,
                    # but not per-file TCGA sample submitter IDs.
                    sample_submitter_id=case_submitter_id,
                    rna_file_id=rna_file_id,
                    rna_file_name=rna_file_name,
                    rna_tsv_path=portable_relative_path(resolved_path, root=repo_root),
                    sample_type=_value_for_index(sample_types, idx),
                    workflow_type=_value_for_index(workflow_types, idx),
                    source=source,
                )
            )

    return sorted(records, key=lambda record: (record.project_id, record.case_submitter_id, record.rna_file_id))


def build_rna_records_from_raw_tree(raw_root: Path, *, repo_root: Path, source: str = "tcga") -> list[RnaFeatureRecord]:
    return build_rna_records_from_raw_tree_limited(raw_root, repo_root=repo_root, source=source)


def build_rna_records_from_raw_tree_limited(
    raw_root: Path,
    *,
    repo_root: Path,
    source: str = "tcga",
    allowed_project_ids: set[str] | None = None,
    max_cases: int | None = None,
) -> list[RnaFeatureRecord]:
    raw_root = raw_root.expanduser().resolve()
    records: list[RnaFeatureRecord] = []
    if not raw_root.exists():
        return records

    completed_cases = 0
    allowed_project_ids = {project.strip() for project in allowed_project_ids or set() if project.strip()}
    for project_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        project_id = _clean_text(project_dir.name)
        if allowed_project_ids and project_id not in allowed_project_ids:
            continue
        for case_dir in sorted(path for path in project_dir.iterdir() if path.is_dir()):
            case_records: list[RnaFeatureRecord] = []
            case_submitter_id = _clean_text(case_dir.name)
            for tsv_path in sorted(case_dir.glob("*.tsv")):
                if not tsv_path.is_file():
                    continue
                rna_file_name = tsv_path.name
                case_records.append(
                    RnaFeatureRecord(
                        project_id=project_id,
                        case_submitter_id=case_submitter_id,
                        sample_submitter_id=case_submitter_id,
                        rna_file_id=infer_rna_file_id_from_name(rna_file_name),
                        rna_file_name=rna_file_name,
                        rna_tsv_path=portable_relative_path(tsv_path, root=repo_root),
                        source=source,
                    )
                )
            if case_records:
                records.extend(case_records)
                completed_cases += 1
                if max_cases is not None and completed_cases >= max_cases:
                    return records

    return records


def _sample_type_sort_rank(sample_type: str) -> int:
    text = sample_type.strip().lower()
    if "primary tumor" in text:
        return 0
    if "recurrent" in text:
        return 1
    if "metastatic" in text:
        return 2
    if "tumor" in text:
        return 3
    if "normal" in text:
        return 10
    return 5


def rna_record_sort_key(record: RnaFeatureRecord) -> tuple[int, tuple[int, str], str, str]:
    return (
        _sample_type_sort_rank(record.sample_type),
        tcga_sample_submitter_sort_key(record.sample_submitter_id),
        record.rna_file_id,
        record.rna_tsv_path,
    )


def select_case_level_rna_records(records: Iterable[RnaFeatureRecord]) -> list[RnaFeatureRecord]:
    grouped: dict[tuple[str, str], list[RnaFeatureRecord]] = {}
    for record in records:
        if not record.project_id or not record.case_submitter_id:
            continue
        grouped.setdefault((record.project_id, record.case_submitter_id), []).append(record)

    selected = [sorted(group, key=rna_record_sort_key)[0] for group in grouped.values() if group]
    return sorted(selected, key=lambda record: (record.project_id, record.case_submitter_id, record.rna_file_id))


def build_case_level_rna_assignments(manifest_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "project_id",
        "patient_id",
        "genomics_rna_bulk_paths",
        "genomics_rna_bulk_feature_path",
        "selected_sample_submitter_id",
        "source_feature_count",
    ]
    required_columns = {
        "project_id",
        "case_submitter_id",
        "sample_submitter_id",
        "rna_tsv_path",
        "feature_path",
        "rna_file_id",
    }
    missing = sorted(required_columns.difference(manifest_df.columns))
    if missing:
        raise ValueError(f"RNA manifest is missing required columns: {missing}")

    if manifest_df.empty:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict[str, object]] = []
    for (project_id, case_submitter_id), group in manifest_df.groupby(
        ["project_id", "case_submitter_id"],
        sort=True,
        dropna=False,
    ):
        patient_id = _clean_text(case_submitter_id)
        if not patient_id:
            continue

        raw_paths = sorted({_clean_text(value) for value in group["rna_tsv_path"].tolist() if _clean_text(value)})
        feature_rows = []
        for row in group.itertuples(index=False):
            feature_path = _clean_text(getattr(row, "feature_path", ""))
            if not feature_path:
                continue
            sample_submitter_id = _clean_text(getattr(row, "sample_submitter_id", ""))
            sample_type = _clean_text(getattr(row, "sample_type", ""))
            rna_file_id = _clean_text(getattr(row, "rna_file_id", ""))
            feature_rows.append(
                (
                    _sample_type_sort_rank(sample_type),
                    tcga_sample_submitter_sort_key(sample_submitter_id),
                    sample_submitter_id,
                    rna_file_id,
                    feature_path,
                )
            )

        if not feature_rows:
            continue

        feature_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        _, _, selected_sample_submitter_id, _, selected_feature_path = feature_rows[0]
        rows.append(
            {
                "project_id": _clean_text(project_id),
                "patient_id": patient_id,
                "genomics_rna_bulk_paths": raw_paths,
                "genomics_rna_bulk_feature_path": selected_feature_path,
                "selected_sample_submitter_id": selected_sample_submitter_id,
                "source_feature_count": len(feature_rows),
            }
        )

    if not rows:
        return pd.DataFrame(columns=output_columns)

    return pd.DataFrame(rows).sort_values(
        by=["project_id", "patient_id", "selected_sample_submitter_id"],
        kind="stable",
    )


def read_tcga_star_log_tpm(tsv_path: str | Path) -> pd.DataFrame:
    """Return one row of log1p(TPM) values keyed by base Ensembl ID."""
    path = Path(tsv_path)
    df = pd.read_csv(path, sep="\t", comment="#")
    required_columns = {"gene_id", "tpm_unstranded"}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"TCGA STAR TSV is missing required columns {missing}: {path}")

    df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
    if "gene_type" in df.columns:
        df = df[df["gene_type"].astype(str).eq("protein_coding")].copy()
    if df.empty:
        raise ValueError(f"No protein-coding ENSG rows found in TCGA STAR TSV: {path}")

    df["ensg_id"] = df["gene_id"].astype(str).str.split(".").str[0]
    df["tpm_unstranded"] = pd.to_numeric(df["tpm_unstranded"], errors="coerce").fillna(0.0)
    df = df.groupby("ensg_id", as_index=False)["tpm_unstranded"].max()
    df["log_tpm"] = np.log1p(df["tpm_unstranded"].to_numpy(dtype=np.float64))
    row = df.set_index("ensg_id")["log_tpm"].to_frame().T.astype(np.float32)
    row.index = [path.stem]
    return row


def align_to_bulkformer_vocab(log_tpm_row: pd.DataFrame, gene_list: list[str]) -> tuple[pd.DataFrame, float]:
    if log_tpm_row.shape[0] != 1:
        raise ValueError(f"Expected one expression row, found {log_tpm_row.shape[0]}.")
    if not gene_list:
        raise ValueError("BulkFormer gene list is empty.")

    unique_gene_list = list(dict.fromkeys(str(gene).strip() for gene in gene_list if str(gene).strip()))
    if len(unique_gene_list) != len(gene_list):
        raise ValueError("BulkFormer gene list contains duplicate or empty Ensembl IDs.")

    present = set(log_tpm_row.columns)
    missing = [gene for gene in gene_list if gene not in present]
    if missing:
        padding = pd.DataFrame(
            np.full((1, len(missing)), -10.0, dtype=np.float32),
            columns=missing,
            index=log_tpm_row.index,
        )
        aligned = pd.concat([log_tpm_row, padding], axis=1)
    else:
        aligned = log_tpm_row

    aligned = aligned.loc[:, gene_list].astype(np.float32)
    mask_prob = len(missing) / len(gene_list)
    return aligned, float(mask_prob)


def build_rna_feature_filename(record: RnaFeatureRecord) -> str:
    sample_token = sanitize_filename_component(
        record.sample_submitter_id or record.case_submitter_id or record.project_id,
        fallback="unknown_sample",
    )
    unique_token = sanitize_filename_component(
        record.rna_file_id or Path(record.rna_file_name).stem or re.sub(r"\W+", "_", record.rna_tsv_path),
        fallback="unknown_file",
    )
    return f"{sample_token}__{unique_token}.pt"


def build_rna_output_path(output_root: Path, record: RnaFeatureRecord) -> Path:
    project_token = sanitize_filename_component(record.project_id or "unknown_project", fallback="unknown_project")
    return output_root / project_token / build_rna_feature_filename(record)
