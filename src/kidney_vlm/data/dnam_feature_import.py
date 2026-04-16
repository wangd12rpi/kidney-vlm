from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ANVIL_DNA_PREFIX = "/anvil/projects/x-cis250966/dna/"
LOCAL_DNA_PREFIX = "/media/volume/patho_meth/"


@dataclass(frozen=True)
class CpGPTCacheRecord:
    cache_hash: str
    beta_path: str
    project_id: str
    case_submitter_id: str
    sample_submitter_id: str
    beta_file_id: str
    beta_file_name: str
    source_index_files: tuple[str, ...]


def normalize_cpgpt_cache_key(key: str | Path) -> str:
    return str(key).replace(ANVIL_DNA_PREFIX, LOCAL_DNA_PREFIX)


def cpgpt_cache_hash_for_beta_path(beta_path: str | Path) -> str:
    normalized = normalize_cpgpt_cache_key(beta_path)
    return hashlib.md5(normalized.encode()).hexdigest()


def resolve_tcga_beta_path(beta_metadata: dict[str, object], *, raw_root: Path) -> str | None:
    beta = beta_metadata.get("local_path") or beta_metadata.get("file_path")
    if beta is None:
        return None
    beta_path = Path(str(beta))
    if not beta_path.is_absolute():
        beta_path = raw_root / beta_path
    return normalize_cpgpt_cache_key(beta_path)


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    unique = sorted({str(value).strip() for value in values if str(value).strip()})
    return tuple(unique)


def _merge_record_sources(existing: CpGPTCacheRecord, index_name: str) -> CpGPTCacheRecord:
    merged_sources = _sorted_unique((*existing.source_index_files, index_name))
    return CpGPTCacheRecord(
        cache_hash=existing.cache_hash,
        beta_path=existing.beta_path,
        project_id=existing.project_id,
        case_submitter_id=existing.case_submitter_id,
        sample_submitter_id=existing.sample_submitter_id,
        beta_file_id=existing.beta_file_id,
        beta_file_name=existing.beta_file_name,
        source_index_files=merged_sources,
    )


def _ensure_compatible_records(existing: CpGPTCacheRecord, candidate: CpGPTCacheRecord) -> None:
    comparable_fields = (
        "cache_hash",
        "beta_path",
        "project_id",
        "case_submitter_id",
        "sample_submitter_id",
        "beta_file_id",
        "beta_file_name",
    )
    mismatched = [
        field
        for field in comparable_fields
        if getattr(existing, field) != getattr(candidate, field)
    ]
    if mismatched:
        raise ValueError(
            "Conflicting CpGPT cache metadata for "
            f"{existing.cache_hash}: mismatched fields={mismatched}"
        )


def parse_cpgpt_index_row(row: dict[str, object], *, raw_root: Path, index_name: str) -> CpGPTCacheRecord | None:
    meth = row.get("methylation_beta")
    if not isinstance(meth, dict):
        return None

    beta_path = resolve_tcga_beta_path(meth, raw_root=raw_root)
    if not beta_path:
        return None

    cache_hash = cpgpt_cache_hash_for_beta_path(beta_path)
    return CpGPTCacheRecord(
        cache_hash=cache_hash,
        beta_path=beta_path,
        project_id=_clean_text(row.get("project_id")),
        case_submitter_id=_clean_text(row.get("case_submitter_id")),
        sample_submitter_id=_clean_text(row.get("sample_submitter_id")),
        beta_file_id=_clean_text(meth.get("file_id")),
        beta_file_name=_clean_text(meth.get("file_name")),
        source_index_files=(index_name,),
    )


def build_cpgpt_hash_index(index_paths: Iterable[Path], *, raw_root: Path) -> dict[str, CpGPTCacheRecord]:
    records: dict[str, CpGPTCacheRecord] = {}
    for index_path in index_paths:
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                record = parse_cpgpt_index_row(row, raw_root=raw_root, index_name=index_path.name)
                if record is None:
                    continue
                existing = records.get(record.cache_hash)
                if existing is None:
                    records[record.cache_hash] = record
                    continue
                _ensure_compatible_records(existing, record)
                records[record.cache_hash] = _merge_record_sources(existing, index_path.name)
    return records


def sanitize_filename_component(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def build_cpgpt_feature_filename(record: CpGPTCacheRecord) -> str:
    sample_token = sanitize_filename_component(
        record.sample_submitter_id or record.case_submitter_id or record.project_id,
        fallback="unknown_sample",
    )
    unique_token = sanitize_filename_component(
        record.beta_file_id or Path(record.beta_file_name).stem or record.cache_hash,
        fallback=record.cache_hash,
    )
    return f"{sample_token}__{unique_token}.pt"


def build_cpgpt_output_path(output_root: Path, record: CpGPTCacheRecord) -> Path:
    project_token = sanitize_filename_component(record.project_id or "unknown_project", fallback="unknown_project")
    return output_root / project_token / build_cpgpt_feature_filename(record)


def tcga_sample_submitter_sort_key(sample_submitter_id: str) -> tuple[int, str]:
    text = str(sample_submitter_id or "").strip().upper()
    token = text.split("-")[-1] if text else ""
    match = re.match(r"(?P<code>\d{2})(?P<vial>[A-Z]?)", token)
    if not match:
        return (999, text)
    return (int(match.group("code")), text)


def build_case_level_dnam_assignments(manifest_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "project_id",
        "case_submitter_id",
        "sample_submitter_id",
        "beta_path",
        "feature_path",
        "beta_file_id",
    }
    missing = sorted(required_columns.difference(manifest_df.columns))
    if missing:
        raise ValueError(f"DNAm manifest is missing required columns: {missing}")

    if manifest_df.empty:
        return pd.DataFrame(
            columns=[
                "project_id",
                "patient_id",
                "genomics_dna_methylation_paths",
                "genomics_dna_methylation_feature_path",
                "selected_sample_submitter_id",
                "source_feature_count",
            ]
        )

    rows: list[dict[str, object]] = []
    group_columns = ["project_id", "case_submitter_id"]
    for (project_id, case_submitter_id), group in manifest_df.groupby(group_columns, sort=True, dropna=False):
        patient_id = str(case_submitter_id or "").strip()
        if not patient_id:
            continue

        raw_paths = sorted(
            {
                str(value).strip()
                for value in group["beta_path"].tolist()
                if str(value).strip()
            }
        )
        feature_rows = []
        for row in group.itertuples(index=False):
            feature_path = str(getattr(row, "feature_path", "") or "").strip()
            if not feature_path:
                continue
            sample_submitter_id = str(getattr(row, "sample_submitter_id", "") or "").strip()
            beta_file_id = str(getattr(row, "beta_file_id", "") or "").strip()
            feature_rows.append(
                (
                    tcga_sample_submitter_sort_key(sample_submitter_id),
                    sample_submitter_id,
                    beta_file_id,
                    feature_path,
                )
            )

        if not feature_rows:
            continue

        feature_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        _, selected_sample_submitter_id, _, selected_feature_path = feature_rows[0]

        rows.append(
            {
                "project_id": str(project_id or "").strip(),
                "patient_id": patient_id,
                "genomics_dna_methylation_paths": raw_paths,
                "genomics_dna_methylation_feature_path": selected_feature_path,
                "selected_sample_submitter_id": selected_sample_submitter_id,
                "source_feature_count": len(feature_rows),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["project_id", "patient_id", "selected_sample_submitter_id"],
        kind="stable",
    )
