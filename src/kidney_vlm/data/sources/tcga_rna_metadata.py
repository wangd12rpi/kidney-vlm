from __future__ import annotations

import gzip
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import requests


PANCAN_RNA_METADATA_URLS = {
    "subtypes": "https://api.gdc.cancer.gov/data/0f31b768-7f67-4fc4-abc3-06ac5bd90bf0",
    "leukocyte_fraction": "https://api.gdc.cancer.gov/data/6f75c9d7-5134-4ed1-b8f3-72856c98a4e8",
    "cibersort": "https://api.gdc.cancer.gov/data/b3df502e-3594-46ef-9f94-d041a20a0b9a",
    "absolute_scores": "https://api.gdc.cancer.gov/data/0e8831f4-dd7e-4673-8624-b4519c2e0d65",
    "absolute_purity": "https://api.gdc.cancer.gov/data/4f277128-f793-4354-a13d-30cc7fe9f6b5",
    "hrd_scores": "https://api.gdc.cancer.gov/data/66dd07d7-6366-4774-83c3-5ad1e22b177e",
}

PANCAN_RNA_METADATA_FILE_NAMES = {
    "subtypes": "TCGASubtype.20170308.tsv",
    "leukocyte_fraction": "TCGA_all_leuk_estimate.masked.20170107.tsv",
    "cibersort": "TCGA.Kallisto.fullIDs.cibersort.relative.tsv",
    "absolute_scores": "ABSOLUTE_scores.tsv",
    "absolute_purity": "TCGA_mastercalls.abs_tables_JSedit.fixed.txt",
    "hrd_scores": "TCGA.HRD_withSampleID.txt",
}

DEFAULT_RNA_METADATA_KEYS = (
    "subtypes",
    "leukocyte_fraction",
    "cibersort",
    "absolute_scores",
    "absolute_purity",
    "hrd_scores",
)


def patient_from_barcode(barcode: str) -> str:
    text = str(barcode).strip().replace(".", "-")
    parts = [part for part in text.split("-") if part]
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return text


def _normalized_keys(keys: list[str] | tuple[str, ...] | None) -> list[str]:
    selected = list(keys or DEFAULT_RNA_METADATA_KEYS)
    normalized: list[str] = []
    for key in selected:
        text = str(key).strip()
        if text and text in PANCAN_RNA_METADATA_FILE_NAMES and text not in normalized:
            normalized.append(text)
    return normalized


def _download_open_access_file(
    *,
    url: str,
    destination: Path,
    skip_existing: bool,
    timeout_seconds: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and destination.exists() and destination.stat().st_size > 0:
        return destination

    temp_path = destination.with_name(f"{destination.name}.part")
    if temp_path.exists():
        temp_path.unlink()

    response = requests.get(url, stream=True, timeout=timeout_seconds)
    response.raise_for_status()
    try:
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        temp_path.replace(destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        response.close()
    return destination


def download_tcga_rna_metadata(
    data_dir: str | Path,
    *,
    skip_existing: bool = True,
    timeout_seconds: int = 300,
    keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    cache_dir = Path(data_dir)
    downloaded: dict[str, Path] = {}
    for key in _normalized_keys(keys):
        destination = cache_dir / PANCAN_RNA_METADATA_FILE_NAMES[key]
        downloaded[key] = _download_open_access_file(
            url=PANCAN_RNA_METADATA_URLS[key],
            destination=destination,
            skip_existing=skip_existing,
            timeout_seconds=timeout_seconds,
        )
    return downloaded


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".gz":
        decompressed = path.with_suffix("")
        if not decompressed.exists():
            with gzip.open(path, "rb") as source, decompressed.open("wb") as target:
                shutil.copyfileobj(source, target)
        path = decompressed
    return pd.read_csv(path, sep="\t")


def _first_matching_column(columns: list[str], patterns: list[str]) -> str | None:
    lowered = {str(column): str(column).lower() for column in columns}
    for pattern in patterns:
        pattern_text = str(pattern).lower()
        for original, lowered_name in lowered.items():
            if pattern_text in lowered_name:
                return original
    return None


def _sample_priority(barcode: str) -> tuple[int, str]:
    parts = str(barcode).strip().replace(".", "-").split("-")
    sample_portion = parts[3] if len(parts) >= 4 else ""
    sample_code = sample_portion[:2]
    try:
        numeric_code = int(sample_code)
    except ValueError:
        numeric_code = 99
    return (numeric_code, str(barcode).strip())


def _format_decimal(value: Any, *, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _format_int_string(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def _format_immune_cell_label(column_name: str) -> str:
    text = str(column_name).strip().replace(".", " ").replace("_", " ")
    return " ".join(text.split())


def _table_or_headerless_three_column(path: Path, *, names: list[str]) -> pd.DataFrame:
    df = _load_table(path)
    if not df.empty and len(df.columns) == len(names):
        sample_col = _first_matching_column(list(df.columns), ["sampleid", "sample"])
        if sample_col is not None:
            return df
    try:
        return pd.read_csv(path, sep="\t", header=None, names=names, low_memory=False)
    except Exception:
        return df


def _choose_best_sample_rows(df: pd.DataFrame, sample_column: str) -> dict[str, pd.Series]:
    by_patient: dict[str, tuple[tuple[int, str], pd.Series]] = {}
    for _, row in df.iterrows():
        sample_barcode = str(row.get(sample_column, "")).strip()
        if not sample_barcode:
            continue
        patient_id = patient_from_barcode(sample_barcode)
        priority = _sample_priority(sample_barcode)
        current = by_patient.get(patient_id)
        if current is None or priority < current[0]:
            by_patient[patient_id] = (priority, row)
    return {patient_id: row for patient_id, (_, row) in by_patient.items()}


def _load_subtypes_index(data_dir: Path, patient_ids: set[str]) -> dict[str, dict[str, str]]:
    df = _load_table(data_dir / PANCAN_RNA_METADATA_FILE_NAMES["subtypes"])
    if df.empty:
        return {}
    sample_col = _first_matching_column(list(df.columns), ["sampleid", "sample"])
    subtype_col = _first_matching_column(list(df.columns), ["Subtype_Selected", "Subtype"])
    mrna_col = _first_matching_column(list(df.columns), ["Subtype_mRNA"])
    dnam_col = _first_matching_column(list(df.columns), ["Subtype_DNAmeth"])
    integrative_col = _first_matching_column(list(df.columns), ["Subtype_Integrative"])
    if sample_col is None:
        return {}

    best_rows = _choose_best_sample_rows(df, sample_col)
    result: dict[str, dict[str, str]] = {}
    for patient_id, row in best_rows.items():
        if patient_id not in patient_ids:
            continue
        record: dict[str, str] = {}
        if subtype_col is not None:
            text = str(row.get(subtype_col, "")).strip()
            if text:
                record["genomics_rna_bulk_molecular_subtype"] = text
                if "msi" in text.lower():
                    record["genomics_msi_status"] = "MSI"
        if mrna_col is not None:
            text = str(row.get(mrna_col, "")).strip()
            if text:
                record["genomics_rna_bulk_subtype_mrna"] = text
        if dnam_col is not None:
            text = str(row.get(dnam_col, "")).strip()
            if text:
                record["genomics_dna_methylation_subtype"] = text
        if integrative_col is not None:
            text = str(row.get(integrative_col, "")).strip()
            if text:
                record["genomics_integrative_subtype"] = text
        if record:
            result[patient_id] = record
    return result


def _index_leukocyte_fraction(data_dir: Path, patient_ids: set[str]) -> dict[str, str]:
    df = _table_or_headerless_three_column(
        data_dir / PANCAN_RNA_METADATA_FILE_NAMES["leukocyte_fraction"],
        names=["CancerType", "SampleID", "leukocyte_fraction"],
    )
    if df.empty:
        return {}
    sample_col = _first_matching_column(list(df.columns), ["sampleid", "sample"])
    value_col = _first_matching_column(list(df.columns), ["leukocyte_fraction", "leukocyte"])
    if sample_col is None or value_col is None:
        return {}

    best_rows = _choose_best_sample_rows(df, sample_col)
    result: dict[str, str] = {}
    for patient_id, row in best_rows.items():
        if patient_id not in patient_ids:
            continue
        value = _format_decimal(row.get(value_col), digits=3)
        if value:
            result[patient_id] = value
    return result


def _index_tumor_purity(data_dir: Path, patient_ids: set[str]) -> dict[str, str]:
    score_df = _load_table(data_dir / PANCAN_RNA_METADATA_FILE_NAMES["absolute_scores"])
    sample_col = _first_matching_column(list(score_df.columns), ["sampleid", "sample"])
    purity_col = _first_matching_column(list(score_df.columns), ["purity"])
    if sample_col is not None and purity_col is not None:
        best_rows = _choose_best_sample_rows(score_df, sample_col)
        result: dict[str, str] = {}
        for patient_id, row in best_rows.items():
            if patient_id not in patient_ids:
                continue
            value = _format_decimal(row.get(purity_col), digits=2)
            if value:
                result[patient_id] = value
        if result:
            return result

    df = _load_table(data_dir / PANCAN_RNA_METADATA_FILE_NAMES["absolute_purity"])
    if df.empty:
        return {}
    sample_col = _first_matching_column(list(df.columns), ["sampleid", "sample", "barcode"])
    purity_col = _first_matching_column(list(df.columns), ["purity"])
    if sample_col is None or purity_col is None:
        return {}

    best_rows = _choose_best_sample_rows(df, sample_col)
    result: dict[str, str] = {}
    for patient_id, row in best_rows.items():
        if patient_id not in patient_ids:
            continue
        value = _format_decimal(row.get(purity_col), digits=2)
        if value:
            result[patient_id] = value
    return result


def _index_aneuploidy_scores(data_dir: Path, patient_ids: set[str]) -> dict[str, str]:
    df = _load_table(data_dir / PANCAN_RNA_METADATA_FILE_NAMES["absolute_scores"])
    if df.empty:
        return {}
    sample_col = _first_matching_column(list(df.columns), ["sampleid", "sample"])
    if sample_col is None and len(df.columns) > 0:
        sample_col = str(df.columns[0])
    aneuploidy_col = _first_matching_column(list(df.columns), ["aneuploidy"])
    if aneuploidy_col is None and "AS" in df.columns:
        aneuploidy_col = "AS"
    if sample_col is None or aneuploidy_col is None:
        return {}

    best_rows = _choose_best_sample_rows(df, sample_col)
    result: dict[str, str] = {}
    for patient_id, row in best_rows.items():
        if patient_id not in patient_ids:
            continue
        value = _format_int_string(row.get(aneuploidy_col))
        if value:
            result[patient_id] = value
    return result


def _index_hrd_scores(data_dir: Path, patient_ids: set[str]) -> dict[str, str]:
    df = _load_table(data_dir / PANCAN_RNA_METADATA_FILE_NAMES["hrd_scores"])
    if df.empty:
        return {}
    sample_col = _first_matching_column(list(df.columns), ["sampleid", "sample", "barcode"])
    if sample_col is None and len(df.columns) > 0:
        sample_col = str(df.columns[0])
    hrd_col = _first_matching_column(list(df.columns), ["hrd"])
    if sample_col is None or hrd_col is None:
        return {}

    best_rows = _choose_best_sample_rows(df, sample_col)
    result: dict[str, str] = {}
    for patient_id, row in best_rows.items():
        if patient_id not in patient_ids:
            continue
        value = _format_int_string(row.get(hrd_col))
        if value:
            result[patient_id] = value
    return result


def _index_top_immune_cells(
    data_dir: Path,
    patient_ids: set[str],
    *,
    top_k: int = 3,
) -> dict[str, dict[str, list[str]]]:
    df = _load_table(data_dir / PANCAN_RNA_METADATA_FILE_NAMES["cibersort"])
    if df.empty:
        return {}
    sample_col = _first_matching_column(list(df.columns), ["sampleid", "sample"])
    if sample_col is None:
        return {}

    excluded_tokens = ("p-value", "correlation", "rmse")
    cell_columns = [
        str(column)
        for column in list(df.columns)
        if str(column) != str(sample_col)
        and all(token not in str(column).lower() for token in excluded_tokens)
    ]
    if not cell_columns:
        return {}

    best_rows = _choose_best_sample_rows(df, sample_col)
    result: dict[str, dict[str, list[str]]] = {}
    for patient_id, row in best_rows.items():
        if patient_id not in patient_ids:
            continue
        ranked: list[tuple[float, str]] = []
        for column in cell_columns:
            try:
                value = float(row.get(column))
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            ranked.append((value, column))
        if not ranked:
            continue
        ranked.sort(key=lambda item: (-item[0], item[1]))
        top = ranked[:top_k]
        result[patient_id] = {
            "genomics_rna_bulk_top_immune_cell_types": [_format_immune_cell_label(label) for _, label in top],
            "genomics_rna_bulk_top_immune_cell_fractions": [f"{value:.3f}" for value, _ in top],
        }
    return result


def build_tcga_rna_metadata_by_patient_id(
    *,
    cases: list[dict[str, Any]],
    data_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    patient_ids = {
        str(case.get("submitter_id", "")).strip()
        for case in cases
        if str(case.get("submitter_id", "")).strip()
    }
    if not patient_ids:
        return {}

    cache_dir = Path(data_dir)
    subtypes_by_patient = _load_subtypes_index(cache_dir, patient_ids)
    leukocyte_by_patient = _index_leukocyte_fraction(cache_dir, patient_ids)
    purity_by_patient = _index_tumor_purity(cache_dir, patient_ids)
    aneuploidy_by_patient = _index_aneuploidy_scores(cache_dir, patient_ids)
    hrd_by_patient = _index_hrd_scores(cache_dir, patient_ids)
    immune_cells_by_patient = _index_top_immune_cells(cache_dir, patient_ids)

    metadata_by_patient: dict[str, dict[str, Any]] = {}
    for patient_id in sorted(patient_ids):
        record: dict[str, Any] = {}
        record.update(subtypes_by_patient.get(patient_id, {}))
        leukocyte_fraction = leukocyte_by_patient.get(patient_id, "")
        if leukocyte_fraction:
            record["genomics_rna_bulk_leukocyte_fraction"] = leukocyte_fraction
        tumor_purity = purity_by_patient.get(patient_id, "")
        if tumor_purity:
            record["genomics_rna_bulk_tumor_purity"] = tumor_purity
        aneuploidy_score = aneuploidy_by_patient.get(patient_id, "")
        if aneuploidy_score:
            record["genomics_aneuploidy_score"] = aneuploidy_score
        hrd_score = hrd_by_patient.get(patient_id, "")
        if hrd_score:
            record["genomics_hrd_score"] = hrd_score
        record.update(immune_cells_by_patient.get(patient_id, {}))
        metadata_by_patient[patient_id] = record
    return metadata_by_patient
