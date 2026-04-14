from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable

import pandas as pd
import requests


DEFAULT_PANCAN_FILE_NAMES = {
    "mc3_maf": "mc3.v0.2.8.PUBLIC.maf.gz",
    "subtypes": "TCGASubtype.20170308.tsv",
    "gistic_thresholded": "all_thresholded.by_genes_whitelisted.tsv",
    "leukocyte_fraction": "TCGA_all_leuk_estimate.masked.20170107.tsv",
    "cibersort": "TCGA.Kallisto.fullIDs.cibersort.relative.tsv",
    "mutation_load": "mutation-load_updated.txt",
    "absolute_scores": "ABSOLUTE_scores.tsv",
    "absolute_purity": "TCGA_mastercalls.abs_tables_JSedit.fixed.txt",
    "hrd_scores": "TCGA.HRD_withSampleID.txt",
    "arm_calls": "PANCAN_ArmCallsAndAneuploidyScore_092817.txt",
    "clinical_followup": "clinical_PANCAN_patient_with_followup.tsv",
    "tcga_clinical_data_resource": "TCGA-CDR-SupplementalTableS1.xlsx",
    "viral_reads": "viral.tsv",
    "pathway_alterations": "TCGA_cancer_pathway_alterations.tsv",
}

# genomics/config.py registers the clinical follow-up URL under the legacy key
# "clinical" rather than "clinical_followup".  This alias map is consulted when
# the primary key resolves to an empty string.
_PANCAN_URL_KEY_ALIASES: dict[str, list[str]] = {
    "clinical_followup": ["clinical"],
}

REQUIRED_PANCAN_KEYS = (
    "mc3_maf",
    "subtypes",
    "gistic_thresholded",
    "leukocyte_fraction",
    "cibersort",
    "mutation_load",
    "absolute_scores",
    "hrd_scores",
)

OPTIONAL_PANCAN_KEYS = (
    "arm_calls",
    "absolute_purity",
    "clinical_followup",
    "tcga_clinical_data_resource",
    "pathway_alterations",
    "viral_reads",
)

DEFAULT_PANCAN_KEYS = REQUIRED_PANCAN_KEYS + OPTIONAL_PANCAN_KEYS

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


@dataclass(frozen=True)
class TcgaGenomicsPrototypeResources:
    cancer_configs: dict[str, Any]
    pancan_urls: dict[str, str]
    nonsynonymous_classes: set[str]
    special_fn_registry: dict[str, Callable[[dict[str, Any], Any], dict[str, str]]]
    msi_threshold_high: float
    msi_threshold_low: float
    ten_pathways: tuple[str, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_tcga_genomics_prototype_resources() -> TcgaGenomicsPrototypeResources:
    genomics_dir = _repo_root() / "genomics"
    config_path = genomics_dir / "config.py"
    extract_path = genomics_dir / "extract.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing genomics prototype config: {config_path}")
    if not extract_path.exists():
        raise FileNotFoundError(f"Missing genomics prototype extract module: {extract_path}")

    config_module = _load_module_from_path("_kidney_vlm_tcga_genomics_config", config_path)

    previous_config_module = sys.modules.get("config")
    sys.modules["config"] = config_module
    try:
        extract_module = _load_module_from_path("_kidney_vlm_tcga_genomics_extract", extract_path)
    finally:
        if previous_config_module is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous_config_module

    return TcgaGenomicsPrototypeResources(
        cancer_configs=dict(getattr(config_module, "CANCER_CONFIGS", {})),
        pancan_urls=dict(getattr(config_module, "PANCAN_URLS", {})),
        nonsynonymous_classes=set(getattr(config_module, "NONSYNONYMOUS_CLASSES", set())),
        special_fn_registry=dict(getattr(extract_module, "SPECIAL_FN_REGISTRY", {})),
        msi_threshold_high=float(getattr(config_module, "MSI_THRESHOLD_HIGH", 3.5)),
        msi_threshold_low=float(getattr(config_module, "MSI_THRESHOLD_LOW", 1.0)),
        ten_pathways=tuple(str(p) for p in getattr(config_module, "TEN_PATHWAYS", [])),
    )


def patient_from_barcode(barcode: str) -> str:
    parts = str(barcode).strip().split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return str(barcode).strip()


def cancer_code_from_project_id(project_id: str) -> str:
    project_text = str(project_id).strip().upper()
    if project_text.startswith("TCGA-"):
        return project_text.split("TCGA-", 1)[1]
    return project_text


def _resolved_pancan_output_path(data_dir: Path, key: str) -> Path:
    if key not in DEFAULT_PANCAN_FILE_NAMES:
        raise KeyError(f"Unsupported PanCancer Atlas key: {key}")
    return data_dir / DEFAULT_PANCAN_FILE_NAMES[key]


def resolve_pancan_cache_path(data_dir: Path, key: str) -> Path:
    raw_path = _resolved_pancan_output_path(data_dir, key)
    if raw_path.suffix == ".gz":
        decompressed = raw_path.with_suffix("")
        if decompressed.exists():
            return decompressed
    return raw_path


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _resolve_gdc_data_url_by_filename(file_name: str, *, timeout_seconds: int) -> str:
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "file_name", "value": file_name}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name",
        "format": "JSON",
        "size": "5",
    }
    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    hits = list((response.json().get("data") or {}).get("hits") or [])
    if not hits:
        raise FileNotFoundError(f"Unable to resolve open-access GDC file by name: {file_name}")
    file_id = str(hits[0].get("file_id", "")).strip()
    if not file_id:
        raise FileNotFoundError(f"GDC response for '{file_name}' did not include a file_id")
    return f"{GDC_DATA_ENDPOINT}/{file_id}"


def _resolve_pancan_download_url(
    *,
    resources: TcgaGenomicsPrototypeResources,
    key: str,
    timeout_seconds: int,
) -> str:
    configured = str(resources.pancan_urls.get(key, "")).strip()
    if not configured:
        for alias in _PANCAN_URL_KEY_ALIASES.get(key, []):
            configured = str(resources.pancan_urls.get(alias, "")).strip()
            if configured:
                break
    if configured:
        if _looks_like_url(configured):
            return configured
        return _resolve_gdc_data_url_by_filename(configured, timeout_seconds=timeout_seconds)
    return _resolve_gdc_data_url_by_filename(DEFAULT_PANCAN_FILE_NAMES[key], timeout_seconds=timeout_seconds)


def _download_open_access_file(
    *,
    url: str,
    dest: Path,
    skip_existing: bool,
    timeout_seconds: int,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.suffix == ".gz":
        decompressed = dest.with_suffix("")
        if skip_existing and decompressed.exists() and decompressed.stat().st_size > 0:
            return decompressed
        if skip_existing and dest.exists() and dest.stat().st_size > 0 and not decompressed.exists():
            with gzip.open(dest, "rb") as source, decompressed.open("wb") as target:
                shutil.copyfileobj(source, target)
            return decompressed
    elif skip_existing and dest.exists() and dest.stat().st_size > 0:
        return dest

    temp_path = dest.with_name(f"{dest.name}.part")
    if temp_path.exists():
        temp_path.unlink()

    response = requests.get(url, stream=True, timeout=timeout_seconds)
    response.raise_for_status()
    try:
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        temp_path.replace(dest)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        response.close()

    if dest.suffix == ".gz":
        decompressed = dest.with_suffix("")
        with gzip.open(dest, "rb") as source, decompressed.open("wb") as target:
            shutil.copyfileobj(source, target)
        return decompressed
    return dest


def download_tcga_pancan_atlas(
    data_dir: str | Path,
    *,
    skip_existing: bool = True,
    timeout_seconds: int = 300,
    keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    resources = load_tcga_genomics_prototype_resources()
    cache_dir = Path(data_dir)
    selected_keys = list(keys or DEFAULT_PANCAN_KEYS)
    downloaded: dict[str, Path] = {}

    for key in selected_keys:
        if key not in DEFAULT_PANCAN_FILE_NAMES:
            raise KeyError(f"Unsupported PanCancer Atlas key: {key}")

        dest = _resolved_pancan_output_path(cache_dir, key)
        try:
            url = _resolve_pancan_download_url(resources=resources, key=key, timeout_seconds=timeout_seconds)
            downloaded[key] = _download_open_access_file(
                url=url,
                dest=dest,
                skip_existing=skip_existing,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            if key in OPTIONAL_PANCAN_KEYS:
                print(f"[warning] Unable to resolve/download optional PanCancer source '{key}'.")
                continue
            raise

    return downloaded


def _ensure_required_pancan_inputs(data_dir: Path, *, keys: list[str] | tuple[str, ...] | None = None) -> None:
    selected_keys = list(keys or REQUIRED_PANCAN_KEYS)
    missing: list[str] = []
    for key in selected_keys:
        if key not in REQUIRED_PANCAN_KEYS:
            continue
        if not resolve_pancan_cache_path(data_dir, key).exists():
            missing.append(key)
    if missing:
        raise FileNotFoundError(
            "Missing required TCGA genomics cache files. "
            f"Expected keys: {missing}. Cache dir: {data_dir}"
        )


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _snake_case(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_").lower()
    return normalized


def _first_matching_column(columns: list[str], patterns: list[str]) -> str:
    lowered = {str(column): str(column).lower() for column in columns}
    for pattern in patterns:
        pattern_lower = str(pattern).lower()
        for original, lower in lowered.items():
            if pattern_lower in lower:
                return original
    return ""


def _load_table(path: Path, *, sep: str = "\t") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, low_memory=False)


def _load_optional_table(data_dir: Path, key: str, *, sep: str = "\t") -> pd.DataFrame:
    path = resolve_pancan_cache_path(data_dir, key)
    if not path.exists():
        return pd.DataFrame()
    return _load_table(path, sep=sep)


def _clean_scalar_text(value: Any, *, max_chars: int = 80) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"na", "nan", "null", "none", "not available", "not reported"}:
        return ""
    if len(text) > max_chars or "\n" in text or "\t" in text:
        return ""
    return text


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_subtypes_index(data_dir: Path, patient_ids: set[str]) -> dict[str, dict[str, Any]]:
    path = resolve_pancan_cache_path(data_dir, "subtypes")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "pan.samplesID" in df.columns:
        df["patient_id"] = df["pan.samplesID"].map(patient_from_barcode)
    elif "sampleID" in df.columns:
        df["patient_id"] = df["sampleID"].map(patient_from_barcode)
    else:
        raise ValueError(f"Subtype table is missing a recognizable sample id column: {path}")

    index: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id or patient_id not in patient_ids or patient_id in index:
            continue
        index[patient_id] = row.to_dict()
    return index


def _load_filtered_maf(
    data_dir: Path,
    *,
    patient_ids: set[str],
    allowed_genes: set[str],
    nonsynonymous_classes: set[str],
) -> dict[str, pd.DataFrame]:
    path = resolve_pancan_cache_path(data_dir, "mc3_maf")
    usecols = [
        "Hugo_Symbol",
        "Variant_Classification",
        "Variant_Type",
        "HGVSp_Short",
        "Tumor_Sample_Barcode",
        "IMPACT",
        "Chromosome",
        "Start_Position",
        "End_Position",
    ]
    frames: list[pd.DataFrame] = []
    read_csv_kwargs: dict[str, Any] = {
        "sep": "\t",
        "comment": "#",
        "low_memory": False,
        "usecols": lambda col: col in usecols,
        "chunksize": 200_000,
    }

    for chunk in pd.read_csv(path, **read_csv_kwargs):
        chunk["patient_id"] = chunk["Tumor_Sample_Barcode"].map(patient_from_barcode)
        mask = (
            chunk["patient_id"].isin(patient_ids)
            & chunk["Variant_Classification"].isin(nonsynonymous_classes)
        )
        if allowed_genes:
            mask &= chunk["Hugo_Symbol"].astype(str).str.upper().isin(allowed_genes)
        filtered = chunk.loc[mask].copy()
        if filtered.empty:
            continue
        filtered["Hugo_Symbol"] = filtered["Hugo_Symbol"].astype(str).str.upper()
        frames.append(filtered)

    if not frames:
        return {}

    filtered_maf = pd.concat(frames, ignore_index=True)
    by_patient: dict[str, pd.DataFrame] = {}
    for patient_id, group in filtered_maf.groupby("patient_id", sort=False):
        by_patient[str(patient_id)] = group.reset_index(drop=True)
    return by_patient


def _load_gistic_table(data_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    path = resolve_pancan_cache_path(data_dir, "gistic_thresholded")
    gistic_df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    for col in ["Locus ID", "Cytoband"]:
        if col in gistic_df.columns:
            gistic_df = gistic_df.drop(columns=[col])

    gistic_col_by_patient: dict[str, str] = {}
    for column in gistic_df.columns:
        patient_id = patient_from_barcode(column)
        if patient_id and patient_id not in gistic_col_by_patient:
            gistic_col_by_patient[patient_id] = str(column)
    return gistic_df, gistic_col_by_patient


def _index_tmb(data_dir: Path, patient_ids: set[str]) -> dict[str, dict[str, str]]:
    df = _load_table(resolve_pancan_cache_path(data_dir, "mutation_load"))
    sample_col = _first_matching_column(list(df.columns), ["barcode", "sample"])
    if not sample_col and len(df.columns) > 0:
        sample_col = str(df.columns[0])
    count_col = _first_matching_column(
        list(df.columns),
        ["non-silent", "nonsilent", "non_silent", "mutation count", "mutation_count", "count", "load"],
    )

    index: dict[str, dict[str, str]] = {}
    if not sample_col:
        return index

    for _, row in df.iterrows():
        patient_id = patient_from_barcode(row.get(sample_col, ""))
        if not patient_id or patient_id not in patient_ids or patient_id in index:
            continue
        result = {"tmb": "not_available", "tmb_class": "not_available"}
        if count_col:
            value = row.get(count_col)
            if pd.notna(value):
                count = float(value)
                tmb = count / 30.0
                if tmb > 20:
                    tmb_class = "high"
                elif tmb > 6:
                    tmb_class = "intermediate"
                else:
                    tmb_class = "low"
                result = {"tmb": f"{tmb:.1f}", "tmb_class": tmb_class}
        index[patient_id] = result
    return index


def _index_leukocyte_fraction(data_dir: Path, patient_ids: set[str]) -> dict[str, str]:
    df = _load_table(resolve_pancan_cache_path(data_dir, "leukocyte_fraction"))
    if df.empty:
        return {}
    sample_col = str(df.columns[0])
    leuk_col = _first_matching_column(list(df.columns), ["leuk", "estimate"])

    index: dict[str, str] = {}
    if not leuk_col:
        return index
    for _, row in df.iterrows():
        patient_id = patient_from_barcode(row.get(sample_col, ""))
        if not patient_id or patient_id not in patient_ids or patient_id in index:
            continue
        value = row.get(leuk_col)
        if pd.notna(value):
            index[patient_id] = f"{float(value):.3f}"
    return index


def _index_top_immune_cells(data_dir: Path, patient_ids: set[str]) -> dict[str, str]:
    df = _load_table(resolve_pancan_cache_path(data_dir, "cibersort"))
    if df.empty:
        return {}

    sample_col = str(df.columns[0])
    index: dict[str, str] = {}
    for _, row in df.iterrows():
        patient_id = patient_from_barcode(row.get(sample_col, ""))
        if not patient_id or patient_id not in patient_ids or patient_id in index:
            continue
        cell_values: dict[str, float] = {}
        for column in df.columns:
            column_text = str(column).lower()
            if column == sample_col:
                continue
            if "p-value" in column_text or "rmse" in column_text or "correlation" in column_text:
                continue
            value = row.get(column)
            if pd.notna(value):
                cell_values[str(column)] = float(value)
        if not cell_values:
            continue
        top3 = sorted(cell_values.items(), key=lambda item: -item[1])[:3]
        formatted = [f"{name} ({fraction:.2f})" for name, fraction in top3 if fraction > 0.01]
        if formatted:
            index[patient_id] = ", ".join(formatted)
    return index


def _index_absolute_scores(data_dir: Path, patient_ids: set[str]) -> dict[str, dict[str, str]]:
    df = _load_table(resolve_pancan_cache_path(data_dir, "absolute_scores"))
    if df.empty:
        return {}

    sample_col = str(df.columns[0])
    index: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        patient_id = patient_from_barcode(row.get(sample_col, ""))
        if not patient_id or patient_id not in patient_ids or patient_id in index:
            continue

        record = {
            "aneuploidy_score": "not_available",
            "whole_genome_doubling": "not_available",
            "tumor_purity": "not_available",
        }
        for column in df.columns:
            column_text = str(column).lower()
            value = row.get(column)
            if pd.isna(value):
                continue
            if "aneuploidy" in column_text:
                record["aneuploidy_score"] = str(int(value))
            elif "wgd" in column_text or "genome_doubling" in column_text:
                if isinstance(value, (int, float)):
                    record["whole_genome_doubling"] = "yes" if float(value) > 0 else "no"
                else:
                    record["whole_genome_doubling"] = str(value).strip().lower()
            elif "purity" in column_text:
                record["tumor_purity"] = f"{float(value):.2f}"
        index[patient_id] = record
    return index


def _index_hrd_scores(data_dir: Path, patient_ids: set[str]) -> dict[str, dict[str, str]]:
    df = _load_table(resolve_pancan_cache_path(data_dir, "hrd_scores"))
    if df.empty:
        return {}

    sample_col = str(df.columns[0])
    hrd_col = _first_matching_column(list(df.columns), ["hrd"])
    index: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        patient_id = patient_from_barcode(row.get(sample_col, ""))
        if not patient_id or patient_id not in patient_ids or patient_id in index:
            continue
        record = {"hrd_score": "not_available"}
        if hrd_col:
            value = row.get(hrd_col)
            if pd.notna(value):
                record["hrd_score"] = str(int(value))
        index[patient_id] = record
    return index


def _find_row_identifier_column(df: pd.DataFrame) -> str:
    candidates = [
        "bcr_patient_barcode",
        "patient_id",
        "submitter_id",
        "sampleID",
        "sample",
        "barcode",
    ]
    sample_col = _first_matching_column(list(df.columns), candidates)
    if sample_col:
        return sample_col
    if len(df.columns) > 0:
        return str(df.columns[0])
    return ""


def _candidate_arm_values(row: dict[str, Any], column_names: list[str], label: str) -> list[int | None]:
    normalized_label = _normalize_token(label)
    normalized_cols = {str(column): _normalize_token(column) for column in column_names}
    matches: list[int | None] = []
    for column, normalized in normalized_cols.items():
        if normalized == normalized_label or normalized.endswith(normalized_label) or normalized_label in normalized:
            raw_value = row.get(column)
            numeric = _safe_float(raw_value)
            if numeric is None:
                cleaned = _clean_scalar_text(raw_value, max_chars=32).lower()
                if cleaned in {"loss", "deleted", "deletion"}:
                    matches.append(-1)
                elif cleaned in {"gain", "gained", "amp", "amplified", "amplification"}:
                    matches.append(1)
                elif cleaned in {"0", "diploid", "neutral", "none"}:
                    matches.append(0)
                elif cleaned:
                    matches.append(None)
                continue
            if numeric > 0:
                matches.append(1)
            elif numeric < 0:
                matches.append(-1)
            else:
                matches.append(0)
    return matches


def _evaluate_arm_rule(row: dict[str, Any], column_names: list[str], chromosome: str, direction: str) -> bool | None:
    chromosome_text = str(chromosome).lower().replace("chr", "")
    p_values = _candidate_arm_values(row, column_names, f"{chromosome_text}p")
    q_values = _candidate_arm_values(row, column_names, f"{chromosome_text}q")
    whole_values = _candidate_arm_values(row, column_names, chromosome_text)

    if direction == "p_loss":
        return any(value == -1 for value in p_values) if p_values else None
    if direction == "q_loss":
        return any(value == -1 for value in q_values) if q_values else None
    if direction == "p_gain":
        return any(value == 1 for value in p_values) if p_values else None
    if direction == "q_gain":
        return any(value == 1 for value in q_values) if q_values else None
    if direction in {"whole_loss", "loss"}:
        if whole_values:
            return any(value == -1 for value in whole_values)
        if p_values and q_values:
            return any(value == -1 for value in p_values) and any(value == -1 for value in q_values)
        return None
    if direction in {"whole_gain", "gain"}:
        if whole_values:
            return any(value == 1 for value in whole_values)
        if p_values and q_values:
            return any(value == 1 for value in p_values) and any(value == 1 for value in q_values)
        return None
    return None


def _index_arm_level_events(
    data_dir: Path,
    *,
    patient_ids: set[str],
    config_by_patient: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    df = _load_optional_table(data_dir, "arm_calls")
    if df.empty:
        return {}

    sample_col = _find_row_identifier_column(df)
    if not sample_col:
        return {}

    results: dict[str, dict[str, Any]] = {}
    column_names = list(df.columns)
    for _, row in df.iterrows():
        patient_id = patient_from_barcode(row.get(sample_col, ""))
        if not patient_id or patient_id not in patient_ids or patient_id in results:
            continue
        config = config_by_patient.get(patient_id)
        arm_event_defs = dict(getattr(config, "arm_level_events", {})) if config is not None else {}
        record: dict[str, Any] = {}
        if arm_event_defs:
            event_values: dict[str, str] = {}
            row_dict = row.to_dict()
            for event_name, rules in arm_event_defs.items():
                evaluations: list[bool | None] = []
                for chromosome, direction in list(rules or []):
                    evaluations.append(_evaluate_arm_rule(row_dict, column_names, str(chromosome), str(direction)))
                if not evaluations or any(value is None for value in evaluations):
                    event_values[str(event_name)] = "not_available"
                else:
                    event_values[str(event_name)] = "yes" if all(bool(value) for value in evaluations) else "no"
            record["arm_level_events"] = event_values

        aneuploidy_col = _first_matching_column(column_names, ["aneuploidy"])
        if aneuploidy_col:
            value = row.get(aneuploidy_col)
            if pd.notna(value):
                record["aneuploidy_score"] = str(int(float(value)))

        if record:
            results[patient_id] = record
    return results


def _add_viral_hit(record: dict[str, Any], label: str) -> None:
    lowered = str(label).lower()
    if "hpv" in lowered or "papilloma" in lowered:
        record["hpv_status"] = "HPV_positive"
        hpv_match = re.search(r"(hpv[-_ ]?\d+[a-z]?)", lowered)
        if hpv_match:
            record.setdefault("hpv_type", hpv_match.group(1).upper().replace(" ", "").replace("_", ""))
        elif not record.get("hpv_type"):
            record["hpv_type"] = "other"
    if "ebv" in lowered or "epstein" in lowered:
        record["ebv_status"] = "positive"
    hbv_hit = "hbv" in lowered or "hepatitis b" in lowered
    hcv_hit = "hcv" in lowered or "hepatitis c" in lowered
    existing = str(record.get("viral_status", "")).strip()
    if hbv_hit and hcv_hit:
        record["viral_status"] = "HBV_HCV"
    elif hbv_hit:
        record["viral_status"] = "HBV" if existing not in {"HCV", "HBV_HCV"} else "HBV_HCV"
    elif hcv_hit:
        record["viral_status"] = "HCV" if existing not in {"HBV", "HBV_HCV"} else "HBV_HCV"


def _is_positive_signal(value: Any) -> bool:
    numeric = _safe_float(value)
    if numeric is not None:
        return numeric > 0
    text = _clean_scalar_text(value, max_chars=32).lower()
    if not text:
        return False
    return text not in {"0", "0.0", "negative", "no", "none", "absent", "not_detected"}


def _index_viral_annotations(data_dir: Path, patient_ids: set[str]) -> dict[str, dict[str, str]]:
    df = _load_optional_table(data_dir, "viral_reads")
    if df.empty:
        return {}

    sample_col = _find_row_identifier_column(df)
    if not sample_col:
        return {}

    virus_col = _first_matching_column(list(df.columns), ["virus"])
    value_col = _first_matching_column(list(df.columns), ["read", "count", "score", "value"])
    by_patient: dict[str, dict[str, str]] = {}

    if virus_col and value_col and virus_col != sample_col:
        for _, row in df.iterrows():
            patient_id = patient_from_barcode(row.get(sample_col, ""))
            if not patient_id or patient_id not in patient_ids:
                continue
            if not _is_positive_signal(row.get(value_col)):
                continue
            record = by_patient.setdefault(patient_id, {})
            _add_viral_hit(record, str(row.get(virus_col, "")))
    else:
        for _, row in df.iterrows():
            patient_id = patient_from_barcode(row.get(sample_col, ""))
            if not patient_id or patient_id not in patient_ids:
                continue
            record = by_patient.setdefault(patient_id, {})
            for column in df.columns:
                if column == sample_col:
                    continue
                if _is_positive_signal(row.get(column)):
                    _add_viral_hit(record, str(column))

    finalized: dict[str, dict[str, str]] = {}
    for patient_id, record in by_patient.items():
        resolved = dict(record)
        resolved.setdefault("hpv_status", "HPV_negative")
        resolved.setdefault("ebv_status", "negative")
        resolved.setdefault("viral_status", "negative")
        finalized[patient_id] = resolved
    return finalized


def _index_pathway_alterations(
    data_dir: Path,
    patient_ids: set[str],
    *,
    ten_pathways: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    df = _load_optional_table(data_dir, "pathway_alterations")
    if df.empty:
        return {}

    sample_col = _find_row_identifier_column(df)
    if not sample_col:
        return {}

    # Build a map from normalised column token → original column name so we can
    # match pathway labels regardless of case/separator differences in the file.
    normalized_to_col: dict[str, str] = {
        _normalize_token(col): col for col in df.columns if col != sample_col
    }
    # Resolve each canonical pathway name to a column in the file (or None).
    pathway_col_map: list[tuple[str, str | None]] = []
    for pathway in ten_pathways:
        col = normalized_to_col.get(_normalize_token(pathway))
        pathway_col_map.append((pathway, col))

    index: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        patient_id = patient_from_barcode(row.get(sample_col, ""))
        if not patient_id or patient_id not in patient_ids or patient_id in index:
            continue
        record: dict[str, str] = {}
        for pathway, col in pathway_col_map:
            if col is None:
                continue
            value = row.get(col)
            if pd.isna(value):
                continue
            try:
                altered = int(float(value)) != 0
            except (TypeError, ValueError):
                altered = str(value).strip().lower() in {"1", "true", "yes", "altered"}
            record[pathway] = "altered" if altered else "intact"
        if record:
            index[patient_id] = record
    return index


def _index_clinical_followup_annotations(data_dir: Path, patient_ids: set[str]) -> dict[str, dict[str, str]]:
    df = _load_optional_table(data_dir, "clinical_followup")
    if df.empty:
        return {}

    sample_col = _find_row_identifier_column(df)
    if not sample_col:
        return {}

    interesting_key_patterns = (
        "histolog",
        "grade",
        "stage",
        "subtype",
        "hpv",
        "ebv",
        "viral",
        "hepat",
        "who",
        "morphology",
    )
    by_patient: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        patient_id = patient_from_barcode(row.get(sample_col, ""))
        if not patient_id or patient_id not in patient_ids:
            continue
        record = by_patient.setdefault(patient_id, {})
        for column in df.columns:
            if column == sample_col:
                continue
            key = _snake_case(column)
            if not key or key in record:
                continue
            if not any(pattern in key for pattern in interesting_key_patterns):
                continue
            value = _clean_scalar_text(row.get(column))
            if value:
                record[key] = value
    return by_patient


def _extract_mutations_for_patient(
    patient_maf: pd.DataFrame | None,
    config: Any | None,
) -> dict[str, Any]:
    driver_genes = [str(gene).strip().upper() for gene in list(getattr(config, "driver_genes", [])) if str(gene).strip()]
    hotspot_variants = dict(getattr(config, "hotspot_variants", {}))

    if patient_maf is None or patient_maf.empty or not driver_genes:
        return {
            "key_somatic_mutations": ["none_in_driver_genes"],
            "mutated_genes": set(),
            "hotspot_hits": {},
            "n_driver_mutations": 0,
        }

    driver_set = set(driver_genes)
    mask = patient_maf["Hugo_Symbol"].astype(str).str.upper().isin(driver_set)
    filtered = patient_maf.loc[mask].copy()
    if filtered.empty:
        return {
            "key_somatic_mutations": ["none_in_driver_genes"],
            "mutated_genes": set(),
            "hotspot_hits": {},
            "n_driver_mutations": 0,
        }

    mutations: list[str] = []
    mutated_genes: set[str] = set()
    hotspot_hits: dict[str, str] = {}
    for _, row in filtered.iterrows():
        gene = str(row.get("Hugo_Symbol", "")).strip().upper()
        variant_class = str(row.get("Variant_Classification", "")).strip()
        hgvsp = row.get("HGVSp_Short", "")
        hgvsp_text = "" if pd.isna(hgvsp) else str(hgvsp).strip()
        hgvsp_clean = hgvsp_text.replace("p.", "") if hgvsp_text else ""
        variant_label = variant_class.replace("_Mutation", "").replace("_", " ").lower()

        if hgvsp_clean:
            mutations.append(f"{gene} p.{hgvsp_clean} ({variant_label})")
        else:
            mutations.append(f"{gene} ({variant_label})")
        mutated_genes.add(gene)

        if gene in hotspot_variants:
            for hotspot in list(hotspot_variants[gene]):
                hotspot_text = str(hotspot).strip()
                if hotspot_text and hotspot_text in hgvsp_clean:
                    hotspot_hits[gene] = hgvsp_clean
                    break

    return {
        "key_somatic_mutations": mutations if mutations else ["none_in_driver_genes"],
        "mutated_genes": mutated_genes,
        "hotspot_hits": hotspot_hits,
        "n_driver_mutations": len(mutations),
    }


def _extract_cna_for_patient(
    gistic_df: pd.DataFrame,
    gistic_col_by_patient: dict[str, str],
    *,
    patient_id: str,
    config: Any | None,
) -> dict[str, Any]:
    cna_genes = [str(gene).strip().upper() for gene in list(getattr(config, "cna_genes", [])) if str(gene).strip()]
    if not cna_genes:
        return {
            "key_copy_number_alterations": ["none_assessed"],
            "cna_by_gene": {},
        }

    column = gistic_col_by_patient.get(patient_id, "")
    if not column:
        return {
            "key_copy_number_alterations": ["data_not_available"],
            "cna_by_gene": {},
        }

    cna_labels = {-2: "deep_deletion", -1: "shallow_deletion", 1: "gain", 2: "amplification"}
    results: list[str] = []
    cna_by_gene: dict[str, str] = {}
    for gene in cna_genes:
        if gene not in gistic_df.index:
            continue
        value = gistic_df.loc[gene, column]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        if pd.isna(value):
            continue
        try:
            normalized = int(float(value))
        except (TypeError, ValueError):
            continue
        if normalized != 0 and normalized in cna_labels:
            label = cna_labels[normalized]
            results.append(f"{gene}: {label}")
            cna_by_gene[gene] = label

    return {
        "key_copy_number_alterations": results if results else ["none_detected"],
        "cna_by_gene": cna_by_gene,
    }


def _extract_subtype_for_patient(row: dict[str, Any] | None) -> dict[str, str]:
    result = {"molecular_subtype": "not_available"}
    if not row:
        return result

    subtype_columns = [
        "Subtype_Selected",
        "Subtype_mRNA",
        "Subtype_DNAmeth",
        "Subtype_protein",
        "Subtype_miRNA",
        "Subtype_CNA",
        "Subtype_Integrative",
    ]
    for column in subtype_columns:
        if column not in row or pd.isna(row[column]):
            continue
        value = str(row[column]).strip()
        if value and value.lower() not in {"na", "nan", "not available"}:
            result["molecular_subtype"] = value
            break

    immune_value = row.get("Immune_Subtype")
    if immune_value is not None and pd.notna(immune_value):
        text = str(immune_value).strip()
        if text:
            result["immune_subtype"] = text
    return result


def _first_annotation_value(annotations: dict[str, str], patterns: list[str]) -> str:
    lowered_items = [(key, _normalize_token(key), value) for key, value in dict(annotations or {}).items()]
    normalized_patterns = [_normalize_token(pattern) for pattern in patterns]
    for pattern in normalized_patterns:
        for _key, normalized_key, value in lowered_items:
            if pattern and pattern in normalized_key:
                cleaned = _clean_scalar_text(value)
                if cleaned:
                    return cleaned
    return ""


def _resolve_msi_status(
    patient_record: dict[str, Any],
    *,
    config: Any | None,
    resources: TcgaGenomicsPrototypeResources,
) -> str:
    subtype_text = str(patient_record.get("molecular_subtype", "")).strip()
    clinical_annotations = dict(patient_record.get("clinical_annotations", {}))
    annotation_value = _first_annotation_value(clinical_annotations, ["msi", "microsatellite"])
    if annotation_value:
        normalized = annotation_value.lower()
        if "msi-h" in normalized or "msih" in normalized or "hypermut" in normalized:
            return "MSI-H"
        if "msi-l" in normalized or "msil" in normalized:
            return "MSI-L"
        if normalized in {"mss", "stable"}:
            return "MSS"
        numeric = _safe_float(annotation_value)
        if numeric is not None:
            if numeric >= resources.msi_threshold_high:
                return "MSI-H"
            if numeric >= resources.msi_threshold_low:
                return "MSI-L"
            return "MSS"
    if subtype_text and "msi" in subtype_text.lower():
        return "MSI-H"
    if bool(getattr(config, "msi_relevant", False)):
        return "not_available"
    return ""


def _replace_placeholder_field(fields: dict[str, str], key: str, value: str) -> None:
    cleaned = _clean_scalar_text(value)
    if not cleaned:
        return
    existing = str(fields.get(key, "")).strip()
    if existing in {"", "check_clinical", "not_available", "unknown"}:
        fields[key] = cleaned


def _infer_histologic_type(patient_record: dict[str, Any], cancer_code: str) -> str:
    diagnosis = str(patient_record.get("primary_diagnosis", "")).lower()
    morphology = str(patient_record.get("morphology", "")).lower()
    clinical_annotations = dict(patient_record.get("clinical_annotations", {}))
    annotation_text = " ".join(str(value).lower() for value in clinical_annotations.values())
    combined = " ".join(part for part in [diagnosis, morphology, annotation_text] if part)

    if cancer_code == "KIRP":
        if "type 1" in combined or "type i" in combined:
            return "type_1"
        if "type 2" in combined or "type ii" in combined:
            return "type_2"
    if cancer_code in {"CESC", "ESCA"}:
        if "adenosquamous" in combined:
            return "adenosquamous"
        if "adenocarcinoma" in combined:
            return "adenocarcinoma"
        if "squamous" in combined:
            return "squamous"
    if cancer_code == "TGCT":
        if "mixed" in combined:
            return "mixed"
        if "seminoma" in combined:
            return "seminoma"
        if "non seminoma" in combined or "nonseminoma" in combined:
            return "non_seminoma"
    if cancer_code == "MESO":
        if "biphasic" in combined:
            return "biphasic"
        if "sarcomatoid" in combined:
            return "sarcomatoid"
        if "epithelioid" in combined:
            return "epithelioid"
    if cancer_code == "KICH":
        if "eosinophilic" in combined:
            return "eosinophilic"
        if "classic" in combined:
            return "classic"
    return ""


def _enrich_special_fields(
    patient_record: dict[str, Any],
    *,
    config: Any | None,
    initial_fields: dict[str, Any] | None,
) -> dict[str, str]:
    fields = {str(key): str(value) for key, value in dict(initial_fields or {}).items()}
    if config is None:
        return fields

    cancer_code = str(getattr(config, "cancer_code", "")).upper()
    cna_by_gene = {str(key): str(value) for key, value in dict(patient_record.get("cna_by_gene", {})).items()}
    arm_level_events = {str(key): str(value) for key, value in dict(patient_record.get("arm_level_events", {})).items()}
    mutated_genes = set(patient_record.get("mutated_genes", set()))

    if cancer_code == "KIRP":
        met_cna = cna_by_gene.get("MET", "")
        if met_cna in {"gain", "amplification"} and fields.get("met_status") == "wildtype":
            fields["met_status"] = "amplified"
        histologic_type = _infer_histologic_type(patient_record, cancer_code)
        if histologic_type:
            fields.setdefault("histologic_type", histologic_type)
    elif cancer_code == "KICH":
        characteristic_losses = [
            key.replace("_loss", "")
            for key, value in arm_level_events.items()
            if key.startswith("chr") and value == "yes"
        ]
        if arm_level_events:
            fields["characteristic_chr_losses"] = (
                ", ".join(sorted(characteristic_losses)) if characteristic_losses else "none_detected"
            )
    elif cancer_code == "BRCA":
        erbb2_cna = cna_by_gene.get("ERBB2", "")
        fields["erbb2_status"] = "amplified" if erbb2_cna == "amplification" else "not_amplified"
    elif cancer_code == "PRAD":
        pten_cna = cna_by_gene.get("PTEN", "")
        if pten_cna in {"deep_deletion", "shallow_deletion"}:
            fields["pten_status"] = "deleted"
        elif "PTEN" in mutated_genes:
            fields["pten_status"] = "mutated"
        else:
            fields.setdefault("pten_status", "intact")
        fields["ar_amplified"] = "yes" if cna_by_gene.get("AR") in {"gain", "amplification"} else "no"

    _replace_placeholder_field(fields, "hpv_status", str(patient_record.get("hpv_status", "")))
    _replace_placeholder_field(fields, "hpv_type", str(patient_record.get("hpv_type", "")))
    _replace_placeholder_field(fields, "viral_status", str(patient_record.get("viral_status", "")))
    _replace_placeholder_field(fields, "ebv_status", str(patient_record.get("ebv_status", "")))

    histologic_type = _infer_histologic_type(patient_record, cancer_code)
    if cancer_code in {"KIRP", "KICH", "CESC", "ESCA", "TGCT", "MESO"} and histologic_type:
        fields.setdefault("histologic_type", histologic_type)

    return fields


def serialize_patient_genomics(patient: dict[str, Any]) -> str:
    lines = ["[GENOMICS]"]
    lines.append(f"cancer_type: {patient.get('cancer_code', 'unknown')}")
    lines.append(f"molecular_subtype: {patient.get('molecular_subtype', 'not_available')}")

    immune_subtype = str(patient.get("immune_subtype", "")).strip()
    if immune_subtype:
        lines.append(f"immune_subtype: {immune_subtype}")

    tmb = patient.get("tmb", "not_available")
    tmb_class = patient.get("tmb_class", "")
    if tmb != "not_available":
        lines.append(f"tmb: {tmb} mut/Mb ({tmb_class})")
    else:
        lines.append("tmb: not_available")

    msi_status = str(patient.get("msi_status", "")).strip()
    if msi_status:
        lines.append(f"msi_status: {msi_status}")

    aneuploidy_score = patient.get("aneuploidy_score", "not_available")
    if aneuploidy_score != "not_available":
        lines.append(f"aneuploidy_score: {aneuploidy_score}")

    whole_genome_doubling = patient.get("whole_genome_doubling", "not_available")
    if whole_genome_doubling != "not_available":
        lines.append(f"whole_genome_doubling: {whole_genome_doubling}")

    tumor_purity = patient.get("tumor_purity", "not_available")
    if tumor_purity != "not_available":
        lines.append(f"tumor_purity: {tumor_purity}")

    mutations = patient.get("key_somatic_mutations", ["not_available"])
    if isinstance(mutations, list):
        lines.append(f"key_somatic_mutations: {', '.join(mutations)}")
    else:
        lines.append(f"key_somatic_mutations: {mutations}")

    cnas = patient.get("key_copy_number_alterations", ["not_available"])
    if isinstance(cnas, list):
        lines.append(f"key_copy_number_alterations: {', '.join(cnas)}")
    else:
        lines.append(f"key_copy_number_alterations: {cnas}")

    arm_level_events = patient.get("arm_level_events", {})
    if arm_level_events:
        lines.append(
            "arm_level_cna: "
            + ", ".join(f"{key}: {value}" for key, value in dict(arm_level_events).items())
        )

    pathway_alterations = patient.get("pathway_alterations", {})
    if pathway_alterations:
        lines.append(
            "pathway_alterations: "
            + ", ".join(f"{key}: {value}" for key, value in dict(pathway_alterations).items())
        )

    hrd_score = patient.get("hrd_score", "not_available")
    if hrd_score != "not_available":
        lines.append(f"hrd_score: {hrd_score}")

    special_fields = patient.get("cancer_specific_fields", {})
    for key, value in dict(special_fields).items():
        lines.append(f"{key}: {value}")

    leukocyte_fraction = patient.get("leukocyte_fraction", "not_available")
    if leukocyte_fraction != "not_available":
        lines.append(f"leukocyte_fraction: {leukocyte_fraction}")

    top_immune_cells = patient.get("top_immune_cells", "not_available")
    if top_immune_cells != "not_available":
        lines.append(f"dominant_immune_cells: {top_immune_cells}")

    lines.append("[/GENOMICS]")
    return "\n".join(lines)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _genomics_fields_for_output(patient_record: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    keep_keys = [
        "patient_id",
        "cancer_code",
        "molecular_subtype",
        "immune_subtype",
        "tmb",
        "tmb_class",
        "msi_status",
        "aneuploidy_score",
        "whole_genome_doubling",
        "tumor_purity",
        "key_somatic_mutations",
        "key_copy_number_alterations",
        "arm_level_events",
        "pathway_alterations",
        "hrd_score",
        "cancer_specific_fields",
        "leukocyte_fraction",
        "top_immune_cells",
        "cna_by_gene",
        "clinical_annotations",
    ]
    for key in keep_keys:
        value = patient_record.get(key)
        if value in (None, "", {}, []):
            continue
        output[key] = _to_serializable(value)
    return output


def build_tcga_genomics_by_patient_id(
    *,
    cases: list[dict[str, Any]],
    data_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    resources = load_tcga_genomics_prototype_resources()
    cache_dir = Path(data_dir)
    _ensure_required_pancan_inputs(cache_dir)

    patient_to_cancer_code: dict[str, str] = {}
    case_by_patient: dict[str, dict[str, Any]] = {}
    for case in cases:
        patient_id = str(case.get("submitter_id", "")).strip()
        if not patient_id:
            continue
        project_id = str((case.get("project") or {}).get("project_id", "")).strip()
        cancer_code = cancer_code_from_project_id(project_id)
        if patient_id not in patient_to_cancer_code:
            patient_to_cancer_code[patient_id] = cancer_code
        case_by_patient.setdefault(patient_id, case)

    patient_ids = set(patient_to_cancer_code)
    if not patient_ids:
        return {}

    config_by_patient: dict[str, Any] = {}
    selected_driver_genes: set[str] = set()
    for patient_id, cancer_code in patient_to_cancer_code.items():
        config = resources.cancer_configs.get(cancer_code)
        config_by_patient[patient_id] = config
        if config is None:
            continue
        selected_driver_genes.update(
            [str(gene).strip().upper() for gene in list(getattr(config, "driver_genes", [])) if str(gene).strip()]
        )

    subtypes_by_patient = _load_subtypes_index(cache_dir, patient_ids)
    maf_by_patient = _load_filtered_maf(
        cache_dir,
        patient_ids=patient_ids,
        allowed_genes=selected_driver_genes,
        nonsynonymous_classes=resources.nonsynonymous_classes,
    )
    gistic_df, gistic_col_by_patient = _load_gistic_table(cache_dir)
    tmb_by_patient = _index_tmb(cache_dir, patient_ids)
    leukocyte_by_patient = _index_leukocyte_fraction(cache_dir, patient_ids)
    immune_cells_by_patient = _index_top_immune_cells(cache_dir, patient_ids)
    absolute_by_patient = _index_absolute_scores(cache_dir, patient_ids)
    hrd_by_patient = _index_hrd_scores(cache_dir, patient_ids)
    arm_by_patient = _index_arm_level_events(cache_dir, patient_ids=patient_ids, config_by_patient=config_by_patient)
    viral_by_patient = _index_viral_annotations(cache_dir, patient_ids)
    clinical_by_patient = _index_clinical_followup_annotations(cache_dir, patient_ids)
    pathway_by_patient = _index_pathway_alterations(
        cache_dir, patient_ids, ten_pathways=resources.ten_pathways
    )

    genomics_by_patient_id: dict[str, dict[str, Any]] = {}
    for patient_id in sorted(patient_ids):
        cancer_code = patient_to_cancer_code.get(patient_id, "")
        config = config_by_patient.get(patient_id)
        case = case_by_patient.get(patient_id, {})
        diagnosis = ((case.get("diagnoses") or [{}])[0] if isinstance(case.get("diagnoses"), list) else {}) or {}

        patient_record: dict[str, Any] = {
            "patient_id": patient_id,
            "cancer_code": cancer_code or "unknown",
            "primary_diagnosis": str(diagnosis.get("primary_diagnosis", "")),
            "morphology": str(diagnosis.get("morphology", "")),
            "leukocyte_fraction": leukocyte_by_patient.get(patient_id, "not_available"),
            "top_immune_cells": immune_cells_by_patient.get(patient_id, "not_available"),
            "clinical_annotations": dict(clinical_by_patient.get(patient_id, {})),
        }
        patient_record.update(_extract_subtype_for_patient(subtypes_by_patient.get(patient_id)))
        patient_record.update(_extract_mutations_for_patient(maf_by_patient.get(patient_id), config))
        patient_record.update(
            _extract_cna_for_patient(
                gistic_df,
                gistic_col_by_patient,
                patient_id=patient_id,
                config=config,
            )
        )
        patient_record.update(tmb_by_patient.get(patient_id, {"tmb": "not_available", "tmb_class": "not_available"}))
        patient_record.update(
            absolute_by_patient.get(
                patient_id,
                {
                    "aneuploidy_score": "not_available",
                    "whole_genome_doubling": "not_available",
                    "tumor_purity": "not_available",
                },
            )
        )
        patient_record.update(hrd_by_patient.get(patient_id, {"hrd_score": "not_available"}))
        patient_record.update(arm_by_patient.get(patient_id, {}))
        patient_record.update(viral_by_patient.get(patient_id, {}))
        pathway_data = pathway_by_patient.get(patient_id)
        if pathway_data:
            patient_record["pathway_alterations"] = pathway_data

        msi_status = _resolve_msi_status(patient_record, config=config, resources=resources)
        if msi_status:
            patient_record["msi_status"] = msi_status

        special_fields: dict[str, str] = {}
        special_fn_name = str(getattr(config, "special_fields_fn", "")).strip() if config is not None else ""
        special_fn = resources.special_fn_registry.get(special_fn_name)
        if special_fn is not None:
            special_fields = dict(special_fn(patient_record, config))
        patient_record["cancer_specific_fields"] = _enrich_special_fields(
            patient_record,
            config=config,
            initial_fields=special_fields,
        )

        genomics_by_patient_id[patient_id] = {
            "patient_id": patient_id,
            "cancer_code": patient_record["cancer_code"],
            "genomics_text": serialize_patient_genomics(patient_record),
            "genomics_fields": _genomics_fields_for_output(patient_record),
        }

    return genomics_by_patient_id


def write_tcga_genomics_jsonl(
    genomics_by_patient_id: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for patient_id in sorted(genomics_by_patient_id):
            handle.write(json.dumps(genomics_by_patient_id[patient_id], sort_keys=True) + "\n")
    return destination
