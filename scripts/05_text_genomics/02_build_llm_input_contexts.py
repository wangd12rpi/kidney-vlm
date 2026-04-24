#!/usr/bin/env python3
"""
Build final LLM-input context files from registry clinical metadata plus
registered text-channel genomics.

This script is intentionally generation-free: it does not call an LLM and it
does not download data. It formats the data already present in the unified
registry into per-case files that can be concatenated into caption-generation
prompts:

    <output_root>/<source>/<project_id>/<patient_id>/
        clinical.txt
        gdisc.txt
        llm_input.txt
        llm_input.json

Only the currently registered text-channel genomics are used:
    * masked somatic mutation MAF files
    * gene-level copy-number files

Segment-level copy-number files, DNAm beta files, RNA, miRNA, report PDFs, and
TCIA metadata are not serialized into the LLM text prompt. Clinical/report/RNA
annotation fields already summarized into the registry may appear in the
clinical block or JSON provenance.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty, write_registry_parquet
from kidney_vlm.genomics import cohort_config as cohort_cfg
from kidney_vlm.genomics.mutation_cna_text_features import (
    NONSILENT_VARIANT_CLASSES,
    ONCOKB_HOTSPOTS,
)
from kidney_vlm.genomics.registry_integration import (
    update_registry_with_llm_input_context_manifest,
)
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)

DEFAULT_REGISTRY_PATH = ROOT / "data" / "registry" / "unified.parquet"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "features" / "llm_input_contexts"
DEFAULT_RAD_PATH_JSON = ROOT / "rad_path.json"
DEFAULT_SOURCE_NAME = "tcga"
DEFAULT_CALLABLE_MB = 38.0


@dataclass(frozen=True)
class CasePaths:
    maf_path: str | None
    gene_cna_path: str | None
    segment_cna_path: str | None


@dataclass(frozen=True)
class MutationCall:
    gene: str
    protein_change: str
    variant_class: str
    hotspot: str
    dbsnp: str
    is_lof: bool


@dataclass(frozen=True)
class MutationFeatures:
    calls_by_gene: dict[str, list[MutationCall]]
    tmb_mutations_per_mb: float | None
    total_nonsilent_snv_count: int | None
    full_maf_variant_count: int | None
    panel_nonsilent_variant_count: int | None
    error: str


@dataclass(frozen=True)
class CnaFeatures:
    calls_by_gene: dict[str, int]
    arm_level_events: list[str]
    error: str


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"none", "nan", "nat", "<na>"}:
        return ""
    return text


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
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


def _optional_float(value: Any) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _format_value(value: Any, *, missing: str = "not_available") -> str:
    text = _clean_text(value)
    return text if text else missing


def _format_float(value: float | None, decimals: int = 1, *, missing: str = "not_available") -> str:
    if value is None or not math.isfinite(float(value)):
        return missing
    return f"{float(value):.{decimals}f}"


def _resolve_path(path_value: str | Path | None, *, root_dir: Path = ROOT) -> Path | None:
    text = _clean_text(path_value)
    if not text or "://" in text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve() if path.exists() else None


def _to_repo_relative(path_value: str | Path | None, *, root_dir: Path = ROOT) -> str:
    path = _resolve_path(path_value, root_dir=root_dir)
    if path is None:
        return _clean_text(path_value)
    try:
        return path.relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _choose_preferred_path(paths: list[str], *, prefer_patterns: list[str]) -> str | None:
    resolved = [path for path in paths if _resolve_path(path) is not None]
    if not resolved:
        return None
    lower_pairs = [(path, path.lower()) for path in resolved]
    for pattern in prefer_patterns:
        pattern_lower = pattern.lower()
        for path, lowered in lower_pairs:
            if pattern_lower in lowered:
                return path
    return sorted(resolved)[0]


def _case_paths(case_row: dict[str, Any]) -> CasePaths:
    maf_path = _choose_preferred_path(
        _as_list(case_row.get("genomics_mutation_paths")),
        prefer_patterns=["aliquot_ensemble_masked", ".maf.gz", ".maf"],
    )
    gene_cna_path = _choose_preferred_path(
        _as_list(case_row.get("genomics_cnv_gene_paths")),
        prefer_patterns=["ascat3", "gistic", "ascat2", "gene_level_copy_number"],
    )
    segment_cna_path = _choose_preferred_path(
        _as_list(case_row.get("genomics_cnv_segment_paths")),
        prefer_patterns=["ascat3", "ascat2", ".seg."],
    )
    return CasePaths(
        maf_path=maf_path,
        gene_cna_path=gene_cna_path,
        segment_cna_path=segment_cna_path,
    )


def _load_patient_ids_from_json(json_path: Path) -> set[str]:
    path = json_path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    values: list[Any]
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        for key in ("patient_ids", "patients", "cases", "case_ids", "submitter_ids"):
            if isinstance(payload.get(key), list):
                values = payload[key]
                break
        else:
            values = list(payload)
    else:
        raise ValueError(f"Unsupported patient JSON shape in {path}")
    return {str(value).strip() for value in values if str(value).strip()}


def _load_config_mutation_panel() -> list[str]:
    """Load the configured pan-cancer mutation panel from conf/data/sources/tcga.yaml.

    Falls back to the union of cohort panels if OmegaConf is unavailable.
    """
    config_path = ROOT / "conf" / "data" / "sources" / "tcga.yaml"
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        values = cfg.data.source.tcga.gdc.mutation_gene_panel
        panel = []
        for value in list(values or []):
            gene = str(value).strip().upper()
            if gene and gene not in panel:
                panel.append(gene)
        if panel:
            return panel
    except Exception:
        pass

    panel: list[str] = []
    for project_id in cohort_cfg.get_all_cohorts():
        for gene in cohort_cfg.get_mutation_panel(project_id):
            gene = str(gene).strip().upper()
            if gene and gene not in panel:
                panel.append(gene)
    return panel


def _project_specific_genes(project_id: str) -> list[str]:
    genes: list[str] = []
    for source in (
        cohort_cfg.SUBTYPE_DEFINING_LOCI.get(project_id, []),
        cohort_cfg.FOCAL_CNA_PANEL.get(project_id, []),
    ):
        for gene in source:
            gene = str(gene).strip().upper()
            if gene and gene not in genes:
                genes.append(gene)
    return genes


def _ordered_union(*groups: list[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for value in group:
            text = str(value).strip().upper()
            if text and text not in out:
                out.append(text)
    return out


def _read_maf(path: Path) -> pd.DataFrame:
    read_kwargs: dict[str, Any] = {"sep": "\t", "comment": "#", "low_memory": False}
    if path.suffix.lower() == ".gz":
        read_kwargs["compression"] = "gzip"
    return pd.read_csv(path, **read_kwargs)


def _is_lof(variant_class: str) -> bool:
    return variant_class in {
        "Nonsense_Mutation",
        "Nonstop_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "Splice_Site",
        "Translation_Start_Site",
    }


def _maf_hotspot_label(row: pd.Series) -> str:
    gene = _clean_text(row.get("Hugo_Symbol")).upper()
    protein_change = _clean_text(row.get("HGVSp_Short") or row.get("HGVSp") or row.get("Amino_acids"))
    curated = ONCOKB_HOTSPOTS.get((gene, protein_change), "")
    if curated:
        return curated
    raw_hotspot = _clean_text(row.get("hotspot") or row.get("Hotspot") or row.get("ONCOKB_HOTSPOT"))
    if raw_hotspot and raw_hotspot.upper() not in {"N", "NO", "FALSE", "0"}:
        return raw_hotspot
    return ""


def extract_mutation_features(
    maf_path: str | None,
    *,
    mutation_panel: list[str],
    callable_mb: float,
) -> MutationFeatures:
    if not maf_path:
        return MutationFeatures({}, None, None, None, None, "not_available")
    path = _resolve_path(maf_path)
    if path is None:
        return MutationFeatures({}, None, None, None, None, f"missing_file:{maf_path}")

    try:
        df = _read_maf(path)
        required = {"Hugo_Symbol", "Variant_Classification"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"MAF missing required columns: {missing}")

        df["Hugo_Symbol"] = df["Hugo_Symbol"].fillna("").astype(str).str.upper().str.strip()
        df["Variant_Classification"] = df["Variant_Classification"].fillna("").astype(str).str.strip()
        nonsilent = df[df["Variant_Classification"].isin(NONSILENT_VARIANT_CLASSES)].copy()
        snv_type = nonsilent.get("Variant_Type")
        if snv_type is not None:
            snv_mask = snv_type.fillna("").astype(str).str.upper().isin({"SNP", "SNV"})
            tmb_count = int(snv_mask.sum())
        else:
            tmb_count = len(nonsilent)

        panel_set = {gene.upper() for gene in mutation_panel}
        panel_df = nonsilent[nonsilent["Hugo_Symbol"].isin(panel_set)].copy()
        calls_by_gene: dict[str, list[MutationCall]] = {}
        for _, row in panel_df.iterrows():
            gene = _clean_text(row.get("Hugo_Symbol")).upper()
            protein_change = _clean_text(
                row.get("HGVSp_Short") or row.get("HGVSp") or row.get("Amino_acids")
            )
            dbsnp = _clean_text(row.get("dbSNP_RS"))
            variant_class = _clean_text(row.get("Variant_Classification"))
            call = MutationCall(
                gene=gene,
                protein_change=protein_change,
                variant_class=variant_class,
                hotspot=_maf_hotspot_label(row),
                dbsnp=dbsnp,
                is_lof=_is_lof(variant_class),
            )
            calls_by_gene.setdefault(gene, []).append(call)

        for gene in calls_by_gene:
            calls_by_gene[gene] = sorted(
                calls_by_gene[gene],
                key=lambda call: (call.variant_class, call.protein_change, call.dbsnp),
            )

        tmb = float(tmb_count) / float(callable_mb) if callable_mb > 0 else None
        return MutationFeatures(
            calls_by_gene=calls_by_gene,
            tmb_mutations_per_mb=tmb,
            total_nonsilent_snv_count=tmb_count,
            full_maf_variant_count=len(df),
            panel_nonsilent_variant_count=len(panel_df),
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        return MutationFeatures({}, None, None, None, None, f"{type(exc).__name__}: {exc}")


def _integer_cn_to_discrete(value: float | None) -> int:
    if value is None or not math.isfinite(float(value)):
        return 0
    value = float(value)
    if value <= 0:
        return -2
    if value <= 1:
        return -1
    if value < 3:
        return 0
    if value < 4:
        return 1
    return 2


def _logratio_to_discrete(value: float | None) -> int:
    if value is None or not math.isfinite(float(value)):
        return 0
    value = float(value)
    if value <= -1.0:
        return -2
    if value <= -0.3:
        return -1
    if value >= 1.0:
        return 2
    if value >= 0.3:
        return 1
    return 0


def _value_series_to_discrete(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series([0] * len(values), index=values.index, dtype="int64")
    max_abs = float(finite.abs().max())
    min_value = float(finite.min())
    max_value = float(finite.max())

    if min_value >= 0 and max_value > 2.5:
        return numeric.map(_integer_cn_to_discrete).fillna(0).astype(int)
    if min_value >= -2.1 and max_value <= 2.1 and max_abs > 1.5:
        return numeric.round().clip(-2, 2).fillna(0).astype(int)
    return numeric.map(_logratio_to_discrete).fillna(0).astype(int)


def _read_gene_cna(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    gene_col = next(
        (col for col in ("gene_name", "Gene Symbol", "Hugo_Symbol", "symbol", "Gene") if col in df.columns),
        None,
    )
    if gene_col is None:
        raise ValueError("gene-level CNA file has no gene-symbol column")

    value_col = next(
        (
            col
            for col in ("copy_number", "GISTIC", "gistic", "seg_mean", "Segment_Mean", "value")
            if col in df.columns
        ),
        None,
    )
    if value_col is None:
        metadata = {
            gene_col,
            "gene_id",
            "Locus ID",
            "Cytoband",
            "chromosome",
            "start",
            "end",
            "min_copy_number",
            "max_copy_number",
        }
        candidates = [col for col in df.columns if col not in metadata]
        if not candidates:
            raise ValueError("gene-level CNA file has no copy-number value column")
        value_col = candidates[0]

    out = df.copy()
    out["gene"] = out[gene_col].fillna("").astype(str).str.upper().str.strip()
    out["raw_value"] = pd.to_numeric(out[value_col], errors="coerce")
    out["call"] = _value_series_to_discrete(out[value_col])
    if "chromosome" in out.columns:
        out["chromosome"] = out["chromosome"].fillna("").astype(str).str.replace("chr", "", regex=False).str.upper()
    if "start" in out.columns:
        out["start"] = pd.to_numeric(out["start"], errors="coerce")
    if "end" in out.columns:
        out["end"] = pd.to_numeric(out["end"], errors="coerce")
    return out


def _chrom_arm_filter(df: pd.DataFrame, event: str) -> pd.Series:
    match = re.match(r"^([0-9XY]+)([pq])?_(gain|loss)$", event)
    if not match or "chromosome" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    chrom, arm, _direction = match.groups()
    mask = df["chromosome"].astype(str).str.upper().eq(chrom.upper())
    if not arm:
        return mask
    if "start" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    arm_info = {
        str(k).upper(): v
        for k, v in {
            "1": (122_026_459, 125_184_587),
            "2": (92_188_145, 94_090_557),
            "3": (90_772_458, 93_655_574),
            "4": (49_708_101, 51_743_951),
            "5": (46_485_900, 50_059_807),
            "6": (58_553_888, 59_829_934),
            "7": (58_169_653, 61_828_234),
            "8": (44_033_744, 47_193_556),
            "9": (43_236_168, 45_518_558),
            "10": (39_686_682, 41_593_521),
            "11": (51_078_348, 54_425_074),
            "12": (34_769_407, 37_185_252),
            "13": (16_000_000, 17_700_000),
            "14": (16_000_000, 17_200_000),
            "15": (17_083_673, 19_725_254),
            "16": (36_311_158, 38_265_669),
            "17": (22_813_679, 26_616_164),
            "18": (15_460_898, 20_861_206),
            "19": (24_631_782, 27_190_874),
            "20": (26_369_569, 28_494_539),
            "21": (11_288_129, 12_915_808),
            "22": (13_285_178, 15_054_318),
            "X": (58_605_498, 62_412_542),
        }.items()
    }
    bounds = arm_info.get(chrom.upper())
    if not bounds:
        return pd.Series([False] * len(df), index=df.index)
    p_end, q_start = bounds
    if arm == "p":
        return mask & df["start"].le(p_end)
    return mask & df["start"].ge(q_start)


def _derive_arm_events_from_gene_cna(
    cna_df: pd.DataFrame,
    *,
    project_id: str,
    threshold: float,
) -> list[str]:
    events: list[str] = []
    for event in cohort_cfg.ARM_LEVEL_CNA_PANEL.get(project_id, []):
        mask = _chrom_arm_filter(cna_df, event)
        sub = cna_df.loc[mask & cna_df["gene"].astype(bool)]
        if sub.empty:
            continue
        direction = event.rsplit("_", 1)[-1]
        if direction == "loss":
            fraction = float((sub["call"] < 0).mean())
        else:
            fraction = float((sub["call"] > 0).mean())
        if fraction >= threshold:
            events.append(event)
    return sorted(set(events))


def extract_cna_features(
    gene_cna_path: str | None,
    *,
    project_id: str,
    cna_panel: list[str],
    arm_event_threshold: float,
) -> CnaFeatures:
    if not gene_cna_path:
        return CnaFeatures({}, [], "not_available")
    path = _resolve_path(gene_cna_path)
    if path is None:
        return CnaFeatures({}, [], f"missing_file:{gene_cna_path}")
    try:
        df = _read_gene_cna(path)
        panel_set = {gene.upper() for gene in cna_panel}
        calls: dict[str, int] = {}
        sub = df[df["gene"].isin(panel_set)].copy()
        sub = sub.drop_duplicates(subset=["gene"], keep="first")
        for row in sub.itertuples(index=False):
            calls[str(row.gene).upper()] = int(row.call)
        for gene in cna_panel:
            calls.setdefault(gene.upper(), 0)

        arm_events = _derive_arm_events_from_gene_cna(
            df,
            project_id=project_id,
            threshold=arm_event_threshold,
        )
        return CnaFeatures(calls, arm_events, "")
    except Exception as exc:  # noqa: BLE001
        return CnaFeatures({}, [], f"{type(exc).__name__}: {exc}")


def _cna_call_label(value: int) -> str:
    if value <= -2:
        return "deep deletion (-2)"
    if value == -1:
        return "shallow deletion (-1)"
    if value == 0:
        return "neutral"
    if value == 1:
        return "gain (+1)"
    if value >= 2:
        return "amplification (+2)"
    return f"unknown ({value})"


def _mutation_call_text(call: MutationCall) -> str:
    protein = call.protein_change or "protein_change_unknown"
    variant = call.variant_class or "variant_class_unknown"
    hotspot = call.hotspot or "not_annotated"
    dbsnp = call.dbsnp or "not_reported"
    flags = []
    if call.is_lof:
        flags.append("LoF")
    flag_text = f" | {';'.join(flags)}" if flags else ""
    return f"{protein} | {variant} | Hotspot:{hotspot} | dbSNP:{dbsnp}{flag_text}"


def _render_gdisc_text(
    *,
    project_id: str,
    mutation_panel: list[str],
    cna_panel: list[str],
    mutation_features: MutationFeatures,
    cna_features: CnaFeatures,
    msi_status: str,
    hrd_score: float | None,
    emit_all_panel_wild_types: bool,
) -> str:
    lines: list[str] = []
    lines.append("DISCRETE GENOMICS:")
    lines.append(f"Project: {project_id}")
    lines.append("")

    lines.append(f"MUTATIONS ({len(mutation_panel)}-gene panel):")
    if mutation_features.error and mutation_features.error != "not_available":
        lines.append(f"  status: error ({mutation_features.error})")
    elif mutation_features.error == "not_available":
        lines.append("  status: not_available")
    else:
        genes_to_emit = list(mutation_panel) if emit_all_panel_wild_types else []
        if not emit_all_panel_wild_types:
            genes_to_emit = [
                gene
                for gene in mutation_panel
                if gene in mutation_features.calls_by_gene or gene in _project_specific_genes(project_id)
            ]
        if not genes_to_emit:
            genes_to_emit = list(mutation_panel)
        for gene in genes_to_emit:
            calls = mutation_features.calls_by_gene.get(gene, [])
            if calls:
                call_text = "; ".join(_mutation_call_text(call) for call in calls)
                lines.append(f"  {gene}: {call_text}")
            else:
                lines.append(f"  {gene}: wild-type")
    lines.append("")

    lines.append("COPY NUMBER ALTERATIONS:")
    if cna_features.error and cna_features.error != "not_available":
        lines.append(f"  status: error ({cna_features.error})")
    elif cna_features.error == "not_available":
        lines.append("  status: not_available")
    else:
        emitted = False
        neutral_genes = set(_project_specific_genes(project_id))
        for gene in cna_panel:
            call = int(cna_features.calls_by_gene.get(gene, 0))
            if call != 0:
                lines.append(f"  {gene}: {_cna_call_label(call)}")
                emitted = True
        for gene in cna_panel:
            call = int(cna_features.calls_by_gene.get(gene, 0))
            if call == 0 and gene in neutral_genes:
                lines.append(f"  {gene}: neutral")
                emitted = True
        for event in cna_features.arm_level_events:
            event_text = event.replace("_", " ")
            lines.append(f"  Arm {event_text}")
            emitted = True
        if not emitted:
            lines.append("  none_detected_in_reported_panel")
    lines.append("")

    lines.append(f"TMB: {_format_float(mutation_features.tmb_mutations_per_mb, 2)} mut/Mb")
    lines.append(f"MSI: {_format_value(msi_status)}")
    lines.append(f"HRD: {_format_float(hrd_score, 1)}")
    return "\n".join(lines).rstrip() + "\n"


def _age_years_from_registry(value: Any) -> float | None:
    number = _optional_float(value)
    if number is None:
        return None
    if number > 150:
        return number / 365.25
    return number


def _clinical_metadata(case_row: dict[str, Any]) -> dict[str, Any]:
    age_years = _age_years_from_registry(case_row.get("age_at_diagnosis"))
    survival_days = _optional_float(case_row.get("task_survival_days"))
    event_text = _clean_text(case_row.get("task_survival_event"))
    survival_event: bool | None
    if event_text.lower() == "true":
        survival_event = True
    elif event_text.lower() == "false":
        survival_event = False
    else:
        survival_event = None

    metadata: dict[str, Any] = {
        "source": _clean_text(case_row.get("source")),
        "sample_id": _clean_text(case_row.get("sample_id")),
        "patient_id": _clean_text(case_row.get("patient_id")),
        "study_id": _clean_text(case_row.get("study_id")),
        "project_id": _clean_text(case_row.get("project_id")),
        "split": _clean_text(case_row.get("split")),
        "diagnosis": {
            "primary_site": _clean_text(case_row.get("primary_site")),
            "disease_type": _clean_text(case_row.get("disease_type")),
            "primary_diagnosis": _clean_text(case_row.get("primary_diagnosis")),
            "morphology": _clean_text(case_row.get("morphology")),
            "kidney_histology_subtype": _clean_text(case_row.get("kidney_histology_subtype")),
        },
        "stage_grade": {
            "tumor_grade": _clean_text(case_row.get("tumor_grade") or case_row.get("task_grade_label")),
            "tumor_stage": _clean_text(case_row.get("tumor_stage")),
            "ajcc_pathologic_stage": _clean_text(
                case_row.get("ajcc_pathologic_stage") or case_row.get("task_stage_label")
            ),
            "ajcc_pathologic_t": _clean_text(case_row.get("ajcc_pathologic_t")),
            "ajcc_pathologic_n": _clean_text(case_row.get("ajcc_pathologic_n")),
            "ajcc_pathologic_m": _clean_text(case_row.get("ajcc_pathologic_m")),
        },
        "demographics": {
            "age_at_diagnosis_years": round(age_years, 1) if age_years is not None else None,
            "gender": _clean_text(case_row.get("gender")),
            "race": _clean_text(case_row.get("race")),
            "ethnicity": _clean_text(case_row.get("ethnicity")),
            "year_of_birth": _clean_text(case_row.get("year_of_birth")),
        },
        "molecular_metadata_from_registry": {
            "rna_molecular_subtype": _clean_text(case_row.get("genomics_rna_bulk_molecular_subtype")),
            "rna_subtype_mrna": _clean_text(case_row.get("genomics_rna_bulk_subtype_mrna")),
            "integrative_subtype": _clean_text(case_row.get("genomics_integrative_subtype")),
            "dna_methylation_subtype": _clean_text(case_row.get("genomics_dna_methylation_subtype")),
            "tumor_purity": _clean_text(case_row.get("genomics_rna_bulk_tumor_purity")),
            "leukocyte_fraction": _clean_text(case_row.get("genomics_rna_bulk_leukocyte_fraction")),
            "aneuploidy_score": _clean_text(case_row.get("genomics_aneuploidy_score")),
            "hrd_score": _clean_text(case_row.get("genomics_hrd_score")),
            "msi_status": _clean_text(case_row.get("genomics_msi_status")),
        },
        "report_provenance": {
            "report_pdf_paths": _as_list(case_row.get("report_pdf_paths")),
            "report_file_ids": _as_list(case_row.get("report_file_ids")),
            "report_file_names": _as_list(case_row.get("report_file_names")),
        },
        "survival_labels": {
            "vital_status": _clean_text(case_row.get("vital_status")),
            "days_to_death": _clean_text(case_row.get("days_to_death")),
            "days_to_last_follow_up": _clean_text(case_row.get("days_to_last_follow_up")),
            "overall_survival_days": survival_days,
            "survival_event": survival_event,
        },
    }
    return metadata


def _render_clinical_text(clinical: dict[str, Any], *, include_survival: bool) -> str:
    diagnosis = clinical["diagnosis"]
    stage_grade = clinical["stage_grade"]
    demographics = clinical["demographics"]
    molecular = clinical["molecular_metadata_from_registry"]

    lines: list[str] = []
    lines.append("CLINICAL METADATA:")
    lines.append(f"  Project: {_format_value(clinical.get('project_id'))}")
    lines.append(f"  Primary site: {_format_value(diagnosis.get('primary_site'))}")
    lines.append(f"  Disease type: {_format_value(diagnosis.get('disease_type'))}")
    lines.append(f"  Primary diagnosis: {_format_value(diagnosis.get('primary_diagnosis'))}")
    lines.append(f"  Morphology: {_format_value(diagnosis.get('morphology'))}")
    lines.append(f"  Kidney histology subtype: {_format_value(diagnosis.get('kidney_histology_subtype'))}")
    lines.append(f"  Tumor grade: {_format_value(stage_grade.get('tumor_grade'))}")
    lines.append(f"  Tumor stage: {_format_value(stage_grade.get('tumor_stage'))}")
    lines.append(f"  AJCC pathologic stage: {_format_value(stage_grade.get('ajcc_pathologic_stage'))}")
    lines.append(f"  AJCC T/N/M: {_format_value(stage_grade.get('ajcc_pathologic_t'))} / {_format_value(stage_grade.get('ajcc_pathologic_n'))} / {_format_value(stage_grade.get('ajcc_pathologic_m'))}")
    lines.append(f"  Age at diagnosis: {_format_float(demographics.get('age_at_diagnosis_years'), 1)} years")
    lines.append(f"  Gender: {_format_value(demographics.get('gender'))}")
    lines.append(f"  Race: {_format_value(demographics.get('race'))}")
    lines.append(f"  Ethnicity: {_format_value(demographics.get('ethnicity'))}")
    lines.append("")
    lines.append("REGISTRY MOLECULAR METADATA:")
    lines.append(f"  RNA molecular subtype: {_format_value(molecular.get('rna_molecular_subtype'))}")
    lines.append(f"  RNA mRNA subtype: {_format_value(molecular.get('rna_subtype_mrna'))}")
    lines.append(f"  Integrative subtype: {_format_value(molecular.get('integrative_subtype'))}")
    lines.append(f"  DNA methylation subtype: {_format_value(molecular.get('dna_methylation_subtype'))}")
    lines.append(f"  Tumor purity: {_format_value(molecular.get('tumor_purity'))}")
    lines.append(f"  Leukocyte fraction: {_format_value(molecular.get('leukocyte_fraction'))}")
    lines.append(f"  Aneuploidy score: {_format_value(molecular.get('aneuploidy_score'))}")
    lines.append(f"  HRD score: {_format_value(molecular.get('hrd_score'))}")
    lines.append(f"  MSI status: {_format_value(molecular.get('msi_status'))}")
    if include_survival and "survival_labels" in clinical:
        survival = clinical["survival_labels"]
        lines.append("")
        lines.append("SURVIVAL LABELS:")
        lines.append(f"  Vital status: {_format_value(survival.get('vital_status'))}")
        lines.append(f"  Overall survival days: {_format_float(survival.get('overall_survival_days'), 0)}")
        event = survival.get("survival_event")
        event_text = "not_available" if event is None else str(bool(event))
        lines.append(f"  Survival event: {event_text}")
    return "\n".join(lines).rstrip() + "\n"


def _render_llm_input_text(
    *,
    clinical_text: str,
    gdisc_text: str,
    include_generation_instructions: bool,
) -> str:
    sections: list[str] = []
    if include_generation_instructions:
        sections.append(
            "You are given structured case metadata and discrete genomics for a TCGA case. "
            "Use these as conditioning context for caption dataset generation. "
            "Do not mention missing fields, file paths, internal IDs, or data processing details in the final caption."
        )
        sections.append("")
    sections.append(clinical_text.rstrip())
    sections.append("")
    sections.append(gdisc_text.rstrip())
    return "\n".join(sections).rstrip() + "\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def process_case(
    *,
    case_row: dict[str, Any],
    output_root: Path,
    source_name: str,
    mutation_panel: list[str],
    callable_mb: float,
    include_survival_in_text: bool,
    include_generation_instructions: bool,
    emit_all_panel_wild_types: bool,
    arm_event_threshold: float,
) -> dict[str, Any]:
    project_id = _clean_text(case_row.get("project_id"))
    patient_id = _clean_text(case_row.get("patient_id"))
    sample_id = _clean_text(case_row.get("sample_id"))
    case_dir = output_root / source_name / project_id / patient_id
    case_dir.mkdir(parents=True, exist_ok=True)

    paths = _case_paths(case_row)
    project_genes = _project_specific_genes(project_id)
    case_mutation_panel = _ordered_union(mutation_panel, project_genes)
    cna_panel = _ordered_union(mutation_panel, cohort_cfg.FOCAL_CNA_PANEL.get(project_id, []), project_genes)

    hrd_score = _optional_float(case_row.get("genomics_hrd_score") or case_row.get("tcga_hrd_score"))
    msi_status = _clean_text(case_row.get("genomics_msi_status") or case_row.get("tcga_msi_status"))

    mutation_features = extract_mutation_features(
        paths.maf_path,
        mutation_panel=case_mutation_panel,
        callable_mb=callable_mb,
    )
    cna_features = extract_cna_features(
        paths.gene_cna_path,
        project_id=project_id,
        cna_panel=cna_panel,
        arm_event_threshold=arm_event_threshold,
    )
    clinical = _clinical_metadata(case_row)
    clinical_text = _render_clinical_text(clinical, include_survival=include_survival_in_text)
    gdisc_text = _render_gdisc_text(
        project_id=project_id,
        mutation_panel=case_mutation_panel,
        cna_panel=cna_panel,
        mutation_features=mutation_features,
        cna_features=cna_features,
        msi_status=msi_status,
        hrd_score=hrd_score,
        emit_all_panel_wild_types=emit_all_panel_wild_types,
    )
    llm_input_text = _render_llm_input_text(
        clinical_text=clinical_text,
        gdisc_text=gdisc_text,
        include_generation_instructions=include_generation_instructions,
    )

    clinical_path = case_dir / "clinical.txt"
    gdisc_path = case_dir / "gdisc.txt"
    llm_input_path = case_dir / "llm_input.txt"
    json_path = case_dir / "llm_input.json"
    clinical_path.write_text(clinical_text, encoding="utf-8")
    gdisc_path.write_text(gdisc_text, encoding="utf-8")
    llm_input_path.write_text(llm_input_text, encoding="utf-8")

    modalities = []
    if paths.maf_path:
        modalities.append("mutation_maf")
    if paths.gene_cna_path:
        modalities.append("copy_number_gene")
    if paths.segment_cna_path:
        modalities.append("copy_number_segment_available_not_serialized")

    payload = {
        "sample_id": sample_id,
        "source": source_name,
        "project_id": project_id,
        "patient_id": patient_id,
        "clinical": clinical,
        "gdisc": {
            "mutation_panel": case_mutation_panel,
            "copy_number_panel": cna_panel,
            "mutation_calls": {
                gene: [call.__dict__ for call in calls]
                for gene, calls in sorted(mutation_features.calls_by_gene.items())
            },
            "copy_number_calls": cna_features.calls_by_gene,
            "arm_level_events": cna_features.arm_level_events,
            "tmb_mutations_per_mb": mutation_features.tmb_mutations_per_mb,
            "tmb_nonsilent_snv_count": mutation_features.total_nonsilent_snv_count,
            "full_maf_variant_count": mutation_features.full_maf_variant_count,
            "panel_nonsilent_variant_count": mutation_features.panel_nonsilent_variant_count,
            "msi_status": msi_status,
            "hrd_score": hrd_score,
        },
        "input_paths": {
            "mutation_maf": _to_repo_relative(paths.maf_path),
            "copy_number_gene": _to_repo_relative(paths.gene_cna_path),
            "copy_number_segment": _to_repo_relative(paths.segment_cna_path),
        },
        "output_paths": {
            "clinical_text_path": _to_repo_relative(clinical_path),
            "gdisc_text_path": _to_repo_relative(gdisc_path),
            "llm_input_text_path": _to_repo_relative(llm_input_path),
            "llm_input_json_path": _to_repo_relative(json_path),
        },
        "available_modalities": modalities,
        "errors": {
            "mutation": mutation_features.error,
            "copy_number": cna_features.error,
        },
        "notes": [
            "DNAm, RNA, miRNA, report PDFs, and TCIA metadata are not serialized into this text prompt.",
            "Segment-level CNA is recorded as available when registered, but arm events are derived from gene-level CNA by default.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    errors = [value for value in [mutation_features.error, cna_features.error] if value and value != "not_available"]
    return {
        "sample_id": sample_id,
        "source": source_name,
        "project_id": project_id,
        "patient_id": patient_id,
        "split": _clean_text(case_row.get("split")),
        "clinical_text_path": _to_repo_relative(clinical_path),
        "gdisc_text_path": _to_repo_relative(gdisc_path),
        "llm_input_text_path": _to_repo_relative(llm_input_path),
        "llm_input_json_path": _to_repo_relative(json_path),
        "mutation_maf_path": _to_repo_relative(paths.maf_path),
        "copy_number_gene_path": _to_repo_relative(paths.gene_cna_path),
        "copy_number_segment_path": _to_repo_relative(paths.segment_cna_path),
        "mutation_available": bool(paths.maf_path),
        "copy_number_gene_available": bool(paths.gene_cna_path),
        "copy_number_segment_available": bool(paths.segment_cna_path),
        "tmb_mutations_per_mb": mutation_features.tmb_mutations_per_mb,
        "panel_nonsilent_variant_count": mutation_features.panel_nonsilent_variant_count,
        "copy_number_non_neutral_count": sum(1 for value in cna_features.calls_by_gene.values() if int(value) != 0),
        "arm_level_event_count": len(cna_features.arm_level_events),
        "errors": "; ".join(errors),
    }


def _filter_registry(
    registry_df: pd.DataFrame,
    *,
    source_name: str,
    case_subset: str,
    case_json_path: Path,
    require_text_genomics: bool,
    max_cases: int | None,
) -> pd.DataFrame:
    if "source" in registry_df.columns:
        out = registry_df[registry_df["source"].fillna("").astype(str).eq(source_name)].copy()
    else:
        out = registry_df.copy()

    if case_subset in {"rad_path", "json"}:
        patient_ids = _load_patient_ids_from_json(case_json_path)
        out = out[out["patient_id"].fillna("").astype(str).isin(patient_ids)].copy()
        print(f"[llm-input] Case subset from {case_json_path}: {len(patient_ids)} requested IDs.")
    elif case_subset != "all":
        raise ValueError("--case-subset must be one of: all, rad_path, json")

    if require_text_genomics:
        def has_text_genomics(row: pd.Series) -> bool:
            return bool(_as_list(row.get("genomics_mutation_paths")) or _as_list(row.get("genomics_cnv_gene_paths")))

        out = out[out.apply(has_text_genomics, axis=1)].copy()

    out = out.sort_values(["project_id", "patient_id"], kind="stable").reset_index(drop=True)
    if max_cases is not None:
        out = out.head(max_cases).copy()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-case LLM input context files from TCGA clinical metadata "
            "and registered MAF/gene-CNA text genomics."
        )
    )
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument(
        "--case-subset",
        choices=["all", "rad_path", "json"],
        default="all",
        help="Use all selected registry rows, rad_path.json, or a custom JSON list.",
    )
    parser.add_argument(
        "--case-json",
        type=Path,
        default=DEFAULT_RAD_PATH_JSON,
        help="JSON list/dict of patient IDs used when --case-subset is rad_path or json.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--require-text-genomics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, skip cases with neither MAF nor gene-CNA paths.",
    )
    parser.add_argument(
        "--include-survival-in-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include survival labels in clinical.txt and llm_input.txt. JSON always records selected metadata.",
    )
    parser.add_argument(
        "--include-generation-instructions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a short caption-generation context instruction at the top of llm_input.txt.",
    )
    parser.add_argument(
        "--emit-all-panel-wild-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit wild-type lines for every mutation-panel gene.",
    )
    parser.add_argument("--callable-mb", type=float, default=DEFAULT_CALLABLE_MB)
    parser.add_argument(
        "--arm-event-threshold",
        type=float,
        default=0.60,
        help="Fraction of genes on a chromosome arm that must show gain/loss to call an arm event.",
    )
    parser.add_argument(
        "--manifest-name",
        default="tcga_llm_input_contexts_manifest.parquet",
        help="Manifest filename written under output root.",
    )
    parser.add_argument(
        "--update-registry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write generated LLM-context paths back into the unified registry.",
    )
    parser.add_argument(
        "--overwrite-existing-registry-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing generated LLM-context registry path fields.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected case counts and exit without writing case files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = args.registry_path.expanduser()
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    registry_df = read_parquet_or_empty(registry_path)
    if registry_df.empty:
        raise RuntimeError(f"Registry is empty or missing: {registry_path}")

    source_df = _filter_registry(
        registry_df,
        source_name=str(args.source_name),
        case_subset=str(args.case_subset),
        case_json_path=args.case_json,
        require_text_genomics=bool(args.require_text_genomics),
        max_cases=args.max_cases,
    )
    print(f"[llm-input] Selected registry rows: {len(source_df)}")
    if source_df.empty:
        print("[llm-input] Nothing to process.")
        return

    mutation_panel = _load_config_mutation_panel()
    print(f"[llm-input] Mutation panel genes: {len(mutation_panel)}")
    if args.dry_run:
        print("[llm-input] Dry run requested; no files written.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    failures = 0

    progress = tqdm(
        source_df.to_dict(orient="records"),
        total=len(source_df),
        desc="Building LLM input contexts",
        unit="case",
    )
    for row in progress:
        project_id = _clean_text(row.get("project_id"))
        patient_id = _clean_text(row.get("patient_id"))
        if not project_id or not patient_id:
            continue
        try:
            manifest_rows.append(
                process_case(
                    case_row=row,
                    output_root=output_root,
                    source_name=str(args.source_name),
                    mutation_panel=mutation_panel,
                    callable_mb=float(args.callable_mb),
                    include_survival_in_text=bool(args.include_survival_in_text),
                    include_generation_instructions=bool(args.include_generation_instructions),
                    emit_all_panel_wild_types=bool(args.emit_all_panel_wild_types),
                    arm_event_threshold=float(args.arm_event_threshold),
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            traceback.print_exc()
            manifest_rows.append(
                {
                    "sample_id": _clean_text(row.get("sample_id")),
                    "source": str(args.source_name),
                    "project_id": project_id,
                    "patient_id": patient_id,
                    "split": _clean_text(row.get("split")),
                    "clinical_text_path": "",
                    "gdisc_text_path": "",
                    "llm_input_text_path": "",
                    "llm_input_json_path": "",
                    "mutation_maf_path": "",
                    "copy_number_gene_path": "",
                    "copy_number_segment_path": "",
                    "mutation_available": False,
                    "copy_number_gene_available": False,
                    "copy_number_segment_available": False,
                    "tmb_mutations_per_mb": None,
                    "panel_nonsilent_variant_count": None,
                    "copy_number_non_neutral_count": None,
                    "arm_level_event_count": None,
                    "errors": f"case_failure:{type(exc).__name__}: {exc}",
                }
            )

    manifest_path = output_root / str(args.manifest_name)
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_parquet(manifest_path, index=False)

    print(f"[llm-input] Manifest written: {manifest_path}")
    print(f"[llm-input] Rows written: {len(manifest_df)}")
    print(f"[llm-input] Case failures: {failures}")
    if not manifest_df.empty:
        print(f"[llm-input] Mutation-available rows: {int(manifest_df['mutation_available'].sum())}")
        print(f"[llm-input] Gene-CNA-available rows: {int(manifest_df['copy_number_gene_available'].sum())}")
    if bool(args.update_registry) and not manifest_df.empty:
        updated_df, stats = update_registry_with_llm_input_context_manifest(
            registry_df,
            manifest_df,
            repo_root=ROOT,
            source_name=str(args.source_name),
            overwrite_existing=bool(args.overwrite_existing_registry_paths),
        )
        write_registry_parquet(updated_df, registry_path, validate=True)
        print("[llm-input] Registry updated.")
        print(f"[llm-input] Matched registry rows: {stats.matched_registry_rows}")
        print(f"[llm-input] Updated registry rows: {stats.updated_registry_rows}")
        print(f"[llm-input] Unmatched context cases: {stats.unmatched_manifest_cases}")


if __name__ == "__main__":
    main()
