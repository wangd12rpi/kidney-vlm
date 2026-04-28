from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from kidney_vlm.genomics import dnam_text_features as dnam_tf
from kidney_vlm.genomics import rna_text_features as rna_tf
from kidney_vlm.genomics import signatures as sig
from kidney_vlm.repo_root import find_repo_root


MISSING_TEXT_VALUES = {"", "nan", "none", "<na>", "na", "n/a", "null"}
REPO_ROOT = find_repo_root(Path(__file__))


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() in MISSING_TEXT_VALUES


def _as_list(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if not _is_missing(item)]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if not _is_missing(item)]
    text = str(value).strip()
    return [text] if text else []


def _resolve_existing_path(path_value: str) -> str:
    path = Path(str(path_value).strip()).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append((Path.cwd() / path).resolve())
        candidates.append((REPO_ROOT / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate.as_posix()
    return ""


def _first_existing_path(value: Any, *, field_name: str) -> str:
    path_values = _as_list(value)
    if not path_values:
        raise FileNotFoundError(f"No raw paths found in registry field '{field_name}'.")
    for path_value in _as_list(value):
        path = _resolve_existing_path(path_value)
        if path:
            return path
    raise FileNotFoundError(
        f"No existing raw paths found in registry field '{field_name}': {path_values[:5]}"
    )


def _format_float(value: float) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not_assessed"
    if not math.isfinite(numeric):
        return "not_assessed"
    return f"{numeric:.3g}"


def _format_percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def build_dnam_text_summary(
    row: Mapping[str, Any],
    *,
    max_beta_values: int = 50_000,
    panel_genes: Sequence[str] | None = None,
) -> str:
    raw_path = _first_existing_path(
        row.get("genomics_dna_methylation_paths"),
        field_name="genomics_dna_methylation_paths",
    )
    return _summarize_dnam_beta_file(
        raw_path,
        project_id=str(row.get("project_id", "")).strip(),
        panel_genes=tuple(
            str(gene).strip().upper()
            for gene in panel_genes or []
            if str(gene).strip()
        ),
        max_beta_values=max_beta_values,
    )


def build_rna_text_summary(
    row: Mapping[str, Any],
    *,
    max_top_genes: int = 8,
    panel_genes: Sequence[str] | None = None,
) -> str:
    raw_path = _first_existing_path(
        row.get("genomics_rna_bulk_paths"),
        field_name="genomics_rna_bulk_paths",
    )
    return _summarize_rna_star_counts_file(
        raw_path,
        project_id=str(row.get("project_id", "")).strip(),
        panel_genes=tuple(
            str(gene).strip().upper()
            for gene in panel_genes or []
            if str(gene).strip()
        ),
        max_top_genes=max_top_genes,
    )


@lru_cache(maxsize=16384)
def _dnam_probe_features(
    raw_path: str,
    *,
    project_id: str,
    panel_genes: tuple[str, ...],
) -> dict[str, Any]:
    betas = dnam_tf.read_tcga_beta_tsv(raw_path)
    promoter = dnam_tf.promoter_methylation_by_gene(betas, project_id)
    if panel_genes:
        promoter = {gene: promoter.get(gene, float("nan")) for gene in panel_genes}
    return {
        "probe_count": int(betas.size),
        "promoter_methylation": promoter,
        "cimp_status": dnam_tf.classify_cimp(betas, project_id),
        "dnam_tumor_purity": dnam_tf.lump_tumor_purity(betas),
    }


@lru_cache(maxsize=16384)
def _summarize_dnam_beta_file(
    raw_path: str,
    *,
    project_id: str,
    panel_genes: tuple[str, ...],
    max_beta_values: int = 50_000,
) -> str:
    beta_values: list[float] = []
    missing_values = 0
    observed_rows = 0
    with Path(raw_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if observed_rows >= max_beta_values:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            observed_rows += 1
            beta_text = fields[1].strip()
            if beta_text.casefold() in MISSING_TEXT_VALUES:
                missing_values += 1
                continue
            try:
                beta = float(beta_text)
            except ValueError:
                missing_values += 1
                continue
            if 0.0 <= beta <= 1.0:
                beta_values.append(beta)
            else:
                missing_values += 1

    if not beta_values:
        raise ValueError(f"No valid DNAm beta values found in raw file: {raw_path}")

    series = pd.Series(beta_values, dtype="float64")
    total = len(beta_values) + missing_values
    missing_fraction = missing_values / total if total else 0.0
    hypo_fraction = float((series < 0.2).mean())
    hyper_fraction = float((series > 0.8).mean())
    intermediate_fraction = 1.0 - hypo_fraction - hyper_fraction
    project_text = f" for project {project_id}" if project_id else ""
    parts = [
        (
            f"DNA methylation raw beta summary{project_text}: "
            f"evaluated {total:,} CpG beta rows from the raw methylation file; "
            f"valid beta values {len(beta_values):,}; "
            f"missing or invalid beta values {_format_percent(missing_fraction)}; "
            f"mean beta {_format_float(float(series.mean()))}; "
            f"median beta {_format_float(float(series.median()))}; "
            f"hypomethylated probes (beta < 0.2) {_format_percent(hypo_fraction)}; "
            f"intermediate probes (0.2 <= beta <= 0.8) "
            f"{_format_percent(intermediate_fraction)}; "
            f"hypermethylated probes (beta > 0.8) {_format_percent(hyper_fraction)}."
        )
    ]
    if project_id:
        features = _dnam_probe_features(
            raw_path,
            project_id=project_id,
            panel_genes=panel_genes,
        )
        parts.append(f"Full DNAm probe count {features['probe_count']:,}.")
        parts.append(f"CIMP status from methylation markers: {features['cimp_status']}.")
        parts.append(
            "DNAm LUMP tumor purity estimate: "
            f"{_format_float(features['dnam_tumor_purity'])}."
        )
        promoter = dict(features["promoter_methylation"] or {})
        if promoter:
            promoter_items = [
                f"{gene} promoter beta {_format_float(value)}"
                for gene, value in promoter.items()
            ]
            parts.append(
                "Promoter methylation for benchmark mutation panel genes: "
                + "; ".join(promoter_items)
                + "."
            )
    return " ".join(parts)


@lru_cache(maxsize=16384)
def _summarize_rna_star_counts_file(
    raw_path: str,
    *,
    project_id: str,
    panel_genes: tuple[str, ...],
    max_top_genes: int = 8,
) -> str:
    frame = pd.read_csv(raw_path, sep="\t", comment="#", low_memory=False)
    required_columns = {
        "gene_id",
        "gene_name",
        "gene_type",
        "unstranded",
        "tpm_unstranded",
    }
    if not required_columns.issubset(frame.columns):
        missing = sorted(required_columns - set(frame.columns))
        raise ValueError(
            f"Raw RNA file is missing required columns {missing}: {raw_path}"
        )

    counts = pd.to_numeric(frame["unstranded"], errors="coerce")
    special_rows = frame["gene_id"].astype(str).str.startswith("N_")
    qc_values = {}
    for label in ["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]:
        match = frame["gene_id"].astype(str).eq(label)
        if match.any():
            value = counts[match].iloc[0]
            if pd.notna(value):
                qc_values[label.removeprefix("N_")] = int(value)

    genes = frame.loc[~special_rows].copy()
    genes["tpm_unstranded"] = pd.to_numeric(genes["tpm_unstranded"], errors="coerce")
    genes["unstranded"] = pd.to_numeric(genes["unstranded"], errors="coerce")
    protein = genes.loc[genes["gene_type"].astype(str).eq("protein_coding")].copy()
    protein = protein.loc[protein["tpm_unstranded"].notna()]
    if protein.empty:
        raise ValueError(
            f"No protein-coding TPM rows found in raw RNA file: {raw_path}"
        )

    expressed = int((protein["tpm_unstranded"] >= 1.0).sum())
    high = int((protein["tpm_unstranded"] >= 10.0).sum())
    median_tpm = float(protein["tpm_unstranded"].median())
    total_counts = int(protein["unstranded"].fillna(0).clip(lower=0).sum())
    gene_names = protein["gene_name"].fillna("").astype(str).str.strip()
    non_mitochondrial = protein.loc[~gene_names.str.upper().str.startswith("MT-")]
    top_source = non_mitochondrial if not non_mitochondrial.empty else protein
    top = top_source.sort_values("tpm_unstranded", ascending=False).head(max_top_genes)
    top_genes = [
        f"{str(row.gene_name).strip()} (TPM {_format_float(float(row.tpm_unstranded))})"
        for row in top.itertuples(index=False)
        if str(row.gene_name).strip()
    ]
    symbol_tpm = protein.groupby("gene_name", as_index=True)["tpm_unstranded"].max()
    panel_expression = [
        f"{gene} TPM {_format_float(symbol_tpm.get(gene, float('nan')))}"
        for gene in panel_genes
    ]
    expression = symbol_tpm.apply(lambda value: math.log1p(max(float(value), 0.0)))
    estimate = rna_tf.estimate_scores(expression)
    proliferation = rna_tf.mean_expression_score(expression, sig.PROLIFERATION_CORE)
    hypoxia = rna_tf.mean_expression_score(expression, sig.HYPOXIA_BUFFA_51)
    emt = (
        rna_tf.mean_expression_score(expression, sig.EMT_CORE_MESENCHYMAL)
        - rna_tf.mean_expression_score(expression, sig.EMT_CORE_EPITHELIAL)
    )
    ifng = rna_tf.mean_expression_score(expression, sig.IFNG_AYERS_6)
    tis = rna_tf.mean_expression_score(expression, sig.TIS_AYERS_18)
    cytolytic = rna_tf.cytolytic_activity_log(expression)
    qc_text = "; ".join(f"{key} {value:,}" for key, value in qc_values.items())
    project_text = f" for project {project_id}" if project_id else ""
    parts = [
        f"RNA-seq raw expression summary{project_text}: "
        f"protein-coding genes measured {len(protein):,}; "
        f"protein-coding genes with TPM >= 1: {expressed:,}; "
        f"protein-coding genes with TPM >= 10: {high:,}; "
        f"median protein-coding TPM {_format_float(median_tpm)}; "
        f"protein-coding unstranded read count sum {total_counts:,}."
    ]
    if top_genes:
        parts.append(
            f"Top expressed non-mitochondrial protein-coding genes by TPM: {', '.join(top_genes)}."
        )
    if panel_expression:
        parts.append(
            "RNA expression for benchmark mutation panel genes: "
            + "; ".join(panel_expression)
            + "."
        )
    parts.append(
        "RNA pathway and microenvironment signatures from log1p(TPM): "
        f"proliferation {_format_float(proliferation)}; "
        f"hypoxia {_format_float(hypoxia)}; "
        f"EMT composite {_format_float(emt)}; "
        f"IFN-gamma {_format_float(ifng)}; "
        f"TIS {_format_float(tis)}; "
        f"cytolytic activity {_format_float(cytolytic)}; "
        f"ESTIMATE stromal {_format_float(estimate.get('stromal'))}; "
        f"ESTIMATE immune {_format_float(estimate.get('immune'))}; "
        f"ESTIMATE tumor purity {_format_float(estimate.get('tumor_purity'))}."
    )
    if qc_text:
        parts.append(f"STAR assignment summary rows: {qc_text}.")
    return " ".join(parts)
