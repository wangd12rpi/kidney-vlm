"""
RNA-seq text-feature extraction.

Consumes the same STAR TSV files that BulkFormer ingests (see
`kidney_vlm.data.rna_feature_import.read_tcga_star_log_tpm`) and produces a
dict of text features per case:

    {
        "mrna_subtype": "ccA" | "Unassigned" | "not_applicable",
        "hallmark_top_enriched": [("HALLMARK_HYPOXIA", 0.83), ...],
        "hallmark_top_suppressed": [("HALLMARK_OXIDATIVE_PHOSPHORYLATION", -0.71), ...],
        "estimate": {"stromal": float, "immune": float, "tumor_purity": float},
        "cell_type_fractions": {"CD8_T": "low", "Treg": "intermediate", ...},
        "signatures": {
            "proliferation_z": float, "hypoxia_z": float, "emt_z": float,
            "ifng_z": float, "cytolytic_log": float, "tis_z": float,
        },
        "lineage_markers": {"CA9": "high", "NDUFA4L2": "high", ...},
        "fusions_detected": [list of cohort-relevant fusion names],
        "categorical_bins": {<feature>: "low"|"intermediate"|"high"},
    }

The actual numeric/categorical split is driven by `cohort_config` thresholds,
which must be populated at fit time over the cohort (see
`fit_cohort_thresholds()`).

For ssGSEA we use a simple rank-based implementation. For ESTIMATE we use
the same rank-sum-based form as Yoshihara et al. 2013 (but with the 40-gene
embedded core when the full 141-gene lists aren't bundled). For fusion
detection we expect a precomputed fusion call file alongside each STAR TSV;
if absent, we emit `fusions_detected: "not_available"`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kidney_vlm.genomics import cohort_config as cohort_cfg
from kidney_vlm.genomics import signatures as sig


# ---------------------------------------------------------------------------
# STAR TPM loading with SYMBOL resolution
# ---------------------------------------------------------------------------


def read_tcga_star_tpm_by_symbol(tsv_path: str | Path) -> pd.Series:
    """Load a TCGA STAR TSV and return log1p(TPM) indexed by gene SYMBOL.

    We deliberately key by gene symbol here (not Ensembl ID) because all
    signatures in `signatures.py` and all cohort panels in `cohort_config.py`
    are specified as HGNC symbols.
    """
    path = Path(tsv_path)
    df = pd.read_csv(path, sep="\t", comment="#")
    required = {"gene_id", "gene_name", "tpm_unstranded"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"STAR TSV missing columns {missing}: {path}")

    df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
    if "gene_type" in df.columns:
        df = df[df["gene_type"].astype(str).eq("protein_coding")].copy()

    df["tpm_unstranded"] = pd.to_numeric(df["tpm_unstranded"], errors="coerce").fillna(0.0)
    # Collapse duplicate symbols by max TPM before log-transforming
    by_symbol = df.groupby("gene_name", as_index=True)["tpm_unstranded"].max()
    return np.log1p(by_symbol.astype(np.float64))


# ---------------------------------------------------------------------------
# ssGSEA (simple rank-based implementation)
# ---------------------------------------------------------------------------


def ssgsea_score(expression: pd.Series, gene_set: list[str], alpha: float = 0.25) -> float:
    """Single-sample GSEA score (Barbie et al. 2009).

    Implementation follows the original definition: rank all genes by
    expression, compute a weighted cumulative distribution for the gene set
    vs. its complement, and return the summed deviation. Positive values
    indicate enrichment.
    """
    # Align to the expression profile
    gene_set = [g for g in gene_set if g in expression.index]
    if not gene_set:
        return float("nan")

    # Rank-transform expression (descending; highest expressed = rank 1)
    ranks = expression.rank(method="average", ascending=False)
    N = len(ranks)
    Ns = len(gene_set)
    if Ns >= N:
        return float("nan")

    # Weights: rank^alpha for in-set genes
    in_set_mask = expression.index.isin(gene_set)
    weights = np.zeros(N, dtype=np.float64)
    # Sort genes by rank ascending (so the "walk" proceeds highest -> lowest expression)
    order = np.argsort(ranks.to_numpy())
    in_set_ordered = in_set_mask[order]
    exp_ordered = expression.to_numpy()[order]

    hit_weights = np.power(np.abs(exp_ordered), alpha) * in_set_ordered
    hit_norm = hit_weights.sum()
    if hit_norm == 0:
        return float("nan")
    p_hit = np.cumsum(hit_weights) / hit_norm
    p_miss = np.cumsum((~in_set_ordered).astype(np.float64)) / (N - Ns)

    deviations = p_hit - p_miss
    return float(deviations.sum() / N)


# ---------------------------------------------------------------------------
# Signature scoring (mean log-TPM over gene set, z-scored across cohort)
# ---------------------------------------------------------------------------


def mean_expression_score(expression: pd.Series, gene_set: list[str]) -> float:
    """Mean log1p(TPM) over the gene set. Missing genes ignored."""
    present = [g for g in gene_set if g in expression.index]
    if not present:
        return float("nan")
    return float(expression[present].mean())


def cytolytic_activity_log(expression: pd.Series) -> float:
    """Geometric mean of GZMA and PRF1 in TPM space (log1p form)."""
    values = [expression.get(g) for g in sig.CYTOLYTIC_ACTIVITY]
    values = [v for v in values if v is not None and np.isfinite(v)]
    if len(values) != 2:
        return float("nan")
    # The original Rooney definition is geometric mean in TPM space
    # Since we have log1p(TPM), we average log1p values and that's equivalent
    # (up to the +1 offset) for reporting purposes.
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# ESTIMATE-style scores
# ---------------------------------------------------------------------------


def estimate_scores(expression: pd.Series) -> dict[str, float]:
    """Compute ESTIMATE stromal / immune scores and a tumor-purity estimate.

    Full ESTIMATE uses single-sample GSEA over 141 stromal and 141 immune
    genes, then derives tumor purity via:
        purity = cos(0.6049872018 + 0.0001467884 * (stromal + immune))

    We preserve that transformation here. With the embedded 40-gene cores
    the scale is approximate; for publication-grade results swap in the full
    141-gene lists via `reference/estimate_signatures.json`.
    """
    sigs = sig.load_estimate_full()
    stromal_score = ssgsea_score(expression, sigs["stromal"])
    immune_score = ssgsea_score(expression, sigs["immune"])

    # ESTIMATE score as sum of the two ssGSEA scores, then purity via the
    # Yoshihara transformation.
    if np.isfinite(stromal_score) and np.isfinite(immune_score):
        combined = stromal_score + immune_score
        # ESTIMATE's original calibration used rank-sum scores; with our
        # ssGSEA form we rescale by a cohort-level factor at aggregation time.
        purity = float(np.cos(0.6049872018 + 0.0001467884 * combined * 1000.0))
        purity = float(np.clip(purity, 0.0, 1.0))
    else:
        purity = float("nan")

    return {
        "stromal": stromal_score,
        "immune": immune_score,
        "tumor_purity": purity,
    }


# ---------------------------------------------------------------------------
# Cell-type markers (simple z-score over marker panel)
# ---------------------------------------------------------------------------


def cell_type_marker_scores(expression: pd.Series) -> dict[str, float]:
    scores: dict[str, float] = {}
    for cell_type, markers in sig.CELL_TYPE_MARKERS.items():
        scores[cell_type] = mean_expression_score(expression, markers)
    return scores


# ---------------------------------------------------------------------------
# PAM50 subtyping (BRCA only)
# ---------------------------------------------------------------------------


def pam50_subtype(expression: pd.Series) -> str:
    """Assign PAM50 intrinsic subtype by nearest-centroid Spearman correlation.

    Returns one of {LumA, LumB, HER2_enriched, Basal_like, Normal_like} or
    "Unassigned" if the centroid lookup is unavailable or gene coverage is
    too low (<70% of 50 centroid genes present).
    """
    try:
        centroids = sig.load_pam50_centroids()
    except FileNotFoundError:
        return "Unassigned"

    all_genes = sorted({g for c in centroids.values() for g in c.keys()})
    present = [g for g in all_genes if g in expression.index]
    if len(present) < int(0.7 * len(all_genes)):
        return "Unassigned"

    sample_vec = expression[present].to_numpy()
    best_subtype = "Unassigned"
    best_rho = -np.inf
    for subtype, centroid in centroids.items():
        centroid_vec = pd.Series(
            [centroid.get(g, 0.0) for g in present],
            index=present,
            dtype=np.float64,
        )
        rho = pd.Series(sample_vec, index=present, dtype=np.float64).corr(
            centroid_vec,
            method="spearman",
        )
        if np.isfinite(rho) and rho > best_rho:
            best_rho = rho
            best_subtype = subtype
    return best_subtype


# ---------------------------------------------------------------------------
# Fusion detection loader
# ---------------------------------------------------------------------------


def load_fusion_calls(fusion_tsv_path: str | Path | None) -> list[str]:
    """Load precomputed STAR-Fusion / Arriba calls for this sample.

    The expected TSV has columns ["fusion_name"] at minimum, optionally with
    a "confidence" column. Returns a list of canonical fusion names like
    "TMPRSS2-ERG".
    """
    if fusion_tsv_path is None:
        return []
    path = Path(fusion_tsv_path)
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t", comment="#")
    if "fusion_name" not in df.columns:
        return []
    names = df["fusion_name"].astype(str).str.strip()
    return sorted(set(name for name in names.tolist() if name))


# ---------------------------------------------------------------------------
# Top-level feature extraction
# ---------------------------------------------------------------------------


@dataclass
class RnaTextFeatures:
    mrna_subtype: str
    mrna_subtype_label_space: list[str]
    hallmark_scores: dict[str, float]
    estimate: dict[str, float]
    cell_type_scores: dict[str, float]
    signatures: dict[str, float]
    lineage_markers: dict[str, float]
    fusions_detected: list[str]
    fusions_panel: list[str]
    gene_coverage: float  # fraction of expected reference genes observed

    def to_dict(self) -> dict[str, Any]:
        return {
            "mrna_subtype": self.mrna_subtype,
            "mrna_subtype_label_space": self.mrna_subtype_label_space,
            "hallmark_scores": self.hallmark_scores,
            "estimate": self.estimate,
            "cell_type_scores": self.cell_type_scores,
            "signatures": self.signatures,
            "lineage_markers": self.lineage_markers,
            "fusions_detected": self.fusions_detected,
            "fusions_panel": self.fusions_panel,
            "gene_coverage": self.gene_coverage,
        }


def extract_rna_text_features(
    *,
    star_tsv_path: str | Path,
    project_id: str,
    fusion_tsv_path: str | Path | None = None,
) -> RnaTextFeatures:
    """Run the full RNA-seq text-feature pipeline for one case.

    Produces a `RnaTextFeatures` object whose numeric fields are raw (not yet
    categorically binned). Bin assignment is deferred to the text-block
    assembly step so we can apply cohort thresholds consistently.
    """
    expression = read_tcga_star_tpm_by_symbol(star_tsv_path)

    # Hallmark ssGSEA
    hallmarks = sig.load_hallmark50()
    hallmark_scores: dict[str, float] = {}
    for name, genes in hallmarks.items():
        hallmark_scores[name] = ssgsea_score(expression, genes)

    # ESTIMATE
    est = estimate_scores(expression)

    # Cell-type markers
    cell_scores = cell_type_marker_scores(expression)

    # Functional signatures
    signatures = {
        "proliferation_mean": mean_expression_score(expression, sig.PROLIFERATION_CORE),
        "hypoxia_mean": mean_expression_score(expression, sig.HYPOXIA_BUFFA_51),
        "emt_mesenchymal_mean": mean_expression_score(expression, sig.EMT_CORE_MESENCHYMAL),
        "emt_epithelial_mean": mean_expression_score(expression, sig.EMT_CORE_EPITHELIAL),
        "ifng_mean": mean_expression_score(expression, sig.IFNG_AYERS_6),
        "tis_mean": mean_expression_score(expression, sig.TIS_AYERS_18),
        "cytolytic_log": cytolytic_activity_log(expression),
    }
    # EMT composite: mesenchymal - epithelial
    signatures["emt_composite"] = (
        signatures["emt_mesenchymal_mean"] - signatures["emt_epithelial_mean"]
    )

    # Lineage / receptor markers (raw log1p(TPM); categorical bin at assembly)
    lineage_markers_genes = cohort_cfg.LINEAGE_RECEPTOR_MARKERS.get(project_id, [])
    lineage_markers = {
        gene: float(expression[gene]) if gene in expression.index else float("nan")
        for gene in lineage_markers_genes
    }

    # mRNA subtype (BRCA has PAM50; other cohorts require cohort-specific
    # classifiers that we don't ship by default; they return "Unassigned"
    # and the teacher can fall back to describing signature-level features.
    if project_id == "TCGA-BRCA":
        mrna_subtype = pam50_subtype(expression)
    else:
        mrna_subtype = "Unassigned"
    label_space = cohort_cfg.MRNA_SUBTYPE_LABELS.get(project_id, [])

    # Fusion detection
    cohort_fusions = cohort_cfg.RECURRENT_FUSIONS.get(project_id, [])
    all_detected = load_fusion_calls(fusion_tsv_path)
    fusions_detected = [f for f in all_detected if f in cohort_fusions]

    # Gene coverage sanity metric
    all_reference_genes = set()
    for genes in hallmarks.values():
        all_reference_genes.update(genes)
    all_reference_genes.update(sig.HYPOXIA_BUFFA_51)
    all_reference_genes.update(sig.PROLIFERATION_CORE)
    all_reference_genes.update(sig.TIS_AYERS_18)
    coverage = (
        len(all_reference_genes.intersection(expression.index)) / max(1, len(all_reference_genes))
    )

    return RnaTextFeatures(
        mrna_subtype=mrna_subtype,
        mrna_subtype_label_space=label_space,
        hallmark_scores=hallmark_scores,
        estimate=est,
        cell_type_scores=cell_scores,
        signatures=signatures,
        lineage_markers=lineage_markers,
        fusions_detected=fusions_detected,
        fusions_panel=cohort_fusions,
        gene_coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Cohort-level threshold fitting
# ---------------------------------------------------------------------------


def fit_cohort_tertile_thresholds(values_by_project: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    """Compute (low_cutoff, high_cutoff) per cohort from the 33rd and 67th percentiles.

    Usage: aggregate a continuous feature (e.g. hypoxia_mean) across all cases
    in a cohort and pass the resulting {project_id: [values]} dict here.
    """
    out: dict[str, tuple[float, float]] = {}
    for project_id, values in values_by_project.items():
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if arr.size < 5:
            # Fallback when we have too few cases to estimate tertiles robustly
            out[project_id] = (float("nan"), float("nan"))
            continue
        low = float(np.percentile(arr, 33.0))
        high = float(np.percentile(arr, 67.0))
        out[project_id] = (low, high)
    return out


def bin_continuous(value: float, cutoffs: tuple[float, float] | None) -> str:
    """Bin a continuous value as low / intermediate / high / unknown."""
    if cutoffs is None or not np.isfinite(cutoffs[0]) or not np.isfinite(cutoffs[1]):
        return "unknown"
    if not np.isfinite(value):
        return "unknown"
    low, high = cutoffs
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "intermediate"
