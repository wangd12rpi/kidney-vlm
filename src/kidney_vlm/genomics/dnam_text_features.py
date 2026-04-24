"""
DNA methylation text-feature extraction.

Consumes raw Illumina 450K / EPIC beta-value TSV files registered by the
TCGA extra-genomics download step and produces per-case text features:

    {
        "methylation_subtype": "KIRC_m1" | "Unassigned" | "not_applicable",
        "cimp_status": "CIMP_high" | "CIMP_low" | "CIMP_negative" | "not_assessed",
        "promoter_methylation": {"VHL": 0.61, "MLH1": 0.08, ...},
        "promoter_methylation_bins": {"VHL": "high", "MLH1": "low", ...},
        "global_mean_beta": 0.46,
        "epigenetic_age_years": 58.3,
        "epigenetic_age_acceleration_years": 3.1,
        "dnam_tumor_purity": 0.71,
        "dnam_immune_fractions": {"CD8_T": "intermediate", ...},
    }

All of these are encoder-derivable (cpGPT is supposed to have learned them
from raw beta values), so they appear only in the teacher's text view; the
student does not see these features as text at inference. The student
reconstructs them from the cpGPT embedding.

The GDC standard beta-value TSV format has two columns (probe_id, beta_value)
with optional header and no index. We parse this robustly.
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
# Beta-value loading
# ---------------------------------------------------------------------------


def read_tcga_beta_tsv(beta_tsv_path: str | Path) -> pd.Series:
    """Load a TCGA GDC-format beta-value TSV. Returns a Series indexed by probe ID."""
    path = Path(beta_tsv_path)
    # GDC beta TSVs typically have no header; two columns: probe_id, beta
    # Some older TCGA level-3 files have a header like "Composite Element REF\tBeta_value".
    df = pd.read_csv(path, sep="\t", comment="#", header=None, engine="c")
    # Heuristic: detect header row
    first_cell = str(df.iloc[0, 0]).strip().lower()
    if first_cell in {"composite element ref", "probe_id", "cg_id", "probeid"} or not first_cell.startswith("cg"):
        df = pd.read_csv(path, sep="\t", comment="#", header=0)
    df.columns = [str(c).strip() for c in df.columns]
    # Find probe column
    probe_col = None
    beta_col = None
    for c in df.columns:
        cl = str(c).lower()
        if probe_col is None and ("probe" in cl or "composite" in cl or cl.startswith("cg")):
            probe_col = c
        if beta_col is None and ("beta" in cl or cl == "1" or cl.endswith("_value")):
            beta_col = c
    if probe_col is None or beta_col is None:
        # Fall back to positional
        probe_col = df.columns[0]
        beta_col = df.columns[1]

    probes = df[probe_col].astype(str).str.strip()
    betas = pd.to_numeric(df[beta_col], errors="coerce")
    out = pd.Series(betas.values, index=probes.values, name="beta").dropna()
    # Collapse any duplicate probes by mean
    if not out.index.is_unique:
        out = out.groupby(level=0).mean()
    return out


# ---------------------------------------------------------------------------
# Promoter methylation per gene
# ---------------------------------------------------------------------------


def promoter_methylation_by_gene(betas: pd.Series, project_id: str) -> dict[str, float]:
    """Mean beta over the TSS1500 + TSS200 probes for each cohort-panel gene.

    Requires `reference/promoter_probes.json` with {gene_symbol: [probe_id,...]}.
    Genes whose probes are entirely missing from this sample return NaN.
    """
    panel = cohort_cfg.PROMOTER_METHYLATION_PANEL.get(project_id, [])
    if not panel:
        return {}
    try:
        probe_map = sig.load_promoter_probes()
    except FileNotFoundError:
        # Without the probe mapping we can't compute promoter methylation.
        # Return NaN placeholders so the assembly step can render "not_assessed".
        return {gene: float("nan") for gene in panel}

    out: dict[str, float] = {}
    for gene in panel:
        probes = probe_map.get(gene, [])
        if not probes:
            out[gene] = float("nan")
            continue
        present = [p for p in probes if p in betas.index]
        if not present:
            out[gene] = float("nan")
            continue
        out[gene] = float(betas[present].mean())
    return out


def bin_promoter_beta(value: float) -> str:
    """Static clinical bins for promoter methylation.

    Thresholds: <0.2 low, 0.2-0.5 intermediate, >=0.5 high. Unlike the RNA
    signatures these are stable enough across cohorts that we use fixed
    cutoffs rather than cohort tertiles.
    """
    if not np.isfinite(value):
        return "not_assessed"
    if value < 0.2:
        return "low"
    if value >= 0.5:
        return "high"
    return "intermediate"


# ---------------------------------------------------------------------------
# CIMP status (cohort-dependent panels)
# ---------------------------------------------------------------------------


# CIMP marker panels per cohort (published clinical panels). Mean beta over
# these markers is thresholded to classify CIMP-high / CIMP-low / CIMP-negative.
# Only applies to cohorts where CIMP has a validated definition.
CIMP_PANELS: dict[str, list[str]] = {
    "TCGA-COAD": ["RUNX3", "CACNA1G", "NEUROG1", "IGF2", "SOCS1"],
    "TCGA-READ": ["RUNX3", "CACNA1G", "NEUROG1", "IGF2", "SOCS1"],
    "TCGA-STAD": ["MLH1", "CDKN2A", "CACNA1G", "NEUROG1"],
    "TCGA-UCEC": ["RASSF1A", "CDKN2A", "MLH1"],
    "TCGA-ESCA": ["RASSF1A", "CDKN2A", "CACNA1G"],
}


def classify_cimp(betas: pd.Series, project_id: str) -> str:
    if project_id not in CIMP_PANELS:
        return "not_assessed"
    try:
        probe_map = sig.load_promoter_probes()
    except FileNotFoundError:
        return "not_assessed"

    markers = CIMP_PANELS[project_id]
    marker_betas: list[float] = []
    for gene in markers:
        probes = probe_map.get(gene, [])
        present = [p for p in probes if p in betas.index]
        if not present:
            continue
        marker_betas.append(float(betas[present].mean()))
    if len(marker_betas) < max(2, len(markers) // 2):
        return "not_assessed"

    n_methylated = sum(1 for v in marker_betas if v >= 0.3)
    if n_methylated >= max(3, int(0.6 * len(marker_betas))):
        return "CIMP_high"
    if n_methylated >= 1:
        return "CIMP_low"
    return "CIMP_negative"


# ---------------------------------------------------------------------------
# Global methylation summary
# ---------------------------------------------------------------------------


def global_mean_beta(betas: pd.Series) -> float:
    if betas.empty:
        return float("nan")
    return float(betas.mean())


# ---------------------------------------------------------------------------
# Epigenetic age (Horvath 2013 353-probe clock)
# ---------------------------------------------------------------------------


def horvath_epigenetic_age(betas: pd.Series) -> float:
    """Return epigenetic age in years using the Horvath 2013 clock.

    Returns NaN if the clock reference isn't bundled or fewer than 300 of
    the 353 probes are present in this sample.
    """
    try:
        coefficients, intercept = sig.load_horvath_clock()
    except FileNotFoundError:
        return float("nan")

    probes = list(coefficients.keys())
    present = [p for p in probes if p in betas.index]
    if len(present) < 300:
        return float("nan")

    # Horvath formula: linear combination of (beta - 0.5) terms, then
    # anti-log transform for age < 20 years; the transformation is detailed
    # in Horvath 2013 Genome Biology.
    linear = intercept + sum(coefficients[p] * float(betas[p]) for p in present)
    # Inverse transformation
    if linear < 0:
        age = (np.exp(linear + 1) - 1) * 20.0
    else:
        age = (linear * 21.0) + 20.0
    return float(age)


def epigenetic_age_acceleration(epigenetic_age: float, chronological_age: float | None) -> float:
    if chronological_age is None or not np.isfinite(epigenetic_age):
        return float("nan")
    return float(epigenetic_age - chronological_age)


# ---------------------------------------------------------------------------
# DNAm-based tumor purity (LUMP, Leukocyte UnMethylation for Purity)
# ---------------------------------------------------------------------------

# The LUMP estimator averages beta values over 44 immune-specific CpGs and
# reports purity = 1 - mean_beta / 0.85. Simplified to the probe list published
# in Aran et al. 2015.
LUMP_PROBES: list[str] = [
    "cg01873645", "cg03046247", "cg03086965", "cg05020959", "cg05475556",
    "cg06081022", "cg06226419", "cg06899649", "cg07075926", "cg07230366",
    "cg07499544", "cg08269036", "cg08532082", "cg08752493", "cg09192967",
    "cg09234118", "cg09580953", "cg09717436", "cg10240487", "cg10402417",
    "cg11121090", "cg11233154", "cg11314684", "cg12089548", "cg12301028",
    "cg12512710", "cg12707721", "cg13176867", "cg13205930", "cg13646990",
    "cg14162967", "cg14580812", "cg15028253", "cg15419831", "cg15633388",
    "cg16345372", "cg16673281", "cg17339856", "cg17582214", "cg18191723",
    "cg18562416", "cg19287457", "cg21126493", "cg21434633",
]


def lump_tumor_purity(betas: pd.Series) -> float:
    """Return tumor purity estimate via LUMP. NaN if too few probes present."""
    present = [p for p in LUMP_PROBES if p in betas.index]
    if len(present) < 30:
        return float("nan")
    mean_beta = float(betas[present].mean())
    purity = 1.0 - (mean_beta / 0.85)
    return float(np.clip(purity, 0.0, 1.0))


# ---------------------------------------------------------------------------
# DNAm immune composition (simplified cell-type-specific CpG panels)
# ---------------------------------------------------------------------------

# A compact panel of cell-type-specific CpGs. Full MethylCIBERSORT requires
# a large signature matrix; we use a reduced proxy here, sufficient for
# coarse categorical assignment.
DNAM_CELLTYPE_PROBES: dict[str, list[str]] = {
    # Placeholder probe IDs should be replaced with a curated reference
    # derived from EpiDISH or MethylCIBERSORT panels. Left empty here so the
    # pipeline falls through to "not_assessed" unless the reference JSON is
    # bundled.
    # See reference/dnam_immune_probes.json for the canonical panel.
}


def dnam_immune_fractions(betas: pd.Series) -> dict[str, float]:
    """Return approximate immune cell-type fractions from DNAm.

    Returns empty dict if no reference panel is bundled.
    """
    if not DNAM_CELLTYPE_PROBES:
        return {}
    out: dict[str, float] = {}
    for cell_type, probes in DNAM_CELLTYPE_PROBES.items():
        present = [p for p in probes if p in betas.index]
        if not present:
            out[cell_type] = float("nan")
        else:
            # Lower beta at cell-type-specific demethylated sites indicates
            # higher presence of that cell type; we invert mean beta.
            out[cell_type] = float(1.0 - betas[present].mean())
    return out


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------


@dataclass
class DnamTextFeatures:
    methylation_subtype: str
    cimp_status: str
    promoter_methylation: dict[str, float]
    global_mean_beta: float
    epigenetic_age_years: float
    epigenetic_age_acceleration_years: float
    dnam_tumor_purity: float
    dnam_immune_fractions: dict[str, float]
    probe_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "methylation_subtype": self.methylation_subtype,
            "cimp_status": self.cimp_status,
            "promoter_methylation": self.promoter_methylation,
            "global_mean_beta": self.global_mean_beta,
            "epigenetic_age_years": self.epigenetic_age_years,
            "epigenetic_age_acceleration_years": self.epigenetic_age_acceleration_years,
            "dnam_tumor_purity": self.dnam_tumor_purity,
            "dnam_immune_fractions": self.dnam_immune_fractions,
            "probe_count": self.probe_count,
        }


def extract_dnam_text_features(
    *,
    beta_tsv_path: str | Path,
    project_id: str,
    chronological_age_years: float | None = None,
    methylation_subtype_label: str | None = None,
) -> DnamTextFeatures:
    """Run the full DNAm text-feature pipeline for one case.

    `methylation_subtype_label` is typically sourced from a precomputed TCGA
    methylation cluster assignment (published per cohort) rather than
    re-clustered here. Pass in "Unassigned" if unknown.
    """
    betas = read_tcga_beta_tsv(beta_tsv_path)
    promoter = promoter_methylation_by_gene(betas, project_id)
    cimp = classify_cimp(betas, project_id)
    epi_age = horvath_epigenetic_age(betas)
    epi_accel = epigenetic_age_acceleration(epi_age, chronological_age_years)
    purity = lump_tumor_purity(betas)
    immune_fracs = dnam_immune_fractions(betas)

    return DnamTextFeatures(
        methylation_subtype=methylation_subtype_label or "Unassigned",
        cimp_status=cimp,
        promoter_methylation=promoter,
        global_mean_beta=global_mean_beta(betas),
        epigenetic_age_years=epi_age,
        epigenetic_age_acceleration_years=epi_accel,
        dnam_tumor_purity=purity,
        dnam_immune_fractions=immune_fracs,
        probe_count=int(betas.size),
    )
