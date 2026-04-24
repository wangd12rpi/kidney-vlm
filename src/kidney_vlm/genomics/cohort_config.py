"""Cohort-specific configuration for TCGA genomics text-feature generation.

The teacher and student text blocks use one stable schema, but the useful
fields are cohort-specific. This module centralizes curated mutation, copy
number, methylation, expression, fusion, and integrated-surrogate panels.

All cohort IDs use the TCGA project_id form, e.g. ``TCGA-KIRC``.
"""
from __future__ import annotations

from dataclasses import dataclass

from kidney_vlm.data.sources.tcga import (
    DEFAULT_PANCANCER_MUTATION_GENE_PANEL,
    DEFAULT_PROJECT_DRIVER_GENE_PANEL_BY_PROJECT,
)


TCGA_PROJECT_IDS = sorted(
    {
        *DEFAULT_PROJECT_DRIVER_GENE_PANEL_BY_PROJECT,
        "TCGA-ACC",
        "TCGA-BLCA",
        "TCGA-BRCA",
        "TCGA-CESC",
        "TCGA-CHOL",
        "TCGA-COAD",
        "TCGA-DLBC",
        "TCGA-ESCA",
        "TCGA-GBM",
        "TCGA-HNSC",
        "TCGA-KICH",
        "TCGA-KIRC",
        "TCGA-KIRP",
        "TCGA-LAML",
        "TCGA-LGG",
        "TCGA-LIHC",
        "TCGA-LUAD",
        "TCGA-LUSC",
        "TCGA-MESO",
        "TCGA-OV",
        "TCGA-PAAD",
        "TCGA-PCPG",
        "TCGA-PRAD",
        "TCGA-READ",
        "TCGA-SARC",
        "TCGA-SKCM",
        "TCGA-STAD",
        "TCGA-TGCT",
        "TCGA-THCA",
        "TCGA-THYM",
        "TCGA-UCEC",
        "TCGA-UCS",
        "TCGA-UVM",
    }
)


# ---------------------------------------------------------------------------
# Canonical gene panels
# ---------------------------------------------------------------------------

# Pan-cancer TSG / driver panel always reported (small; these appear as
# wild-type calls when not mutated in the case).
PAN_CANCER_CORE_PANEL: list[str] = [
    "TP53",
    "PTEN",
    "PIK3CA",
    "KRAS",
    "BRAF",
    "CDKN2A",
    "RB1",
    "MYC",
    "ARID1A",
    "ARID1B",
    "SMAD4",
    "APC",
]

# Subtype-defining loci by cohort. Wild-type calls are serialized for these
# genes so absence of evidence is not confused with evidence of absence.
SUBTYPE_DEFINING_LOCI: dict[str, list[str]] = {
    "TCGA-BLCA": ["TP53", "FGFR3", "RB1", "ERBB2", "KDM6A", "STAG2", "ARID1A"],
    "TCGA-BRCA": ["TP53", "PIK3CA", "GATA3", "MAP3K1", "CDH1", "AKT1", "BRCA1", "BRCA2"],
    "TCGA-CESC": ["PIK3CA", "EP300", "FBXW7", "HLA-B", "PTEN", "TP53", "STK11"],
    "TCGA-COAD": ["APC", "TP53", "KRAS", "PIK3CA", "SMAD4", "BRAF", "FBXW7", "TCF7L2"],
    "TCGA-ESCA": ["TP53", "CDKN2A", "NFE2L2", "PIK3CA", "NOTCH1", "KMT2D", "ERBB2"],
    "TCGA-KICH": ["TP53", "PTEN"],
    "TCGA-KIRC": ["VHL", "PBRM1", "BAP1", "SETD2", "KDM5C", "MTOR", "TP53"],
    "TCGA-KIRP": ["MET", "SETD2", "NF2", "FH", "CDKN2A", "TERT"],
    "TCGA-LIHC": ["TP53", "CTNNB1", "AXIN1", "ARID1A", "ARID2", "RB1", "ALB"],
    "TCGA-LUAD": ["KRAS", "EGFR", "TP53", "STK11", "KEAP1", "BRAF", "MET", "ALK", "ROS1"],
    "TCGA-LUSC": ["TP53", "CDKN2A", "PIK3CA", "NFE2L2", "KEAP1", "FGFR1", "PTEN"],
    "TCGA-OV": ["TP53", "BRCA1", "BRCA2", "NF1", "RB1", "CDK12"],
    "TCGA-PRAD": ["TP53", "PTEN", "SPOP", "FOXA1", "TMPRSS2", "ERG", "AR"],
    "TCGA-READ": ["APC", "TP53", "KRAS", "PIK3CA", "SMAD4", "BRAF", "FBXW7"],
    "TCGA-SARC": ["TP53", "ATRX", "RB1", "CDKN2A", "MDM2"],
    "TCGA-STAD": ["TP53", "ARID1A", "PIK3CA", "CDH1", "KRAS", "ERBB2", "SMAD4"],
    "TCGA-THCA": ["BRAF", "NRAS", "HRAS", "KRAS", "TP53", "TERT", "EIF1AX"],
    "TCGA-UCEC": ["PTEN", "PIK3CA", "ARID1A", "CTNNB1", "TP53", "PIK3R1", "KRAS", "POLE"],
}


# Promoter methylation panels per cohort. These are the genes for which we
# report promoter beta (mean of probes in the TSS1500 + TSS200 window, using
# the Illumina 450K manifest). cpGPT receives the full beta array, so these
# are encoder-derivable; they appear in the teacher's text only.
PROMOTER_METHYLATION_PANEL: dict[str, list[str]] = {
    "TCGA-BLCA": ["CDKN2A", "RASSF1A", "MGMT", "APC", "HOXA9"],
    "TCGA-BRCA": ["BRCA1", "RASSF1A", "ESR1", "CDKN2A", "GSTP1"],
    "TCGA-CESC": ["CDKN2A", "RASSF1A", "MGMT", "CADM1", "MAL"],
    "TCGA-COAD": ["MLH1", "MGMT", "CDKN2A", "APC", "RASSF1A"],
    "TCGA-ESCA": ["CDKN2A", "MGMT", "APC", "RASSF1A", "CDH1"],
    "TCGA-KICH": ["VHL", "CDKN2A", "SFRP1", "RASSF1A"],
    "TCGA-KIRC": ["VHL", "CDKN2A", "SFRP1", "RASSF1A", "MGMT"],
    "TCGA-KIRP": ["CDKN2A", "RASSF1A", "SFRP1", "MGMT"],
    "TCGA-LIHC": ["CDKN2A", "RASSF1A", "GSTP1", "APC", "MGMT"],
    "TCGA-LUAD": ["CDKN2A", "RASSF1A", "MGMT", "APC", "CDH13"],
    "TCGA-LUSC": ["CDKN2A", "RASSF1A", "MGMT", "DAPK1"],
    "TCGA-OV": ["BRCA1", "RASSF1A", "CDKN2A", "MGMT"],
    "TCGA-PRAD": ["GSTP1", "APC", "RASSF1A", "RARB", "MGMT"],
    "TCGA-READ": ["MLH1", "MGMT", "CDKN2A", "APC", "RASSF1A"],
    "TCGA-SARC": ["CDKN2A", "RASSF1A", "MGMT"],
    "TCGA-STAD": ["MLH1", "CDKN2A", "RASSF1A", "MGMT", "CDH1"],
    "TCGA-THCA": ["TSHR", "RASSF1A", "CDKN2A"],
    "TCGA-UCEC": ["MLH1", "PTEN", "CDKN2A", "RASSF1A", "MGMT"],
}


# Gene-level copy number panels (focal CNA events reported by GISTIC2/ASCAT2).
FOCAL_CNA_PANEL: dict[str, list[str]] = {
    "TCGA-BLCA": ["CDKN2A", "MDM2", "CCND1", "ERBB2", "FGFR3", "E2F3", "PPARG"],
    "TCGA-BRCA": ["ERBB2", "MYC", "CCND1", "FGFR1", "CDKN2A", "PTEN", "MDM2"],
    "TCGA-CESC": ["MYC", "CDKN2A", "PIK3CA", "TP63"],
    "TCGA-COAD": ["MYC", "CDKN2A", "SMAD4", "APC"],
    "TCGA-ESCA": ["CCND1", "ERBB2", "MYC", "CDKN2A", "MDM2", "FGFR1"],
    "TCGA-KICH": ["CDKN2A"],
    "TCGA-KIRC": ["CDKN2A", "MYC", "PTEN"],
    "TCGA-KIRP": ["CDKN2A", "MET"],
    "TCGA-LIHC": ["MYC", "CCND1", "CDKN2A", "FGF19", "TERT"],
    "TCGA-LUAD": ["MYC", "CDKN2A", "NKX2-1", "MDM2", "TERT", "EGFR", "MET", "CCND1"],
    "TCGA-LUSC": ["SOX2", "CDKN2A", "PIK3CA", "MYC", "FGFR1", "NFE2L2"],
    "TCGA-OV": ["MYC", "CCNE1", "CDKN2A", "MECOM", "KRAS"],
    "TCGA-PRAD": ["MYC", "PTEN", "RB1", "CDKN1B", "TP53"],
    "TCGA-READ": ["MYC", "CDKN2A", "SMAD4", "APC"],
    "TCGA-SARC": ["MDM2", "CDK4", "CDKN2A", "RB1"],
    "TCGA-STAD": ["ERBB2", "CCNE1", "MYC", "CDKN2A", "KRAS", "MET"],
    "TCGA-THCA": ["CDKN2A"],
    "TCGA-UCEC": ["MYC", "CDKN2A", "ERBB2", "CCNE1", "FGFR3"],
}


# Arm-level CNA events the cohort is known to recurrently carry. These are
# reported as "present / absent" from segment-level aggregation.
ARM_LEVEL_CNA_PANEL: dict[str, list[str]] = {
    "TCGA-BLCA": ["9p_loss", "9q_loss", "5q_gain", "20q_gain"],
    "TCGA-BRCA": ["1q_gain", "8q_gain", "16q_loss", "17p_loss", "11q_loss"],
    "TCGA-CESC": ["3q_gain", "1q_gain", "11q_loss"],
    "TCGA-COAD": ["7p_gain", "8q_gain", "13q_gain", "18q_loss", "17p_loss"],
    "TCGA-ESCA": ["3q_gain", "8q_gain", "5q_loss", "18q_loss"],
    "TCGA-KICH": ["1p_loss", "2p_loss", "6p_loss", "10q_loss", "13q_loss", "17p_loss"],
    "TCGA-KIRC": ["3p_loss", "5q_gain", "14q_loss", "9p_loss"],
    "TCGA-KIRP": ["7_gain", "17_gain", "8p_loss", "9p_loss"],
    "TCGA-LIHC": ["1q_gain", "8q_gain", "8p_loss", "17p_loss", "4q_loss"],
    "TCGA-LUAD": ["1q_gain", "8q_gain", "3q_gain", "9p_loss", "17p_loss"],
    "TCGA-LUSC": ["3q_gain", "5p_gain", "8q_gain", "3p_loss", "8p_loss", "9p_loss"],
    "TCGA-OV": ["8q_gain", "3q_gain", "20q_gain", "4q_loss", "16q_loss"],
    "TCGA-PRAD": ["8q_gain", "8p_loss", "13q_loss", "16q_loss", "17p_loss"],
    "TCGA-READ": ["7p_gain", "8q_gain", "13q_gain", "18q_loss", "17p_loss"],
    "TCGA-SARC": ["5p_gain", "8q_gain", "17p_loss"],
    "TCGA-STAD": ["8q_gain", "20q_gain", "3q_gain", "17p_loss", "18q_loss"],
    "TCGA-THCA": [],
    "TCGA-UCEC": ["1q_gain", "8q_gain", "17p_loss"],
}


# mRNA transcriptomic subtype label spaces per cohort. These are canonical
# TCGA marker-paper clusters. Values are informational; the actual subtype
# assignment requires a cohort-specific classifier trained on TCGA expression.
MRNA_SUBTYPE_LABELS: dict[str, list[str]] = {
    "TCGA-BLCA": ["luminal_papillary", "luminal", "luminal_infiltrated", "basal_squamous", "neuronal"],
    "TCGA-BRCA": ["LumA", "LumB", "HER2_enriched", "Basal_like", "Normal_like"],
    "TCGA-CESC": ["keratin_low_squamous", "keratin_high_squamous", "adenocarcinoma_rich"],
    "TCGA-COAD": ["CMS1", "CMS2", "CMS3", "CMS4"],
    "TCGA-ESCA": ["ESCA_1_squamous", "ESCA_2_adeno_chromosomal", "ESCA_3_adeno_metabolic"],
    "TCGA-KICH": ["eosinophilic", "classical"],
    "TCGA-KIRC": ["ccA", "ccB"],
    "TCGA-KIRP": ["C1_papillary_type1", "C2a_papillary_type2", "C2b_papillary_type2", "C2c_papillary_type2"],
    "TCGA-LIHC": ["iCluster1", "iCluster2", "iCluster3"],
    "TCGA-LUAD": ["TRU", "PI", "PP"],
    "TCGA-LUSC": ["classical", "basal", "secretory", "primitive"],
    "TCGA-OV": ["differentiated", "immunoreactive", "mesenchymal", "proliferative"],
    "TCGA-PRAD": ["ERG", "ETV1", "ETV4", "FLI1", "SPOP_mutant", "FOXA1_mutant", "IDH1_mutant"],
    "TCGA-READ": ["CMS1", "CMS2", "CMS3", "CMS4"],
    "TCGA-SARC": ["DDLPS", "LMS", "UPS", "MFS", "SS", "MPNST"],
    "TCGA-STAD": ["EBV", "MSI", "GS", "CIN"],
    "TCGA-THCA": ["BRAF_like", "RAS_like"],
    "TCGA-UCEC": ["POLE_ultramutated", "MSI_hypermutated", "CN_low", "CN_high"],
}


# Lineage / receptor marker genes reported with categorical expression levels.
LINEAGE_RECEPTOR_MARKERS: dict[str, list[str]] = {
    "TCGA-BLCA": ["KRT5", "KRT14", "KRT20", "UPK3A", "PPARG", "FOXA1", "GATA3"],
    "TCGA-BRCA": ["ESR1", "PGR", "ERBB2", "MKI67", "FOXA1", "GATA3"],
    "TCGA-CESC": ["KRT5", "KRT14", "TP63", "CDKN2A"],
    "TCGA-COAD": ["CDX2", "KRT20", "MUC2", "VIL1"],
    "TCGA-ESCA": ["TP63", "KRT5", "CDX2", "MUC2"],
    "TCGA-KICH": ["KIT", "FOXI1", "AQP6"],
    "TCGA-KIRC": ["CA9", "NDUFA4L2", "VEGFA", "EPAS1"],
    "TCGA-KIRP": ["MET", "HGF", "CA9"],
    "TCGA-LIHC": ["AFP", "GPC3", "ALB", "KRT19"],
    "TCGA-LUAD": ["NKX2-1", "NAPSA", "SFTPC", "KRT7"],
    "TCGA-LUSC": ["TP63", "KRT5", "KRT6A", "SOX2"],
    "TCGA-OV": ["PAX8", "WT1", "KRT7", "ESR1"],
    "TCGA-PRAD": ["AR", "KLK3", "NKX3-1", "CHGA", "SYP", "NCAM1"],
    "TCGA-READ": ["CDX2", "KRT20", "MUC2", "VIL1"],
    "TCGA-SARC": ["MYOD1", "MYOG", "DES", "CD34", "S100B"],
    "TCGA-STAD": ["CDX2", "CDH17", "MUC5AC", "MUC6"],
    "TCGA-THCA": ["TG", "TPO", "TSHR", "PAX8"],
    "TCGA-UCEC": ["ESR1", "PGR", "PAX8", "VIM", "CDH1"],
}


# Cohort-relevant recurrent fusions for STAR-Fusion / Arriba. Reported as
# detected / not_detected. Non-exhaustive; see the RNA-seq pipeline for
# optional expansion.
RECURRENT_FUSIONS: dict[str, list[str]] = {
    "TCGA-BLCA": ["FGFR3-TACC3"],
    "TCGA-BRCA": ["MYB-NFIB", "ETV6-NTRK3"],
    "TCGA-CESC": [],
    "TCGA-COAD": ["PTPRK-RSPO3", "EIF3E-RSPO2", "NTRK1_fusions"],
    "TCGA-ESCA": [],
    "TCGA-KICH": [],
    "TCGA-KIRC": ["SFPQ-TFE3"],
    "TCGA-KIRP": ["SFPQ-TFE3", "ALK_fusions"],
    "TCGA-LIHC": [],
    "TCGA-LUAD": ["EML4-ALK", "CD74-ROS1", "KIF5B-RET", "NTRK1_fusions", "FGFR3-TACC3"],
    "TCGA-LUSC": ["FGFR3-TACC3"],
    "TCGA-OV": [],
    "TCGA-PRAD": ["TMPRSS2-ERG", "TMPRSS2-ETV1", "TMPRSS2-ETV4", "SLC45A3-ERG"],
    "TCGA-READ": ["PTPRK-RSPO3", "EIF3E-RSPO2"],
    "TCGA-SARC": ["SS18-SSX1", "SS18-SSX2", "FUS-DDIT3", "EWSR1_fusions"],
    "TCGA-STAD": ["CLDN18-ARHGAP26"],
    "TCGA-THCA": ["CCDC6-RET", "NCOA4-RET", "ETV6-NTRK3"],
    "TCGA-UCEC": [],
}


# Integrated surrogate applicability. Determines which surrogate panels are
# computed per cohort; `False` means we serialize `not_assessed`.
@dataclass(frozen=True)
class IntegratedSurrogateConfig:
    msi_like: bool = False
    hrd_like: bool = False
    hormone_receptor_concordance: bool = False
    vhl_pathway_inactivation: bool = False
    cimp_status: bool = False


INTEGRATED_SURROGATES: dict[str, IntegratedSurrogateConfig] = {
    "TCGA-BLCA": IntegratedSurrogateConfig(),
    "TCGA-BRCA": IntegratedSurrogateConfig(hrd_like=True, hormone_receptor_concordance=True),
    "TCGA-CESC": IntegratedSurrogateConfig(),
    "TCGA-COAD": IntegratedSurrogateConfig(msi_like=True, cimp_status=True),
    "TCGA-ESCA": IntegratedSurrogateConfig(cimp_status=True),
    "TCGA-KICH": IntegratedSurrogateConfig(),
    "TCGA-KIRC": IntegratedSurrogateConfig(vhl_pathway_inactivation=True),
    "TCGA-KIRP": IntegratedSurrogateConfig(),
    "TCGA-LIHC": IntegratedSurrogateConfig(),
    "TCGA-LUAD": IntegratedSurrogateConfig(),
    "TCGA-LUSC": IntegratedSurrogateConfig(),
    "TCGA-OV": IntegratedSurrogateConfig(hrd_like=True),
    "TCGA-PRAD": IntegratedSurrogateConfig(),
    "TCGA-READ": IntegratedSurrogateConfig(msi_like=True, cimp_status=True),
    "TCGA-SARC": IntegratedSurrogateConfig(),
    "TCGA-STAD": IntegratedSurrogateConfig(msi_like=True, cimp_status=True),
    "TCGA-THCA": IntegratedSurrogateConfig(),
    "TCGA-UCEC": IntegratedSurrogateConfig(msi_like=True),
}


# Non-coding RNA signatures that fall outside BulkFormer's vocabulary but are
# cohort-relevant. Used by the miRNA pipeline to decide what to report as text.
NCRNA_SIGNATURES: dict[str, list[str]] = {
    "TCGA-BRCA": ["miR-200_family", "miR-21", "miR-155", "HOTAIR"],
    "TCGA-KIRC": ["miR-210_hypoxia", "miR-21", "miR-155"],
    "TCGA-KIRP": ["miR-210_hypoxia"],
    "TCGA-LUAD": ["miR-210_hypoxia", "miR-200_family", "MALAT1"],
    "TCGA-LUSC": ["miR-210_hypoxia", "MALAT1"],
    "TCGA-LIHC": ["miR-122", "miR-21"],
    "TCGA-OV": ["miR-200_family", "HOTAIR"],
    "TCGA-PRAD": ["miR-21", "miR-141", "miR-200_family"],
    "TCGA-COAD": ["miR-21", "miR-155"],
    "TCGA-READ": ["miR-21", "miR-155"],
    "TCGA-STAD": ["miR-21", "HOTAIR"],
    "TCGA-UCEC": ["miR-200_family"],
    "TCGA-BLCA": ["miR-21", "miR-200_family"],
    "TCGA-ESCA": ["miR-21"],
    "TCGA-CESC": ["miR-21"],
    "TCGA-SARC": [],
    "TCGA-THCA": ["miR-146b", "miR-221"],
    "TCGA-KICH": [],
}


# Categorical binning thresholds for continuous features. These are
# intentionally quantile-based and computed at data-generation time over the
# cohort, so the thresholds become frozen lookups; the pipeline does not
# re-estimate them per run.
@dataclass
class CohortQuantileThresholds:
    """Populated at fit time with cohort-specific tertile cutoffs.

    The fields map to continuous scores produced by the RNA and DNAm pipelines.
    Each tuple is (low_threshold, high_threshold); values below the first
    cutoff are "low", between the cutoffs "intermediate", above the second
    "high".
    """

    hypoxia_score: tuple[float, float] | None = None
    proliferation_score: tuple[float, float] | None = None
    emt_score: tuple[float, float] | None = None
    ifng_score: tuple[float, float] | None = None
    cytolytic_score: tuple[float, float] | None = None
    tis_score: tuple[float, float] | None = None
    estimate_stromal: tuple[float, float] | None = None
    estimate_immune: tuple[float, float] | None = None
    estimate_tumor_purity: tuple[float, float] | None = None


def get_all_cohorts() -> list[str]:
    """Return every TCGA cohort the pipeline may encounter."""
    return list(TCGA_PROJECT_IDS)


def get_mutation_panel(project_id: str) -> list[str]:
    """Return the cohort mutation panel with curated genes first.

    Curated subtype loci stay at the front because their wild-type status is
    often useful in generated text. The existing TCGA source driver panel is
    then appended as a broad fallback, followed by a small pan-cancer core.
    Unknown projects receive the source package's pan-cancer driver union.
    """
    project_key = str(project_id).strip()
    subtype_panel = SUBTYPE_DEFINING_LOCI.get(project_key, [])
    project_panel = DEFAULT_PROJECT_DRIVER_GENE_PANEL_BY_PROJECT.get(project_key, [])
    fallback_panel = project_panel or DEFAULT_PANCANCER_MUTATION_GENE_PANEL
    return list(
        dict.fromkeys(
            str(gene).strip().upper()
            for gene in [*subtype_panel, *fallback_panel, *PAN_CANCER_CORE_PANEL]
            if str(gene).strip()
        )
    )
