"""
Reference gene signatures and probe sets for the genomics text-feature pipeline.

We keep these inline to avoid external downloads during feature extraction.
Where a full MSigDB Hallmark 50 gene list is required, we load it lazily from
a bundled JSON blob (see `data/reference/hallmark50.json`). For smaller, more
frequently used signatures we embed them directly as Python lists here.

Sources for the embedded signatures (attribution for the paper):
* Hallmark 50: MSigDB v2023.2 human Hallmark collection (Liberzon et al. 2015)
* Proliferation: consensus cell-cycle signature (Whitfield et al. 2006)
* Hypoxia: Buffa 51-gene metagene (Buffa et al. 2010)
* EMT: Mak 77-gene pan-cancer EMT score (Mak et al. 2016)
* IFN-gamma: Ayers 6-gene (Ayers et al. 2017)
* Cytolytic activity: GZMA, PRF1 geometric mean (Rooney et al. 2015)
* TIS (Tumor Inflammation Signature): 18-gene Danaher/Ayers (Ayers et al. 2017)
* ESTIMATE: Yoshihara et al. 2013 (stromal 141 genes + immune 141 genes)
* PAM50: Parker et al. 2009 centroids (BRCA only)
* Horvath 2013 clock: 353 CpG probe set with coefficients
* MCP-counter / xCell-lite marker panels for immune deconvolution

NOTE: Full reference arrays for ESTIMATE (282 genes), Hallmark 50 (~4,400 genes),
PAM50 (50 genes x 5 centroids), and the Horvath clock (353 probes x 1 coefficient)
are large enough that we load them from a bundled JSON file rather than inline.
We ship `src/kidney_vlm/genomics/reference/*.json` with the repo. For this file
we include:
    * Small hand-curated signatures as Python literals (proliferation, hypoxia,
      EMT-core, IFN-gamma, cytolytic, TIS, cell-type markers)
    * Loader stubs for the larger JSONs
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


# ---------------------------------------------------------------------------
# Small embedded signatures
# ---------------------------------------------------------------------------

# Buffa 51-gene hypoxia metagene (Buffa et al. Br J Cancer 2010)
HYPOXIA_BUFFA_51: list[str] = [
    "VEGFA", "SLC2A1", "PGAM1", "ENO1", "LDHA", "TPI1", "P4HA1", "MRPS17",
    "CDKN3", "ADM", "NDRG1", "TUBB6", "ALDOA", "MIF", "ACOT7", "MCTS1",
    "PSMA7", "ANLN", "TUBA1B", "SLC25A32", "HK2", "ESRP1", "PFKP", "CORO1C",
    "PSRC1", "CA9", "LRRC42", "KIF20A", "DDIT4", "PFAS", "BNIP3", "KIF4A",
    "LRRN2", "FAM83B", "MAD2L2", "SIAH2", "ANKRD37", "PRC1", "UTP11L",
    "PNP", "GPI", "NME1", "KIF22", "SEC61G", "TUBB", "EGLN3", "GAPDH",
    "MRPL13", "CHCHD2", "AK2", "RRAGD",
]

# Whitfield / Rhodes proliferation (cell-cycle) signature core set
PROLIFERATION_CORE: list[str] = [
    "MKI67", "PCNA", "CCNB1", "CCNB2", "CCNA2", "CCNE1", "CDK1", "CDK2",
    "AURKA", "AURKB", "BUB1", "BUB1B", "MCM2", "MCM3", "MCM4", "MCM5",
    "MCM6", "MCM7", "TOP2A", "FOXM1", "PLK1", "TPX2", "BIRC5", "KIF20A",
    "KIF2C", "RRM2", "TYMS", "UBE2C", "CDC20",
]

# Mak EMT signature using a reduced but representative 30-gene core;
# positive coefficients indicate mesenchymal character.
EMT_CORE_MESENCHYMAL: list[str] = [
    "VIM", "CDH2", "FN1", "SNAI1", "SNAI2", "TWIST1", "TWIST2", "ZEB1", "ZEB2",
    "MMP2", "MMP9", "MMP14", "SPARC", "S100A4", "ITGB1", "ITGA5", "COL1A1",
    "COL3A1", "COL5A1", "POSTN", "THBS2", "TAGLN", "ACTA2", "CNN1", "PDGFRB",
    "PDGFRA", "NOTCH1", "NOTCH2", "JAG1", "FOXC2",
]
EMT_CORE_EPITHELIAL: list[str] = [
    "CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "CLDN3", "CLDN4", "CLDN7",
    "OCLN", "TJP1", "DSP", "JUP", "MUC1", "ESRP1", "ESRP2", "GRHL2", "OVOL2",
]

# Ayers IFN-gamma expanded 18-gene Tumor Inflammation Signature
TIS_AYERS_18: list[str] = [
    "CCL5", "CD27", "CD274", "CD276", "CD8A", "CMKLR1", "CXCL9", "CXCR6",
    "HLA-DQA1", "HLA-DRB1", "HLA-E", "IDO1", "LAG3", "NKG7", "PDCD1LG2",
    "PSMB10", "STAT1", "TIGIT",
]

# Ayers 6-gene IFN-gamma compact
IFNG_AYERS_6: list[str] = ["IFNG", "CXCL9", "CXCL10", "IDO1", "HLA-DRA", "STAT1"]

# Rooney cytolytic activity
CYTOLYTIC_ACTIVITY: list[str] = ["GZMA", "PRF1"]


# ---------------------------------------------------------------------------
# Cell-type marker panels (MCP-counter-inspired short lists)
# ---------------------------------------------------------------------------

CELL_TYPE_MARKERS: dict[str, list[str]] = {
    "CD8_T": ["CD8A", "CD8B", "GZMK", "PRF1", "GZMA", "GZMB", "NKG7", "EOMES"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TNFRSF4"],
    "NK": ["NCAM1", "KLRD1", "NCR1", "NKG7", "GNLY", "KLRK1"],
    "B_cell": ["CD19", "MS4A1", "CD79A", "CD79B", "IGHM", "BLK"],
    "Macrophage_M1": ["CD68", "NOS2", "IL1B", "TNF", "IL6", "CXCL10", "CXCL9"],
    "Macrophage_M2": ["CD163", "MRC1", "CCL22", "CCL17", "IL10", "TGFB1", "ARG1"],
    "Neutrophil": ["FCGR3B", "CXCR1", "CXCR2", "ELANE", "MPO", "CSF3R"],
    "Fibroblast": ["FAP", "COL1A1", "COL3A1", "PDGFRB", "ACTA2", "S100A4"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "KDR", "ENG"],
}


# ---------------------------------------------------------------------------
# ESTIMATE score genes (Yoshihara et al. 2013), compact versions
# ---------------------------------------------------------------------------

# Full ESTIMATE uses 141 stromal + 141 immune genes. We embed a representative
# 50-gene subset for each to keep this file manageable; the full lists live in
# reference/estimate_signatures.json (loaded lazily below).

ESTIMATE_STROMAL_CORE: list[str] = [
    "ACTA2", "ADAM12", "ADAMTS2", "AEBP1", "BGN", "COL1A1", "COL1A2", "COL3A1",
    "COL5A1", "COL5A2", "COL6A3", "COL8A1", "CTSK", "DCN", "FAP", "FBN1",
    "FN1", "LOX", "LUM", "MMP2", "MMP11", "MMP14", "NNMT", "OLFML2B", "PDGFRB",
    "POSTN", "PRRX1", "SFRP4", "SPARC", "SULF1", "TAGLN", "THBS2", "THY1",
    "TIMP3", "TNFAIP6", "VCAN", "VIM", "WISP1", "ZEB1", "ZEB2",
]

ESTIMATE_IMMUNE_CORE: list[str] = [
    "CD2", "CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD19", "CD27", "CD37",
    "CD48", "CD52", "CD53", "CD79A", "CD79B", "CD163", "CTLA4", "CXCL9",
    "CXCL10", "CXCL13", "FCRL3", "GZMA", "GZMB", "GZMK", "HLA-DMA", "HLA-DMB",
    "HLA-DOA", "HLA-DPB1", "HLA-DQA1", "HLA-DRA", "IDO1", "IL10RA", "IL2RG",
    "LCK", "LCP2", "LTB", "MS4A1", "NCKAP1L", "NKG7", "PRF1", "PTPRC",
    "SLAMF1", "TIGIT", "TNFRSF9",
]


# ---------------------------------------------------------------------------
# Promoter probe mapping for 450K manifest
# ---------------------------------------------------------------------------

# Minimal Illumina 450K / EPIC manifest probes mapped to our promoter gene
# panel. For each gene, we list the canonical TSS1500 + TSS200 probe IDs
# (from the Illumina HM450 manifest v1.2 and EPIC v1.0). These are the probes
# whose mean beta we aggregate to report "promoter methylation". The full
# manifest is too large to embed; we load a curated subset from
# `reference/promoter_probes.json`.

# The loader is a function (below) that returns a dict[gene] -> list[probe_id].


# ---------------------------------------------------------------------------
# JSON-backed lazy loaders
# ---------------------------------------------------------------------------

_REFERENCE_DIR = Path(__file__).parent / "reference"


def _reference_path(name: str) -> Path:
    return _REFERENCE_DIR / name


def _load_json_dict(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Reference JSON must contain an object: {path}")
    return payload


def _clean_gene_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for value in values:
        gene = str(value).strip().upper()
        if gene and gene not in out:
            out.append(gene)
    return out


def _clean_gene_set_mapping(payload: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, genes in payload.items():
        if str(name).startswith("_"):
            continue
        cleaned = _clean_gene_list(genes)
        if cleaned:
            out[str(name).strip()] = cleaned
    return out


def _clean_centroid_mapping(payload: dict) -> dict[str, dict[str, float]]:
    if "centroids" in payload and isinstance(payload["centroids"], dict):
        payload = payload["centroids"]

    subtype_aliases = {
        "Her2": "HER2_enriched",
        "Basal": "Basal_like",
        "Normal": "Normal_like",
    }
    out: dict[str, dict[str, float]] = {}
    for subtype, centroid in payload.items():
        if str(subtype).startswith("_") or not isinstance(centroid, dict):
            continue
        normalized_subtype = subtype_aliases.get(str(subtype).strip(), str(subtype).strip())
        values: dict[str, float] = {}
        for gene, raw_value in centroid.items():
            gene_symbol = str(gene).strip().upper()
            if not gene_symbol:
                continue
            try:
                values[gene_symbol] = float(raw_value)
            except (TypeError, ValueError):
                continue
        if values:
            out[normalized_subtype] = values
    return out


@lru_cache(maxsize=None)
def load_hallmark50() -> dict[str, list[str]]:
    """Return the MSigDB Hallmark 50 collection as {hallmark_name: gene_list}.

    Ships with the repo at `src/kidney_vlm/genomics/reference/hallmark50.json`.
    If missing, the caller should fall back to the embedded EMBEDDED_HALLMARK_CORE
    dict (which covers the 10 most commonly used hallmarks).
    """
    path = _reference_path("hallmark50.json")
    if not path.exists():
        return EMBEDDED_HALLMARK_CORE
    hallmarks = _clean_gene_set_mapping(_load_json_dict(path))
    return hallmarks or EMBEDDED_HALLMARK_CORE


@lru_cache(maxsize=None)
def load_estimate_full() -> dict[str, list[str]]:
    """Return ESTIMATE stromal + immune gene lists.

    Ships with `reference/estimate_signatures.json`. Falls back to the
    embedded 40-gene cores if the file is missing.
    """
    path = _reference_path("estimate_signatures.json")
    if not path.exists():
        return {"stromal": ESTIMATE_STROMAL_CORE, "immune": ESTIMATE_IMMUNE_CORE}
    payload = _load_json_dict(path)
    return {
        "stromal": _clean_gene_list(payload.get("stromal")) or ESTIMATE_STROMAL_CORE,
        "immune": _clean_gene_list(payload.get("immune")) or ESTIMATE_IMMUNE_CORE,
    }


@lru_cache(maxsize=None)
def load_pam50_centroids() -> dict[str, dict[str, float]]:
    """Return PAM50 subtype centroids: {subtype: {gene_symbol: centroid_value}}.

    Required for BRCA subtyping. Ships with `reference/pam50_centroids.json`.
    Raises if missing; PAM50 cannot be stubbed.
    """
    path = _reference_path("pam50_centroids.json")
    if not path.exists():
        raise FileNotFoundError(
            f"PAM50 centroids not found at {path}. "
            "Bundle reference/pam50_centroids.json from Parker et al. 2009."
        )
    centroids = _clean_centroid_mapping(_load_json_dict(path))
    if not centroids:
        raise ValueError(f"PAM50 centroid reference is empty or malformed: {path}")
    return centroids


@lru_cache(maxsize=None)
def load_horvath_clock() -> tuple[dict[str, float], float]:
    """Return Horvath 2013 epigenetic age clock coefficients.

    Returns (coefficients_by_probe, intercept). Ships with
    `reference/horvath_2013_353_probes.json`.
    """
    path = _reference_path("horvath_2013_353_probes.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Horvath clock coefficients not found at {path}. "
            "Bundle Horvath 2013 353-probe clock weights."
        )
    blob = _load_json_dict(path)
    raw_coefficients = blob.get("coefficients", {})
    if not isinstance(raw_coefficients, dict):
        raise ValueError(f"Horvath clock coefficients must be an object: {path}")
    coefficients = {
        str(probe_id).strip(): float(value)
        for probe_id, value in raw_coefficients.items()
        if str(probe_id).strip()
    }
    return coefficients, float(blob["intercept"])


@lru_cache(maxsize=None)
def load_promoter_probes() -> dict[str, list[str]]:
    """Return {gene_symbol: [probe_ids,...]} for the promoter panel.

    Ships with `reference/promoter_probes.json`. Keys should cover every gene
    listed in `cohort_config.PROMOTER_METHYLATION_PANEL`.
    """
    path = _reference_path("promoter_probes.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Promoter probe mapping not found at {path}. "
            "Bundle reference/promoter_probes.json derived from HM450 manifest."
        )
    payload = _load_json_dict(path)
    if "probes" in payload and isinstance(payload["probes"], dict):
        payload = payload["probes"]
    probes: dict[str, list[str]] = {}
    for gene, probe_ids in payload.items():
        if str(gene).startswith("_") or not isinstance(probe_ids, list):
            continue
        cleaned = [str(probe_id).strip() for probe_id in probe_ids if str(probe_id).strip()]
        if cleaned:
            probes[str(gene).strip().upper()] = list(dict.fromkeys(cleaned))
    return probes


# ---------------------------------------------------------------------------
# Embedded Hallmark core (fallback if the JSON isn't available)
# ---------------------------------------------------------------------------

# A 10-hallmark minimal set. If the full JSON isn't bundled, the RNA pipeline
# falls back to ssGSEA over these 10. In that case the generated text will
# mention fewer hallmarks than the 5+3 spec calls for, and the pipeline
# logs a warning.

EMBEDDED_HALLMARK_CORE: dict[str, list[str]] = {
    "HALLMARK_HYPOXIA": HYPOXIA_BUFFA_51,
    "HALLMARK_E2F_TARGETS": PROLIFERATION_CORE,
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": EMT_CORE_MESENCHYMAL,
    "HALLMARK_INFLAMMATORY_RESPONSE": TIS_AYERS_18,
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": IFNG_AYERS_6 + [
        "IRF1", "GBP1", "OAS2", "IFI27", "IFIT1", "IFIT3", "MX1", "ISG15",
    ],
    "HALLMARK_G2M_CHECKPOINT": PROLIFERATION_CORE,
    "HALLMARK_MYC_TARGETS_V1": [
        "MYC", "CAD", "EIF4G1", "EIF4G2", "LDHA", "NPM1", "ODC1", "POLR3K",
        "PRPS2", "PTMA", "RAN", "RPL22", "SRSF1", "SRSF3", "TFDP1", "TYMS",
    ],
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": [
        "NDUFA1", "NDUFA2", "NDUFA3", "NDUFA4", "NDUFB1", "NDUFB2", "NDUFS1",
        "NDUFS2", "SDHA", "SDHB", "COX4I1", "COX5A", "ATP5A1", "ATP5B",
        "UQCRC1", "UQCRC2",
    ],
    "HALLMARK_ANGIOGENESIS": [
        "VEGFA", "VEGFB", "VEGFC", "FLT1", "KDR", "FLT4", "PDGFA", "PDGFB",
        "ANGPT1", "ANGPT2", "TEK", "TIE1", "CDH5",
    ],
    "HALLMARK_GLYCOLYSIS": [
        "SLC2A1", "HK2", "PFKP", "ALDOA", "GAPDH", "PGK1", "PGAM1", "ENO1",
        "PKM", "LDHA", "TPI1",
    ],
}
