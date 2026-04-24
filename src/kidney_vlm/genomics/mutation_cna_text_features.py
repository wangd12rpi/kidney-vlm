"""
Mutation, copy-number, TMB, MSI, and HRD text-feature extraction.

These features fall under the "text-channel" category in the hybrid genomics
design: the student cannot recover them from cpGPT or BulkFormer embeddings,
so they are serialized as structured text and shown to both the teacher LLM
at generation time and the student at training/inference time.

Inputs:
    * Masked MAF file (from GDC; MuTect2/VarScan2/MuSE/Pindel union or MC3)
    * Gene-level GISTIC2 CNA TSV (values: -2/-1/0/+1/+2)
    * Allele-specific / masked copy-number segment TSV (for arm-level calls)
    * Optional MSI and HRD scores (from GDC clinical / supplementary data)

Outputs:
    {
        "mutations": [{"gene": "VHL", "protein_change": "p.R167W",
                       "variant_class": "Missense_Mutation",
                       "oncokb_level": "Level_1"}, ...],
        "wild_type_calls": ["TP53", "BAP1", ...],
        "focal_cnas": {"CDKN2A": -2, "MYC": +1, ...},
        "arm_level_cnas": ["3p_loss", "5q_gain", ...],
        "structural_rearrangements_dna": "none_detected" | [...],
        "tmb_mutations_per_mb": 1.8,
        "msi_status": "MSS" | "MSI-L" | "MSI-H" | "not_assessed",
        "hrd_score": 42.0,
        "ncrna_findings": ["miR-210_upregulation_hypoxia_associated", ...],
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kidney_vlm.genomics import cohort_config as cohort_cfg


# ---------------------------------------------------------------------------
# MAF parsing
# ---------------------------------------------------------------------------

# Variant classifications we treat as "non-silent" (protein-altering) for TMB
# and panel reporting. Silent, intronic, flanking, and 3'/5' UTR variants are
# excluded.
NONSILENT_VARIANT_CLASSES: set[str] = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Splice_Region",
    "Translation_Start_Site",
}


# Minimal OncoKB-style hotspot map. The real OncoKB annotation is API-based;
# for pipeline simplicity we carry a small curated map of well-established
# Level-1 / Level-2 hotspots. Upgrade by hitting the OncoKB API at runtime
# and joining on (gene, protein_change) if a key is available.
ONCOKB_HOTSPOTS: dict[tuple[str, str], str] = {
    ("BRAF", "p.V600E"): "Level_1",
    ("BRAF", "p.V600K"): "Level_1",
    ("EGFR", "p.L858R"): "Level_1",
    ("EGFR", "p.T790M"): "Level_1",
    ("KRAS", "p.G12D"): "Level_1",
    ("KRAS", "p.G12C"): "Level_1",
    ("KRAS", "p.G12V"): "Level_1",
    ("KRAS", "p.G13D"): "Level_1",
    ("PIK3CA", "p.H1047R"): "Level_1",
    ("PIK3CA", "p.E545K"): "Level_1",
    ("PIK3CA", "p.E542K"): "Level_1",
    ("IDH1", "p.R132H"): "Level_1",
    ("IDH1", "p.R132C"): "Level_1",
    ("IDH2", "p.R172K"): "Level_1",
    ("TP53", "p.R175H"): "Level_2",
    ("TP53", "p.R248Q"): "Level_2",
    ("TP53", "p.R248W"): "Level_2",
    ("TP53", "p.R273H"): "Level_2",
    ("TP53", "p.R273C"): "Level_2",
    ("VHL", "p.R167W"): "Level_2",
    ("VHL", "p.R167Q"): "Level_2",
    ("NRAS", "p.Q61R"): "Level_1",
    ("NRAS", "p.Q61K"): "Level_1",
    ("HRAS", "p.Q61R"): "Level_2",
    ("AR", "p.T878A"): "Level_2",
    ("AR", "p.L702H"): "Level_2",
    ("FGFR3", "p.S249C"): "Level_1",
    ("FGFR3", "p.R248C"): "Level_1",
    ("ERBB2", "p.L755S"): "Level_2",
    ("ERBB2", "p.V777L"): "Level_2",
    ("SPOP", "p.F133V"): "Level_2",
    ("SPOP", "p.F133L"): "Level_2",
}


def _looks_like_loss_of_function(variant_class: str) -> bool:
    return variant_class in {
        "Nonsense_Mutation",
        "Nonstop_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "Splice_Site",
        "Translation_Start_Site",
    }


def parse_maf(maf_path: str | Path, panel_genes: set[str]) -> pd.DataFrame:
    """Return a DataFrame of non-silent mutations in `panel_genes`.

    Columns:
        gene (Hugo_Symbol), variant_class, protein_change, chromosome, start,
        end, reference_allele, tumor_allele, is_lof, oncokb_level.
    """
    path = Path(maf_path)
    # Detect whether the MAF is gzipped
    read_kwargs: dict[str, Any] = {"sep": "\t", "comment": "#", "low_memory": False}
    if path.suffix.lower() == ".gz":
        read_kwargs["compression"] = "gzip"

    df = pd.read_csv(path, **read_kwargs)
    required = {"Hugo_Symbol", "Variant_Classification"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"MAF file missing columns {missing}: {path}")

    df = df[df["Hugo_Symbol"].astype(str).isin(panel_genes)].copy()
    df = df[df["Variant_Classification"].astype(str).isin(NONSILENT_VARIANT_CLASSES)].copy()

    # Standardize HGVSp column
    hgvsp_col = None
    for candidate in ("HGVSp_Short", "HGVSp", "HGVSc", "Amino_acids"):
        if candidate in df.columns:
            hgvsp_col = candidate
            break
    df["protein_change"] = (
        df[hgvsp_col].astype(str).str.strip() if hgvsp_col is not None else ""
    )

    df["is_lof"] = df["Variant_Classification"].map(_looks_like_loss_of_function)
    df["oncokb_level"] = df.apply(
        lambda row: ONCOKB_HOTSPOTS.get(
            (str(row["Hugo_Symbol"]).strip(), str(row.get("protein_change", "")).strip()),
            "",
        ),
        axis=1,
    )

    keep_cols = [
        "Hugo_Symbol", "Variant_Classification", "protein_change",
        "is_lof", "oncokb_level",
    ]
    optional_cols = ["Chromosome", "Start_Position", "End_Position",
                     "Reference_Allele", "Tumor_Seq_Allele2"]
    for c in optional_cols:
        if c in df.columns:
            keep_cols.append(c)
    return df[keep_cols].rename(columns={"Hugo_Symbol": "gene",
                                          "Variant_Classification": "variant_class"})


def panel_mutation_summary(maf_df: pd.DataFrame, panel_genes: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (mutations_serialized, wild_type_calls).

    Each mutation dict has keys: gene, protein_change, variant_class,
    oncokb_level, is_lof. Genes in the panel with no non-silent variants
    appear in `wild_type_calls`.
    """
    mutations: list[dict[str, Any]] = []
    mutated_genes: set[str] = set()
    for row in maf_df.itertuples(index=False):
        gene = str(row.gene).strip()
        mutated_genes.add(gene)
        mutations.append(
            {
                "gene": gene,
                "protein_change": str(row.protein_change).strip(),
                "variant_class": str(row.variant_class).strip(),
                "oncokb_level": str(row.oncokb_level).strip(),
                "is_lof": bool(row.is_lof),
            }
        )
    wild_type = [g for g in panel_genes if g not in mutated_genes]
    return mutations, wild_type


def compute_tmb_per_mb(maf_df_full: pd.DataFrame, panel_size_mb: float = 38.0) -> float:
    """Crude TMB estimate from a MAF.

    `maf_df_full` should be the full MAF for this case (not filtered to the
    cohort panel). panel_size_mb is the sequenced footprint; WXS default
    is ~38 Mb. For TMB-from-panel estimation the value will differ; update
    if using a targeted panel.
    """
    if maf_df_full.empty:
        return 0.0
    nonsilent = maf_df_full[
        maf_df_full["Variant_Classification"].astype(str).isin(NONSILENT_VARIANT_CLASSES)
    ]
    n = len(nonsilent)
    return float(n) / float(panel_size_mb)


# ---------------------------------------------------------------------------
# Copy-number parsing
# ---------------------------------------------------------------------------


def parse_gistic_gene_level(cna_tsv_path: str | Path, panel_genes: list[str]) -> dict[str, int]:
    """Return focal CN calls {gene: -2|-1|0|+1|+2} for panel genes.

    Supports both GISTIC2 gene-level output and ASCAT2 gene-level output.
    GDC gene-level CN TSV typically has columns:
        gene_id, gene_name, chromosome, start, end, copy_number (integer)
        OR
        Gene Symbol, Locus ID, Cytoband, <sample_barcode>
    """
    path = Path(cna_tsv_path)
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)

    # Identify gene-symbol column
    gene_col = None
    for cand in ("gene_name", "Gene Symbol", "Hugo_Symbol", "symbol", "Gene"):
        if cand in df.columns:
            gene_col = cand
            break
    if gene_col is None:
        raise ValueError(f"Gene-level CN TSV missing a gene-symbol column: {path}")

    # Identify value column
    value_col = None
    for cand in ("copy_number", "GISTIC", "seg_mean", "min_copy_number", "max_copy_number"):
        if cand in df.columns:
            value_col = cand
            break
    if value_col is None:
        # Assume the last non-metadata column is the sample's value
        candidate_cols = [c for c in df.columns if c not in {gene_col, "Locus ID", "Cytoband", "chromosome", "start", "end"}]
        if not candidate_cols:
            raise ValueError(f"Gene-level CN TSV missing a value column: {path}")
        value_col = candidate_cols[0]

    df = df[[gene_col, value_col]].copy()
    df.columns = ["gene", "value"]
    df["gene"] = df["gene"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Convert to GISTIC2-style discrete calls if the input is continuous
    if df["value"].abs().max() > 2.5:
        # Looks like raw integer copy number; convert assuming diploid ref = 2
        df["call"] = df["value"].apply(_integer_cn_to_gistic_call)
    elif df["value"].abs().max() > 1.5:
        # Already in -2..+2 range
        df["call"] = df["value"].round().clip(-2, 2).astype("Int64")
    else:
        # Continuous log-ratio; use Yoshihara/ASCAT-style cutoffs
        df["call"] = df["value"].apply(_log_ratio_to_gistic_call)

    panel_set = set(panel_genes)
    sub = df[df["gene"].isin(panel_set)].copy()
    sub = sub.drop_duplicates(subset=["gene"], keep="first")
    out: dict[str, int] = {
        row.gene: int(row.call) if pd.notna(row.call) else 0
        for row in sub.itertuples(index=False)
    }
    # Genes in panel but not in file get 0 (neutral)
    for gene in panel_genes:
        out.setdefault(gene, 0)
    return out


def _integer_cn_to_gistic_call(value: float) -> int:
    if not np.isfinite(value):
        return 0
    if value <= 0:
        return -2
    if value == 1:
        return -1
    if value == 2:
        return 0
    if value == 3:
        return 1
    return 2


def _log_ratio_to_gistic_call(value: float) -> int:
    if not np.isfinite(value):
        return 0
    if value <= -1.0:
        return -2
    if value <= -0.3:
        return -1
    if value >= 1.0:
        return 2
    if value >= 0.3:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Arm-level CN from segment file
# ---------------------------------------------------------------------------


# Chromosome-arm coordinates (hg38). Approximate arm boundaries via
# centromere positions. These are used to aggregate segment calls into
# arm-level gain/loss events.
CHROMOSOME_ARMS_HG38: dict[str, tuple[int, int, int]] = {
    # chr: (p_end, q_start, q_end) - positions in bp
    "1": (122_026_459, 125_184_587, 248_956_422),
    "2": (92_188_145, 94_090_557, 242_193_529),
    "3": (90_772_458, 93_655_574, 198_295_559),
    "4": (49_708_101, 51_743_951, 190_214_555),
    "5": (46_485_900, 50_059_807, 181_538_259),
    "6": (58_553_888, 59_829_934, 170_805_979),
    "7": (58_169_653, 61_828_234, 159_345_973),
    "8": (44_033_744, 47_193_556, 145_138_636),
    "9": (43_236_168, 45_518_558, 138_394_717),
    "10": (39_686_682, 41_593_521, 133_797_422),
    "11": (51_078_348, 54_425_074, 135_086_622),
    "12": (34_769_407, 37_185_252, 133_275_309),
    "13": (16_000_000, 17_700_000, 114_364_328),
    "14": (16_000_000, 17_200_000, 107_043_718),
    "15": (17_083_673, 19_725_254, 101_991_189),
    "16": (36_311_158, 38_265_669, 90_338_345),
    "17": (22_813_679, 26_616_164, 83_257_441),
    "18": (15_460_898, 20_861_206, 80_373_285),
    "19": (24_631_782, 27_190_874, 58_617_616),
    "20": (26_369_569, 28_494_539, 64_444_167),
    "21": (11_288_129, 12_915_808, 46_709_983),
    "22": (13_285_178, 15_054_318, 50_818_468),
    "X": (58_605_498, 62_412_542, 156_040_895),
}


def arm_level_cna_from_segments(
    segment_tsv_path: str | Path,
    project_id: str,
    *,
    gain_threshold: float = 0.2,
    loss_threshold: float = -0.2,
    arm_coverage_threshold: float = 0.5,
) -> list[str]:
    """Aggregate segment-level CN calls into arm-level gain/loss events.

    Returns event strings like "3p_loss", "5q_gain" filtered to the cohort's
    panel in `ARM_LEVEL_CNA_PANEL`. Events outside the panel are dropped to
    keep the text block lean.
    """
    path = Path(segment_tsv_path)
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    rename = {}
    for cand in ("Chromosome", "chromosome"):
        if cand in df.columns:
            rename[cand] = "chromosome"
            break
    for cand in ("Start", "Start_Position", "start"):
        if cand in df.columns:
            rename[cand] = "start"
            break
    for cand in ("End", "End_Position", "end"):
        if cand in df.columns:
            rename[cand] = "end"
            break
    for cand in ("Segment_Mean", "seg.mean", "seg_mean", "Copy_Number"):
        if cand in df.columns:
            rename[cand] = "value"
            break
    df = df.rename(columns=rename)
    needed = {"chromosome", "start", "end", "value"}
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f"Segment TSV missing columns {missing}: {path}")

    df["chromosome"] = df["chromosome"].astype(str).str.replace("chr", "", regex=False)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["chromosome", "start", "end", "value"])

    detected_events: list[str] = []
    for chrom, (p_end, q_start, q_end) in CHROMOSOME_ARMS_HG38.items():
        sub = df[df["chromosome"] == chrom]
        if sub.empty:
            continue
        for arm, arm_start, arm_end in [("p", 1, p_end), ("q", q_start, q_end)]:
            arm_length = arm_end - arm_start
            if arm_length <= 0:
                continue

            # Weighted mean segment value restricted to the arm
            overlap_total = 0
            weighted_sum = 0.0
            for seg in sub.itertuples(index=False):
                seg_start = max(int(seg.start), arm_start)
                seg_end = min(int(seg.end), arm_end)
                overlap = max(0, seg_end - seg_start)
                if overlap == 0:
                    continue
                overlap_total += overlap
                weighted_sum += overlap * float(seg.value)

            if overlap_total / arm_length < arm_coverage_threshold:
                continue
            arm_mean = weighted_sum / overlap_total
            if arm_mean >= gain_threshold:
                detected_events.append(f"{chrom}{arm}_gain")
            elif arm_mean <= loss_threshold:
                detected_events.append(f"{chrom}{arm}_loss")

    cohort_panel = set(cohort_cfg.ARM_LEVEL_CNA_PANEL.get(project_id, []))
    if cohort_panel:
        detected_events = [ev for ev in detected_events if ev in cohort_panel]
    return sorted(set(detected_events))


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------


@dataclass
class MutationCnaTextFeatures:
    mutations: list[dict[str, Any]]
    wild_type_calls: list[str]
    focal_cnas: dict[str, int]
    arm_level_cnas: list[str]
    structural_rearrangements_dna: list[str]
    tmb_mutations_per_mb: float
    msi_status: str
    hrd_score: float
    ncrna_findings: list[str]
    mutation_panel: list[str]
    focal_cna_panel: list[str]
    arm_level_panel: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutations": self.mutations,
            "wild_type_calls": self.wild_type_calls,
            "focal_cnas": self.focal_cnas,
            "arm_level_cnas": self.arm_level_cnas,
            "structural_rearrangements_dna": self.structural_rearrangements_dna,
            "tmb_mutations_per_mb": self.tmb_mutations_per_mb,
            "msi_status": self.msi_status,
            "hrd_score": self.hrd_score,
            "ncrna_findings": self.ncrna_findings,
            "mutation_panel": self.mutation_panel,
            "focal_cna_panel": self.focal_cna_panel,
            "arm_level_panel": self.arm_level_panel,
        }


def extract_mutation_cna_text_features(
    *,
    maf_path: str | Path | None,
    gene_cna_path: str | Path | None,
    segment_cna_path: str | Path | None,
    project_id: str,
    msi_status: str | None = None,
    hrd_score: float | None = None,
    structural_rearrangements_dna: list[str] | None = None,
    ncrna_findings: list[str] | None = None,
) -> MutationCnaTextFeatures:
    mutation_panel = cohort_cfg.get_mutation_panel(project_id)
    focal_panel = cohort_cfg.FOCAL_CNA_PANEL.get(project_id, [])
    arm_panel = cohort_cfg.ARM_LEVEL_CNA_PANEL.get(project_id, [])

    # Mutations
    if maf_path is not None and Path(maf_path).exists():
        maf_full = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)
        panel_maf = parse_maf(maf_path, panel_genes=set(mutation_panel))
        mutations, wild_type = panel_mutation_summary(panel_maf, mutation_panel)
        tmb = compute_tmb_per_mb(maf_full)
    else:
        mutations = []
        wild_type = []
        tmb = float("nan")

    # Focal CNAs
    if gene_cna_path is not None and Path(gene_cna_path).exists():
        focal = parse_gistic_gene_level(gene_cna_path, panel_genes=focal_panel)
    else:
        focal = {gene: 0 for gene in focal_panel}

    # Arm-level CNAs
    if segment_cna_path is not None and Path(segment_cna_path).exists():
        arm_events = arm_level_cna_from_segments(segment_cna_path, project_id)
    else:
        arm_events = []

    return MutationCnaTextFeatures(
        mutations=mutations,
        wild_type_calls=wild_type,
        focal_cnas=focal,
        arm_level_cnas=arm_events,
        structural_rearrangements_dna=structural_rearrangements_dna or [],
        tmb_mutations_per_mb=tmb,
        msi_status=(msi_status or "not_assessed").strip(),
        hrd_score=float(hrd_score) if hrd_score is not None else float("nan"),
        ncrna_findings=ncrna_findings or [],
        mutation_panel=mutation_panel,
        focal_cna_panel=focal_panel,
        arm_level_panel=arm_panel,
    )
