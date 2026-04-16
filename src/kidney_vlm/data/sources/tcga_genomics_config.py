from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


ALL_TCGA_PROJECTS = [
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
]

PANCAN_URLS = {
    "mc3_maf": "https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc",
    "subtypes": "https://api.gdc.cancer.gov/data/0f31b768-7f67-4fc4-abc3-06ac5bd90bf0",
    "gistic_thresholded": "https://api.gdc.cancer.gov/data/7d64377f-2cea-4ee3-917f-8fcfbcd999e7",
    "leukocyte_fraction": "https://api.gdc.cancer.gov/data/6f75c9d7-5134-4ed1-b8f3-72856c98a4e8",
    "cibersort": "https://api.gdc.cancer.gov/data/b3df502e-3594-46ef-9f94-d041a20a0b9a",
    "mutation_load": "https://api.gdc.cancer.gov/data/ff3f962c-3573-44ae-a8f4-e5ac0aea64b6",
    "absolute_scores": "https://api.gdc.cancer.gov/data/0e8831f4-dd7e-4673-8624-b4519c2e0d65",
    "absolute_purity": "https://api.gdc.cancer.gov/data/4f277128-f793-4354-a13d-30cc7fe9f6b5",
    "hrd_scores": "https://api.gdc.cancer.gov/data/66dd07d7-6366-4774-83c3-5ad1e22b177e",
    "clinical": "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81",
}

TEN_PATHWAYS = [
    "RTK_RAS",
    "PI3K",
    "TP53",
    "Cell_Cycle",
    "WNT",
    "HIPPO",
    "MYC",
    "NOTCH",
    "NRF2",
    "TGF_Beta",
]

NONSYNONYMOUS_CLASSES = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Nonstop_Mutation",
    "Translation_Start_Site",
}

MSI_THRESHOLD_HIGH = 3.5
MSI_THRESHOLD_LOW = 1.0


@dataclass
class CancerConfig:
    project: str
    cancer_code: str
    driver_genes: List[str]
    cna_genes: List[str]
    arm_level_events: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    hotspot_variants: Dict[str, List[str]] = field(default_factory=dict)
    special_fields_fn: Optional[str] = None
    msi_relevant: bool = False
    notes: str = ""


CANCER_CONFIGS: Dict[str, CancerConfig] = {}

CANCER_CONFIGS["KIRC"] = CancerConfig(
    project="TCGA-KIRC",
    cancer_code="KIRC",
    driver_genes=[
        "VHL",
        "PBRM1",
        "BAP1",
        "SETD2",
        "KDM5C",
        "PTEN",
        "MTOR",
        "TP53",
        "PIK3CA",
        "TCEB1",
        "ARID1A",
        "SMARCA4",
        "NFE2L2",
        "STAG2",
        "TSC1",
        "TSC2",
        "MET",
        "KDM6A",
    ],
    cna_genes=["VHL", "CDKN2A", "MYC", "MDM4", "SQSTM1"],
    arm_level_events={"chr3p_loss": [("chr3", "p_loss")]},
    special_fields_fn="special_kirc",
)

CANCER_CONFIGS["KIRP"] = CancerConfig(
    project="TCGA-KIRP",
    cancer_code="KIRP",
    driver_genes=[
        "MET",
        "SETD2",
        "NF2",
        "STAG2",
        "NFE2L2",
        "FAT1",
        "BAP1",
        "PBRM1",
        "KDM6A",
        "SMARCB1",
        "TP53",
        "CDKN2A",
        "FH",
        "ARID2",
        "KMT2C",
        "KMT2D",
        "CUL3",
        "KEAP1",
        "TERT",
    ],
    cna_genes=["MET", "CDKN2A", "CIITA", "NF2"],
    arm_level_events={
        "chr7_gain": [("chr7", "gain")],
        "chr17_gain": [("chr17", "gain")],
    },
    special_fields_fn="special_kirp",
)

CANCER_CONFIGS["KICH"] = CancerConfig(
    project="TCGA-KICH",
    cancer_code="KICH",
    driver_genes=["TP53", "PTEN", "TERT", "MTOR", "TSC1", "TSC2", "CDKN2A", "FOXI1"],
    cna_genes=[],
    arm_level_events={f"chr{chromosome}_loss": [(f"chr{chromosome}", "whole_loss")] for chromosome in [1, 2, 6, 10, 13, 17, 21]},
    special_fields_fn="special_kich",
    notes="Small cohort (n~66). Genomically quiet.",
)

CANCER_CONFIGS["BRCA"] = CancerConfig(
    project="TCGA-BRCA",
    cancer_code="BRCA",
    driver_genes=[
        "PIK3CA",
        "TP53",
        "GATA3",
        "CDH1",
        "MAP3K1",
        "MAP2K4",
        "AKT1",
        "PTEN",
        "CBFB",
        "RB1",
        "RUNX1",
        "ERBB2",
        "KMT2C",
        "TBX3",
        "FOXA1",
        "PIK3R1",
        "ARID1A",
        "NF1",
        "SF3B1",
        "NCOR1",
        "CTCF",
        "MED12",
        "AKT2",
        "BRCA1",
        "BRCA2",
    ],
    cna_genes=[
        "ERBB2",
        "MYC",
        "CCND1",
        "FGFR1",
        "PIK3CA",
        "CDKN2A",
        "PTEN",
        "RB1",
        "MAP2K4",
        "MDM2",
    ],
    hotspot_variants={"PIK3CA": ["E545K", "H1047R", "E542K"]},
    special_fields_fn="special_brca",
)

CANCER_CONFIGS["PRAD"] = CancerConfig(
    project="TCGA-PRAD",
    cancer_code="PRAD",
    driver_genes=[
        "SPOP",
        "TP53",
        "FOXA1",
        "IDH1",
        "PTEN",
        "MED12",
        "CDKN1B",
        "ATM",
        "BRCA2",
        "BRCA1",
        "CDK12",
        "APC",
        "CTNNB1",
        "PIK3CA",
        "PIK3CB",
        "RB1",
        "KMT2C",
        "KMT2D",
        "KDM6A",
        "AKT1",
    ],
    cna_genes=["PTEN", "RB1", "TP53", "MYC", "AR", "CHD1", "NKX3-1", "CDKN2A", "ERG"],
    special_fields_fn="special_prad",
)

CANCER_CONFIGS["LUAD"] = CancerConfig(
    project="TCGA-LUAD",
    cancer_code="LUAD",
    driver_genes=[
        "KRAS",
        "EGFR",
        "TP53",
        "STK11",
        "KEAP1",
        "NF1",
        "BRAF",
        "PIK3CA",
        "RBM10",
        "MGA",
        "MET",
        "ERBB2",
        "RB1",
        "CDKN2A",
        "SMARCA4",
        "SETD2",
        "ARID1A",
        "ATM",
        "APC",
        "U2AF1",
        "RIT1",
        "MAP2K1",
        "NRAS",
    ],
    cna_genes=["NKX2-1", "TERT", "MYC", "EGFR", "MET", "ERBB2", "FGFR1", "CDKN2A", "STK11"],
    hotspot_variants={
        "KRAS": ["G12C", "G12V", "G12D", "G12A", "G12S", "G13", "Q61"],
        "EGFR": ["L858R", "T790M", "G719", "L861Q", "S768I"],
        "BRAF": ["V600E"],
    },
    special_fields_fn="special_luad",
)

CANCER_CONFIGS["LUSC"] = CancerConfig(
    project="TCGA-LUSC",
    cancer_code="LUSC",
    driver_genes=[
        "TP53",
        "CDKN2A",
        "PTEN",
        "PIK3CA",
        "NFE2L2",
        "KEAP1",
        "RB1",
        "NOTCH1",
        "NOTCH2",
        "HLA-A",
        "FBXW7",
        "FAT1",
        "KMT2D",
        "HRAS",
        "NF1",
        "ARID1A",
        "EP300",
        "CREBBP",
        "NSD1",
    ],
    cna_genes=["SOX2", "PIK3CA", "TP63", "FGFR1", "NSD3", "CDKN2A", "PTEN", "RB1", "EGFR", "PDGFRA"],
    special_fields_fn="special_lusc",
)

CANCER_CONFIGS["GBM"] = CancerConfig(
    project="TCGA-GBM",
    cancer_code="GBM",
    driver_genes=[
        "IDH1",
        "IDH2",
        "TP53",
        "PTEN",
        "EGFR",
        "NF1",
        "RB1",
        "PIK3CA",
        "PIK3R1",
        "PDGFRA",
        "CDK4",
        "CDKN2A",
        "MDM2",
        "ATRX",
        "TERT",
        "CIC",
        "FUBP1",
        "H3F3A",
        "HIST1H3B",
    ],
    cna_genes=["EGFR", "PDGFRA", "CDK4", "MDM2", "CDKN2A", "CDKN2B", "PTEN", "RB1", "NF1"],
    arm_level_events={
        "chr7_gain": [("chr7", "gain")],
        "chr10_loss": [("chr10", "whole_loss")],
    },
    hotspot_variants={"IDH1": ["R132H", "R132C", "R132S", "R132G", "R132L"]},
    special_fields_fn="special_gbm",
)

CANCER_CONFIGS["LGG"] = CancerConfig(
    project="TCGA-LGG",
    cancer_code="LGG",
    driver_genes=[
        "IDH1",
        "IDH2",
        "TP53",
        "ATRX",
        "CIC",
        "FUBP1",
        "NOTCH1",
        "PIK3CA",
        "PIK3R1",
        "PTEN",
        "EGFR",
        "NF1",
        "SMARCA4",
        "CDKN2A",
        "H3F3A",
    ],
    cna_genes=["CDKN2A", "EGFR", "PDGFRA", "CDK4", "MDM2"],
    arm_level_events={"1p19q_codeletion": [("chr1", "p_loss"), ("chr19", "q_loss")]},
    hotspot_variants={"IDH1": ["R132H", "R132C", "R132S"]},
    special_fields_fn="special_lgg",
)

CANCER_CONFIGS["COAD"] = CancerConfig(
    project="TCGA-COAD",
    cancer_code="COAD",
    driver_genes=[
        "APC",
        "TP53",
        "KRAS",
        "PIK3CA",
        "FBXW7",
        "SMAD4",
        "NRAS",
        "BRAF",
        "TCF7L2",
        "SOX9",
        "ARID1A",
        "ACVR2A",
        "TGFBR2",
        "MSH6",
        "MSH3",
        "ATM",
        "PTEN",
        "AMER1",
        "CTNNB1",
        "RNF43",
    ],
    cna_genes=["MYC", "ERBB2", "EGFR", "SMAD4", "APC", "CDKN2A", "PTEN"],
    hotspot_variants={"BRAF": ["V600E"], "KRAS": ["G12", "G13", "Q61"]},
    special_fields_fn="special_coad",
    msi_relevant=True,
)

CANCER_CONFIGS["READ"] = CancerConfig(
    project="TCGA-READ",
    cancer_code="READ",
    driver_genes=CANCER_CONFIGS["COAD"].driver_genes,
    cna_genes=CANCER_CONFIGS["COAD"].cna_genes,
    hotspot_variants=CANCER_CONFIGS["COAD"].hotspot_variants,
    special_fields_fn="special_coad",
    msi_relevant=True,
    notes="Molecularly identical to COAD. Process as COADREAD.",
)

CANCER_CONFIGS["OV"] = CancerConfig(
    project="TCGA-OV",
    cancer_code="OV",
    driver_genes=["TP53", "BRCA1", "BRCA2", "NF1", "RB1", "CDK12", "CSMD3", "FAT3", "GABRA6"],
    cna_genes=["CCNE1", "MYC", "MECOM", "PIK3CA", "KRAS", "RB1", "NF1", "PTEN", "BRCA1", "BRCA2", "CDK12"],
    special_fields_fn="special_ov",
)

CANCER_CONFIGS["UCEC"] = CancerConfig(
    project="TCGA-UCEC",
    cancer_code="UCEC",
    driver_genes=[
        "PTEN",
        "PIK3CA",
        "PIK3R1",
        "CTNNB1",
        "ARID1A",
        "TP53",
        "KRAS",
        "FBXW7",
        "PPP2R1A",
        "ARID5B",
        "CTCF",
        "RPL22",
        "FGFR2",
        "SOX17",
        "POLE",
        "MSH6",
        "MSH2",
        "MLH1",
        "PMS2",
        "RB1",
        "CCND1",
        "ERBB2",
    ],
    cna_genes=["MYC", "ERBB2", "CCNE1", "FGFR3", "CDKN2A", "PTEN", "TP53"],
    hotspot_variants={"POLE": ["P286R", "V411L", "S297F", "A456P", "S459F"]},
    special_fields_fn="special_ucec",
    msi_relevant=True,
)

CANCER_CONFIGS["BLCA"] = CancerConfig(
    project="TCGA-BLCA",
    cancer_code="BLCA",
    driver_genes=[
        "TP53",
        "KDM6A",
        "ARID1A",
        "PIK3CA",
        "RB1",
        "ERBB3",
        "ERBB2",
        "FGFR3",
        "ELF3",
        "CDKN1A",
        "STAG2",
        "EP300",
        "CREBBP",
        "NFE2L2",
        "FBXW7",
        "TSC1",
        "HRAS",
        "RXRA",
        "KMT2D",
        "CDKN2A",
        "FAT1",
    ],
    cna_genes=["FGFR3", "ERBB2", "EGFR", "CCND1", "E2F3", "PPARG", "CDKN2A", "RB1", "PTEN"],
    special_fields_fn="special_blca",
    msi_relevant=True,
)

CANCER_CONFIGS["HNSC"] = CancerConfig(
    project="TCGA-HNSC",
    cancer_code="HNSC",
    driver_genes=[
        "TP53",
        "CDKN2A",
        "FAT1",
        "NOTCH1",
        "PIK3CA",
        "CASP8",
        "NSD1",
        "HRAS",
        "TGFBR2",
        "HLA-A",
        "HLA-B",
        "FBXW7",
        "NFE2L2",
        "AJUBA",
        "EP300",
        "RB1",
        "PTEN",
        "KMT2D",
        "EPHA2",
        "RAC1",
    ],
    cna_genes=["CCND1", "FADD", "EGFR", "PIK3CA", "SOX2", "CDKN2A", "PTEN"],
    special_fields_fn="special_hnsc",
)

CANCER_CONFIGS["LIHC"] = CancerConfig(
    project="TCGA-LIHC",
    cancer_code="LIHC",
    driver_genes=[
        "TP53",
        "CTNNB1",
        "AXIN1",
        "ARID1A",
        "ARID2",
        "ALB",
        "APOB",
        "BAP1",
        "NFE2L2",
        "KEAP1",
        "RPS6KA3",
        "PIK3CA",
        "TSC1",
        "TSC2",
        "RB1",
        "CDKN2A",
        "TERT",
        "IL6ST",
        "LZTR1",
    ],
    cna_genes=["MYC", "CCND1", "MET", "VEGFA", "CDKN2A", "RB1", "AXIN1", "PTEN"],
    special_fields_fn="special_lihc",
)

CANCER_CONFIGS["STAD"] = CancerConfig(
    project="TCGA-STAD",
    cancer_code="STAD",
    driver_genes=[
        "TP53",
        "CDH1",
        "ARID1A",
        "PIK3CA",
        "KRAS",
        "RHOA",
        "APC",
        "CTNNB1",
        "SMAD4",
        "FBXW7",
        "ERBB3",
        "ERBB2",
        "MUC6",
        "RNF43",
        "SOX9",
        "BCOR",
        "KMT2D",
        "TGFBR2",
    ],
    cna_genes=["ERBB2", "EGFR", "MYC", "CCNE1", "VEGFA", "KRAS", "FGFR2", "CDKN2A", "SMAD4"],
    special_fields_fn="special_stad",
    msi_relevant=True,
)

CANCER_CONFIGS["ESCA"] = CancerConfig(
    project="TCGA-ESCA",
    cancer_code="ESCA",
    driver_genes=[
        "TP53",
        "CDKN2A",
        "NFE2L2",
        "PIK3CA",
        "NOTCH1",
        "KMT2D",
        "FAT1",
        "ERBB2",
        "ARID1A",
        "SMAD4",
        "KRAS",
        "EP300",
        "EPPK1",
        "FBXW7",
        "RB1",
    ],
    cna_genes=["ERBB2", "CCND1", "SOX2", "TP63", "VEGFA", "EGFR", "GATA4", "CDKN2A"],
    special_fields_fn="special_esca",
)

CANCER_CONFIGS["PAAD"] = CancerConfig(
    project="TCGA-PAAD",
    cancer_code="PAAD",
    driver_genes=[
        "KRAS",
        "TP53",
        "CDKN2A",
        "SMAD4",
        "BRCA2",
        "BRCA1",
        "PALB2",
        "ARID1A",
        "RNF43",
        "TGFBR2",
        "ATM",
        "KDM6A",
        "GNAS",
        "STK11",
        "BRAF",
        "MAP2K4",
        "RBM10",
    ],
    cna_genes=["KRAS", "MYC", "GATA6", "CDKN2A", "SMAD4", "TP53", "BRCA2"],
    hotspot_variants={"KRAS": ["G12D", "G12V", "G12R", "G12C", "Q61H"]},
    special_fields_fn="special_paad",
    msi_relevant=True,
)

CANCER_CONFIGS["SKCM"] = CancerConfig(
    project="TCGA-SKCM",
    cancer_code="SKCM",
    driver_genes=["BRAF", "NRAS", "NF1", "TP53", "CDKN2A", "PTEN", "RAC1", "IDH1", "RB1", "PPP6C", "DDX3X", "MAP2K1", "ARID2", "KIT"],
    cna_genes=["BRAF", "CCND1", "CDK4", "MDM2", "MITF", "KIT", "CDKN2A", "PTEN"],
    hotspot_variants={"BRAF": ["V600E", "V600K", "V600R"], "NRAS": ["Q61R", "Q61K", "Q61L", "Q61H"]},
    special_fields_fn="special_skcm",
)

CANCER_CONFIGS["THCA"] = CancerConfig(
    project="TCGA-THCA",
    cancer_code="THCA",
    driver_genes=["BRAF", "NRAS", "HRAS", "KRAS", "EIF1AX", "PPM1D", "CHEK2", "DICER1", "TERT"],
    cna_genes=["CDKN2A", "CCND1"],
    hotspot_variants={"BRAF": ["V600E"]},
    special_fields_fn="special_thca",
    notes="Very low TMB. Genomically quiet.",
)

CANCER_CONFIGS["CESC"] = CancerConfig(
    project="TCGA-CESC",
    cancer_code="CESC",
    driver_genes=["PIK3CA", "EP300", "FBXW7", "PTEN", "HLA-A", "HLA-B", "ERBB3", "KRAS", "ARID1A", "NFE2L2", "MAPK1", "CBFB", "TP53", "STK11", "CASP8"],
    cna_genes=["CD274", "PIK3CA", "SOX2", "MYC", "CDKN2A"],
    special_fields_fn="special_cesc",
)

CANCER_CONFIGS["PCPG"] = CancerConfig(
    project="TCGA-PCPG",
    cancer_code="PCPG",
    driver_genes=["HRAS", "RET", "VHL", "NF1", "MAX", "TMEM127", "SDHA", "SDHB", "SDHC", "SDHD", "SDHAF2", "FH", "EPAS1", "CSDE1", "MAML3", "ATRX"],
    cna_genes=["SDHB", "VHL", "NF1", "RET"],
    special_fields_fn="special_pcpg",
)

CANCER_CONFIGS["ACC"] = CancerConfig(
    project="TCGA-ACC",
    cancer_code="ACC",
    driver_genes=["TP53", "CTNNB1", "ZNRF3", "DAXX", "MEN1", "PRKAR1A", "RPL22", "TERT", "CDKN2A", "RB1", "APC"],
    cna_genes=[],
    special_fields_fn="special_acc",
    notes="Small cohort (n~79).",
)

CANCER_CONFIGS["SARC"] = CancerConfig(
    project="TCGA-SARC",
    cancer_code="SARC",
    driver_genes=["TP53", "ATRX", "RB1", "NF1", "PTEN", "PIK3CA", "MDM2", "CDK4", "CDKN2A", "CTNNB1", "APC", "KMT2D", "MED12"],
    cna_genes=["MDM2", "CDK4", "HMGA2", "RB1", "CDKN2A", "PTEN", "MYC"],
    special_fields_fn="special_sarc",
    notes="Histologically heterogeneous. Subtype = histologic type.",
)

CANCER_CONFIGS["TGCT"] = CancerConfig(
    project="TCGA-TGCT",
    cancer_code="TGCT",
    driver_genes=["KIT", "KRAS", "NRAS", "TP53", "CDC27", "BRAF"],
    cna_genes=["KIT", "KRAS", "CCND2"],
    arm_level_events={"chr12p_gain": [("chr12", "p_gain")]},
    special_fields_fn="special_tgct",
    notes="Very low somatic mutation rate. CNA-driven.",
)

CANCER_CONFIGS["DLBC"] = CancerConfig(
    project="TCGA-DLBC",
    cancer_code="DLBC",
    driver_genes=[
        "MYD88",
        "CD79B",
        "EZH2",
        "CREBBP",
        "KMT2D",
        "TNFAIP3",
        "BCL2",
        "CARD11",
        "PIM1",
        "TP53",
        "GNA13",
        "SOCS1",
        "B2M",
        "HIST1H1E",
        "IRF4",
        "PRDM1",
        "BTG1",
        "EP300",
    ],
    cna_genes=["BCL2", "BCL6", "MYC", "REL", "CDKN2A"],
    hotspot_variants={"MYD88": ["L265P"]},
    special_fields_fn="special_dlbc",
    notes="Small cohort (n~48). Hematologic — no solid tumor imaging.",
)

CANCER_CONFIGS["LAML"] = CancerConfig(
    project="TCGA-LAML",
    cancer_code="LAML",
    driver_genes=[
        "FLT3",
        "NPM1",
        "DNMT3A",
        "IDH1",
        "IDH2",
        "TET2",
        "RUNX1",
        "TP53",
        "NRAS",
        "KRAS",
        "CEBPA",
        "WT1",
        "KIT",
        "U2AF1",
        "SRSF2",
        "SF3B1",
        "STAG2",
        "RAD21",
        "SMC1A",
        "SMC3",
        "ASXL1",
        "EZH2",
        "PHF6",
        "BCOR",
        "KMT2A",
    ],
    cna_genes=["MECOM", "MYC", "KMT2A"],
    special_fields_fn="special_laml",
    notes="Hematologic malignancy — no solid tumor imaging.",
)

CANCER_CONFIGS["MESO"] = CancerConfig(
    project="TCGA-MESO",
    cancer_code="MESO",
    driver_genes=["BAP1", "NF2", "TP53", "SETD2", "CDKN2A", "DDX3X", "ULK2", "LATS2", "SETDB1"],
    cna_genes=["BAP1", "NF2", "CDKN2A", "LATS2"],
    special_fields_fn="special_meso",
)

CANCER_CONFIGS["UVM"] = CancerConfig(
    project="TCGA-UVM",
    cancer_code="UVM",
    driver_genes=["GNAQ", "GNA11", "SF3B1", "EIF1AX", "BAP1", "CYSLTR2", "PLCB4", "SRSF2"],
    cna_genes=[],
    arm_level_events={
        "chr3_monosomy": [("chr3", "whole_loss")],
        "chr8q_gain": [("chr8", "q_gain")],
        "chr6p_gain": [("chr6", "p_gain")],
    },
    special_fields_fn="special_uvm",
)

CANCER_CONFIGS["CHOL"] = CancerConfig(
    project="TCGA-CHOL",
    cancer_code="CHOL",
    driver_genes=["IDH1", "IDH2", "TP53", "KRAS", "ARID1A", "BAP1", "PBRM1", "SMAD4", "BRAF", "PIK3CA", "NRAS", "FGFR2", "EPHA2"],
    cna_genes=["ERBB2", "CDKN2A", "MYC", "FGFR2"],
    special_fields_fn="special_chol",
    notes="Very small cohort (n~36).",
)

CANCER_CONFIGS["UCS"] = CancerConfig(
    project="TCGA-UCS",
    cancer_code="UCS",
    driver_genes=["TP53", "PIK3CA", "PPP2R1A", "FBXW7", "PTEN", "KRAS", "CHD4", "ARID1A", "CSMD3", "RB1"],
    cna_genes=["MYC", "CCNE1", "ERBB2", "CDKN2A", "PTEN"],
    special_fields_fn="special_ucs",
    notes="Small cohort (n~57). Molecularly similar to CN-high UCEC.",
)

CANCER_CONFIGS["THYM"] = CancerConfig(
    project="TCGA-THYM",
    cancer_code="THYM",
    driver_genes=["GTF2I", "HRAS", "TP53", "NRAS", "KIT", "BCOR", "CDKN2A"],
    cna_genes=["CDKN2A", "MYC", "BCL2"],
    hotspot_variants={"GTF2I": ["L424H"]},
    special_fields_fn="special_thym",
    notes="Very low mutation rate. GTF2I L424H is dominant event.",
)
