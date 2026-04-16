from __future__ import annotations

from typing import Any, Dict

from .tcga_genomics_config import CancerConfig


def special_kirc(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    bap1_pbrm1 = (
        "both"
        if ("BAP1" in mutated_genes and "PBRM1" in mutated_genes)
        else "BAP1_mutant"
        if "BAP1" in mutated_genes
        else "PBRM1_mutant"
        if "PBRM1" in mutated_genes
        else "neither"
    )
    mtor_genes = {"MTOR", "TSC1", "TSC2", "PTEN", "PIK3CA"}
    return {
        "bap1_pbrm1_status": bap1_pbrm1,
        "mtor_pathway_altered": "yes" if mutated_genes & mtor_genes else "no",
        "vhl_status": "mutated" if "VHL" in mutated_genes else "wildtype",
    }


def special_kirp(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    nrf2 = {"NFE2L2", "KEAP1", "CUL3"}
    return {
        "met_status": "mutated" if "MET" in mutated_genes else "wildtype",
        "nrf2_pathway_altered": "yes" if mutated_genes & nrf2 else "no",
    }


def special_kich(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    tp53_pten = (
        "both"
        if ("TP53" in mutated_genes and "PTEN" in mutated_genes)
        else "TP53_mutant"
        if "TP53" in mutated_genes
        else "PTEN_mutant"
        if "PTEN" in mutated_genes
        else "neither"
    )
    return {"tp53_pten_status": tp53_pten}


def special_brca(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    brca = (
        "BRCA1_mutant"
        if "BRCA1" in mutated_genes
        else "BRCA2_mutant"
        if "BRCA2" in mutated_genes
        else "wildtype"
    )
    return {
        "pik3ca_mutated": "yes" if "PIK3CA" in mutated_genes else "no",
        "tp53_mutated": "yes" if "TP53" in mutated_genes else "no",
        "cdh1_mutated": "yes" if "CDH1" in mutated_genes else "no",
        "brca_hrd_status": brca,
    }


def special_prad(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {
        "spop_mutated": "yes" if "SPOP" in mutated_genes else "no",
        "tp53_mutated": "yes" if "TP53" in mutated_genes else "no",
        "pten_status": "mutated" if "PTEN" in mutated_genes else "intact",
    }


def special_luad(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    hotspot_hits = patient.get("hotspot_hits", {})
    kras_variant = hotspot_hits.get("KRAS", "wildtype") if "KRAS" in mutated_genes else "wildtype"
    egfr_variant = hotspot_hits.get("EGFR", "other") if "EGFR" in mutated_genes else "wildtype"
    co_mutation = "none"
    if "KRAS" in mutated_genes:
        if "TP53" in mutated_genes:
            co_mutation = "KRAS_TP53"
        elif "STK11" in mutated_genes:
            co_mutation = "KRAS_STK11"
        elif "KEAP1" in mutated_genes:
            co_mutation = "KRAS_KEAP1"
    return {
        "kras_variant": kras_variant,
        "egfr_variant": egfr_variant,
        "stk11_mutated": "yes" if "STK11" in mutated_genes else "no",
        "keap1_mutated": "yes" if "KEAP1" in mutated_genes else "no",
        "kras_co_mutation": co_mutation,
    }


def special_lusc(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    nrf2 = {"NFE2L2", "KEAP1"}
    pi3k = {"PIK3CA", "PTEN"}
    return {
        "nrf2_pathway_altered": "yes" if mutated_genes & nrf2 else "no",
        "pi3k_pathway_altered": "yes" if mutated_genes & pi3k else "no",
    }


def special_gbm(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    hotspot_hits = patient.get("hotspot_hits", {})
    idh_status = "wildtype"
    if "IDH1" in mutated_genes:
        idh_status = hotspot_hits.get("IDH1", "IDH1_other")
        if "R132H" in idh_status:
            idh_status = "IDH1_R132H"
        elif "IDH1" in idh_status:
            idh_status = "IDH1_other"
    elif "IDH2" in mutated_genes:
        idh_status = "IDH2_mutant"
    return {
        "idh_status": idh_status,
        "cdkn2a_homdel": "not_assessed",
    }


def special_lgg(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    hotspot_hits = patient.get("hotspot_hits", {})
    idh_status = "wildtype"
    if "IDH1" in mutated_genes:
        idh_status = hotspot_hits.get("IDH1", "IDH1_other")
        if "R132H" in idh_status:
            idh_status = "IDH1_R132H"
    elif "IDH2" in mutated_genes:
        idh_status = "IDH2_mutant"
    return {
        "idh_status": idh_status,
        "atrx_status": "mutated" if "ATRX" in mutated_genes else "wildtype",
        "cic_fubp1_status": (
            "both"
            if ("CIC" in mutated_genes and "FUBP1" in mutated_genes)
            else "CIC_mutant"
            if "CIC" in mutated_genes
            else "FUBP1_mutant"
            if "FUBP1" in mutated_genes
            else "neither"
        ),
    }


def special_coad(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    hotspot_hits = patient.get("hotspot_hits", {})
    return {
        "braf_v600e": "yes" if hotspot_hits.get("BRAF", "") == "V600E" else "no",
        "kras_mutated": "yes" if "KRAS" in mutated_genes else "no",
        "apc_mutated": "yes" if "APC" in mutated_genes else "no",
    }


def special_ov(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    brca = (
        "BRCA1_mutant"
        if "BRCA1" in mutated_genes
        else "BRCA2_mutant"
        if "BRCA2" in mutated_genes
        else "wildtype"
    )
    return {
        "tp53_mutated": "yes" if "TP53" in mutated_genes else "no",
        "brca_status": brca,
    }


def special_ucec(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {
        "pole_mutated": "yes" if "POLE" in mutated_genes else "no",
        "pten_mutated": "yes" if "PTEN" in mutated_genes else "no",
        "tp53_mutated": "yes" if "TP53" in mutated_genes else "no",
        "pik3ca_mutated": "yes" if "PIK3CA" in mutated_genes else "no",
    }


def special_blca(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {
        "fgfr3_status": "mutated" if "FGFR3" in mutated_genes else "wildtype",
        "rb1_altered": "yes" if "RB1" in mutated_genes else "no",
    }


def special_hnsc(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {
        "tp53_mutated": "yes" if "TP53" in mutated_genes else "no",
        "hpv_status": "check_clinical",
    }


def special_lihc(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    axis = (
        "both"
        if ("TP53" in mutated_genes and "CTNNB1" in mutated_genes)
        else "TP53_mutant"
        if "TP53" in mutated_genes
        else "CTNNB1_mutant"
        if "CTNNB1" in mutated_genes
        else "neither"
    )
    return {
        "tp53_ctnnb1_axis": axis,
        "wnt_pathway_activated": "yes" if ("CTNNB1" in mutated_genes or "AXIN1" in mutated_genes) else "no",
    }


def special_stad(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {
        "cdh1_mutated": "yes" if "CDH1" in mutated_genes else "no",
        "rhoa_mutated": "yes" if "RHOA" in mutated_genes else "no",
    }


def special_esca(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {"nrf2_pathway_altered": "yes" if ("NFE2L2" in mutated_genes or "KEAP1" in mutated_genes) else "no"}


def special_paad(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    hotspot_hits = patient.get("hotspot_hits", {})
    kras_variant = hotspot_hits.get("KRAS", "other") if "KRAS" in mutated_genes else "wildtype"
    brca = (
        "BRCA2_mutant"
        if "BRCA2" in mutated_genes
        else "BRCA1_mutant"
        if "BRCA1" in mutated_genes
        else "PALB2_mutant"
        if "PALB2" in mutated_genes
        else "wildtype"
    )
    return {
        "kras_mutated": "yes" if "KRAS" in mutated_genes else "no",
        "kras_variant": kras_variant,
        "smad4_status": "mutated" if "SMAD4" in mutated_genes else "intact",
        "brca_hrd_status": brca,
    }


def special_skcm(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    hotspot_hits = patient.get("hotspot_hits", {})
    braf_variant = hotspot_hits.get("BRAF", "other") if "BRAF" in mutated_genes else "wildtype"
    genomic_subtype = (
        "BRAF_hotspot"
        if "BRAF" in mutated_genes
        else "NRAS_hotspot"
        if "NRAS" in mutated_genes
        else "NF1_mutant"
        if "NF1" in mutated_genes
        else "Triple_wildtype"
    )
    return {
        "genomic_subtype": genomic_subtype,
        "braf_variant": braf_variant,
    }


def special_thca(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    hotspot_hits = patient.get("hotspot_hits", {})
    ras = (
        "NRAS"
        if "NRAS" in mutated_genes
        else "HRAS"
        if "HRAS" in mutated_genes
        else "KRAS"
        if "KRAS" in mutated_genes
        else "wildtype"
    )
    return {
        "braf_v600e": "yes" if hotspot_hits.get("BRAF", "") == "V600E" else "no",
        "ras_mutated": ras,
    }


def special_cesc(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    return {"hpv_type": "check_clinical"}


def special_pcpg(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    sdh_genes = {"SDHA", "SDHB", "SDHC", "SDHD", "SDHAF2"}
    sdh_hit = mutated_genes & sdh_genes
    sdh_status = list(sdh_hit)[0] + "_mutant" if sdh_hit else "intact"
    return {
        "sdh_status": sdh_status,
        "ret_status": "mutated" if "RET" in mutated_genes else "wildtype",
        "vhl_status": "mutated" if "VHL" in mutated_genes else "wildtype",
    }


def special_acc(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {
        "tp53_ctnnb1_axis": (
            "TP53_mutant"
            if "TP53" in mutated_genes
            else "CTNNB1_mutant"
            if "CTNNB1" in mutated_genes
            else "neither"
        ),
    }


def special_sarc(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {"atrx_status": "mutated" if "ATRX" in mutated_genes else "wildtype"}


def special_tgct(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {"kit_mutated": "yes" if "KIT" in mutated_genes else "no"}


def special_dlbc(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    hotspot_hits = patient.get("hotspot_hits", {})
    return {
        "myd88_l265p": "yes" if hotspot_hits.get("MYD88", "") == "L265P" else "no",
        "ezh2_mutated": "yes" if "EZH2" in mutated_genes else "no",
    }


def special_laml(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    idh_status = (
        "IDH1_mutant"
        if "IDH1" in mutated_genes
        else "IDH2_mutant"
        if "IDH2" in mutated_genes
        else "wildtype"
    )
    flt3_npm1 = (
        "FLT3_NPM1"
        if ("FLT3" in mutated_genes and "NPM1" in mutated_genes)
        else "FLT3_only"
        if "FLT3" in mutated_genes
        else "NPM1_only"
        if "NPM1" in mutated_genes
        else "neither"
    )
    return {
        "npm1_mutated": "yes" if "NPM1" in mutated_genes else "no",
        "idh_status": idh_status,
        "dnmt3a_mutated": "yes" if "DNMT3A" in mutated_genes else "no",
        "tp53_mutated": "yes" if "TP53" in mutated_genes else "no",
        "flt3_npm1_combination": flt3_npm1,
    }


def special_meso(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {
        "bap1_status": "mutated" if "BAP1" in mutated_genes else "intact",
        "nf2_status": "mutated" if "NF2" in mutated_genes else "intact",
    }


def special_uvm(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    gnaq_gna11 = (
        "GNAQ_mutant"
        if "GNAQ" in mutated_genes
        else "GNA11_mutant"
        if "GNA11" in mutated_genes
        else "other"
    )
    return {
        "gnaq_gna11_status": gnaq_gna11,
        "bap1_status": "mutated" if "BAP1" in mutated_genes else "wildtype",
        "sf3b1_status": "mutated" if "SF3B1" in mutated_genes else "wildtype",
        "eif1ax_status": "mutated" if "EIF1AX" in mutated_genes else "wildtype",
    }


def special_chol(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    idh_status = (
        "IDH1_mutant"
        if "IDH1" in mutated_genes
        else "IDH2_mutant"
        if "IDH2" in mutated_genes
        else "wildtype"
    )
    return {"idh_status": idh_status}


def special_ucs(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {
        "tp53_mutated": "yes" if "TP53" in mutated_genes else "no",
        "ppp2r1a_mutated": "yes" if "PPP2R1A" in mutated_genes else "no",
    }


def special_thym(patient: Dict[str, Any], config: CancerConfig) -> Dict[str, str]:
    mutated_genes = patient.get("mutated_genes", set())
    return {"gtf2i_mutated": "yes" if "GTF2I" in mutated_genes else "no"}


SPECIAL_FN_REGISTRY = {
    "special_kirc": special_kirc,
    "special_kirp": special_kirp,
    "special_kich": special_kich,
    "special_brca": special_brca,
    "special_prad": special_prad,
    "special_luad": special_luad,
    "special_lusc": special_lusc,
    "special_gbm": special_gbm,
    "special_lgg": special_lgg,
    "special_coad": special_coad,
    "special_ov": special_ov,
    "special_ucec": special_ucec,
    "special_blca": special_blca,
    "special_hnsc": special_hnsc,
    "special_lihc": special_lihc,
    "special_stad": special_stad,
    "special_esca": special_esca,
    "special_paad": special_paad,
    "special_skcm": special_skcm,
    "special_thca": special_thca,
    "special_cesc": special_cesc,
    "special_pcpg": special_pcpg,
    "special_acc": special_acc,
    "special_sarc": special_sarc,
    "special_tgct": special_tgct,
    "special_dlbc": special_dlbc,
    "special_laml": special_laml,
    "special_meso": special_meso,
    "special_uvm": special_uvm,
    "special_chol": special_chol,
    "special_ucs": special_ucs,
    "special_thym": special_thym,
}
