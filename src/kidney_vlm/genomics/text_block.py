"""Final text-block assembler for genomics caption generation.

Takes the three feature dicts produced by the DNAm, RNA-seq, and
mutation/CNA extractors and renders them as a single flat text block with
stable field order, consistent units, and explicit "not_assessed" markers.

This is the block that the teacher model sees when generating captions and VQA. At
training/inference time the student only sees the "text_channel" section
(which is a strict subset: specifically the mutation, CNA, structural
rearrangement, TMB, MSI, HRD, and ncRNA fields).

Public API:
    assemble_teacher_text_block(dnam_features, rna_features, mut_cna_features,
                                 integrated_surrogates, project_id) -> str
    assemble_student_text_block(mut_cna_features) -> str
"""
from __future__ import annotations

from typing import Any

import numpy as np

from kidney_vlm.genomics import cohort_config as cohort_cfg
from kidney_vlm.genomics import rna_text_features as rna_tf


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_float(value: float, decimals: int = 2, na_str: str = "not_assessed") -> str:
    if value is None:
        return na_str
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return na_str
    if not np.isfinite(fv):
        return na_str
    return f"{fv:.{decimals}f}"


def _fmt_signed(value: float, decimals: int = 2, na_str: str = "not_assessed") -> str:
    if value is None or not np.isfinite(value):
        return na_str
    return f"{value:+.{decimals}f}"


def _fmt_list(values: list[str], empty: str = "none") -> str:
    if not values:
        return empty
    return ", ".join(values)


def _fmt_focal_cna_call(value: int) -> str:
    if value == -2:
        return "deep_deletion (-2)"
    if value == -1:
        return "shallow_loss (-1)"
    if value == 0:
        return "neutral (0)"
    if value == 1:
        return "gain (+1)"
    if value == 2:
        return "amplification (+2)"
    return f"unknown ({value})"


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------


def assemble_teacher_text_block(
    *,
    dnam_features: dict[str, Any] | None,
    rna_features: dict[str, Any] | None,
    mut_cna_features: dict[str, Any] | None,
    integrated_surrogates: dict[str, Any] | None,
    project_id: str,
    cohort_thresholds: cohort_cfg.CohortQuantileThresholds | None = None,
) -> str:
    """Return the full text block shown to the teacher at generation time.

    Any of the four feature dicts may be None (e.g. a case lacks DNAm); in
    that case the corresponding section renders as `not_assessed` lines so
    the teacher is never silently missing a field.
    """
    lines: list[str] = []
    lines.append(f"[project_id]")
    lines.append(f"  {project_id}")
    lines.append("")

    lines.extend(_render_dnam_section(dnam_features, project_id))
    lines.append("")
    lines.extend(_render_rna_section(rna_features, project_id, cohort_thresholds))
    lines.append("")
    lines.extend(_render_text_channel_section(mut_cna_features, project_id))
    lines.append("")
    lines.extend(_render_integrated_surrogates_section(integrated_surrogates, project_id))

    return "\n".join(lines).rstrip() + "\n"


def assemble_student_text_block(
    *,
    mut_cna_features: dict[str, Any] | None,
    project_id: str,
) -> str:
    """Return the restricted text block shown to the student at train/inference.

    Strict subset of the teacher block: only the text-channel section
    (mutations, CNAs, structural rearrangements, TMB, MSI, HRD, ncRNA).
    DNAm and RNA sections are omitted because the student reconstructs
    them from cpGPT and BulkFormer embeddings.
    """
    lines: list[str] = [f"[project_id]", f"  {project_id}", ""]
    lines.extend(_render_text_channel_section(mut_cna_features, project_id))
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_dnam_section(features: dict[str, Any] | None, project_id: str) -> list[str]:
    lines = ["[DNAm_features]  # encoder-derivable; teacher-only"]
    if features is None:
        lines.append("  status: not_assessed")
        return lines

    subtype = features.get("methylation_subtype", "Unassigned")
    lines.append(f"  methylation_subtype: {subtype}")
    lines.append(f"  cimp_status: {features.get('cimp_status', 'not_assessed')}")

    promoter = features.get("promoter_methylation", {}) or {}
    panel = cohort_cfg.PROMOTER_METHYLATION_PANEL.get(project_id, [])
    lines.append("  promoter_methylation:")
    if not panel:
        lines.append("    (no panel defined for cohort)")
    else:
        for gene in panel:
            val = promoter.get(gene, float("nan"))
            from kidney_vlm.genomics.dnam_text_features import bin_promoter_beta
            bin_label = bin_promoter_beta(val)
            lines.append(f"    {gene}: {_fmt_float(val, 2)} ({bin_label})")

    lines.append(f"  global_mean_beta: {_fmt_float(features.get('global_mean_beta'), 3)}")
    lines.append(
        f"  epigenetic_age_years: {_fmt_float(features.get('epigenetic_age_years'), 1)}"
    )
    lines.append(
        f"  epigenetic_age_acceleration_years: "
        f"{_fmt_signed(features.get('epigenetic_age_acceleration_years'), 1)}"
    )
    lines.append(
        f"  dnam_tumor_purity: {_fmt_float(features.get('dnam_tumor_purity'), 2)}"
    )

    immune = features.get("dnam_immune_fractions", {}) or {}
    lines.append("  dnam_immune_fractions:")
    if not immune:
        lines.append("    not_assessed")
    else:
        for cell_type, value in immune.items():
            lines.append(f"    {cell_type}: {_fmt_float(value, 2)}")

    return lines


def _render_rna_section(
    features: dict[str, Any] | None,
    project_id: str,
    thresholds: cohort_cfg.CohortQuantileThresholds | None,
) -> list[str]:
    lines = ["[RNA_features]  # encoder-derivable; teacher-only; protein-coding only"]
    if features is None:
        lines.append("  status: not_assessed")
        return lines

    subtype = features.get("mrna_subtype", "Unassigned")
    label_space = features.get("mrna_subtype_label_space") or cohort_cfg.MRNA_SUBTYPE_LABELS.get(
        project_id, []
    )
    lines.append(f"  mrna_subtype: {subtype}")
    if label_space:
        lines.append(f"  mrna_subtype_label_space: {_fmt_list(label_space)}")

    # Hallmark top enriched / suppressed
    hallmarks = features.get("hallmark_scores", {}) or {}
    sorted_hallmarks = sorted(
        [(k, v) for k, v in hallmarks.items() if np.isfinite(v)],
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_enriched = sorted_hallmarks[:5]
    top_suppressed = sorted_hallmarks[-3:][::-1] if len(sorted_hallmarks) >= 3 else []
    lines.append(
        "  hallmark_top_enriched: "
        + _fmt_list([f"{name} ({score:+.2f})" for name, score in top_enriched])
    )
    lines.append(
        "  hallmark_top_suppressed: "
        + _fmt_list([f"{name} ({score:+.2f})" for name, score in top_suppressed])
    )

    # ESTIMATE
    estimate = features.get("estimate", {}) or {}
    lines.append(
        f"  estimate_stromal: {_fmt_float(estimate.get('stromal'), 3)}"
    )
    lines.append(
        f"  estimate_immune: {_fmt_float(estimate.get('immune'), 3)}"
    )
    lines.append(
        f"  estimate_tumor_purity: {_fmt_float(estimate.get('tumor_purity'), 2)}"
    )

    # Functional signatures (raw + categorical)
    signatures = features.get("signatures", {}) or {}
    sig_rows: list[tuple[str, float, tuple[float, float] | None]] = [
        ("proliferation_score", signatures.get("proliferation_mean", float("nan")),
         thresholds.proliferation_score if thresholds else None),
        ("hypoxia_score", signatures.get("hypoxia_mean", float("nan")),
         thresholds.hypoxia_score if thresholds else None),
        ("emt_score", signatures.get("emt_composite", float("nan")),
         thresholds.emt_score if thresholds else None),
        ("ifng_score", signatures.get("ifng_mean", float("nan")),
         thresholds.ifng_score if thresholds else None),
        ("cytolytic_score", signatures.get("cytolytic_log", float("nan")),
         thresholds.cytolytic_score if thresholds else None),
        ("tis_score", signatures.get("tis_mean", float("nan")),
         thresholds.tis_score if thresholds else None),
    ]
    lines.append("  functional_signatures:")
    for name, raw, cutoffs in sig_rows:
        bin_label = rna_tf.bin_continuous(raw, cutoffs) if cutoffs else "unknown"
        lines.append(f"    {name}: {_fmt_float(raw, 2)} ({bin_label})")

    # Immune cell-type markers (categorical only; raw values are too noisy
    # with the compact marker panels to report as numerics).
    cell_scores = features.get("cell_type_scores", {}) or {}
    lines.append("  immune_cell_marker_scores (mean log1p(TPM)):")
    if not cell_scores:
        lines.append("    not_assessed")
    else:
        for cell_type in ["CD8_T", "Treg", "NK", "B_cell", "Macrophage_M1", "Macrophage_M2", "Neutrophil", "Fibroblast", "Endothelial"]:
            v = cell_scores.get(cell_type)
            if v is None:
                continue
            lines.append(f"    {cell_type}: {_fmt_float(v, 2)}")

    # Lineage / receptor markers
    lineage = features.get("lineage_markers", {}) or {}
    lines.append("  lineage_receptor_markers (log1p(TPM)):")
    if not lineage:
        lines.append("    not_applicable")
    else:
        for gene, value in lineage.items():
            lines.append(f"    {gene}: {_fmt_float(value, 2)}")

    # Fusions
    panel = features.get("fusions_panel", []) or []
    detected = features.get("fusions_detected", []) or []
    if panel:
        lines.append(f"  fusion_panel: {_fmt_list(panel)}")
        lines.append(
            f"  fusions_detected_rna: {_fmt_list(detected, empty='none_detected')}"
        )
    else:
        lines.append("  fusion_panel: none_defined_for_cohort")

    return lines


def _render_text_channel_section(features: dict[str, Any] | None, project_id: str) -> list[str]:
    lines = ["[text_channel_features]  # seen by both teacher and student"]
    if features is None:
        lines.append("  status: not_assessed")
        return lines

    mutation_panel = features.get("mutation_panel") or cohort_cfg.get_mutation_panel(project_id)
    focal_panel = features.get("focal_cna_panel") or cohort_cfg.FOCAL_CNA_PANEL.get(project_id, [])

    # Mutations
    lines.append("  mutations:")
    mutations = features.get("mutations", []) or []
    if not mutations:
        lines.append("    none_detected_in_panel")
    else:
        for m in mutations:
            gene = m.get("gene", "?")
            pc = m.get("protein_change", "")
            vc = m.get("variant_class", "")
            level = m.get("oncokb_level", "")
            flags = []
            if level:
                flags.append(f"OncoKB_{level}")
            if m.get("is_lof"):
                flags.append("LoF")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            pc_str = f" {pc}" if pc else ""
            lines.append(f"    {gene}:{pc_str} [{vc}]{flag_str}")

    # Wild-type calls
    wild_type = features.get("wild_type_calls", []) or []
    if wild_type:
        lines.append("  wild_type_panel_genes:")
        lines.append(f"    {_fmt_list(wild_type)}")

    # Focal CNAs
    lines.append("  focal_CNAs:")
    focal = features.get("focal_cnas", {}) or {}
    any_nonzero = any(v != 0 for v in focal.values())
    if not focal_panel:
        lines.append("    no_panel_defined")
    elif not any_nonzero:
        lines.append("    all_neutral")
    else:
        for gene in focal_panel:
            call = focal.get(gene, 0)
            if call == 0:
                continue
            lines.append(f"    {gene}: {_fmt_focal_cna_call(int(call))}")

    # Arm-level CNAs
    arm = features.get("arm_level_cnas", []) or []
    lines.append(f"  arm_level_CNAs: {_fmt_list(arm, empty='none_detected')}")

    # Structural rearrangements (DNA-level; RNA-detected fusions are in the
    # RNA section because those are encoder-derivable via the expression
    # downstream signatures).
    structural = features.get("structural_rearrangements_dna", []) or []
    lines.append(
        f"  structural_rearrangements_dna: "
        f"{_fmt_list(structural, empty='none_detected')}"
    )

    # TMB
    tmb = features.get("tmb_mutations_per_mb")
    lines.append(f"  tmb_mutations_per_mb: {_fmt_float(tmb, 2)}")

    # MSI
    msi = features.get("msi_status", "not_assessed") or "not_assessed"
    lines.append(f"  msi_status: {msi}")

    # HRD
    hrd = features.get("hrd_score")
    lines.append(f"  hrd_score: {_fmt_float(hrd, 1)}")

    # Non-coding RNA findings
    ncrna = features.get("ncrna_findings", []) or []
    lines.append(
        f"  noncoding_rna_findings: {_fmt_list(ncrna, empty='none_reported')}"
    )

    return lines


def _render_integrated_surrogates_section(
    features: dict[str, Any] | None, project_id: str
) -> list[str]:
    lines = ["[integrated_surrogates]  # teacher-only; derived from DNAm + RNA (+ mutations)"]
    applicability = cohort_cfg.INTEGRATED_SURROGATES.get(
        project_id, cohort_cfg.IntegratedSurrogateConfig()
    )
    features = features or {}

    def _val(key: str, applicable: bool) -> str:
        if not applicable:
            return "not_applicable"
        value = features.get(key)
        if value is None:
            return "not_assessed"
        return str(value)

    lines.append(f"  msi_surrogate: {_val('msi_surrogate', applicability.msi_like)}")
    lines.append(f"  hrd_surrogate: {_val('hrd_surrogate', applicability.hrd_like)}")
    lines.append(
        f"  hormone_receptor_concordance: "
        f"{_val('hormone_receptor_concordance', applicability.hormone_receptor_concordance)}"
    )
    lines.append(
        f"  vhl_pathway_inactivation_surrogate: "
        f"{_val('vhl_pathway_inactivation', applicability.vhl_pathway_inactivation)}"
    )
    lines.append(
        f"  cimp_status_surrogate: "
        f"{_val('cimp_status_surrogate', applicability.cimp_status)}"
    )
    return lines


# ---------------------------------------------------------------------------
# Integrated surrogate derivation
# ---------------------------------------------------------------------------


def derive_integrated_surrogates(
    *,
    dnam_features: dict[str, Any] | None,
    rna_features: dict[str, Any] | None,
    mut_cna_features: dict[str, Any] | None,
    project_id: str,
) -> dict[str, Any]:
    """Derive the cohort-applicable integrated surrogates from the three streams.

    Returns only the surrogates applicable to the cohort (per
    `INTEGRATED_SURROGATES`); others are omitted so the assembler can render
    them as "not_applicable".
    """
    config = cohort_cfg.INTEGRATED_SURROGATES.get(project_id, cohort_cfg.IntegratedSurrogateConfig())
    out: dict[str, Any] = {}

    promoter = (dnam_features or {}).get("promoter_methylation", {}) or {}
    hallmarks = (rna_features or {}).get("hallmark_scores", {}) or {}

    if config.msi_like:
        mlh1_beta = promoter.get("MLH1", float("nan"))
        ifng_score = hallmarks.get("HALLMARK_INTERFERON_GAMMA_RESPONSE", float("nan"))
        if np.isfinite(mlh1_beta) and np.isfinite(ifng_score):
            if mlh1_beta > 0.4 and ifng_score > 0:
                out["msi_surrogate"] = "MSI-H-like"
            else:
                out["msi_surrogate"] = "MSS-like"
        else:
            out["msi_surrogate"] = "not_assessed"

    if config.hrd_like:
        brca1_beta = promoter.get("BRCA1", float("nan"))
        has_brca_mut = any(
            m.get("gene") in {"BRCA1", "BRCA2"}
            for m in (mut_cna_features or {}).get("mutations", []) or []
        )
        if has_brca_mut or (np.isfinite(brca1_beta) and brca1_beta > 0.4):
            out["hrd_surrogate"] = "HRD-like"
        elif np.isfinite(brca1_beta):
            out["hrd_surrogate"] = "HR-proficient-like"
        else:
            out["hrd_surrogate"] = "not_assessed"

    if config.hormone_receptor_concordance:
        lineage = (rna_features or {}).get("lineage_markers", {}) or {}
        esr1 = lineage.get("ESR1", float("nan"))
        pgr = lineage.get("PGR", float("nan"))
        erbb2 = lineage.get("ERBB2", float("nan"))
        parts: list[str] = []
        if np.isfinite(esr1):
            parts.append(f"ER_{'high' if esr1 > 5 else 'low'}")
        if np.isfinite(pgr):
            parts.append(f"PR_{'high' if pgr > 4 else 'low'}")
        if np.isfinite(erbb2):
            parts.append(f"HER2_{'high' if erbb2 > 8 else 'low'}")
        out["hormone_receptor_concordance"] = "; ".join(parts) if parts else "not_assessed"

    if config.vhl_pathway_inactivation:
        vhl_beta = promoter.get("VHL", float("nan"))
        vhl_mut = any(
            m.get("gene") == "VHL"
            for m in (mut_cna_features or {}).get("mutations", []) or []
        )
        hypoxia = hallmarks.get("HALLMARK_HYPOXIA", float("nan"))
        score = 0
        if vhl_mut:
            score += 2
        if np.isfinite(vhl_beta) and vhl_beta > 0.4:
            score += 1
        if np.isfinite(hypoxia) and hypoxia > 0:
            score += 1
        if score >= 3:
            out["vhl_pathway_inactivation"] = "likely_inactivated"
        elif score >= 1:
            out["vhl_pathway_inactivation"] = "possibly_inactivated"
        else:
            out["vhl_pathway_inactivation"] = "not_inactivated"

    if config.cimp_status:
        cimp = (dnam_features or {}).get("cimp_status", "not_assessed")
        out["cimp_status_surrogate"] = cimp

    return out
