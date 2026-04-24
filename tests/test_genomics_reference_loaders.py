from __future__ import annotations

from pathlib import Path

import numpy as np

from kidney_vlm.genomics import cohort_config, dnam_text_features, signatures


def test_cohort_config_keeps_full_tcga_coverage_and_curated_front_matter() -> None:
    assert "TCGA-ACC" in cohort_config.get_all_cohorts()
    assert "TCGA-KIRC" in cohort_config.get_all_cohorts()

    kirc_panel = cohort_config.get_mutation_panel("TCGA-KIRC")
    assert kirc_panel[:3] == ["VHL", "PBRM1", "BAP1"]
    assert "TP53" in kirc_panel

    fallback_panel = cohort_config.get_mutation_panel("TCGA-ACC")
    assert fallback_panel
    assert "TP53" in fallback_panel


def test_reference_loaders_strip_metadata_and_unwrap_envelopes() -> None:
    hallmarks = signatures.load_hallmark50()
    assert len(hallmarks) == 50
    assert "_source" not in hallmarks
    assert "_note" not in hallmarks
    assert all(name.startswith("HALLMARK_") for name in hallmarks)
    assert all(isinstance(genes, list) and genes for genes in hallmarks.values())

    estimate = signatures.load_estimate_full()
    assert sorted(estimate) == ["immune", "stromal"]
    assert len(estimate["stromal"]) > 100
    assert len(estimate["immune"]) > 100

    centroids = signatures.load_pam50_centroids()
    assert "_source" not in centroids
    assert "HER2_enriched" in centroids
    assert "Basal_like" in centroids
    assert "Normal_like" in centroids
    assert all(isinstance(values, dict) and values for values in centroids.values())

    probes = signatures.load_promoter_probes()
    assert "_source" not in probes
    assert "probes" not in probes
    assert "VHL" in probes
    assert all(probe.startswith("cg") for probe in probes["VHL"])

    coefficients, intercept = signatures.load_horvath_clock()
    assert len(coefficients) == 353
    assert np.isfinite(intercept)


def test_dnam_beta_loader_and_promoter_probe_mapping_work_without_header(tmp_path: Path) -> None:
    vhl_probe = signatures.load_promoter_probes()["VHL"][0]
    beta_path = tmp_path / "beta.tsv"
    beta_path.write_text(f"{vhl_probe}\t0.82\ncg00000000\t0.11\n", encoding="utf-8")

    betas = dnam_text_features.read_tcga_beta_tsv(beta_path)
    assert betas[vhl_probe] == 0.82

    promoter = dnam_text_features.promoter_methylation_by_gene(betas, "TCGA-KIRC")
    assert np.isclose(promoter["VHL"], 0.82)
    assert np.isnan(dnam_text_features.horvath_epigenetic_age(betas))
