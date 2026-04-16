from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "dnam_proj_train" / "02_gen_dnam_case_captions.py"
    spec = importlib.util.spec_from_file_location("dnam_case_caption_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_dnam_metadata_lines_excludes_tcga_sample_barcode() -> None:
    module = _load_script_module()
    row = {
        "project_id": "TCGA-GBM",
        "primary_site": "Brain",
        "primary_diagnosis": "Glioblastoma",
        "age_at_diagnosis": "16425",
        "gender": "female",
        "genomics_dna_methylation_paths": ["/tmp/tumor.txt"],
        "project_driver_gene_mutations": ["EGFR", "PTEN"],
        "mutated_gene_symbols": ["EGFR", "PTEN", "TP53", "MDM4"],
        "mutation_query_succeeded": True,
    }
    beta_stats = {
        "probe_count": 23475,
        "mean": 0.2561,
        "std": 0.3322,
        "median": 0.0534,
        "q25": 0.0271,
        "q75": 0.4620,
        "low_frac": 0.669,
        "mid_frac": 0.185,
        "high_frac": 0.146,
    }

    metadata_lines = module._build_dnam_metadata_lines(
        row,
        selected_sample_id="TCGA-02-0001-01C",
        beta_stats=beta_stats,
        low_threshold=0.2,
        high_threshold=0.8,
        max_driver_mutations_to_list=5,
        max_additional_positive_mutations_to_list=4,
        include_zero_mutation_counts_in_prompt=False,
        metadata_fields=["project_id", "primary_site", "primary_diagnosis", "age_at_diagnosis", "gender"],
    )
    metadata_block = "\n".join(metadata_lines)

    assert "TCGA-02-0001-01C" not in metadata_block
    assert "selected_dnam_sample_type: primary tumor" in metadata_block
    assert "positive_project_driver_mutations: EGFR, PTEN" in metadata_block
    assert "additional_positive_mutations: TP53, MDM4" in metadata_block
    assert "dnam_beta_mean: 0.2561" in metadata_block


def test_build_caption_request_prompt_mentions_no_internal_ids_requirement() -> None:
    module = _load_script_module()
    prompt = module._build_caption_request_prompt(
        instruction="Describe the DNA methylation profile.",
        caption_prompt_variant="Summarize the DNAm profile.",
        caption_length_instruction="Write 4-6 sentences.",
        metadata_lines=["project_id: TCGA-GBM", "selected_dnam_sample_type: primary tumor"],
    )

    assert "Generate one grounded DNA methylation caption" in prompt
    assert "Use it only as source material to summarize the DNAm case." in prompt
