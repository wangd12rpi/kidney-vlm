from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "05_text_genomics" / "02_build_llm_input_contexts.py"
    spec = importlib.util.spec_from_file_location("llm_input_contexts_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_process_case_writes_clinical_gdisc_and_llm_context_files(tmp_path: Path) -> None:
    module = _load_script_module()

    maf_path = tmp_path / "case.maf"
    maf_path.write_text(
        "\n".join(
            [
                "Hugo_Symbol\tVariant_Classification\tVariant_Type\tHGVSp_Short\tdbSNP_RS",
                "TP53\tMissense_Mutation\tSNP\tp.R175H\trs1",
                "VHL\tSilent\tSNP\tp.P25P\trs2",
            ]
        ),
        encoding="utf-8",
    )
    cna_path = tmp_path / "gene_cna.tsv"
    cna_path.write_text(
        "\n".join(
            [
                "gene_name\tcopy_number\tchromosome\tstart\tend",
                "TP53\t5\t17\t7565097\t7590856",
                "VHL\t2\t3\t10141778\t10153667",
            ]
        ),
        encoding="utf-8",
    )

    result = module.process_case(
        case_row={
            "sample_id": "TCGA-AA-0001",
            "source": "tcga",
            "project_id": "TCGA-KIRC",
            "patient_id": "TCGA-AA-0001",
            "split": "test",
            "primary_site": "Kidney",
            "primary_diagnosis": "Clear cell renal cell carcinoma",
            "age_at_diagnosis": 60 * 365.25,
            "gender": "female",
            "genomics_mutation_paths": [str(maf_path)],
            "genomics_cnv_gene_paths": [str(cna_path)],
            "genomics_msi_status": "MSS",
            "genomics_hrd_score": "12",
        },
        output_root=tmp_path / "contexts",
        source_name="tcga",
        mutation_panel=["TP53", "VHL"],
        callable_mb=40.0,
        include_survival_in_text=False,
        include_generation_instructions=True,
        emit_all_panel_wild_types=True,
        arm_event_threshold=0.6,
    )

    llm_input_path = Path(result["llm_input_text_path"])
    json_path = Path(result["llm_input_json_path"])
    assert llm_input_path.is_file()
    assert json_path.is_file()

    llm_text = llm_input_path.read_text(encoding="utf-8")
    assert "CLINICAL METADATA:" in llm_text
    assert "DISCRETE GENOMICS:" in llm_text
    assert "TP53: p.R175H | Missense_Mutation | Hotspot:Level_2 | dbSNP:rs1" in llm_text
    assert "TP53: amplification (+2)" in llm_text
    assert "TMB: 0.03 mut/Mb" in llm_text
    assert "MSI: MSS" in llm_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["available_modalities"] == ["mutation_maf", "copy_number_gene"]
    assert payload["gdisc"]["panel_nonsilent_variant_count"] == 1
    assert payload["gdisc"]["copy_number_calls"]["TP53"] == 2
    assert result["mutation_available"] is True
    assert result["copy_number_gene_available"] is True
