from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "05_text_genomics" / "02_build_llm_input_contexts.py"
    spec = importlib.util.spec_from_file_location("llm_input_contexts_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Feature:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def test_process_case_writes_clinical_and_genomics_files(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    maf_path = tmp_path / "case.maf"
    maf_path.write_text("placeholder\n", encoding="utf-8")
    cna_path = tmp_path / "gene_cna.tsv"
    cna_path.write_text("placeholder\n", encoding="utf-8")

    monkeypatch.setattr(
        module.mut_ext,
        "extract_mutation_cna_text_features",
        lambda **_: _Feature({"msi_status": "MSS", "hrd_score": 12.0}),
    )
    monkeypatch.setattr(module.text_block, "derive_integrated_surrogates", lambda **_: {"status": "ok"})
    monkeypatch.setattr(module.text_block, "assemble_teacher_text_block", lambda **_: "genomics text\n")

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

    clinical_path = Path(result["clinical_text_path"])
    genomics_path = Path(result["genomics_text_path"])
    assert clinical_path.is_file()
    assert genomics_path.is_file()

    clinical_text = clinical_path.read_text(encoding="utf-8")
    assert "CLINICAL METADATA:" in clinical_text
    assert "Primary diagnosis: Clear cell renal cell carcinoma" in clinical_text

    genomics_text = genomics_path.read_text(encoding="utf-8")
    assert genomics_text == "genomics text\n"

    case_dir = clinical_path.parent
    assert not (case_dir / "gdisc.txt").exists()
    assert not (case_dir / "llm_input.txt").exists()
    assert not (case_dir / "llm_input.json").exists()

    assert result["available_modalities"] == ["mutation_maf", "copy_number_gene"]
    assert result["mutation_available"] is True
    assert result["copy_number_gene_available"] is True


def test_filter_registry_require_text_genomics_keeps_any_supported_modality() -> None:
    module = _load_script_module()

    registry_df = pd.DataFrame(
        [
            {
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "genomics_rna_bulk_paths": ["data/rna.tsv"],
            },
            {
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0002",
                "genomics_dna_methylation_paths": ["data/beta.tsv"],
            },
            {
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0003",
                "genomics_mutation_paths": ["data/case.maf"],
            },
            {
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0004",
            },
        ]
    )

    filtered = module._filter_registry(
        registry_df,
        source_name="tcga",
        case_subset="all",
        case_json_path=Path("unused.json"),
        require_text_genomics=True,
        max_cases=None,
    )

    assert filtered["patient_id"].tolist() == [
        "TCGA-AA-0001",
        "TCGA-AA-0002",
        "TCGA-AA-0003",
    ]
