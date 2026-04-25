from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "05_text_genomics" / "02_build_genomics_text_blocks.py"
    spec = importlib.util.spec_from_file_location("genomics_text_block_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Feature:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def test_process_case_writes_teacher_student_blocks_from_registry_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()

    rna_normal = tmp_path / "data" / "raw" / "tcga" / "rna_bulk" / "normal.tsv"
    rna_tumor = tmp_path / "data" / "raw" / "tcga" / "rna_bulk" / "tumor.tsv"
    beta_path = tmp_path / "data" / "raw" / "tcga" / "dna_methylation" / "beta.tsv"
    maf_path = tmp_path / "data" / "raw" / "tcga" / "mutation_maf" / "case.maf"
    gene_cna_path = tmp_path / "data" / "raw" / "tcga" / "copy_number_gene" / "case.tsv"
    segment_cna_path = tmp_path / "data" / "raw" / "tcga" / "copy_number_segment" / "case.tsv"
    for path in [rna_normal, rna_tumor, beta_path, maf_path, gene_cna_path, segment_cna_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    calls: dict[str, dict[str, Any]] = {}

    def fake_dnam(**kwargs):
        calls["dnam"] = kwargs
        return _Feature({"methylation_subtype": kwargs["methylation_subtype_label"]})

    def fake_rna(**kwargs):
        calls["rna"] = kwargs
        return _Feature({"mrna_subtype": "Unassigned"})

    def fake_mut_cna(**kwargs):
        calls["mut_cna"] = kwargs
        return _Feature({"tmb_mutations_per_mb": 1.5})

    monkeypatch.setattr(module.dnam_ext, "extract_dnam_text_features", fake_dnam)
    monkeypatch.setattr(module.rna_ext, "extract_rna_text_features", fake_rna)
    monkeypatch.setattr(module.mut_ext, "extract_mutation_cna_text_features", fake_mut_cna)
    monkeypatch.setattr(module.text_block, "derive_integrated_surrogates", lambda **_: {"status": "ok"})
    monkeypatch.setattr(module.text_block, "assemble_teacher_text_block", lambda **_: "teacher text\n")
    monkeypatch.setattr(module.text_block, "assemble_student_text_block", lambda **_: "student text\n")

    def rel(path: Path) -> str:
        return path.relative_to(tmp_path).as_posix()

    result = module.process_case(
        project_id="TCGA-KIRC",
        patient_id="TCGA-AA-0001",
        case_row={
            "genomics_rna_bulk_paths": [rel(rna_normal), rel(rna_tumor)],
            "genomics_rna_bulk_sample_types": ["Solid Tissue Normal", "Primary Tumor"],
            "genomics_dna_methylation_paths": [rel(beta_path)],
            "genomics_mutation_paths": [rel(maf_path)],
            "genomics_cnv_gene_paths": [rel(gene_cna_path)],
            "genomics_cnv_segment_paths": [rel(segment_cna_path)],
            "genomics_dna_methylation_subtype": "ccRCC_m2",
            "genomics_msi_status": "MSS",
            "genomics_hrd_score": "27",
            "age_at_diagnosis_years": "61.5",
        },
        extra_genomics_by_patient={},
        output_root=tmp_path / "features" / "genomics_text_blocks" / "tcga",
        root_dir=tmp_path,
    )

    assert calls["rna"]["star_tsv_path"] == str(rna_tumor.resolve())
    assert calls["dnam"]["beta_tsv_path"] == str(beta_path.resolve())
    assert calls["dnam"]["chronological_age_years"] == 61.5
    assert calls["dnam"]["methylation_subtype_label"] == "ccRCC_m2"
    assert calls["mut_cna"]["maf_path"] == str(maf_path.resolve())
    assert calls["mut_cna"]["gene_cna_path"] == str(gene_cna_path.resolve())
    assert calls["mut_cna"]["segment_cna_path"] == str(segment_cna_path.resolve())
    assert calls["mut_cna"]["msi_status"] == "MSS"
    assert calls["mut_cna"]["hrd_score"] == 27.0
    assert result["available_modalities"] == [
        "dna_methylation_beta",
        "rna_bulk",
        "mutation_maf",
        "copy_number_gene",
        "copy_number_segment",
    ]
    assert Path(result["teacher_text_path"]).read_text(encoding="utf-8") == "teacher text\n"
    assert Path(result["student_text_path"]).read_text(encoding="utf-8") == "student text\n"
    assert Path(result["genomics_json_path"]).is_file()


def test_manifest_paths_override_missing_registry_genomics_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()

    maf_path = tmp_path / "manifest" / "case.maf"
    maf_path.parent.mkdir(parents=True)
    maf_path.write_text("placeholder\n", encoding="utf-8")

    calls: dict[str, dict[str, Any]] = {}

    def fake_mut_cna(**kwargs):
        calls["mut_cna"] = kwargs
        return _Feature({})

    monkeypatch.setattr(module.mut_ext, "extract_mutation_cna_text_features", fake_mut_cna)
    monkeypatch.setattr(module.text_block, "derive_integrated_surrogates", lambda **_: {})
    monkeypatch.setattr(module.text_block, "assemble_teacher_text_block", lambda **_: "teacher\n")
    monkeypatch.setattr(module.text_block, "assemble_student_text_block", lambda **_: "student\n")

    result = module.process_case(
        project_id="TCGA-KIRC",
        patient_id="TCGA-AA-0001",
        case_row={"genomics_mutation_paths": ["missing.maf"]},
        extra_genomics_by_patient={
            ("TCGA-KIRC", "TCGA-AA-0001"): {"mutation_maf": [str(maf_path)]}
        },
        output_root=tmp_path / "out",
        root_dir=tmp_path,
    )

    assert calls["mut_cna"]["maf_path"] == str(maf_path.resolve())
    assert result["available_modalities"] == ["mutation_maf"]


def test_consume_legacy_opt_in_rejects_accidental_script_use() -> None:
    module = _load_script_module()

    try:
        module._consume_legacy_opt_in([])
    except SystemExit as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected SystemExit for missing legacy opt-in flag.")

    assert "--allow-legacy-output" in message
    assert "deprecated" in message.lower()


def test_consume_legacy_opt_in_strips_flag_from_overrides() -> None:
    module = _load_script_module()

    cleaned = module._consume_legacy_opt_in(
        ["--allow-legacy-output", "data.source.download.enabled=false"]
    )

    assert cleaned == ["data.source.download.enabled=false"]
