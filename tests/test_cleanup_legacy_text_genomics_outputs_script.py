from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "05_text_genomics" / "03_cleanup_legacy_text_genomics_outputs.py"
    spec = importlib.util.spec_from_file_location("cleanup_legacy_text_genomics_outputs_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cleanup_clears_legacy_columns_only_when_canonical_outputs_exist(tmp_path: Path) -> None:
    module = _load_script_module()
    case_dir = tmp_path / "data" / "features" / "llm_input_contexts" / "tcga" / "TCGA-KIRC" / "TCGA-AA-0001"
    clinical_path = case_dir / "clinical.txt"
    genomics_path = case_dir / "genomics.txt"
    legacy_teacher = case_dir / "teacher.txt"
    legacy_student = case_dir / "student.txt"
    legacy_json = case_dir / "features.json"
    for path in (clinical_path, genomics_path, legacy_teacher, legacy_student, legacy_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    registry_df = pd.DataFrame(
        [
            {
                "sample_id": "TCGA-AA-0001",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0001",
                "genomics_clinical_text_path": clinical_path.relative_to(tmp_path).as_posix(),
                "genomics_genomics_text_path": genomics_path.relative_to(tmp_path).as_posix(),
                "genomics_teacher_text_path": legacy_teacher.relative_to(tmp_path).as_posix(),
                "genomics_student_text_path": legacy_student.relative_to(tmp_path).as_posix(),
                "genomics_json_path": legacy_json.relative_to(tmp_path).as_posix(),
                "genomics_json_errors": "legacy-error",
            },
            {
                "sample_id": "TCGA-AA-0002",
                "source": "tcga",
                "project_id": "TCGA-KIRC",
                "patient_id": "TCGA-AA-0002",
                "genomics_clinical_text_path": "",
                "genomics_genomics_text_path": "",
                "genomics_teacher_text_path": "data/features/legacy/teacher.txt",
                "genomics_json_errors": "keep-me",
            },
        ]
    )

    updated_df, delete_candidates, stats = module.cleanup_legacy_registry_outputs(
        registry_df,
        source_name="tcga",
        repo_root=tmp_path,
        require_canonical_outputs=True,
    )

    first = updated_df.iloc[0]
    second = updated_df.iloc[1]
    assert stats["matched_rows"] == 2
    assert stats["cleared_rows"] == 1
    assert stats["skipped_missing_canonical"] == 1
    assert first["genomics_teacher_text_path"] == ""
    assert first["genomics_student_text_path"] == ""
    assert first["genomics_json_path"] == ""
    assert first["genomics_json_errors"] == ""
    assert second["genomics_teacher_text_path"] == "data/features/legacy/teacher.txt"
    assert second["genomics_json_errors"] == "keep-me"
    assert sorted(path.name for path in delete_candidates) == ["features.json", "student.txt", "teacher.txt"]


def test_delete_paths_removes_existing_files_and_counts_missing(tmp_path: Path) -> None:
    module = _load_script_module()
    existing = tmp_path / "teacher.txt"
    missing = tmp_path / "student.txt"
    existing.write_text("placeholder\n", encoding="utf-8")

    deleted, missing_count = module._delete_paths([existing, missing])

    assert deleted == 1
    assert missing_count == 1
    assert not existing.exists()
