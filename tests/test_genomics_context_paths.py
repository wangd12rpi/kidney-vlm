from __future__ import annotations

from kidney_vlm.genomics.context_paths import (
    resolve_clinical_text_path,
    resolve_genomics_text_path,
)


def test_resolve_clinical_text_path_prefers_canonical_column() -> None:
    row = {
        "genomics_clinical_text_path": "data/features/clinical.txt",
        "genomics_llm_input_text_path": "data/features/llm_input.txt",
    }

    assert resolve_clinical_text_path(row) == "data/features/clinical.txt"


def test_resolve_genomics_text_path_prefers_canonical_column() -> None:
    row = {
        "genomics_genomics_text_path": "data/features/genomics.txt",
        "genomics_teacher_text_path": "data/features/teacher.txt",
        "genomics_gdisc_text_path": "data/features/gdisc.txt",
    }

    assert resolve_genomics_text_path(row) == "data/features/genomics.txt"


def test_resolve_genomics_text_path_falls_back_to_teacher_then_legacy_prompt_files() -> None:
    teacher_row = {
        "genomics_teacher_text_path": "data/features/teacher.txt",
        "genomics_gdisc_text_path": "data/features/gdisc.txt",
        "genomics_llm_input_text_path": "data/features/llm_input.txt",
    }
    gdisc_row = {
        "genomics_gdisc_text_path": "data/features/gdisc.txt",
        "genomics_llm_input_text_path": "data/features/llm_input.txt",
    }

    assert resolve_genomics_text_path(teacher_row) == "data/features/teacher.txt"
    assert resolve_genomics_text_path(gdisc_row) == "data/features/gdisc.txt"


def test_resolve_genomics_text_path_can_disable_legacy_fallback() -> None:
    row = {"genomics_teacher_text_path": "data/features/teacher.txt"}

    assert resolve_genomics_text_path(row, allow_legacy_fallback=False) == ""
