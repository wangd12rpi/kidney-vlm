from __future__ import annotations

from pathlib import Path

from kidney_vlm.pathology import report_filters


def test_sample_ids_with_missing_pathology_report_forms_uses_signature_helper(monkeypatch, tmp_path: Path) -> None:
    def fake_is_missing_pathology_report_form(path_value, *, repo_root):
        return "missing" in str(path_value)

    monkeypatch.setattr(
        report_filters,
        "is_missing_pathology_report_form",
        fake_is_missing_pathology_report_form,
    )

    rows = [
        {"sample_id": "case-good", "report_pdf_paths": ["good.pdf"]},
        {"sample_id": "case-bad", "report_pdf_paths": ["missing.pdf"]},
        {"sample_id": "case-empty", "report_pdf_paths": []},
    ]

    sample_ids = report_filters.sample_ids_with_missing_pathology_report_forms(
        rows,
        repo_root=tmp_path,
    )

    assert sample_ids == {"case-bad"}
