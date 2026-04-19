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


def test_is_missing_pathology_report_form_remaps_legacy_repo_and_report_path(monkeypatch, tmp_path: Path) -> None:
    live_pdf = (
        tmp_path
        / "data/raw/tcga/report_pathology/TCGA-GBM/TCGA-02-0003/TCGA-02-0003.legacy.PDF"
    )
    live_pdf.parent.mkdir(parents=True, exist_ok=True)
    live_pdf.write_bytes(b"%PDF-1.4\n")

    captured: list[str] = []

    def fake_pdf_head_text(path_key: str) -> str:
        captured.append(path_key)
        return "TCGA Missing Pathology Report Form"

    monkeypatch.setattr(report_filters, "_pdf_head_text", fake_pdf_head_text)

    legacy_root = tmp_path / "old-worktree"
    legacy_path = legacy_root / "data/raw/tcga/reports/TCGA-GBM/TCGA-02-0003/TCGA-02-0003.legacy.PDF"
    assert report_filters.is_missing_pathology_report_form(legacy_path, repo_root=tmp_path) is True
    assert captured == [str(live_pdf.resolve())]


def test_is_missing_pathology_report_form_raises_when_path_cannot_be_resolved(tmp_path: Path) -> None:
    missing_path = "data/raw/tcga/reports/TCGA-GBM/TCGA-02-9999/not-there.PDF"
    try:
        report_filters.is_missing_pathology_report_form(missing_path, repo_root=tmp_path)
    except FileNotFoundError as exc:
        assert "could not be resolved" in str(exc)
    else:
        raise AssertionError("Expected unresolved pathology report path to raise FileNotFoundError.")
