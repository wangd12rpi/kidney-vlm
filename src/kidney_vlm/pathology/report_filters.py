from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

MISSING_PATHOLOGY_REPORT_SIGNATURES = (
    "tcga missing pathology report form",
    "pathology report is not available",
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_local_path(path_value: str, *, repo_root: Path) -> Path:
    path = Path(str(path_value).strip()).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


@lru_cache(maxsize=32768)
def _pdf_head_text(path_key: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required. Install it with: uv add pypdf") from exc

    pdf_path = Path(path_key)
    if not pdf_path.exists():
        return ""

    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return ""

    chunks: list[str] = []
    for page in reader.pages[:1]:
        text = (page.extract_text() or "").strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def is_missing_pathology_report_form(path_value: str | Path, *, repo_root: Path) -> bool:
    normalized_path = _normalize_local_path(str(path_value), repo_root=repo_root)
    text = _pdf_head_text(str(normalized_path)).lower()
    if not text:
        return False
    return any(signature in text for signature in MISSING_PATHOLOGY_REPORT_SIGNATURES)


def sample_ids_with_missing_pathology_report_forms(
    rows: Iterable[dict[str, Any]],
    *,
    repo_root: Path,
    sample_id_key: str = "sample_id",
    report_paths_key: str = "report_pdf_paths",
    progress_desc: str | None = None,
    total: int | None = None,
) -> set[str]:
    sample_ids: set[str] = set()
    iterable: Iterable[dict[str, Any]] = rows
    if progress_desc:
        iterable = tqdm(
            rows,
            total=total,
            desc=progress_desc,
            unit="row",
            leave=False,
            dynamic_ncols=True,
        )
    for row in iterable:
        sample_id = str(row.get(sample_id_key, "")).strip()
        if not sample_id:
            continue
        report_paths = _as_list(row.get(report_paths_key))
        if any(is_missing_pathology_report_form(path_value, repo_root=repo_root) for path_value in report_paths):
            sample_ids.add(sample_id)
    return sample_ids
