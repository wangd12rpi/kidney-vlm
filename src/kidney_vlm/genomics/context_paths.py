from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _first_nonempty(
    row: Mapping[str, Any],
    columns: list[str],
) -> str:
    for column in columns:
        text = _clean_text(row.get(column))
        if text:
            return text
    return ""


def resolve_clinical_text_path(
    row: Mapping[str, Any],
) -> str:
    """Return the canonical clinical text path for a registry row."""
    return _first_nonempty(row, ["genomics_clinical_text_path"])


def resolve_genomics_text_path(
    row: Mapping[str, Any],
    *,
    allow_legacy_fallback: bool = True,
) -> str:
    """Return the preferred genomics text path for a registry row.

    Canonical path order:
      1. genomics_genomics_text_path

    Legacy fallback order (when enabled):
      2. genomics_teacher_text_path
      3. genomics_gdisc_text_path
      4. genomics_llm_input_text_path
    """
    canonical = _first_nonempty(row, ["genomics_genomics_text_path"])
    if canonical or not allow_legacy_fallback:
        return canonical
    return _first_nonempty(
        row,
        [
            "genomics_teacher_text_path",
            "genomics_gdisc_text_path",
            "genomics_llm_input_text_path",
        ],
    )
