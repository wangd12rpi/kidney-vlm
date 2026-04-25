"""Utilities for TCGA genomics text-context extraction."""

from .context_paths import resolve_clinical_text_path, resolve_genomics_text_path

__all__ = [
    "resolve_clinical_text_path",
    "resolve_genomics_text_path",
]
