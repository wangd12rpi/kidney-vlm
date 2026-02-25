from __future__ import annotations

from pathlib import Path


def find_repo_root(start_path: str | Path) -> Path:
    """Resolve repository root by walking parents until a .git marker is found."""
    candidate = Path(start_path).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent

    raise FileNotFoundError(f"Unable to locate repository root from start path: {start_path}")
