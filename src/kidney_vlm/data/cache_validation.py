from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheValidationResult:
    path: Path
    exists: bool
    size_bytes: int
    valid: bool
    reason: str


def validate_cached_file(path: str | Path, min_size_bytes: int = 1, validate_size: bool = True) -> CacheValidationResult:
    file_path = Path(path)
    if not file_path.exists():
        return CacheValidationResult(file_path, False, 0, False, "missing")
    if not file_path.is_file():
        return CacheValidationResult(file_path, False, 0, False, "not_a_file")

    size_bytes = file_path.stat().st_size
    if validate_size and size_bytes < min_size_bytes:
        return CacheValidationResult(file_path, True, size_bytes, False, "too_small")
    return CacheValidationResult(file_path, True, size_bytes, True, "ok")
