from __future__ import annotations

from pathlib import Path
from typing import Any

from kidney_vlm.vqa.stage_config import clean_text, slugify_label

MAX_CACHE_FILENAME_CHARS = 240


def llm_cache_slug(model_name_or_path: str) -> str:
    parts = [part for part in str(model_name_or_path).strip().split("/") if part]
    return slugify_label(parts[-1] if parts else model_name_or_path, default="llm")


def repo_relative_path(repo_root: Path, path_value: str | Path) -> Path:
    text = clean_text(path_value)
    if not text:
        raise ValueError("Cache paths must not be empty.")
    path = Path(text).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (repo_root / path).resolve()
    try:
        return resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Cache input path must live inside the project root: {path_value}") from exc


def safe_cache_component(value: Any) -> str:
    text = clean_text(value)
    if not text:
        raise ValueError("Cannot build cache path from an empty component.")
    pieces: list[str] = []
    for character in text:
        if character == "/":
            pieces.append("__")
        elif character == "\\":
            pieces.append("__")
        elif character.isspace():
            pieces.append("_")
        elif character in {"<", ">", "|", "\0"}:
            pieces.append("_")
        else:
            pieces.append(character)
    return "".join(pieces).strip("._") or "value"


def projector_cache_dir_name(*, repo_root: Path, modality: str, checkpoint_path: str | Path) -> str:
    relative_checkpoint = repo_relative_path(repo_root, checkpoint_path)
    return safe_cache_component(f"{modality}__{relative_checkpoint.parent.name}__{relative_checkpoint.name}")


def feature_cache_relative_path(feature_ref: str | Path) -> Path:
    text = clean_text(feature_ref)
    if not text:
        raise ValueError("Cannot build prefix cache path for an empty feature reference.")
    if Path(text).expanduser().is_absolute():
        raise ValueError(f"Feature cache references must be project-relative, got absolute path: {text}")

    if "::" in text:
        left, right = text.split("::", 1)
        left_component = safe_cache_component(left)
        right_filename = f"{safe_cache_component(right)}.pt"
        if len(right_filename) > MAX_CACHE_FILENAME_CHARS:
            raise ValueError(f"Feature cache filename is too long without hashing: {right_filename}")
        return Path(f"{left_component}__ref") / right_filename

    filename = f"{safe_cache_component(text)}.pt"
    if len(filename) > MAX_CACHE_FILENAME_CHARS:
        raise ValueError(f"Feature cache filename is too long without hashing: {filename}")
    return Path(filename)


def prefix_cache_path(
    *,
    repo_root: Path,
    cache_root: str | Path,
    model_name_or_path: str,
    modality: str,
    checkpoint_path: str | Path,
    feature_ref: str | Path,
) -> Path:
    relative_cache_root = repo_relative_path(repo_root, cache_root)
    return (
        repo_root.resolve()
        / relative_cache_root
        / llm_cache_slug(model_name_or_path)
        / projector_cache_dir_name(repo_root=repo_root, modality=modality, checkpoint_path=checkpoint_path)
        / feature_cache_relative_path(feature_ref)
    )


def discover_prefix_cache_path(
    *,
    repo_root: Path,
    cache_root: str | Path,
    model_name_or_path: str,
    modality: str,
    feature_ref: str | Path,
) -> Path:
    relative_cache_root = repo_relative_path(repo_root, cache_root)
    model_root = repo_root.resolve() / relative_cache_root / llm_cache_slug(model_name_or_path)
    feature_path = feature_cache_relative_path(feature_ref)
    candidates = sorted(
        cache_dir / feature_path
        for cache_dir in model_root.glob(f"{modality}__*")
        if (cache_dir / feature_path).is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Cached {modality} prefix not found under {model_root} for feature reference {clean_text(feature_ref)!r}"
        )
    if len(candidates) > 1:
        relative_candidates = []
        for path in candidates:
            try:
                relative_candidates.append(path.relative_to(repo_root.resolve()).as_posix())
            except ValueError:
                relative_candidates.append(str(path))
        raise ValueError(
            f"Ambiguous cached {modality} prefix for feature reference {clean_text(feature_ref)!r}: {relative_candidates}"
        )
    return candidates[0]
