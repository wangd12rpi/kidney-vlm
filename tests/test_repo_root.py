from __future__ import annotations

from pathlib import Path

from kidney_vlm.repo_root import find_repo_root


def test_find_repo_root_from_file_and_directory() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    from_file = find_repo_root(Path(__file__))
    from_directory = find_repo_root(repo_root / "scripts" / "data")

    assert from_file == repo_root
    assert from_directory == repo_root
    assert (repo_root / ".git").exists()
