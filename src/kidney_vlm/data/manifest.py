from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_short_sha(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip() or "unknown"
    except Exception:
        return "unknown"


def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_run_manifest(
    manifests_root: str | Path,
    repo_root: str | Path,
    source_name: str,
    source_row_count: int,
    staging_path: str | Path,
    unified_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    manifests_dir = Path(manifests_root)
    repo_path = Path(repo_root)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _now_utc_stamp()
    git_sha = _git_short_sha(repo_path)
    manifest = {
        "timestamp_utc": timestamp,
        "git_short_sha": git_sha,
        "source": source_name,
        "source_row_count": int(source_row_count),
        "staging_path": str(Path(staging_path)),
        "unified_path": str(Path(unified_path)),
    }
    if extra:
        manifest["extra"] = extra

    output_path = manifests_dir / f"{timestamp}-{git_sha}.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
