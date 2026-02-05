from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class TCGACase:
    project_id: str
    case_id: str
    case_submitter_id: str
    label: str
    svs_path: Path
    tiles_dir: Path


def load_tcga_path_index(index_path: str | Path) -> List[TCGACase]:
    index_path = Path(index_path)
    cases: List[TCGACase] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                TCGACase(
                    project_id=obj["project_id"],
                    case_id=obj["case_id"],
                    case_submitter_id=obj["case_submitter_id"],
                    label=obj["label"],
                    svs_path=Path(obj["raw_svs_path"]),
                    tiles_dir=Path(obj["tiles_dir"]),
                )
            )
    return cases


def get_tile_paths(case: TCGACase, max_tiles: int) -> List[Path]:
    # Prefer manifest ordering if available.
    manifest = case.tiles_dir / "tiles_manifest.jsonl"
    if manifest.exists():
        out: List[Path] = []
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if len(out) >= max_tiles:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.append(case.tiles_dir / obj["tile_relpath"])
        out = [p for p in out if p.exists()]
        return out

    # Fallback: glob
    tiles = sorted(case.tiles_dir.glob("*.png")) + sorted(case.tiles_dir.glob("*.jpg"))
    return [p for p in tiles[:max_tiles] if p.exists()]
