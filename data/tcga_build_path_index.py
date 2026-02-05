"""Build a JSONL index for TCGA kidney pathology slides (SVS) via the GDC API.

Outputs:
- data/tcga_path_index.jsonl (one record per case, 1 SVS selected)

Notes
- Only open-access SVS slide images are included.
- Default: select 1 SVS per case (prefer Primary Tumor sample code 01, then DX1).

Run:
  python data/tcga_build_path_index.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kidney_vlm.gdc import iter_gdc_files
from kidney_vlm.jsonl import write_jsonl


# -----------------
# Config (edit me)
# -----------------
PROJECTS = ["TCGA-KIRC"]
OUT_INDEX = Path("data/tcga_path_index.jsonl")
RAW_ROOT = Path("data/raw/tcga/path")

# Query paging
PAGE_SIZE = 500

# Selection policy: 1 SVS per case
PREFER_TCGA_SAMPLE_CODES = ["01"]  # 01 = Primary Tumor
PREFER_DX_NUMBER = 1  # prefer DX1 if present


# -----------------
# Helpers
# -----------------

def parse_tcga_sample_code(file_name: str) -> str | None:
    """Extract TCGA sample type code from slide filename.

    Typical filename:
      TCGA-AB-1234-01Z-00-DX1.<UUID>.svs

    sample token is the 4th dash-separated token, like 01Z.
    sample code is first two chars: 01.
    """
    base = file_name.split(".")[0]
    parts = base.split("-")
    if len(parts) < 4:
        return None
    token = parts[3]
    return token[:2]


def parse_dx_number(file_name: str) -> int | None:
    m = re.search(r"-DX(\d+)", file_name)
    if not m:
        return None
    return int(m.group(1))


def rank_slide(file_name: str) -> Tuple[int, int, str]:
    """Lower is better."""
    sample_code = parse_tcga_sample_code(file_name)
    dx_num = parse_dx_number(file_name)

    sample_rank = 1
    if sample_code in PREFER_TCGA_SAMPLE_CODES:
        sample_rank = 0

    dx_rank = 1
    if dx_num is not None and dx_num == PREFER_DX_NUMBER:
        dx_rank = 0

    return (sample_rank, dx_rank, file_name)


# -----------------
# Main
# -----------------

def main() -> None:
    filters: Dict[str, Any] = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": PROJECTS}},
            {"op": "=", "content": {"field": "files.data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "files.data_format", "value": "SVS"}},
            {"op": "=", "content": {"field": "files.experimental_strategy", "value": "Tissue Slide"}},
            {"op": "=", "content": {"field": "files.access", "value": "open"}},
        ],
    }

    fields = [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "data_type",
        "data_format",
        "experimental_strategy",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.diagnoses.primary_diagnosis",
        "cases.diagnoses.tumor_stage",
        "cases.diagnoses.tumor_grade",
    ]

    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    n_hits = 0
    for hit in iter_gdc_files(filters=filters, fields=fields, page_size=PAGE_SIZE):
        n_hits += 1
        cases = hit.get("cases") or []
        if not cases:
            continue
        case = cases[0]
        case_id = case.get("case_id")
        if not case_id:
            continue
        by_case[case_id].append(hit)

    print(f"GDC hits (files): {n_hits}")
    print(f"Unique cases: {len(by_case)}")

    rows: List[Dict[str, Any]] = []

    per_project_cases = defaultdict(int)

    for case_id, hits in sorted(by_case.items(), key=lambda x: x[0]):
        # Choose best slide for this case
        hits_sorted = sorted(hits, key=lambda h: rank_slide(h.get("file_name", "")))
        h0 = hits_sorted[0]

        file_id = h0["file_id"]
        file_name = h0["file_name"]
        file_size = int(h0.get("file_size") or 0)
        md5sum = h0.get("md5sum")

        case = (h0.get("cases") or [])[0]
        case_submitter_id = case.get("submitter_id")
        project_id = (case.get("project") or {}).get("project_id")
        if not project_id:
            project_id = "UNKNOWN"

        # Optional clinical info (may be list)
        diagnoses = case.get("diagnoses") or []
        primary_diagnosis = diagnoses[0].get("primary_diagnosis") if diagnoses else None
        tumor_stage = diagnoses[0].get("tumor_stage") if diagnoses else None
        tumor_grade = diagnoses[0].get("tumor_grade") if diagnoses else None

        label = project_id.replace("TCGA-", "")

        out_dir = RAW_ROOT / project_id / case_submitter_id
        raw_svs_path = out_dir / file_name
        tiles_dir = out_dir / "tiles"

        per_project_cases[project_id] += 1

        rows.append(
            {
                "dataset": "tcga",
                "modality": "path",
                "project_id": project_id,
                "label": label,
                "case_id": case_id,
                "case_submitter_id": case_submitter_id,
                "file_id": file_id,
                "file_name": file_name,
                "file_size": file_size,
                "md5sum": md5sum,
                "raw_svs_path": str(raw_svs_path),
                "tiles_dir": str(tiles_dir),
                "labels": {
                    "primary_diagnosis": primary_diagnosis,
                    "tumor_stage": tumor_stage,
                    "tumor_grade": tumor_grade,
                    "selected_from_n_slides": len(hits),
                },
            }
        )

    write_jsonl(OUT_INDEX, rows)

    print(f"\nWrote index: {OUT_INDEX} ({len(rows)} cases)")
    for k in sorted(per_project_cases.keys()):
        print(f"  {k}: {per_project_cases[k]} cases")


if __name__ == "__main__":
    main()
