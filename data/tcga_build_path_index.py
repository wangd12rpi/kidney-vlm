"""Build a JSONL index for TCGA kidney pathology slides (SVS) via the GDC API.

Outputs:
- data/tcga_path_index.jsonl (one record per case, 1 SVS selected)

Notes
- Only open-access SVS slide images are included.
- Default: select 1 SVS per case (prefer Primary Tumor sample code 01, then DX1).
- labels{} is intentionally "task-rich" so you can evaluate multiple typical TCGA WSI tasks.

Common WSI tasks (weakly supervised / slide-level supervision) often use:
- tumor subtype / histology (here: KICH/KIRC/KIRP)
- tumor grade (G1-G4 where available)
- tumor stage (coarse Stage I-IV where available)
- survival/vital status (Alive/Dead) and recurrence flags
- basic demographics (gender/age bins)

Run:
  python data/tcga_build_path_index.py
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kidney_vlm.gdc import iter_gdc_files
from kidney_vlm.jsonl import write_jsonl


# -----------------
# Config (edit me)
# -----------------
# Start small by default (KICH). You can expand to all 3:
#   ["TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP"]
PROJECTS = ["TCGA-KICH"]

OUT_INDEX = Path("data/tcga_path_index.jsonl")
RAW_ROOT = Path("data/raw/tcga/path")

# Query paging
PAGE_SIZE = 500

# Selection policy: 1 SVS per case
PREFER_TCGA_SAMPLE_CODES = ["01"]  # 01 = Primary Tumor; set ["01","11"] if you want tumor-vs-normal tasks
PREFER_DX_NUMBER = 1  # prefer DX1 if present

# Derived label bins
AGE_BINS = [
    ("<50", 0, 50),
    ("50-59", 50, 60),
    ("60-69", 60, 70),
    ("70+", 70, 200),
]


# -----------------
# Helpers: slide parsing
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
# Helpers: label normalization
# -----------------
def _norm_str(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def normalize_gender(x: Any) -> str | None:
    s = _norm_str(x)
    if not s:
        return None
    t = s.strip().lower()
    if t == "male":
        return "Male"
    if t == "female":
        return "Female"
    return s


def normalize_vital_status(x: Any) -> str | None:
    s = _norm_str(x)
    if not s:
        return None
    t = s.strip().lower()
    if t == "alive":
        return "Alive"
    if t == "dead":
        return "Dead"
    return None


def normalize_yes_no(x: Any) -> str | None:
    s = _norm_str(x)
    if not s:
        return None
    t = s.strip().lower()
    if t in ["yes", "y", "true", "1"]:
        return "Yes"
    if t in ["no", "n", "false", "0"]:
        return "No"
    return None


def normalize_tumor_grade(raw: Any) -> str | None:
    s = _norm_str(raw)
    if not s:
        return None
    t = s.upper()

    m = re.search(r"\bG\s*([1-4])\b", t)
    if m:
        return f"G{m.group(1)}"

    m = re.search(r"\bGRADE\s*([1-4])\b", t)
    if m:
        return f"G{m.group(1)}"

    m = re.search(r"\bGRADE\s*([IV]{1,3}|IV)\b", t)
    if m:
        roman = m.group(1)
        roman_to_num = {"I": 1, "II": 2, "III": 3, "IV": 4}
        n = roman_to_num.get(roman)
        if n:
            return f"G{n}"

    return None


def normalize_tumor_stage_coarse(raw: Any) -> str | None:
    """Coarse map to STAGE I/II/III/IV."""
    s = _norm_str(raw)
    if not s:
        return None
    t = s.upper()

    m = re.search(r"STAGE\s*([IVX]+|\d+)", t)
    if not m:
        return None

    tok = m.group(1)
    if tok.isdigit():
        n = int(tok)
        num_to_roman = {1: "I", 2: "II", 3: "III", 4: "IV"}
        r = num_to_roman.get(n)
        return f"STAGE {r}" if r else None

    roman = tok
    roman_to_num = {"I": 1, "II": 2, "III": 3, "IV": 4}
    # If "IIIA" etc, keep prefix roman block
    roman = re.sub(r"[^IVX].*$", "", roman)
    n = roman_to_num.get(roman)
    if not n:
        return None
    num_to_roman = {1: "I", 2: "II", 3: "III", 4: "IV"}
    return f"STAGE {num_to_roman[n]}"


def primary_diagnosis_coarse(raw: Any) -> str | None:
    s = _norm_str(raw)
    if not s:
        return None
    t = s.lower()
    if "clear cell" in t:
        return "CLEAR CELL RCC"
    if "papillary" in t:
        return "PAPILLARY RCC"
    if "chromophobe" in t:
        return "CHROMOPHOBE RCC"
    return None


def age_years_from_days(age_at_diagnosis_days: Any) -> float | None:
    s = _norm_str(age_at_diagnosis_days)
    if not s:
        return None
    try:
        days = float(s)
    except ValueError:
        return None
    return days / 365.25


def age_bin_from_years(age_years: float | None) -> str | None:
    if age_years is None:
        return None
    for name, lo, hi in AGE_BINS:
        if age_years >= lo and age_years < hi:
            return name
    return None


def pick_best_diagnosis(diagnoses: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not diagnoses:
        return None

    keys = [
        "primary_diagnosis",
        "tumor_stage",
        "tumor_grade",
        "age_at_diagnosis",
        "vital_status",
        "days_to_death",
        "days_to_last_follow_up",
        "days_to_recurrence",
        "progression_or_recurrence",
        "prior_malignancy",
    ]

    def score(d: Dict[str, Any]) -> int:
        s = 0
        for k in keys:
            v = d.get(k)
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            s += 1
        return s

    return sorted(diagnoses, key=score, reverse=True)[0]


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

    # IMPORTANT: Keep fields conservative (known-stable), but task-rich.
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
        # diagnoses (task labels)
        "cases.diagnoses.primary_diagnosis",
        "cases.diagnoses.tumor_stage",
        "cases.diagnoses.tumor_grade",
        "cases.diagnoses.age_at_diagnosis",
        "cases.diagnoses.vital_status",
        "cases.diagnoses.days_to_death",
        "cases.diagnoses.days_to_last_follow_up",
        "cases.diagnoses.days_to_recurrence",
        "cases.diagnoses.last_known_disease_status",
        "cases.diagnoses.prior_malignancy",
        "cases.diagnoses.progression_or_recurrence",
        "cases.diagnoses.classification_of_tumor",
        "cases.diagnoses.morphology",
        "cases.diagnoses.site_of_resection_or_biopsy",
        "cases.diagnoses.tissue_or_organ_of_origin",
        # demographic (task labels)
        "cases.demographic.gender",
        "cases.demographic.race",
        "cases.demographic.ethnicity",
        "cases.demographic.year_of_birth",
        "cases.demographic.year_of_death",
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
        hits_sorted = sorted(hits, key=lambda h: rank_slide(h.get("file_name", "")))
        h0 = hits_sorted[0]

        file_id = h0["file_id"]
        file_name = h0["file_name"]
        file_size = int(h0.get("file_size") or 0)
        md5sum = h0.get("md5sum")

        case = (h0.get("cases") or [])[0]
        case_submitter_id = case.get("submitter_id")
        project_id = (case.get("project") or {}).get("project_id") or "UNKNOWN"

        label_project = project_id.replace("TCGA-", "")

        slide_sample_code = parse_tcga_sample_code(file_name)
        slide_dx_number = parse_dx_number(file_name)

        diagnoses = case.get("diagnoses") or []
        dx = pick_best_diagnosis(diagnoses) or {}
        dx_best_of_n = len(diagnoses)

        primary_diagnosis = dx.get("primary_diagnosis")
        tumor_stage_raw = dx.get("tumor_stage")
        tumor_grade_raw = dx.get("tumor_grade")
        age_at_diagnosis_days = dx.get("age_at_diagnosis")
        vital_status_raw = dx.get("vital_status")
        days_to_death = dx.get("days_to_death")
        days_to_last_follow_up = dx.get("days_to_last_follow_up")
        days_to_recurrence = dx.get("days_to_recurrence")
        last_known_disease_status = dx.get("last_known_disease_status")
        prior_malignancy_raw = dx.get("prior_malignancy")
        progression_or_recurrence_raw = dx.get("progression_or_recurrence")
        classification_of_tumor = dx.get("classification_of_tumor")
        morphology = dx.get("morphology")
        site_of_resection_or_biopsy = dx.get("site_of_resection_or_biopsy")
        tissue_or_organ_of_origin = dx.get("tissue_or_organ_of_origin")

        tumor_grade = normalize_tumor_grade(tumor_grade_raw)
        tumor_stage_coarse = normalize_tumor_stage_coarse(tumor_stage_raw)
        vital_status = normalize_vital_status(vital_status_raw)

        age_years = age_years_from_days(age_at_diagnosis_days)
        age_bin = age_bin_from_years(age_years)

        prior_malignancy = normalize_yes_no(prior_malignancy_raw)
        progression_or_recurrence = normalize_yes_no(progression_or_recurrence_raw)

        primary_dx_coarse = primary_diagnosis_coarse(primary_diagnosis)

        demo = case.get("demographic") or {}
        if isinstance(demo, list):
            demo = demo[0] if demo else {}
        gender = normalize_gender(demo.get("gender"))
        race = _norm_str(demo.get("race"))
        ethnicity = _norm_str(demo.get("ethnicity"))
        year_of_birth = demo.get("year_of_birth")
        year_of_death = demo.get("year_of_death")

        out_dir = RAW_ROOT / project_id / case_submitter_id
        raw_svs_path = out_dir / file_name
        tiles_dir = out_dir / "tiles"

        per_project_cases[project_id] += 1

        rows.append(
            {
                "dataset": "tcga",
                "modality": "path",
                "project_id": project_id,
                "label": label_project,
                "case_id": case_id,
                "case_submitter_id": case_submitter_id,
                "file_id": file_id,
                "file_name": file_name,
                "file_size": file_size,
                "md5sum": md5sum,
                "raw_svs_path": str(raw_svs_path),
                "tiles_dir": str(tiles_dir),
                "labels": {
                    # convenient always-present task
                    "tcga_kidney_subtype": label_project,
                    "tcga_project_id": project_id,
                    # slide filename-derived labels
                    "slide_sample_code": slide_sample_code,
                    "slide_dx_number": slide_dx_number,
                    # raw clinical labels
                    "primary_diagnosis": primary_diagnosis,
                    "tumor_stage_raw": tumor_stage_raw,
                    "tumor_grade_raw": tumor_grade_raw,
                    "age_at_diagnosis_days": age_at_diagnosis_days,
                    "vital_status_raw": vital_status_raw,
                    "days_to_death": days_to_death,
                    "days_to_last_follow_up": days_to_last_follow_up,
                    "days_to_recurrence": days_to_recurrence,
                    "last_known_disease_status": last_known_disease_status,
                    "prior_malignancy_raw": prior_malignancy_raw,
                    "progression_or_recurrence_raw": progression_or_recurrence_raw,
                    "classification_of_tumor": classification_of_tumor,
                    "morphology": morphology,
                    "site_of_resection_or_biopsy": site_of_resection_or_biopsy,
                    "tissue_or_organ_of_origin": tissue_or_organ_of_origin,
                    # normalized / derived labels (recommended for evaluation)
                    "primary_diagnosis_coarse": primary_dx_coarse,
                    "tumor_stage_coarse": tumor_stage_coarse,
                    "tumor_grade": tumor_grade,
                    "vital_status": vital_status,
                    "age_at_diagnosis_years": round(age_years, 2) if age_years is not None else None,
                    "age_bin": age_bin,
                    "prior_malignancy": prior_malignancy,
                    "progression_or_recurrence": progression_or_recurrence,
                    # demographics
                    "gender": gender,
                    "race": race,
                    "ethnicity": ethnicity,
                    "year_of_birth": year_of_birth,
                    "year_of_death": year_of_death,
                    # bookkeeping
                    "selected_from_n_slides": len(hits),
                    "selected_from_n_diagnoses": dx_best_of_n,
                },
            }
        )

    write_jsonl(OUT_INDEX, rows)

    print(f"\nWrote index: {OUT_INDEX} ({len(rows)} cases)")
    for k in sorted(per_project_cases.keys()):
        print(f"  {k}: {per_project_cases[k]} cases")


if __name__ == "__main__":
    main()
