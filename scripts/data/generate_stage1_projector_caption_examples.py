#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and print example stage-1 projector caption strings from paired TCGA+TCIA cases.",
    )
    parser.add_argument(
        "--registry",
        default="/Users/wdn/tsa/kidney-vlm/data/registry/unified.parquet",
        help="Path to unified parquet registry.",
    )
    parser.add_argument(
        "--source",
        default="tcga",
        help="Source filter (default: tcga).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of examples to print.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--download-dir",
        default="/Users/wdn/tsa/kidney-vlm/data/tmp/report_pdfs",
        help="Where to save fetched PDF reports.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download PDFs, only print download URLs.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout for GDC requests.",
    )
    parser.add_argument(
        "--max-pathology-items",
        type=int,
        default=3,
        help="Max pathology items included per caption body.",
    )
    parser.add_argument(
        "--max-radiology-items",
        type=int,
        default=3,
        help="Max radiology items included per caption body.",
    )
    return parser.parse_args()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(x) for x in converted if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def has_items(value: Any) -> bool:
    return len(as_list(value)) > 0


def query_report_metadata(case_ids: list[str], timeout_seconds: int) -> dict[str, dict[str, str]]:
    if not case_ids:
        return {}

    payload = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.case_id",
                        "value": case_ids,
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_format",
                        "value": ["PDF"],
                    },
                },
            ],
        },
        "fields": "file_id,file_name,data_category,data_type,data_format,cases.case_id,cases.submitter_id",
        "size": max(100, len(case_ids) * 10),
    }

    response = requests.post(
        "https://api.gdc.cancer.gov/files",
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    hits = response.json().get("data", {}).get("hits", [])

    by_case: dict[str, list[dict[str, str]]] = {}
    for hit in hits:
        file_id = str(hit.get("file_id", "")).strip()
        if not file_id:
            continue
        file_name = str(hit.get("file_name", "")).strip()
        data_type = str(hit.get("data_type", "")).strip()
        data_category = str(hit.get("data_category", "")).strip()
        for case in hit.get("cases", []):
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("case_id", "")).strip()
            if not case_id:
                continue
            by_case.setdefault(case_id, []).append(
                {
                    "file_id": file_id,
                    "file_name": file_name,
                    "data_type": data_type,
                    "data_category": data_category,
                }
            )

    def score(entry: dict[str, str]) -> tuple[int, int]:
        dt = entry.get("data_type", "").lower()
        fn = entry.get("file_name", "").lower()
        priority = 0
        if "pathology report" in dt:
            priority += 3
        if "report" in dt:
            priority += 2
        if fn.endswith(".pdf"):
            priority += 1
        return (priority, -len(fn))

    selected: dict[str, dict[str, str]] = {}
    for case_id, entries in by_case.items():
        selected[case_id] = sorted(entries, key=score, reverse=True)[0]

    return selected


def download_report_pdf(file_id: str, output_path: Path, timeout_seconds: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def format_caption(
    row: pd.Series,
    pathology_paths: list[str],
    radiology_paths: list[str],
    report_meta: dict[str, str] | None,
    local_report_path: str | None,
) -> str:
    report_url = ""
    if report_meta and report_meta.get("file_id"):
        report_url = f"https://api.gdc.cancer.gov/data/{report_meta['file_id']}"

    biomarkers_text = str(row.get("biomarkers_text", "")).strip()
    if not biomarkers_text:
        biomarkers_text = "not_available"

    lines = [
        f"Case summary: project={row.get('project_id','')}; patient_id={row.get('patient_id','')}; case_id={row.get('study_id','')}; source=TCGA+TCIA.",
        f"Ground truth: primary_diagnosis={row.get('primary_diagnosis','')}; tumor_grade={row.get('tumor_grade','')}; stage={row.get('ajcc_pathologic_stage','')}; vital_status={row.get('vital_status','')}; disease_type={row.get('disease_type','')}.",
        f"Demographics: gender={row.get('gender','')}; race={row.get('race','')}; ethnicity={row.get('ethnicity','')}.",
        f"Biomarkers and clinical tags: {biomarkers_text}",
        f"Pathology context ({len(pathology_paths)} slides): {', '.join(pathology_paths) if pathology_paths else 'none'}",
        f"Radiology context ({len(radiology_paths)} studies): {', '.join(radiology_paths) if radiology_paths else 'none'}",
        "Training target (stage-1 projector): produce a dense multimodal caption maximizing biomarkers, morphology, radiology phenotype, and diagnostic evidence.",
        f"Report PDF file_id: {(report_meta or {}).get('file_id','missing')}",
        f"Report PDF filename: {(report_meta or {}).get('file_name','missing')}",
        f"Report PDF local_path: {local_report_path or 'not_downloaded'}",
        f"Report PDF url: {report_url or 'missing'}",
        "Report text to paste into caption later: <PASTE_REPORT_TEXT_HERE>",
    ]

    return " ".join(lines)


def main() -> None:
    args = parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry parquet not found: {registry_path}")

    df = pd.read_parquet(registry_path)
    if "source" not in df.columns:
        raise ValueError("Registry missing 'source' column.")

    df = df[df["source"] == str(args.source)].copy()
    if df.empty:
        raise ValueError(f"No rows found for source='{args.source}'.")

    paired_mask = df["pathology_wsi_paths"].map(has_items) & df["radiology_image_paths"].map(has_items)
    paired_df = df[paired_mask].copy()

    if paired_df.empty:
        raise ValueError("No paired pathology+radiology rows found for requested source.")

    requested = max(1, int(args.examples))
    sample_n = min(requested, len(paired_df))
    sampled = paired_df.sample(n=sample_n, random_state=int(args.seed)).reset_index(drop=True)

    case_ids = [str(x) for x in sampled["study_id"].tolist() if str(x).strip()]
    report_meta_by_case = query_report_metadata(case_ids=case_ids, timeout_seconds=int(args.timeout_seconds))

    print("=" * 120)
    print("Stage-1 Projector Caption Examples (LLaVA-Med style supervision text)")
    print(f"Registry: {registry_path}")
    print(f"Source: {args.source}")
    print(f"Paired rows available: {len(paired_df)}")
    print(f"Examples printed: {sample_n}")
    print("=" * 120)

    download_dir = Path(args.download_dir)

    for i, (_, row) in enumerate(sampled.iterrows(), start=1):
        case_id = str(row.get("study_id", "")).strip()
        patient_id = str(row.get("patient_id", "")).strip()

        pathology_paths = as_list(row.get("pathology_wsi_paths", []))[: max(1, int(args.max_pathology_items))]
        radiology_paths = as_list(row.get("radiology_image_paths", []))[: max(1, int(args.max_radiology_items))]

        report_meta = report_meta_by_case.get(case_id)
        local_report_path: str | None = None

        if report_meta and (not args.no_download):
            file_id = report_meta.get("file_id", "").strip()
            file_name = report_meta.get("file_name", "report.pdf").strip() or "report.pdf"
            if file_id:
                safe_name = file_name.replace("/", "_")
                out_path = download_dir / patient_id / safe_name
                try:
                    local_report_path = str(download_report_pdf(file_id=file_id, output_path=out_path, timeout_seconds=int(args.timeout_seconds)))
                except requests.RequestException as exc:
                    local_report_path = f"download_failed: {exc}"

        caption = format_caption(
            row=row,
            pathology_paths=pathology_paths,
            radiology_paths=radiology_paths,
            report_meta=report_meta,
            local_report_path=local_report_path,
        )

        print(f"\n--- Example {i}/{sample_n} ---")
        print(f"patient_id: {patient_id}")
        print(f"case_id: {case_id}")
        print("caption_text:")
        print(textwrap.fill(caption, width=140))


if __name__ == "__main__":
    main()
