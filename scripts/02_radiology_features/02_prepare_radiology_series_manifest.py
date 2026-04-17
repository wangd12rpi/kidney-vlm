#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.radiology.dicom_qc import Config as DicomQCConfig
from kidney_vlm.script_config import load_script_cfg
from kidney_vlm.radiology.tcga_series_manifest import (
    build_tcia_series_result_without_qc,
    empty_series_manifest_frame,
    ensure_series_manifest_df,
    extract_tcia_series_zip,
    infer_modality_from_qc_result,
    read_series_manifest,
    series_entries_from_registry,
    to_registry_relative_path,
    write_series_manifest,
    write_tcia_qc_detail,
    write_tcia_qc_report,
    run_tcia_series_qc,
)


ROOT = find_repo_root(Path(__file__))


def _qc_config_from_cfg(cfg) -> DicomQCConfig:
    qc_cfg = cfg.radiology.qc
    return DicomQCConfig(
        min_rows=int(qc_cfg.min_rows),
        min_cols=int(qc_cfg.min_cols),
        min_usable_images=int(qc_cfg.min_usable_images),
        max_rejected_fraction_for_series=float(qc_cfg.max_rejected_fraction_for_series),
        reject_multiframe_series=bool(qc_cfg.reject_multiframe_series),
        reject_derived_series=bool(qc_cfg.reject_derived_series),
        decode_pixels=bool(qc_cfg.decode_pixels),
        force_read=bool(qc_cfg.force_read),
        verbose=False,
    )


def main() -> None:
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="02_radiology_features/02_prepare_radiology_series_manifest.yaml",
        overrides=sys.argv[1:],
    )

    registry_path = Path(str(cfg.radiology.registry_path))
    manifest_path = Path(str(cfg.radiology.series_manifest_path))
    detail_root = Path(str(cfg.radiology.qc.detail_root))
    qc_report_path = Path(str(cfg.radiology.qc.report_jsonl_path))
    skip_existing_extraction = bool(cfg.radiology.skip_existing_extraction)
    skip_existing_qc = bool(cfg.radiology.skip_existing_qc)
    run_qc = bool(cfg.radiology.qc.enabled)
    series_limit_text = str(cfg.radiology.get("series_limit", "")).strip()
    series_limit = int(series_limit_text) if series_limit_text else None

    registry_df = read_parquet_or_empty(registry_path)
    manifest_df = read_series_manifest(manifest_path)
    existing_by_zip = {
        str(row["source_zip_relpath"]).strip(): row.to_dict()
        for _, row in manifest_df.iterrows()
        if str(row.get("source_zip_relpath", "")).strip()
    }

    records = series_entries_from_registry(registry_df, root_dir=ROOT)
    if series_limit is not None:
        records = records[:series_limit]

    if not records:
        write_series_manifest(empty_series_manifest_frame(), manifest_path)
        print("No radiology series zip files were found in the registry.")
        return

    qc_cfg = _qc_config_from_cfg(cfg)
    prepared_rows: list[dict[str, object]] = []
    qc_report_entries: list[dict[str, object]] = []

    for record in tqdm(records, desc="Preparing radiology series", unit="series"):
        source_zip_relpath = str(record["source_zip_relpath"]).strip()
        existing_row = dict(existing_by_zip.get(source_zip_relpath, {}))
        if skip_existing_qc and bool(existing_row.get("processed_qc")):
            prepared_rows.append(existing_row)
            continue

        zip_path = Path(str(record["source_zip_path"]))
        if not zip_path.is_file():
            print(f"[warning] Missing radiology series zip, skipping: {zip_path}")
            continue

        extracted_root = extract_tcia_series_zip(
            zip_path=zip_path,
            extracted_root=zip_path.with_suffix(""),
            skip_existing=skip_existing_extraction,
        )
        qc_result = (
            run_tcia_series_qc(extracted_root=extracted_root, qc_cfg=qc_cfg)
            if run_qc
            else build_tcia_series_result_without_qc(extracted_root=extracted_root)
        )
        accepted = bool(qc_result["accepted"])
        selected_series_dir = str(qc_result.get("selected_series_dir", "")).strip()
        source_series_dir_relpath = ""
        if selected_series_dir:
            source_series_dir_relpath = to_registry_relative_path(ROOT, selected_series_dir)
        elif qc_result.get("series_reject_record"):
            source_series_dir_relpath = to_registry_relative_path(
                ROOT,
                str((qc_result["series_reject_record"] or {}).get("series_dir", "")).strip(),
            )

        manifest_row = existing_row or {}
        manifest_row.update(record)
        manifest_row.update(
            {
                "series_zip_path": str(zip_path),
                "extracted_root": str(extracted_root),
                "selected_series_dir": selected_series_dir,
                "source_series_dir_relpath": source_series_dir_relpath,
                "accepted": accepted,
                "processed_qc": True,
                "reject_reason": str((qc_result.get("series_reject_record") or {}).get("reject_reason", "")).strip(),
                "reject_details": str((qc_result.get("series_reject_record") or {}).get("reject_details", "")).strip(),
                "candidate_series_dirs": list(qc_result.get("candidate_series_dirs", []) or []),
                "usable_dicom_paths": list(qc_result.get("usable_image_paths", []) or []),
                "source_file_paths": list(qc_result.get("source_file_paths", []) or []),
                "all_image_records": list(qc_result.get("all_image_records", []) or []),
                "accepted_image_records": [
                    image_record
                    for image_record in list(qc_result.get("all_image_records", []) or [])
                    if not str((image_record or {}).get("reject_reason", "")).strip()
                ],
                "image_reject_records": list(qc_result.get("image_reject_records", []) or []),
                "slice_count": len(list(qc_result.get("usable_image_paths", []) or [])),
            }
        )
        if not str(manifest_row.get("modality", "")).strip():
            manifest_row["modality"] = infer_modality_from_qc_result(qc_result)

        detail_payload = dict(manifest_row)
        detail_payload["qc_result"] = qc_result
        detail_path = write_tcia_qc_detail(detail_payload, detail_root=detail_root)
        manifest_row["qc_detail_path"] = to_registry_relative_path(ROOT, detail_path)
        prepared_rows.append(manifest_row)
        qc_report_entries.append(
            {
                "collection": manifest_row["collection"],
                "patient_id": manifest_row["patient_id"],
                "study_instance_uid": manifest_row["study_instance_uid"],
                "series_instance_uid": manifest_row["series_instance_uid"],
                "modality": manifest_row["modality"],
                "accepted": manifest_row["accepted"],
                "reject_reason": manifest_row["reject_reason"],
                "reject_details": manifest_row["reject_details"],
                "source_zip_relpath": manifest_row["source_zip_relpath"],
                "source_series_dir_relpath": manifest_row["source_series_dir_relpath"],
                "qc_detail_path": manifest_row["qc_detail_path"],
            }
        )

    output_df = ensure_series_manifest_df(pd.DataFrame(prepared_rows))
    write_series_manifest(output_df, manifest_path)
    if qc_report_entries:
        write_tcia_qc_report(qc_report_entries, qc_report_path)

    accepted_count = int(output_df["accepted"].sum()) if not output_df.empty else 0
    print(f"Radiology series rows written: {len(output_df)}")
    print(f"Accepted radiology series: {accepted_count}")
    print(f"Series manifest: {manifest_path}")
    print(f"QC detail root: {detail_root}")
    print(f"QC report: {qc_report_path}")


if __name__ == "__main__":
    main()
