#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.registry_io import read_parquet_or_empty
from kidney_vlm.data.rna_feature_import import (
    RnaFeatureRecord,
    align_to_bulkformer_vocab,
    build_rna_output_path,
    build_rna_records_from_raw_tree_limited,
    build_rna_records_from_registry,
    portable_relative_path,
    read_tcga_star_log_tpm,
    resolve_local_path,
    select_case_level_rna_records,
)
from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.rna.bulkformer_runtime import BULKFORMER_VARIANTS, bulkformer_sample_hidden_dim, load_bulkformer
from kidney_vlm.script_config import load_script_cfg

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="04_rna_features/01_extract_bulkformer_tcga_rna_features.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _nonempty_project_filter(values: Any) -> set[str]:
    return {str(value).strip() for value in list(values or []) if str(value).strip()}


def _load_gene_list(gene_info_path: Path, *, expected_gene_length: int) -> list[str]:
    if not gene_info_path.exists():
        raise FileNotFoundError(f"BulkFormer gene info CSV not found: {gene_info_path}")
    gene_info = pd.read_csv(gene_info_path)
    if "ensg_id" not in gene_info.columns:
        raise ValueError(f"BulkFormer gene info CSV is missing 'ensg_id': {gene_info_path}")
    gene_list = gene_info["ensg_id"].astype(str).tolist()
    if len(gene_list) != expected_gene_length:
        raise RuntimeError(
            f"BulkFormer gene vocab size ({len(gene_list)}) does not match "
            f"variant gene_length ({expected_gene_length})."
        )
    return gene_list


def _load_input_records(cfg) -> tuple[list[RnaFeatureRecord], str]:
    input_source = str(cfg.rna_features.get("input_source", "auto")).strip().lower()
    if input_source not in {"auto", "registry", "raw_tree"}:
        raise ValueError("input_source must be one of: auto, registry, raw_tree")
    allowed_project_ids = _nonempty_project_filter(cfg.rna_features.allowed_project_ids)
    first_n_cases = cfg.rna_features.get("first_n_cases")
    max_raw_cases = max(0, int(first_n_cases)) if first_n_cases is not None else None

    registry_records: list[RnaFeatureRecord] = []
    if input_source in {"auto", "registry"}:
        registry_path = _resolve_path(cfg.rna_features.source_registry_path)
        registry_df = read_parquet_or_empty(registry_path)
        registry_records = build_rna_records_from_registry(registry_df, repo_root=ROOT)
        if registry_records or input_source == "registry":
            return registry_records, "registry"

    raw_root = _resolve_path(cfg.rna_features.raw_rna_root)
    raw_records = build_rna_records_from_raw_tree_limited(
        raw_root,
        repo_root=ROOT,
        allowed_project_ids=allowed_project_ids,
        max_cases=max_raw_cases,
    )
    if raw_records or input_source == "raw_tree":
        return raw_records, "raw_tree"

    return [], "auto"


def _feature_exists_and_valid(path: Path, *, expected_dim: int) -> bool:
    if not path.exists():
        return False
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not torch.is_tensor(tensor):
        return False
    return tuple(tensor.shape) == (expected_dim,) and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())


def _aggregate_gene_embeddings(gene_emb: torch.Tensor, aggregate_type: str) -> torch.Tensor:
    aggregate_type = str(aggregate_type).strip().lower()
    if aggregate_type == "max":
        return gene_emb.amax(dim=1)
    if aggregate_type == "mean":
        return gene_emb.mean(dim=1)
    if aggregate_type == "median":
        return gene_emb.median(dim=1).values
    if aggregate_type == "all":
        return gene_emb.amax(dim=1) + gene_emb.mean(dim=1) + gene_emb.median(dim=1).values
    raise ValueError("aggregate_type must be one of: max, mean, median, all")


def _run_embedding_batch(
    *,
    model,
    expr_arrays: list[np.ndarray],
    mask_prob: float,
    aggregate_type: str,
    device: torch.device,
) -> torch.Tensor:
    expr_tensor = torch.from_numpy(np.stack(expr_arrays, axis=0)).to(device=device, dtype=torch.float32)
    use_autocast = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_autocast):
        gene_emb = model(expr_tensor, mask_prob=mask_prob, output_expr=False)
        sample_emb = _aggregate_gene_embeddings(gene_emb, aggregate_type)
    return sample_emb.detach().cpu().float()


def _compute_mask_prob(record: RnaFeatureRecord, gene_list: list[str]) -> float:
    tsv_path = resolve_local_path(record.rna_tsv_path, root=ROOT)
    log_tpm_row = read_tcga_star_log_tpm(tsv_path)
    _aligned_row, mask_prob = align_to_bulkformer_vocab(log_tpm_row, gene_list)
    return mask_prob


def _write_failure_rows(failure_rows: list[dict[str, object]], failure_csv_path: Path) -> None:
    if not failure_rows:
        return
    failure_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failure_rows).sort_values(
        by=["project_id", "case_submitter_id", "rna_file_id"],
        kind="stable",
    ).to_csv(failure_csv_path, index=False)


def _manifest_row_for_existing(
    *,
    record: RnaFeatureRecord,
    output_path: Path,
    cfg,
    bundle,
    input_source: str,
    source_file_count: int,
    mask_prob: float | None = None,
) -> dict[str, object]:
    return {
        "project_id": record.project_id,
        "case_submitter_id": record.case_submitter_id,
        "sample_submitter_id": record.sample_submitter_id,
        "rna_file_id": record.rna_file_id,
        "rna_file_name": record.rna_file_name,
        "rna_tsv_path": record.rna_tsv_path,
        "feature_path": portable_relative_path(output_path, root=ROOT),
        "feature_filename": output_path.name,
        "bulkformer_variant": bundle.variant,
        "bulkformer_hidden_dim": bundle.hidden_dim,
        "bulkformer_sample_dim": bundle.sample_dim,
        "aggregate_type": str(cfg.rna_features.aggregate_type),
        "mask_prob": mask_prob,
        "gene_missing_rate": mask_prob,
        "source_checkpoint": portable_relative_path(bundle.checkpoint_path, root=ROOT),
        "selection_rule": "case_level_prefer_primary_tumor_then_tcga_sample_submitter_sort_key",
        "source_file_count": source_file_count,
        "input_source": input_source,
        "sample_type": record.sample_type,
        "workflow_type": record.workflow_type,
    }


def main() -> None:
    cfg = load_cfg()
    feature_cfg = cfg.rna_features

    variant = str(feature_cfg.bulkformer_variant).strip()
    if variant not in BULKFORMER_VARIANTS:
        raise ValueError(f"Unknown BulkFormer variant {variant}; known variants: {sorted(BULKFORMER_VARIANTS)}")
    aggregate_type = str(feature_cfg.aggregate_type).strip().lower()
    if aggregate_type not in {"max", "mean", "median", "all"}:
        raise ValueError("aggregate_type must be one of: max, mean, median, all")

    output_features_dir = _resolve_path(feature_cfg.output_features_dir)
    output_manifest_parquet = _resolve_path(feature_cfg.output_manifest_parquet)
    output_manifest_csv = _resolve_path(feature_cfg.output_manifest_csv)
    failure_csv_path = _resolve_path(feature_cfg.failure_csv)
    device = torch.device(str(feature_cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Configured device {device} requires CUDA, but torch.cuda.is_available() is false.")

    records, input_source = _load_input_records(cfg)
    if not records:
        raise RuntimeError(
            "No RNA STAR TSV records found. Populate genomics_rna_bulk_paths in the registry "
            "or place files under data/raw/tcga/rna_bulk/<project>/<case>/*.tsv."
        )

    allowed_project_ids = _nonempty_project_filter(feature_cfg.allowed_project_ids)
    if allowed_project_ids:
        records = [record for record in records if record.project_id in allowed_project_ids]
    if not records:
        raise RuntimeError("No RNA records remain after applying allowed_project_ids.")

    source_file_count_by_case = {
        key: len(group)
        for key, group in _group_records_by_case(records).items()
    }
    selected_records = select_case_level_rna_records(records)
    first_n_cases = feature_cfg.get("first_n_cases")
    if first_n_cases is not None:
        selected_records = selected_records[: max(0, int(first_n_cases))]
    if not selected_records:
        raise RuntimeError("No case-level RNA records selected.")

    expected_gene_length = int(BULKFORMER_VARIANTS[variant]["gene_length"])
    gene_list = _load_gene_list(_resolve_path(feature_cfg.bulkformer_gene_info), expected_gene_length=expected_gene_length)

    print(f"Repo root: {ROOT}")
    print(f"Input source: {input_source}")
    print(f"Input RNA files discovered: {len(records)}")
    print(f"Case-level RNA files selected: {len(selected_records)}")
    print(f"BulkFormer variant: {variant}")
    print(f"Aggregate type: {aggregate_type}")
    print(f"Expected sample dim: {bulkformer_sample_hidden_dim(variant)}")
    print(f"Output features dir: {output_features_dir}")

    bundle = load_bulkformer(
        bulkformer_root=_resolve_path(feature_cfg.bulkformer_root),
        variant=variant,
        checkpoint_path=_resolve_path(feature_cfg.bulkformer_checkpoint),
        graph_path=_resolve_path(feature_cfg.bulkformer_graph),
        weights_path=_resolve_path(feature_cfg.bulkformer_weights),
        gene_emb_path=_resolve_path(feature_cfg.bulkformer_gene_emb),
        device=device,
    )

    batch_size = max(1, int(feature_cfg.batch_size))
    overwrite_existing = bool(feature_cfg.overwrite_existing)
    validate_existing = bool(feature_cfg.validate_existing)
    continue_on_error = bool(feature_cfg.continue_on_error)

    manifest_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    batch_items: list[dict[str, object]] = []
    written_count = 0
    skipped_existing_count = 0

    def flush_batch() -> None:
        nonlocal written_count
        if not batch_items:
            return
        grouped_by_mask: dict[float, list[dict[str, object]]] = defaultdict(list)
        for item in batch_items:
            grouped_by_mask[round(float(item["mask_prob"]), 8)].append(item)

        for mask_prob, group in grouped_by_mask.items():
            embeddings = _run_embedding_batch(
                model=bundle.model,
                expr_arrays=[item["expr_array"] for item in group],
                mask_prob=mask_prob,
                aggregate_type=aggregate_type,
                device=device,
            )
            if embeddings.shape != (len(group), bundle.sample_dim):
                raise RuntimeError(
                    f"Unexpected BulkFormer embedding shape {tuple(embeddings.shape)}; "
                    f"expected ({len(group)}, {bundle.sample_dim})."
                )
            for item, embedding in zip(group, embeddings, strict=True):
                output_path = item["output_path"]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(embedding.contiguous().to(dtype=torch.float32), output_path)
                written_count += 1
                manifest_rows.append(
                    _manifest_row_for_existing(
                        record=item["record"],
                        output_path=output_path,
                        cfg=cfg,
                        bundle=bundle,
                        input_source=input_source,
                        source_file_count=item["source_file_count"],
                        mask_prob=float(item["mask_prob"]),
                    )
                )
        batch_items.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    loop = tqdm(selected_records, total=len(selected_records), desc="Extracting BulkFormer RNA features", unit="case")
    for record in loop:
        output_path = build_rna_output_path(output_features_dir, record)
        source_file_count = source_file_count_by_case.get((record.project_id, record.case_submitter_id), 1)
        if output_path.exists() and not overwrite_existing:
            if not validate_existing or _feature_exists_and_valid(output_path, expected_dim=bundle.sample_dim):
                try:
                    mask_prob = _compute_mask_prob(record, gene_list)
                except Exception as exc:
                    failure_rows.append(
                        {
                            "project_id": record.project_id,
                            "case_submitter_id": record.case_submitter_id,
                            "sample_submitter_id": record.sample_submitter_id,
                            "rna_file_id": record.rna_file_id,
                            "rna_tsv_path": record.rna_tsv_path,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    _write_failure_rows(failure_rows, failure_csv_path)
                    if not continue_on_error:
                        raise
                    mask_prob = None
                skipped_existing_count += 1
                manifest_rows.append(
                    _manifest_row_for_existing(
                        record=record,
                        output_path=output_path,
                        cfg=cfg,
                        bundle=bundle,
                        input_source=input_source,
                        source_file_count=source_file_count,
                        mask_prob=mask_prob,
                    )
                )
                continue

        try:
            tsv_path = resolve_local_path(record.rna_tsv_path, root=ROOT)
            log_tpm_row = read_tcga_star_log_tpm(tsv_path)
            aligned_row, mask_prob = align_to_bulkformer_vocab(log_tpm_row, gene_list)
            batch_items.append(
                {
                    "record": record,
                    "output_path": output_path,
                    "expr_array": aligned_row.to_numpy(dtype=np.float32)[0],
                    "mask_prob": mask_prob,
                    "source_file_count": source_file_count,
                }
            )
            if len(batch_items) >= batch_size:
                flush_batch()
        except Exception as exc:
            failure_rows.append(
                {
                    "project_id": record.project_id,
                    "case_submitter_id": record.case_submitter_id,
                    "sample_submitter_id": record.sample_submitter_id,
                    "rna_file_id": record.rna_file_id,
                    "rna_tsv_path": record.rna_tsv_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_failure_rows(failure_rows, failure_csv_path)
            if not continue_on_error:
                raise

    flush_batch()
    _write_failure_rows(failure_rows, failure_csv_path)

    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.empty:
        raise RuntimeError("No RNA feature manifest rows were produced.")
    manifest_df = manifest_df.sort_values(
        by=["project_id", "case_submitter_id", "sample_submitter_id", "rna_file_id", "feature_filename"],
        kind="stable",
    )
    output_manifest_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_parquet(output_manifest_parquet, index=False)
    if bool(feature_cfg.write_csv_manifest):
        manifest_df.to_csv(output_manifest_csv, index=False)

    print("BulkFormer RNA feature extraction complete.")
    print(f"Manifest rows written: {len(manifest_df)}")
    print(f"Feature files written: {written_count}")
    print(f"Existing feature files skipped: {skipped_existing_count}")
    print(f"Failures: {len(failure_rows)}")
    print(f"Manifest parquet: {output_manifest_parquet}")
    if bool(feature_cfg.write_csv_manifest):
        print(f"Manifest csv: {output_manifest_csv}")
    if failure_rows:
        print(f"Failure csv: {failure_csv_path}")


def _group_records_by_case(records: list[RnaFeatureRecord]) -> dict[tuple[str, str], list[RnaFeatureRecord]]:
    grouped: dict[tuple[str, str], list[RnaFeatureRecord]] = {}
    for record in records:
        grouped.setdefault((record.project_id, record.case_submitter_id), []).append(record)
    return grouped


if __name__ == "__main__":
    main()
