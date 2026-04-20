#!/usr/bin/env python3
"""Phase 0 smoke test for the BulkFormer integration.

Runs a single TCGA STAR gene-counts TSV end-to-end through BulkFormer and
prints the resulting sample-level embedding shape. Intended as a one-shot
sanity check before wiring the full feature-extraction pipeline in Phase 1.

Verifies:
  1. Vendor checkpoint + graph + ESM2 gene embedding load without errors.
  2. log1p(tpm_unstranded) from a real TCGA STAR TSV aligns cleanly to the
     BulkFormer 20,010-gene vocabulary (mask_prob should be ~0 on TCGA).
  3. Sample-level forward pass yields the expected [1, dim+3] tensor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.rna.bulkformer_runtime import (
    BULKFORMER_VARIANTS,
    BulkFormerBundle,
    bulkformer_sample_hidden_dim,
    load_bulkformer,
)

ROOT = find_repo_root(Path(__file__))


def _find_sample_tcga_tsv() -> Path:
    rna_dir = ROOT / "data" / "raw" / "tcga" / "rna_bulk"
    if not rna_dir.exists():
        raise FileNotFoundError(f"Missing TCGA RNA directory: {rna_dir}")
    for cohort in sorted(rna_dir.iterdir()):
        if not cohort.is_dir():
            continue
        for case in sorted(cohort.iterdir()):
            if not case.is_dir():
                continue
            matches = sorted(case.glob("*.rna_seq.augmented_star_gene_counts.tsv"))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"No TCGA STAR TSV found under {rna_dir}.")


def _read_tcga_star_log_tpm(tsv_path: Path) -> pd.DataFrame:
    """Return a one-row DataFrame of log1p(tpm_unstranded) keyed by base Ensembl ID."""
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")]
    df = df.loc[df["gene_type"].astype(str) == "protein_coding"].copy()
    df["ensg_id"] = df["gene_id"].astype(str).str.split(".").str[0]
    df["tpm_unstranded"] = pd.to_numeric(df["tpm_unstranded"], errors="coerce").fillna(0.0)
    # STAR GENCODE v36 has a handful of PAR_Y duplicates once version suffixes
    # are stripped; collapse by max TPM to keep the gene vector unique.
    df = df.groupby("ensg_id", as_index=False)["tpm_unstranded"].max()
    df["log_tpm"] = np.log1p(df["tpm_unstranded"].to_numpy(dtype=np.float64))
    row = df.set_index("ensg_id")["log_tpm"].to_frame().T
    row.index = [tsv_path.stem]
    return row


def _align_to_bulkformer_vocab(
    log_tpm_row: pd.DataFrame,
    gene_list: list[str],
) -> tuple[pd.DataFrame, float]:
    """Align a one-row expression frame to the BulkFormer gene vocabulary.

    Missing genes get the vendor-mandated -10 placeholder. Returns the aligned
    frame plus mask_prob, the fraction of the vocabulary that was imputed.
    """
    present = set(log_tpm_row.columns)
    missing = [gene for gene in gene_list if gene not in present]
    if missing:
        pad = pd.DataFrame(
            np.full((log_tpm_row.shape[0], len(missing)), -10.0, dtype=np.float32),
            columns=missing,
            index=log_tpm_row.index,
        )
        aligned = pd.concat([log_tpm_row, pad], axis=1)
    else:
        aligned = log_tpm_row
    aligned = aligned.loc[:, gene_list]
    mask_prob = len(missing) / len(gene_list)
    return aligned, float(mask_prob)


def _run_sample_level_embedding(
    bundle: BulkFormerBundle,
    aligned_row: pd.DataFrame,
    *,
    mask_prob: float,
    device: torch.device,
) -> np.ndarray:
    expr_array = aligned_row.to_numpy(dtype=np.float32)
    expr_tensor = torch.from_numpy(expr_array).to(device=device, dtype=torch.float32)
    use_autocast = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_autocast):
        gene_emb = bundle.model(expr_tensor, mask_prob=mask_prob, output_expr=False)
    gene_emb_np = gene_emb.detach().cpu().float().numpy()
    return gene_emb_np.mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tsv",
        type=str,
        default="",
        help="Path to a TCGA STAR *augmented_star_gene_counts.tsv. "
             "Defaults to the first TSV under data/raw/tcga/rna_bulk.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="93M",
        choices=sorted(BULKFORMER_VARIANTS),
    )
    parser.add_argument(
        "--bulkformer-root",
        type=str,
        default=str(ROOT / "external" / "BulkFormer"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to the BulkFormer .pt checkpoint. "
             "Defaults to external/BulkFormer/checkpoints/BulkFormer_<variant>.pt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    tsv_path = Path(args.tsv).expanduser().resolve() if args.tsv else _find_sample_tcga_tsv()
    bulkformer_root = Path(args.bulkformer_root).expanduser().resolve()
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else (bulkformer_root / "checkpoints" / f"BulkFormer_{args.variant}.pt").resolve()
    )
    graph_path = bulkformer_root / "data" / "G_tcga.pt"
    weights_path = bulkformer_root / "data" / "G_tcga_weight.pt"
    gene_emb_path = bulkformer_root / "data" / "esm2_feature_concat.pt"
    gene_info_path = bulkformer_root / "data" / "bulkformer_gene_info.csv"
    device = torch.device(args.device)

    print(f"Repo root: {ROOT}")
    print(f"BulkFormer root: {bulkformer_root}")
    print(f"Variant: {args.variant}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Sample TSV: {tsv_path}")

    print("[1/5] Loading BulkFormer gene vocabulary...")
    if not gene_info_path.exists():
        raise FileNotFoundError(f"Missing BulkFormer gene info CSV: {gene_info_path}")
    gene_info = pd.read_csv(gene_info_path)
    gene_list = gene_info["ensg_id"].astype(str).tolist()
    expected_vocab = int(BULKFORMER_VARIANTS[args.variant]["gene_length"])
    if len(gene_list) != expected_vocab:
        raise RuntimeError(
            f"Gene vocabulary size ({len(gene_list)}) does not match variant "
            f"gene_length ({expected_vocab})."
        )
    print(f"    Gene vocabulary size: {len(gene_list)}")

    print("[2/5] Parsing TCGA STAR TSV into log1p(TPM) vector...")
    log_tpm_row = _read_tcga_star_log_tpm(tsv_path)
    print(f"    Protein-coding genes parsed: {log_tpm_row.shape[1]}")

    print("[3/5] Aligning to BulkFormer vocabulary with -10 padding...")
    aligned_row, mask_prob = _align_to_bulkformer_vocab(log_tpm_row, gene_list)
    print(f"    Aligned shape: {aligned_row.shape} | mask_prob: {mask_prob:.6f}")

    print("[4/5] Loading BulkFormer model weights...")
    bundle = load_bulkformer(
        bulkformer_root=bulkformer_root,
        variant=args.variant,
        checkpoint_path=checkpoint_path,
        graph_path=graph_path,
        weights_path=weights_path,
        gene_emb_path=gene_emb_path,
        device=device,
    )
    expected_dim = bulkformer_sample_hidden_dim(args.variant)
    print(f"    Model loaded. Expected sample-level dim: {expected_dim}")

    print("[5/5] Running sample-level forward pass...")
    embedding = _run_sample_level_embedding(
        bundle,
        aligned_row,
        mask_prob=mask_prob,
        device=device,
    )
    print(f"    Output shape: {tuple(embedding.shape)} | dtype: {embedding.dtype}")
    if embedding.shape != (1, expected_dim):
        raise RuntimeError(
            f"Unexpected output shape {tuple(embedding.shape)}; expected (1, {expected_dim})."
        )
    if not np.isfinite(embedding).all():
        raise RuntimeError("Sample-level embedding contains non-finite values.")
    print(
        f"    Finite check passed | mean: {embedding.mean():.4f} | "
        f"std: {embedding.std():.4f} | min: {embedding.min():.4f} | max: {embedding.max():.4f}"
    )

    print()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
