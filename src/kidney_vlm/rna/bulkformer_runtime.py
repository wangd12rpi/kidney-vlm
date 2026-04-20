from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


BULKFORMER_VARIANTS: dict[str, dict[str, Any]] = {
    "37M": {"dim": 128, "bins": 0, "gb_repeat": 1, "p_repeat": 1, "bin_head": 12, "full_head": 8, "gene_length": 20010},
    "50M": {"dim": 256, "bins": 0, "gb_repeat": 1, "p_repeat": 2, "bin_head": 12, "full_head": 8, "gene_length": 20010},
    "93M": {"dim": 512, "bins": 0, "gb_repeat": 1, "p_repeat": 6, "bin_head": 12, "full_head": 8, "gene_length": 20010},
    "127M": {"dim": 640, "bins": 0, "gb_repeat": 1, "p_repeat": 8, "bin_head": 12, "full_head": 8, "gene_length": 20010},
    "147M": {"dim": 640, "bins": 0, "gb_repeat": 1, "p_repeat": 12, "bin_head": 12, "full_head": 8, "gene_length": 20010},
}


# Sample-level features concat three global summary scalars onto the per-gene
# hidden state (see utils/BulkFormer.py: [gene_emb, mask_scalar, expr_mean,
# nonzero_ratio]), so the pooled sample embedding is dim+3 wide.
_SAMPLE_DIM_EXTRA = 3


_ZENODO_HINT = (
    "Download the auxiliary BulkFormer files from the vendor's Zenodo record "
    "(DOI 10.5281/zenodo.15744294) and place them under external/BulkFormer/data/:\n"
    "  - G_tcga.pt\n"
    "  - G_tcga_weight.pt\n"
    "  - esm2_feature_concat.pt"
)


@dataclass
class BulkFormerBundle:
    model: Any
    variant: str
    hidden_dim: int
    sample_dim: int
    checkpoint_path: Path
    bulkformer_root: Path


def bulkformer_sample_hidden_dim(variant: str) -> int:
    if variant not in BULKFORMER_VARIANTS:
        raise ValueError(f"Unknown BulkFormer variant: {variant}. Known: {sorted(BULKFORMER_VARIANTS)}.")
    return int(BULKFORMER_VARIANTS[variant]["dim"]) + _SAMPLE_DIM_EXTRA


def _ensure_vendor_import_path(bulkformer_root: Path) -> None:
    as_str = str(bulkformer_root)
    if as_str not in sys.path:
        sys.path.insert(0, as_str)


def _require_file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required BulkFormer file not found: {path}\n{hint}")
    return path


def _strip_module_prefix(state_dict: dict[str, Any]) -> OrderedDict:
    cleaned: OrderedDict = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[len("module."):] if key.startswith("module.") else key] = value
    return cleaned


def load_bulkformer(
    *,
    bulkformer_root: Path,
    variant: str,
    checkpoint_path: Path,
    graph_path: Path,
    weights_path: Path,
    gene_emb_path: Path,
    device: torch.device | str,
) -> BulkFormerBundle:
    """Load a BulkFormer checkpoint without editing the vendor config."""
    if variant not in BULKFORMER_VARIANTS:
        raise ValueError(f"Unknown BulkFormer variant: {variant}. Known: {sorted(BULKFORMER_VARIANTS)}.")

    bulkformer_root = Path(bulkformer_root).resolve()
    _require_file(bulkformer_root, "BulkFormer source tree is missing.")
    _require_file(checkpoint_path, f"Checkpoint expected at {checkpoint_path}.")
    _require_file(graph_path, _ZENODO_HINT)
    _require_file(weights_path, _ZENODO_HINT)
    _require_file(gene_emb_path, _ZENODO_HINT)

    _ensure_vendor_import_path(bulkformer_root)
    try:
        from torch_geometric.typing import SparseTensor  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError(
            "torch_geometric is required for BulkFormer. "
            "Install project deps (e.g. `uv sync`) after updating pyproject.toml."
        ) from exc
    try:
        from utils.BulkFormer import BulkFormer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import BulkFormer from the vendor tree. "
            f"Check that {bulkformer_root / 'utils' / 'BulkFormer.py'} exists and that "
            "performer-pytorch / torch_geometric are installed."
        ) from exc

    device_obj = torch.device(device)

    graph_index = torch.load(str(graph_path), map_location="cpu", weights_only=False)
    graph_weights = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    gene_emb = torch.load(str(gene_emb_path), map_location="cpu", weights_only=False)

    graph = SparseTensor(row=graph_index[1], col=graph_index[0], value=graph_weights).t().to(device_obj)

    params = dict(BULKFORMER_VARIANTS[variant])
    params["graph"] = graph
    params["gene_emb"] = gene_emb
    model = BulkFormer(**params).to(device_obj)

    raw_state = torch.load(str(checkpoint_path), map_location=device_obj, weights_only=False)
    if hasattr(raw_state, "state_dict"):
        raw_state = raw_state.state_dict()
    if not isinstance(raw_state, dict):
        raise RuntimeError(
            f"Unexpected checkpoint payload type {type(raw_state).__name__} at {checkpoint_path}."
        )
    cleaned = _strip_module_prefix(raw_state)
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    return BulkFormerBundle(
        model=model,
        variant=variant,
        hidden_dim=int(params["dim"]),
        sample_dim=int(params["dim"]) + _SAMPLE_DIM_EXTRA,
        checkpoint_path=Path(checkpoint_path).resolve(),
        bulkformer_root=bulkformer_root,
    )
