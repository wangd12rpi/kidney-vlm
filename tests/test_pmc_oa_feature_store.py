from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pytest

from kidney_vlm.radiology.pmc_oa_feature_store import (
    build_pmc_oa_feature_index,
    build_pmc_oa_lookup_by_image_name,
    format_pmc_oa_feature_ref,
    load_pmc_oa_feature_array,
)

pytest.importorskip("h5py")


def _write_store(path: Path) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        samples = handle.create_group("samples")
        sample = samples.create_group("pmc_oa-sample-a")
        image_0 = sample.create_dataset("image_0", data=np.arange(8, dtype=np.float32))
        image_0.attrs["image_path"] = "data/raw/pmc_oa/images/PMC1_fig1.jpg"
        image_0.attrs["sample_id"] = "pmc_oa-sample-a"
        image_0.attrs["feature_type"] = "pooled_embedding"
        image_0.attrs["model_name"] = "google/medsiglip-448"

        image_1 = sample.create_dataset("image_1", data=np.arange(8, 16, dtype=np.float32))
        image_1.attrs["image_path"] = "data/raw/pmc_oa/images/PMC1_fig2.jpg"
        image_1.attrs["sample_id"] = "pmc_oa-sample-a"
        image_1.attrs["feature_type"] = "pooled_embedding"
        image_1.attrs["model_name"] = "google/medsiglip-448"


def _make_local_tmpdir(name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_root = repo_root / ".test_artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    path = artifacts_root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_build_pmc_oa_feature_index_and_load_array() -> None:
    root_dir = _make_local_tmpdir("pmc_oa_feature_store")
    try:
        store_path = root_dir / "data" / "processes" / "pmc_oa" / "pmc_oa_features.h5"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        _write_store(store_path)

        index = build_pmc_oa_feature_index(root_dir=root_dir, store_path=store_path)

        assert index["image_name"].tolist() == ["PMC1_fig1.jpg", "PMC1_fig2.jpg"]
        lookup = build_pmc_oa_lookup_by_image_name(index)
        assert lookup["PMC1_fig1.jpg"]["sample_id"] == "pmc_oa-sample-a"

        embedding_ref = format_pmc_oa_feature_ref(
            root_dir=root_dir,
            store_path=store_path,
            sample_id="pmc_oa-sample-a",
            image_key="image_1",
        )
        array = load_pmc_oa_feature_array(root_dir, embedding_ref)

        assert array.shape == (1, 8)
        assert array.tolist() == [list(np.arange(8, 16, dtype=np.float32))]
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)
