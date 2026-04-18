from __future__ import annotations

from pathlib import Path

import numpy as np

from kidney_vlm.radiology.pmc_oa_feature_store import (
    build_pmc_oa_feature_index,
    build_pmc_oa_lookup_by_image_name,
    format_pmc_oa_feature_ref,
    load_pmc_oa_feature_array,
    parse_pmc_oa_feature_ref,
)


def _write_store(path: Path) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        samples = handle.create_group("samples")

        sample_a = samples.create_group("pmc_oa-sample-a")
        dataset_a = sample_a.create_dataset("image_0", data=np.arange(4, dtype=np.float32))
        dataset_a.attrs["image_path"] = "/tmp/images/PMC1_fig1.jpg"
        dataset_a.attrs["sample_id"] = "pmc_oa-sample-a"
        dataset_a.attrs["feature_type"] = "pooled_embedding"
        dataset_a.attrs["model_name"] = "google/medsiglip-448"

        sample_b = samples.create_group("pmc_oa-sample-b")
        dataset_b = sample_b.create_dataset("image_0", data=np.arange(4, 8, dtype=np.float32))
        dataset_b.attrs["image_path"] = "/tmp/images/PMC2_fig2.jpg"
        dataset_b.attrs["sample_id"] = "pmc_oa-sample-b"
        dataset_b.attrs["feature_type"] = "pooled_embedding"
        dataset_b.attrs["model_name"] = "google/medsiglip-448"


def test_format_and_parse_pmc_oa_feature_ref(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    store_path = root_dir / "data" / "processes" / "pmc_oa" / "pmc_oa_features.h5"

    ref = format_pmc_oa_feature_ref(
        root_dir=root_dir,
        store_path=store_path,
        sample_id="pmc_oa-sample-a",
        image_key="image_0",
    )
    parsed = parse_pmc_oa_feature_ref(ref)

    assert parsed.store_path == "data/processes/pmc_oa/pmc_oa_features.h5"
    assert parsed.sample_id == "pmc_oa-sample-a"
    assert parsed.image_key == "image_0"


def test_load_pmc_oa_feature_array_returns_single_token_batch(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    store_path = root_dir / "data" / "processes" / "pmc_oa" / "pmc_oa_features.h5"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    _write_store(store_path)

    ref = format_pmc_oa_feature_ref(
        root_dir=root_dir,
        store_path=store_path,
        sample_id="pmc_oa-sample-a",
        image_key="image_0",
    )
    array = load_pmc_oa_feature_array(root_dir, ref)

    assert array.shape == (1, 4)
    assert array.dtype == np.float32
    assert array.tolist() == [[0.0, 1.0, 2.0, 3.0]]


def test_build_pmc_oa_feature_index_and_lookup(tmp_path: Path) -> None:
    root_dir = tmp_path / "repo"
    store_path = root_dir / "data" / "processes" / "pmc_oa" / "pmc_oa_features.h5"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    _write_store(store_path)

    feature_index = build_pmc_oa_feature_index(root_dir=root_dir, store_path=store_path)
    lookup = build_pmc_oa_lookup_by_image_name(feature_index)

    assert feature_index["image_name"].tolist() == ["PMC1_fig1.jpg", "PMC2_fig2.jpg"]
    assert lookup["PMC1_fig1.jpg"]["sample_id"] == "pmc_oa-sample-a"
    assert lookup["PMC1_fig1.jpg"]["embedding_ref"] == (
        "data/processes/pmc_oa/pmc_oa_features.h5::sample=pmc_oa-sample-a::image=image_0"
    )
