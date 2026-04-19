from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("torch")


class _Tokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in str(text)]}


def _write_series_store(path: Path) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "features",
            data=np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ],
                dtype=np.float32,
            ),
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "png_relpaths",
            data=np.array(
                [
                    "data/processes/radiology/pngs/TCGA-KIRC/patient-1/study-1/series-a/slice-001.png",
                    "data/processes/radiology/pngs/TCGA-KIRC/patient-1/study-1/series-a/slice-002.png",
                    "data/processes/radiology/pngs/TCGA-KIRC/patient-1/study-1/series-b/slice-001.png",
                ],
                dtype=object,
            ),
            dtype=string_dtype,
        )


def _write_pmc_store(path: Path) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        samples = handle.create_group("samples")
        sample = samples.create_group("pmc_oa-sample-a")
        dataset = sample.create_dataset("image_0", data=np.arange(8, dtype=np.float32))
        dataset.attrs["image_path"] = "data/raw/pmc_oa/images/PMC1_fig1.jpg"
        dataset.attrs["sample_id"] = "pmc_oa-sample-a"
        dataset.attrs["feature_type"] = "pooled_embedding"
        dataset.attrs["model_name"] = "google/medsiglip-448"


def _make_local_tmpdir(name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_root = repo_root / ".test_artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    path = artifacts_root / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_radiology_projector_collator_loads_series_feature_refs() -> None:
    import torch
    from kidney_vlm.radiology.feature_registry import format_series_embedding_ref
    from kidney_vlm.training.collator import RadiologyProjectorQACollator

    root_dir = _make_local_tmpdir("radiology_collator_series")
    try:
        store_path = root_dir / "data" / "processes" / "radiology" / "features" / "features_tcga.h5"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        _write_series_store(store_path)

        collator = RadiologyProjectorQACollator(
            tokenizer=_Tokenizer(),
            root_dir=root_dir,
            max_slice_tokens=4,
        )
        embedding_ref = format_series_embedding_ref(
            root_dir=root_dir,
            store_path=store_path,
            series_dir=root_dir / "data" / "processes" / "radiology" / "pngs" / "TCGA-KIRC" / "patient-1" / "study-1" / "series-a",
        )

        batch = collator(
            [
                {
                    "sample_id": "tcga-case-1",
                    "project_id": "TCGA-KIRC",
                    "source": "tcga",
                    "patient_id": "patient-1",
                    "study_id": "study-1",
                    "answer": "Example caption.",
                    "radiology_embedding_paths": [embedding_ref],
                }
            ]
        )

        assert batch["radiology_features"].shape == (1, 2, 3)
        assert torch.equal(batch["radiology_feature_mask"], torch.tensor([[1, 1]]))
        assert batch["sample_id"] == ["tcga-case-1"]
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)


def test_radiology_projector_collator_loads_pmc_oa_refs() -> None:
    import torch
    from kidney_vlm.radiology.pmc_oa_feature_store import format_pmc_oa_feature_ref
    from kidney_vlm.training.collator import RadiologyProjectorQACollator

    root_dir = _make_local_tmpdir("radiology_collator_pmc")
    try:
        store_path = root_dir / "data" / "processes" / "pmc_oa" / "pmc_oa_features.h5"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pmc_store(store_path)

        collator = RadiologyProjectorQACollator(
            tokenizer=_Tokenizer(),
            root_dir=root_dir,
            max_slice_tokens=4,
        )
        embedding_ref = format_pmc_oa_feature_ref(
            root_dir=root_dir,
            store_path=store_path,
            sample_id="pmc_oa-sample-a",
            image_key="image_0",
        )

        batch = collator(
            [
                {
                    "sample_id": "pmc_oa-sample-a",
                    "project_id": "pmc_oa",
                    "source": "pmc_oa",
                    "patient_id": "PMC1",
                    "study_id": "article-1",
                    "image_name": "PMC1_fig1.jpg",
                    "answer": "Example caption.",
                    "radiology_embedding_paths": [embedding_ref],
                }
            ]
        )

        assert batch["radiology_features"].shape == (1, 1, 8)
        assert torch.equal(batch["radiology_feature_mask"], torch.tensor([[1]]))
        assert batch["image_name"] == ["PMC1_fig1.jpg"]
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)


def test_radiology_projector_collator_has_five_default_prompt_texts() -> None:
    pytest.importorskip("torch")
    from kidney_vlm.training.collator import RadiologyProjectorQACollator

    collator = RadiologyProjectorQACollator(
        tokenizer=_Tokenizer(),
        root_dir=".",
    )

    assert len(collator.prompt_texts) == 5
