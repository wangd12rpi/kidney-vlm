from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kidney_vlm.radiology.pmc_oa_feature_store import format_pmc_oa_feature_ref
from kidney_vlm.training.collator import RadiologyProjectorQACollator


class _Tokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in str(text)]}


def _write_store(path: Path) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        samples = handle.create_group("samples")
        sample = samples.create_group("pmc_oa-sample-a")
        dataset = sample.create_dataset("image_0", data=np.arange(8, dtype=np.float32))
        dataset.attrs["image_path"] = "/tmp/images/PMC1_fig1.jpg"
        dataset.attrs["sample_id"] = "pmc_oa-sample-a"
        dataset.attrs["feature_type"] = "pooled_embedding"
        dataset.attrs["model_name"] = "google/medsiglip-448"


def test_radiology_projector_collator_loads_pmc_oa_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root_dir = tmp_path / "repo"
    store_path = root_dir / "data" / "processes" / "pmc_oa" / "pmc_oa_features.h5"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    _write_store(store_path)

    selected_prompt = "Write a radiology report-style caption for this image.\nCaption:"
    monkeypatch.setattr("kidney_vlm.training.collator.random.choice", lambda options: selected_prompt)

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
                "modality": "radiology",
                "pmcid": "PMC1",
                "url_name": "url-1",
                "answer": "Example caption.",
                "radiology_embedding_paths": [embedding_ref],
            }
        ]
    )

    assert batch["input_ids"].shape[0] == 1
    assert batch["radiology_features"].shape == (1, 1, 8)
    assert torch.equal(batch["radiology_feature_mask"], torch.tensor([[1]]))
    assert batch["sample_id"] == ["pmc_oa-sample-a"]
    assert batch["pmcid"] == ["PMC1"]


def test_radiology_projector_collator_has_ten_default_prompt_texts() -> None:
    collator = RadiologyProjectorQACollator(
        tokenizer=_Tokenizer(),
        root_dir=".",
    )

    assert len(collator.prompt_texts) == 10
