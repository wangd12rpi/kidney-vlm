from __future__ import annotations

from collections import UserDict
from pathlib import Path
import numpy as np
import pytest
import torch

from kidney_vlm.training.collator import (
    DNAMProjectorQACollator,
    PathologyProjectorQACollator,
    QACollator,
    _infer_global_coord_step,
    _apply_patch_token_dropout,
    _load_h5_patch_features,
    _spatial_bucket_keys,
)


REAL_PATHOLOGY_H5 = Path(
    "data/features/features_conch_v15/TCGA-02-0003-01Z-00-DX1.6171b175-0972-4e84-9997-2f1ce75f4407.h5"
)


class _DummyTokenizer:
    def __call__(self, texts, **_kwargs):
        return {"input_ids": [0 for _ in texts]}


def test_collator_exposes_new_embedding_fields_and_legacy_aliases() -> None:
    collator = QACollator(tokenizer=_DummyTokenizer(), max_length=32)
    batch = collator(
        [
            {
                "question": "q",
                "answer": "a",
                "pathology_tile_embedding_paths": np.array(["/tmp/tile.npy"], dtype=object),
                "pathology_slide_embedding_paths": np.array(["/tmp/slide.npy"], dtype=object),
                "radiology_embedding_paths": np.array(["/tmp/rad.npy"], dtype=object),
                "pathology_wsi_paths": [],
                "radiology_image_paths": [],
                "pathology_mask_paths": [],
                "radiology_mask_paths": [],
            }
        ]
    )

    assert batch["pathology_tile_embedding_paths"] == [["/tmp/tile.npy"]]
    assert batch["pathology_slide_embedding_paths"] == [["/tmp/slide.npy"]]
    assert batch["radiology_embedding_paths"] == [["/tmp/rad.npy"]]


class _ProjectorTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in str(text)]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, chat_template_kwargs=None):
        assert tokenize is True
        assert chat_template_kwargs == {"enable_thinking": False}
        pieces = []
        for message in messages:
            role = str(message["role"]).upper()
            pieces.append(f"<{role}>{message['content']}")
        if add_generation_prompt:
            pieces.append("<ASSISTANT>")
        return [ord(char) for char in "".join(pieces)]


def test_projector_collator_uses_chat_template_with_hardcoded_prompt_text(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_prompt = "Write a pathology caption."
    monkeypatch.setattr(
        "kidney_vlm.training.collator.random.choice",
        lambda options: selected_prompt,
    )
    collator = PathologyProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=".",
    )

    input_ids, labels = collator._build_text_pair(
        {
            "instruction": "THIS SHOULD BE IGNORED",
            "answer": "Clear cell renal neoplasm.",
        }
    )

    expected_prompt = f"<USER>{selected_prompt}<ASSISTANT>"
    expected_full = f"{expected_prompt}Clear cell renal neoplasm."
    expected_prompt_ids = [ord(char) for char in expected_prompt]
    expected_full_ids = [ord(char) for char in expected_full]

    assert input_ids == expected_full_ids
    assert labels == ([-100] * len(expected_prompt_ids)) + expected_full_ids[len(expected_prompt_ids) :]


def test_projector_collator_has_five_default_prompt_texts() -> None:
    collator = PathologyProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=".",
    )

    assert len(collator.prompt_texts) == 5


def test_projector_collator_accepts_batch_encoding_like_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_prompt = "Describe the pathology image."
    monkeypatch.setattr(
        "kidney_vlm.training.collator.random.choice",
        lambda options: selected_prompt,
    )

    class _BatchEncodingLikeTokenizer(_ProjectorTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, chat_template_kwargs=None):
            encoded = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs,
            )
            return {"input_ids": torch.tensor([encoded], dtype=torch.long)}

    collator = PathologyProjectorQACollator(
        tokenizer=_BatchEncodingLikeTokenizer(),
        root_dir=".",
    )

    input_ids, labels = collator._build_text_pair(
        {
            "instruction": "THIS SHOULD BE IGNORED",
            "answer": "Example pathology caption.",
        }
    )

    expected_prompt = f"<USER>{selected_prompt}<ASSISTANT>"
    expected_full = f"{expected_prompt}Example pathology caption."
    expected_prompt_ids = [ord(char) for char in expected_prompt]
    expected_full_ids = [ord(char) for char in expected_full]

    assert input_ids == expected_full_ids
    assert labels == ([-100] * len(expected_prompt_ids)) + expected_full_ids[len(expected_prompt_ids) :]


def test_projector_collator_accepts_mapping_like_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_prompt = "Describe the pathology image."
    monkeypatch.setattr(
        "kidney_vlm.training.collator.random.choice",
        lambda options: selected_prompt,
    )

    class _MappingLikeTokenizer(_ProjectorTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, chat_template_kwargs=None):
            encoded = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs,
            )
            return UserDict({"input_ids": encoded})

    collator = PathologyProjectorQACollator(
        tokenizer=_MappingLikeTokenizer(),
        root_dir=".",
    )

    input_ids, labels = collator._build_text_pair(
        {
            "instruction": "THIS SHOULD BE IGNORED",
            "answer": "Example pathology caption.",
        }
    )

    expected_prompt = f"<USER>{selected_prompt}<ASSISTANT>"
    expected_full = f"{expected_prompt}Example pathology caption."
    expected_prompt_ids = [ord(char) for char in expected_prompt]
    expected_full_ids = [ord(char) for char in expected_full]

    assert input_ids == expected_full_ids
    assert labels == ([-100] * len(expected_prompt_ids)) + expected_full_ids[len(expected_prompt_ids) :]


def test_dnam_projector_collator_loads_pt_features(tmp_path) -> None:
    feature_path = tmp_path / "sample.pt"
    torch.save(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32), feature_path)

    collator = DNAMProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=tmp_path,
    )
    batch = collator(
        [
            {
                "sample_id": "tcga-1",
                "project_id": "TCGA-GBM",
                "source": "tcga",
                "answer": "Example DNAm caption.",
                "genomics_dna_methylation_feature_path": feature_path.name,
            }
        ]
    )

    assert batch["dnam_features"].shape == (1, 1, 3)
    assert batch["dnam_feature_mask"].tolist() == [[1]]


def test_dnam_projector_collator_has_default_prompt_texts() -> None:
    collator = DNAMProjectorQACollator(
        tokenizer=_ProjectorTokenizer(),
        root_dir=".",
    )

    assert len(collator.prompt_texts) == 5


def test_apply_patch_token_dropout_keeps_selected_tokens_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kidney_vlm.training.collator.torch.rand",
        lambda size, device=None: torch.tensor([0.9, 0.1, 0.8, 0.2], device=device),
    )
    patch_tensor = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    dropped = _apply_patch_token_dropout(patch_tensor, dropout_prob=0.5)

    assert torch.equal(
        dropped,
        torch.tensor(
            [
                [1.0, 10.0],
                [3.0, 30.0],
            ]
        ),
    )


def test_apply_patch_token_dropout_keeps_one_token_when_all_would_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kidney_vlm.training.collator.torch.rand",
        lambda size, device=None: torch.tensor([0.1, 0.4, 0.3], device=device),
    )
    patch_tensor = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    dropped = _apply_patch_token_dropout(patch_tensor, dropout_prob=0.9)

    assert torch.equal(dropped, torch.tensor([[2.0, 20.0]]))


def test_load_h5_patch_features_mean_pools_nearby_2d_coords(tmp_path) -> None:
    import h5py

    path = tmp_path / "slide-a.h5"
    features = np.array(
        [
            [1.0, 1.0],  # (0, 0)
            [3.0, 3.0],  # (10, 0)
            [5.0, 5.0],  # (0, 10)
            [7.0, 7.0],  # (10, 10)
            [10.0, 10.0],  # (20, 0)
            [12.0, 12.0],  # (30, 0)
            [14.0, 14.0],  # (20, 10)
            [16.0, 16.0],  # (30, 10)
        ],
        dtype=np.float32,
    )
    coords = np.array(
        [
            [0, 0],
            [10, 0],
            [0, 10],
            [10, 10],
            [20, 0],
            [30, 0],
            [20, 10],
            [30, 10],
        ],
        dtype=np.int64,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)

    tensor = _load_h5_patch_features(
        path,
        max_patch_tokens=0,
        compression_method="mean_pool",
        compression_kernel_size=2,
    )

    expected = torch.tensor(
        [
            [4.0, 4.0],
            [13.0, 13.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(tensor, expected)


def test_load_h5_patch_features_stride_subsamples_nearby_2d_coords(tmp_path) -> None:
    import h5py

    path = tmp_path / "slide-a.h5"
    features = np.array(
        [
            [1.0, 1.0],
            [3.0, 3.0],
            [5.0, 5.0],
            [7.0, 7.0],
            [10.0, 10.0],
            [12.0, 12.0],
            [14.0, 14.0],
            [16.0, 16.0],
        ],
        dtype=np.float32,
    )
    coords = np.array(
        [
            [0, 0],
            [10, 0],
            [0, 10],
            [10, 10],
            [20, 0],
            [30, 0],
            [20, 10],
            [30, 10],
        ],
        dtype=np.int64,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)

    tensor = _load_h5_patch_features(
        path,
        max_patch_tokens=0,
        compression_method="stride",
        compression_kernel_size=2,
    )

    expected = torch.tensor(
        [
            [1.0, 1.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(tensor, expected)


def test_load_h5_patch_features_mean_pool_respects_large_coordinate_gaps(tmp_path) -> None:
    import h5py

    path = tmp_path / "slide-gap.h5"
    features = np.array(
        [
            [1.0, 1.0],
            [3.0, 3.0],
            [10.0, 10.0],
            [12.0, 12.0],
        ],
        dtype=np.float32,
    )
    coords = np.array(
        [
            [0, 0],
            [0, 512],
            [4096, 0],
            [4096, 512],
        ],
        dtype=np.int64,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)

    bucket_keys = _spatial_bucket_keys(coords, 2)
    assert bucket_keys is not None
    assert bucket_keys.tolist() == [[0, 0], [0, 0], [4, 0], [4, 0]]

    tensor = _load_h5_patch_features(
        path,
        max_patch_tokens=0,
        compression_method="mean_pool",
        compression_kernel_size=2,
    )

    expected = torch.tensor(
        [
            [2.0, 2.0],
            [11.0, 11.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(tensor, expected)


def test_load_h5_patch_features_real_slide_mean_pools_actual_2x2_neighbors() -> None:
    import h5py

    if not REAL_PATHOLOGY_H5.exists():
        pytest.skip(f"Real pathology feature file not available: {REAL_PATHOLOGY_H5}")

    with h5py.File(REAL_PATHOLOGY_H5, "r") as handle:
        features = np.asarray(handle["features"])
        coords = np.asarray(handle["coords"])

    step = _infer_global_coord_step(coords)
    assert step == 512

    bucket_keys = _spatial_bucket_keys(coords, 2)
    assert bucket_keys is not None

    pooled = _load_h5_patch_features(
        REAL_PATHOLOGY_H5,
        max_patch_tokens=0,
        compression_method="mean_pool",
        compression_kernel_size=2,
    )
    stride = _load_h5_patch_features(
        REAL_PATHOLOGY_H5,
        max_patch_tokens=0,
        compression_method="stride",
        compression_kernel_size=2,
    )

    unique_keys = np.unique(bucket_keys, axis=0)
    assert pooled.shape[0] == unique_keys.shape[0]
    assert stride.shape[0] == unique_keys.shape[0]

    rng = np.random.default_rng(0)
    sampled_keys = unique_keys[rng.choice(unique_keys.shape[0], size=min(20, unique_keys.shape[0]), replace=False)]
    x_min = int(coords[:, 0].min())
    y_min = int(coords[:, 1].min())
    for key in sampled_keys:
        bucket_mask = np.all(bucket_keys == key, axis=1)
        bucket_coords = coords[np.flatnonzero(bucket_mask)]
        grid_x = (bucket_coords[:, 0] - x_min) // step
        grid_y = (bucket_coords[:, 1] - y_min) // step
        assert 1 <= bucket_coords.shape[0] <= 4
        assert (grid_x.max() - grid_x.min()) < 2
        assert (grid_y.max() - grid_y.min()) < 2
        unique_x = np.unique(bucket_coords[:, 0])
        unique_y = np.unique(bucket_coords[:, 1])
        expected_coords = {(int(x), int(y)) for x in unique_x for y in unique_y}
        actual_coords = {tuple(map(int, row)) for row in bucket_coords.tolist()}
        assert actual_coords == expected_coords

    first_bucket_mask = np.all(bucket_keys == np.array([0, 0]), axis=1)
    first_bucket_indices = np.flatnonzero(first_bucket_mask)
    assert coords[first_bucket_indices].tolist() == [
        [0, 0],
        [0, 512],
        [512, 0],
        [512, 512],
    ]

    expected_first_mean = torch.from_numpy(features[first_bucket_indices].mean(axis=0, dtype=np.float32))
    expected_first_stride = torch.from_numpy(features[first_bucket_indices[0]].astype(np.float32, copy=False))

    assert torch.allclose(pooled[0], expected_first_mean, atol=1e-6)
    assert torch.allclose(stride[0], expected_first_stride, atol=1e-6)
