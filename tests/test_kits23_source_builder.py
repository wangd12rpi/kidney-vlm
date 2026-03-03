from __future__ import annotations

from pathlib import Path

from kidney_vlm.data.sources.kits23 import _discover_case_dirs, assign_split, build_kits23_registry_rows


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_assign_split_is_deterministic() -> None:
    split_1 = assign_split("case_00001", {"train": 0.8, "val": 0.1, "test": 0.1})
    split_2 = assign_split("case_00001", {"train": 0.8, "val": 0.1, "test": 0.1})
    assert split_1 == split_2
    assert split_1 in {"train", "val", "test"}


def test_discover_case_dirs_direct_layout(tmp_path: Path) -> None:
    (tmp_path / "case_00000").mkdir()
    (tmp_path / "case_00001").mkdir()

    discovered = _discover_case_dirs(tmp_path)
    assert [path.name for path in discovered] == ["case_00000", "case_00001"]


def test_discover_case_dirs_nested_dataset_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "case_00000").mkdir(parents=True)
    (dataset_root / "case_00001").mkdir(parents=True)

    discovered = _discover_case_dirs(tmp_path)
    assert [path.name for path in discovered] == ["case_00000", "case_00001"]


def test_build_kits23_registry_rows_uses_required_lists_and_pairs(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    images_root = pairs_root / "images"
    masks_root = pairs_root / "masks"

    _touch(images_root / "case_00001" / "case_00001_slice_0000.png")
    _touch(masks_root / "case_00001" / "case_00001_slice_0000.png")
    _touch(images_root / "case_00001" / "case_00001_slice_0001.png")
    _touch(masks_root / "case_00001" / "case_00001_slice_0001.png")
    _touch(images_root / "case_00002" / "case_00002_slice_0003.png")
    _touch(masks_root / "case_00002" / "case_00002_slice_0003.png")
    _touch(images_root / "case_00003" / "case_00003_slice_0002.png")

    frame = build_kits23_registry_rows(
        pairs_root=pairs_root,
        source_name="kits23",
        split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        slice_axis=0,
        show_progress=False,
    )

    assert len(frame) == 3
    assert set(frame["patient_id"]) == {"case_00001", "case_00002"}
    assert frame["sample_id"].nunique() == 3

    case_1 = frame[frame["patient_id"] == "case_00001"]
    assert len(case_1["split"].unique()) == 1

    for _, row in frame.iterrows():
        assert isinstance(row["pathology_wsi_paths"], list)
        assert isinstance(row["radiology_image_paths"], list)
        assert isinstance(row["pathology_mask_paths"], list)
        assert isinstance(row["radiology_mask_paths"], list)
        assert isinstance(row["pathology_tile_embedding_paths"], list)
        assert isinstance(row["pathology_slide_embedding_paths"], list)
        assert isinstance(row["radiology_embedding_paths"], list)
        assert len(row["pathology_wsi_paths"]) == 0
        assert len(row["radiology_image_paths"]) == 1
        assert len(row["pathology_mask_paths"]) == 0
        assert len(row["radiology_mask_paths"]) == 1
        assert row["source"] == "kits23"


def test_build_kits23_registry_rows_attaches_existing_embedding_paths(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    images_root = pairs_root / "images"
    masks_root = pairs_root / "masks"
    embeddings_root = pairs_root / "embeddings" / "medsiglip-448"

    image_rel = Path("case_00001/case_00001_slice_0000.png")
    _touch(images_root / image_rel)
    _touch(masks_root / image_rel)

    embedding_rel = image_rel.with_suffix(".pt")
    _touch(embeddings_root / embedding_rel)

    frame = build_kits23_registry_rows(
        pairs_root=pairs_root,
        source_name="kits23",
        split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        slice_axis=0,
        embeddings_root=embeddings_root,
        show_progress=False,
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert len(row["radiology_embedding_paths"]) == 1
    assert row["radiology_embedding_paths"][0] == str((embeddings_root / embedding_rel).resolve())
