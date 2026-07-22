import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


CONVERTER_PATH = Path(__file__).parents[1] / "examples" / "oncovlm" / "prepare_pathology_mcq.py"
SPEC = importlib.util.spec_from_file_location("prepare_pathology_mcq", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)

REPO_ROOT = Path(__file__).parents[3]
PRIVATE_DATA_AVAILABLE = all(
    path.is_file()
    for path in (
        REPO_ROOT / "data" / "vqa" / "merged_vqa.parquet",
        REPO_ROOT / "data" / "registry" / "unified.parquet",
    )
)


@pytest.fixture(scope="module")
def datasets():
    if not PRIVATE_DATA_AVAILABLE:
        pytest.skip("private kidney-vlm data is not available")
    return converter.prepare_datasets(repo_root=REPO_ROOT, max_images=4)


def test_full_counts_and_deterministic_subsets(datasets):
    assert {name: len(frame) for name, frame in datasets.items()} == {
        "train": 3_873,
        "train_batch8": 3_880,
        "val": 322,
        "smoke32": 32,
        "pilot128": 128,
        "val_smoke4": 4,
        "val_monitor64": 64,
    }


def test_train_validation_cases_are_disjoint(datasets):
    train_cases = set(datasets["train"]["case_id"])
    val_cases = set(datasets["val"]["case_id"])
    assert train_cases.isdisjoint(val_cases)


def test_real_batch_aligned_view_preserves_all_training_questions(datasets):
    train = datasets["train"]
    padded = datasets["train_batch8"]
    assert len(padded) % 8 == 0
    assert padded["question_id"].is_unique
    assert set(train["question_id"]).issubset(padded["question_id"])
    pd.testing.assert_frame_equal(padded.iloc[: len(train)].reset_index(drop=True), train)


@pytest.mark.parametrize("split", ["train", "val"])
def test_rows_have_unique_ids_exact_options_and_existing_images(datasets, split):
    frame = datasets[split]
    assert frame["question_id"].is_unique

    for row in frame.itertuples(index=False):
        option_texts = [
            line[3:]
            for line in row.problem.splitlines()
            if len(line) > 3 and line[:3] in {"A. ", "B. ", "C. ", "D. "}
        ]
        assert row.answer in option_texts
        assert 1 <= len(row.images) <= 4
        assert row.problem.count("<image>") == len(row.images)
        assert all(Path(path).is_file() for path in row.images)


def test_evenly_spaced_image_selection_keeps_endpoints():
    paths = [f"roi_{index}.png" for index in range(10)]
    assert converter.select_evenly_spaced(paths, 4) == [
        "roi_0.png",
        "roi_3.png",
        "roi_6.png",
        "roi_9.png",
    ]


def test_two_roi_selection_and_prompt_placeholders(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"roi_{index}.png"
        path.touch()
        paths.append(str(path))

    selected = converter.select_evenly_spaced(paths, 2)
    row = pd.Series(
        {
            "question": "Which finding best matches?",
            "option_a": "Alpha",
            "option_b": "Beta",
            "option_c": "Gamma",
            "option_d": "Delta",
        }
    )
    problem = converter.build_prompt(row, len(selected))

    assert selected == [paths[0], paths[-1]]
    assert problem.count("<image>") == 2


def test_deterministic_sample_repeats_question_ids():
    frame = pd.DataFrame({"question_id": [91, 12, 77, 3, 44, 68, 25, 59, 36, 10]})

    first = converter.deterministic_sample(frame, 4)
    second = converter.deterministic_sample(frame, 4)

    assert first["question_id"].tolist() == [3, 77, 59, 36]
    assert first["question_id"].tolist() == second["question_id"].tolist()


def test_batch_aligned_training_view_contains_every_source_row():
    frame = pd.DataFrame(
        {
            "question_id": [f"q{index}" for index in range(9)],
            "case_id": [f"c{index}" for index in range(9)],
        }
    )

    padded = converter.pad_to_batch_size(frame, 8)

    assert len(padded) == 16
    assert padded["question_id"].is_unique
    assert set(frame["question_id"]).issubset(padded["question_id"])
    assert padded.iloc[9:]["question_id"].str.contains("__batch_pad_").all()


def test_write_datasets_schema_and_manifest(tmp_path):
    image_path = tmp_path / "roi.png"
    image_path.touch()
    row = {
        "problem": "<image>\n\nQuestion: Q?\n\nOptions:\nA. A\nB. B\nC. C\nD. D",
        "answer": "Alpha",
        "images": [str(image_path)],
        "question_id": "q1",
        "case_id": "c1",
    }
    datasets = {"train": pd.DataFrame([row], columns=converter.OUTPUT_COLUMNS)}
    output_dir = tmp_path / "prepared"

    converter.write_datasets(datasets, repo_root=tmp_path, output_dir=output_dir, max_images=2)

    written = pd.read_parquet(output_dir / "train.parquet")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert written.columns.tolist() == converter.OUTPUT_COLUMNS
    assert written.loc[0, "images"].tolist() == [str(image_path)]
    assert manifest["seed"] == 417
    assert manifest["max_images"] == 2
    assert manifest["counts"] == {"train": 1}
    assert set(manifest["sources"]) == {"vqa", "registry"}
    assert set(manifest["outputs"]) == {"train"}


@pytest.mark.skipif(not PRIVATE_DATA_AVAILABLE, reason="private kidney-vlm data is not available")
def test_two_roi_real_data_counts_and_placeholders():
    two_roi = converter.prepare_datasets(repo_root=REPO_ROOT, max_images=2)
    assert {name: len(frame) for name, frame in two_roi.items()} == {
        "train": 3_873,
        "train_batch8": 3_880,
        "val": 322,
        "smoke32": 32,
        "pilot128": 128,
        "val_smoke4": 4,
        "val_monitor64": 64,
    }
    for frame in two_roi.values():
        assert frame.columns.tolist() == converter.OUTPUT_COLUMNS
        assert frame["images"].map(len).between(1, 2).all()
        assert all(row.problem.count("<image>") == len(row.images) for row in frame.itertuples())


def test_image_limit_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        converter.select_evenly_spaced(["roi.png"], 0)
