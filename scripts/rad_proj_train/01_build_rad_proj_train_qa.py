#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.radiology.pmc_oa_feature_store import build_pmc_oa_lookup_by_image_name, read_or_build_pmc_oa_feature_index
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:
        raise RuntimeError("hydra is required for radiology projector QA building.") from exc

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config", overrides=["qa_genereation=rad_proj_train_qa"])
    OmegaConf.set_struct(cfg, False)
    return cfg


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _project_relative_or_absolute(path: Path) -> str:
    absolute_path = path if path.is_absolute() else (ROOT / path)
    try:
        return absolute_path.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute_path.as_posix()


def _resolve_repo_path_preserve_symlink(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _normalize_split_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"validation", "valid", "val", "dev"}:
        return "validation"
    if normalized == "test":
        return "test"
    return "train"


def _load_caption_split_frame(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PMC-OA caption split not found: {path}")
    frame = pd.read_json(path, lines=True)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["split"] = _normalize_split_name(split_name)
    return frame


def _build_training_rows(
    caption_frame: pd.DataFrame,
    *,
    feature_index_by_image_name: dict[str, dict[str, object]],
    image_root_dir: Path,
    require_existing_image_files: bool,
    default_instruction: str,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    rows: list[dict[str, object]] = []
    missing_feature_images: list[str] = []
    missing_image_files: list[str] = []

    for _, row in caption_frame.iterrows():
        image_name = str(row.get("image", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if not image_name or not caption:
            continue

        feature_row = feature_index_by_image_name.get(image_name)
        if feature_row is None:
            missing_feature_images.append(image_name)
            continue

        image_path = image_root_dir / image_name
        if require_existing_image_files and not image_path.exists():
            missing_image_files.append(image_name)
            continue

        sample_id = str(feature_row["sample_id"]).strip()
        rows.append(
            {
                "qa_row_id": sample_id,
                "sample_id": sample_id,
                "source": "pmc_oa",
                "project_id": "pmc_oa",
                "patient_id": str(row.get("pmcid", "")).strip(),
                "study_id": str(row.get("url_name", "")).strip(),
                "split": str(row.get("split", "train")).strip(),
                "series_stem": sample_id,
                "modality": "radiology",
                "pmcid": str(row.get("pmcid", "")).strip(),
                "url_name": str(row.get("url_name", "")).strip(),
                "image_name": image_name,
                "radiology_image_paths": [_project_relative_or_absolute(image_path)],
                "radiology_image_modalities": ["figure"],
                "radiology_embedding_paths": [str(feature_row["embedding_ref"]).strip()],
                "instruction": default_instruction,
                "question": default_instruction,
                "caption": caption,
                "answer": caption,
            }
        )

    return rows, missing_feature_images, missing_image_files


def _build_output_frame(
    *,
    existing_output: pd.DataFrame,
    generated_rows: list[dict[str, object]],
    overwrite_output: bool,
) -> pd.DataFrame:
    generated_df = pd.DataFrame(generated_rows)
    if not existing_output.empty and not overwrite_output:
        final_df = pd.concat([existing_output, generated_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["qa_row_id"], keep="last").reset_index(drop=True)
        return final_df
    return generated_df


def main() -> None:
    cfg = load_cfg()
    qa_cfg = cfg.qa_genereation

    feature_store_path = _resolve_path(qa_cfg.feature_store_path)
    feature_index_path = _resolve_path(qa_cfg.feature_index_path)
    image_root_dir = _resolve_repo_path_preserve_symlink(qa_cfg.image_root_dir)
    train_jsonl_path = _resolve_path(qa_cfg.train_jsonl_path)
    validation_jsonl_path = _resolve_path(qa_cfg.validation_jsonl_path)
    test_jsonl_path = _resolve_path(qa_cfg.test_jsonl_path)
    output_path = _resolve_path(qa_cfg.output_parquet_path)

    feature_index = read_or_build_pmc_oa_feature_index(
        root_dir=ROOT,
        store_path=feature_store_path,
        index_path=feature_index_path,
        rebuild=bool(qa_cfg.get("rebuild_feature_index", False)),
    )
    if feature_index.empty:
        raise RuntimeError(f"PMC-OA feature index is empty: {feature_index_path}")
    feature_index_by_image_name = build_pmc_oa_lookup_by_image_name(feature_index)

    train_frame = _load_caption_split_frame(train_jsonl_path, "train")
    validation_frame = _load_caption_split_frame(validation_jsonl_path, "validation")
    test_frame = _load_caption_split_frame(test_jsonl_path, "test")
    caption_frame = pd.concat([train_frame, validation_frame, test_frame], ignore_index=True)

    rows, missing_feature_images, missing_image_files = _build_training_rows(
        caption_frame,
        feature_index_by_image_name=feature_index_by_image_name,
        image_root_dir=image_root_dir,
        require_existing_image_files=bool(qa_cfg.get("require_existing_image_files", False)),
        default_instruction=str(qa_cfg.get("instruction", "Describe the radiology image.")).strip(),
    )
    if not rows:
        raise RuntimeError("No PMC-OA radiology projector rows were generated.")

    overwrite_output = bool(qa_cfg.get("overwrite_output", False))
    existing_output = pd.DataFrame()
    if output_path.exists() and not overwrite_output:
        existing_output = pd.read_parquet(output_path)

    final_df = _build_output_frame(
        existing_output=existing_output,
        generated_rows=rows,
        overwrite_output=overwrite_output,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    print(f"PMC-OA feature store: {feature_store_path}")
    print(f"PMC-OA feature index: {feature_index_path}")
    print(f"Feature index rows: {len(feature_index)}")
    print(f"Caption rows loaded: {len(caption_frame)}")
    print(f"Rows written: {len(final_df)}")
    print(f"Output parquet: {output_path}")
    print(f"Missing feature matches: {len(missing_feature_images)}")
    print(f"Missing image files: {len(missing_image_files)}")

    fail_on_missing_features = bool(qa_cfg.get("fail_on_missing_features", True))
    if missing_feature_images:
        print("Missing feature image examples:")
        for image_name in missing_feature_images[:10]:
            print(f"  {image_name}")
        if fail_on_missing_features:
            raise RuntimeError(
                "Some PMC-OA caption rows did not have matching feature-store entries. "
                "Set qa_genereation.fail_on_missing_features=false to keep partial output."
            )

    if missing_image_files:
        print("Missing image file examples:")
        for image_name in missing_image_files[:10]:
            print(f"  {image_name}")

    print_first_n = int(qa_cfg.get("print_first_n", 0) or 0)
    for row in rows[:print_first_n]:
        print("-" * 80)
        print(f"qa_row_id: {row['qa_row_id']}")
        print(f"split: {row['split']}")
        print(f"image_name: {row['image_name']}")
        print(f"pmcid: {row['pmcid']}")
        print(f"caption: {row['caption']}")


if __name__ == "__main__":
    main()
