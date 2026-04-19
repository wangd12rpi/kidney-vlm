#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.data.sources.pmc_oa import build_pmc_oa_caption_rows, load_pmc_oa_caption_frame
from kidney_vlm.radiology.pmc_oa_feature_store import build_pmc_oa_lookup_by_image_name, read_or_build_pmc_oa_feature_index
from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    from kidney_vlm.script_config import load_script_cfg

    return load_script_cfg(
        repo_root=ROOT,
        config_relative_path="02_radiology_proj/01_import_pmc_oa_captions.yaml",
        overrides=sys.argv[1:],
    )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _build_output_frame(
    *,
    existing_output: pd.DataFrame,
    generated_rows: list[dict[str, object]],
    overwrite_output: bool,
) -> pd.DataFrame:
    generated_df = pd.DataFrame(generated_rows)
    if not existing_output.empty and not overwrite_output:
        final_df = pd.concat([existing_output, generated_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["radiology_caption_row_id"], keep="last").reset_index(drop=True)
        return final_df
    return generated_df


def main() -> None:
    cfg = load_cfg()
    stage_cfg = cfg.radiology_proj

    feature_store_path = _resolve_path(stage_cfg.feature_store_path)
    feature_index_path = _resolve_path(stage_cfg.feature_index_path)
    image_root_dir = _resolve_path(stage_cfg.image_root_dir)
    train_jsonl_path = _resolve_path(stage_cfg.train_jsonl_path)
    validation_jsonl_path = _resolve_path(stage_cfg.validation_jsonl_path)
    test_jsonl_path = _resolve_path(stage_cfg.test_jsonl_path)
    output_path = _resolve_path(stage_cfg.output_parquet_path)

    feature_index = read_or_build_pmc_oa_feature_index(
        root_dir=ROOT,
        store_path=feature_store_path,
        index_path=feature_index_path,
        rebuild=bool(stage_cfg.get("rebuild_feature_index", False)),
    )
    if feature_index.empty:
        raise RuntimeError(f"PMC-OA feature index is empty: {feature_index_path}")
    feature_index_by_image_name = build_pmc_oa_lookup_by_image_name(feature_index)

    caption_frame = load_pmc_oa_caption_frame(
        train_jsonl_path=train_jsonl_path,
        validation_jsonl_path=validation_jsonl_path,
        test_jsonl_path=test_jsonl_path,
    )
    if caption_frame.empty:
        raise RuntimeError("PMC-OA caption splits are empty.")

    first_n = stage_cfg.get("first_n")
    if first_n not in (None, "", "null"):
        caption_frame = caption_frame.head(int(first_n)).reset_index(drop=True)

    rows, missing_feature_images, missing_image_files = build_pmc_oa_caption_rows(
        caption_frame,
        root_dir=ROOT,
        feature_index_by_image_name=feature_index_by_image_name,
        image_root_dir=image_root_dir,
        require_existing_image_files=bool(stage_cfg.get("require_existing_image_files", False)),
        default_instruction=str(stage_cfg.get("instruction", "Describe the radiology image.")).strip(),
        caption_model=str(stage_cfg.get("caption_model", "pmc_oa_human")).strip(),
    )
    if not rows:
        raise RuntimeError("No PMC-OA radiology caption rows were generated.")

    overwrite_output = bool(stage_cfg.get("overwrite_output", False))
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

    if missing_feature_images:
        print("Missing feature image examples:")
        for image_name in missing_feature_images[:10]:
            print(f"  {image_name}")
        if bool(stage_cfg.get("fail_on_missing_features", True)):
            raise RuntimeError(
                "Some PMC-OA caption rows did not have matching feature-store entries. "
                "Set radiology_proj.fail_on_missing_features=false to allow partial output."
            )

    if missing_image_files:
        print("Missing image file examples:")
        for image_name in missing_image_files[:10]:
            print(f"  {image_name}")

    print_first_n = int(stage_cfg.get("print_first_n", 0) or 0)
    for row in rows[:print_first_n]:
        print("-" * 80)
        print(f"radiology_caption_row_id: {row['radiology_caption_row_id']}")
        print(f"sample_id: {row['sample_id']}")
        print(f"split: {row['split']}")
        print(f"image_name: {row['image_name']}")
        print(f"caption: {row['caption']}")


if __name__ == "__main__":
    main()
