#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.script_config import load_script_cfg

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def _string_list(values: Any) -> list[str]:
    items: list[str] = []
    for value in list(values or []):
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return items


def _source_value(mapping: Any, source_name: str) -> Any:
    if source_name not in mapping:
        raise KeyError(f"Missing pathology_features.uni value for source '{source_name}'.")
    return mapping[source_name]


def _discover_archives(*, repo_id: str, repo_type: str, dataset_subfolder: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    repo_files = api.list_repo_files(
        repo_id,
        repo_type=repo_type,
        token=True,
    )
    archives = [
        Path(repo_file).name
        for repo_file in repo_files
        if str(repo_file).startswith(f"{DATASET_SUBFOLDER}/") and str(repo_file).endswith(".tar.gz")
    ]
    return sorted(set(archives))


def _selected_archives(
    *,
    selected_archives: list[str],
    repo_id: str,
    repo_type: str,
    dataset_subfolder: str,
) -> list[str]:
    if selected_archives:
        return sorted(set(selected_archives))
    return _discover_archives(
        repo_id=repo_id,
        repo_type=repo_type,
        dataset_subfolder=dataset_subfolder,
    )


def main() -> None:
    cfg = load_script_cfg(
        repo_root=ROOT,
        config_relative_path="01_pathology_features/default.yaml",
        overrides=sys.argv[1:],
    )
    uni_cfg = cfg.pathology_features.uni
    source_name = str(uni_cfg.source).strip().lower()
    repo_id = str(uni_cfg.repo_id)
    repo_type = str(uni_cfg.repo_type)
    dataset_subfolder = str(_source_value(uni_cfg.dataset_subfolders, source_name))
    local_archive_dir = Path(str(_source_value(uni_cfg.archive_roots, source_name)))
    selected_archives = _string_list(_source_value(uni_cfg.selected_archives, source_name))
    force_download = bool(uni_cfg.force_download)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for UNI archive download.") from exc

    archives = _selected_archives(
        selected_archives=selected_archives,
        repo_id=repo_id,
        repo_type=repo_type,
        dataset_subfolder=dataset_subfolder,
    )
    if not archives:
        raise RuntimeError(f"No UNI archives selected for source '{source_name}'.")

    local_archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source_name}")
    print(f"Repo: {repo_id}")
    print(f"Dataset subfolder: {dataset_subfolder}")
    print(f"Archive dir: {local_archive_dir}")
    print(f"Archives selected: {len(archives)}")
    if selected_archives:
        print("Selection mode: explicit archive list from YAML")
    else:
        print(f"Selection mode: all {source_name.upper()} archives discovered from the gated repo")

    loop = tqdm(archives, total=len(archives), desc=f"Downloading UNI {source_name.upper()} archives", unit="archive")
    for archive_name in loop:
        loop.set_postfix_str(archive_name)
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{dataset_subfolder}/{archive_name}",
            repo_type=repo_type,
            local_dir=local_archive_dir,
            token=True,
            force_download=force_download,
        )

    print("UNI archive download complete.")
    print(f"Archives available under: {local_archive_dir / dataset_subfolder}")


if __name__ == "__main__":
    main()
