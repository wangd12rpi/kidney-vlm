#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


# Download settings
REPO_ID = "MahmoodLab/UNI2-h-features"
REPO_TYPE = "dataset"
DATASET_SUBFOLDER = "TCGA"
LOCAL_ARCHIVE_DIR = ROOT / "data" / "staging" / "uni2_tcga_archives"

# If empty, download every TCGA archive listed in the gated dataset.
PROJECT_ARCHIVES: list[str] = [
    # "TCGA-KICH.tar.gz",
    # "TCGA-KIRC.tar.gz",
    # "TCGA-KIRP.tar.gz",
]

FORCE_DOWNLOAD = False


def _discover_tcga_archives() -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    repo_files = api.list_repo_files(
        REPO_ID,
        repo_type=REPO_TYPE,
        token=True,
    )
    archives = [
        Path(repo_file).name
        for repo_file in repo_files
        if str(repo_file).startswith(f"{DATASET_SUBFOLDER}/") and str(repo_file).endswith(".tar.gz")
    ]
    return sorted(set(archives))


def _selected_archives() -> list[str]:
    if PROJECT_ARCHIVES:
        return sorted({str(name).strip() for name in PROJECT_ARCHIVES if str(name).strip()})
    return _discover_tcga_archives()


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for UNI archive download.") from exc

    archives = _selected_archives()
    if not archives:
        raise RuntimeError("No UNI TCGA archives selected for download.")

    LOCAL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Repo: {REPO_ID}")
    print(f"Dataset subfolder: {DATASET_SUBFOLDER}")
    print(f"Archive dir: {LOCAL_ARCHIVE_DIR}")
    print(f"Archives selected: {len(archives)}")
    if PROJECT_ARCHIVES:
        print("Selection mode: explicit archive list from PROJECT_ARCHIVES")
    else:
        print("Selection mode: all TCGA archives discovered from the gated repo")

    loop = tqdm(archives, total=len(archives), desc="Downloading UNI TCGA archives", unit="archive")
    for archive_name in loop:
        loop.set_postfix_str(archive_name)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{DATASET_SUBFOLDER}/{archive_name}",
            repo_type=REPO_TYPE,
            local_dir=LOCAL_ARCHIVE_DIR,
            token=True,
            force_download=FORCE_DOWNLOAD,
        )

    print("UNI archive download complete.")
    print(f"Archives available under: {LOCAL_ARCHIVE_DIR / DATASET_SUBFOLDER}")


if __name__ == "__main__":
    main()
