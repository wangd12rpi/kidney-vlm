"""Download TCGA pathology SVS files listed in data/tcga_path_index.jsonl.

Resumable behavior:
- If the target file exists and matches expected file_size, it is skipped.
- Otherwise it is downloaded to *.part then renamed.

Run:
  python data/tcga_download_path_from_index.py
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

from kidney_vlm.jsonl import read_jsonl


# -----------------
# Config (edit me)
# -----------------
INDEX_PATH = Path("data/tcga_path_index.jsonl")
VERIFY_MD5 = False  # slow, optional
SLEEP_BETWEEN_FILES_S = 0.0  # set to >0 if you want to be nice to the API

CHUNK_SIZE = 1024 * 1024  # 1MB
TIMEOUT_S = 120


def stream_download(url: str, out_path: Path, expected_size: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        if expected_size and total and expected_size != total:
            print(f"[warn] size mismatch header={total} expected={expected_size} for {out_path.name}")

        pbar_total = expected_size or total or None
        with tmp_path.open("wb") as f, tqdm(
            total=pbar_total,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    tmp_path.replace(out_path)


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    rows = read_jsonl(INDEX_PATH)
    print(f"Index rows: {len(rows)}")

    for obj in rows:
        file_id = obj["file_id"]
        out_path = Path(obj["raw_svs_path"])
        expected_size = int(obj.get("file_size") or 0)
        expected_md5 = obj.get("md5sum")

        if out_path.exists() and expected_size > 0 and out_path.stat().st_size == expected_size:
            continue

        url = f"https://api.gdc.cancer.gov/data/{file_id}"
        print(f"\nDownloading: {obj['project_id']} {obj['case_submitter_id']} {obj['file_name']}")
        stream_download(url, out_path, expected_size if expected_size > 0 else None)

        if VERIFY_MD5 and expected_md5:
            got = md5_of_file(out_path)
            if got.lower() != str(expected_md5).lower():
                raise RuntimeError(f"MD5 mismatch for {out_path}: got={got} expected={expected_md5}")

        if SLEEP_BETWEEN_FILES_S > 0:
            time.sleep(SLEEP_BETWEEN_FILES_S)

    print("Done")


if __name__ == "__main__":
    main()
