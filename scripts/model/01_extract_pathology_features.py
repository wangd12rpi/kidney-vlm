#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.repo_root import find_repo_root
from kidney_vlm.pathology.trident_adapter import TridentAdapter

ROOT = find_repo_root(Path(__file__))
os.environ["KIDNEY_VLM_ROOT"] = str(ROOT)


def load_cfg():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        return compose(config_name="config")


def main() -> None:
    cfg = load_cfg()
    trident_root = Path(str(cfg.model.pathology.trident_root))
    adapter = TridentAdapter(trident_root=trident_root)

    print(f"TRIDENT root: {trident_root}")
    try:
        adapter.import_core()
        print("TRIDENT import check passed.")
    except Exception as exc:
        print(f"TRIDENT import check failed: {exc}")

    print(
        "Feature extraction pipeline is scaffolded only. "
        "Implement concrete patch/slide processing after pinning the exact TRIDENT API snapshot."
    )


if __name__ == "__main__":
    main()
