#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
SRC = BOOTSTRAP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidney_vlm.segmentation.pathology_segmentation_runner import main


if __name__ == "__main__":
    main()
