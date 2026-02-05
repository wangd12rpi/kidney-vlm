from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class KidneyStoneSample:
    image_path: Path
    label: str


def load_kidney_stone_dataset(root: str | Path) -> List[KidneyStoneSample]:
    """Scan kidney stone dataset folder.

    Expected structure (your existing one):
      root/
        Stone/*.jpg
        Non-Stone/*.jpg

    Returns a flat list of samples.
    """
    root = Path(root)
    samples: List[KidneyStoneSample] = []

    for label_dir, label in [("Stone", "Stone"), ("Non-Stone", "Non-Stone")]:
        d = root / label_dir
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpg")):
            samples.append(KidneyStoneSample(image_path=p, label=label))
        for p in sorted(d.glob("*.png")):
            samples.append(KidneyStoneSample(image_path=p, label=label))

    return samples
