from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


class TridentAdapter:
    """Thin adapter around a vendored TRIDENT checkout.

    This wrapper keeps integration points explicit and avoids locking this repository
    to uncertain TRIDENT internals until APIs are finalized.
    """

    def __init__(self, trident_root: str | Path):
        self.trident_root = Path(trident_root)

    def ensure_on_path(self) -> None:
        if not self.trident_root.exists():
            raise FileNotFoundError(
                f"TRIDENT root not found at '{self.trident_root}'. "
                "Place a TRIDENT checkout under external/trident."
            )
        root_str = str(self.trident_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def import_core(self) -> Any:
        self.ensure_on_path()
        try:
            return importlib.import_module("trident")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Failed to import TRIDENT after adding external/trident to sys.path."
            ) from exc

    def load_encoders(self, slide_encoder: str, patch_encoder: str, device: str = "cpu") -> tuple[Any, Any]:
        self.ensure_on_path()
        patch_mod = importlib.import_module("trident.patch_encoder_models")
        slide_mod = importlib.import_module("trident.slide_encoder_models")

        create_patch = getattr(patch_mod, "create_patch_encoder", None)
        create_slide = getattr(slide_mod, "create_slide_encoder", None)
        if create_patch is None or create_slide is None:
            raise NotImplementedError(
                "TRIDENT encoder factory APIs are not confirmed in this scaffold. "
                "Update this adapter after pinning the exact TRIDENT snapshot/API."
            )

        patch_model = create_patch(patch_encoder, device=device)
        slide_model = create_slide(slide_encoder, device=device)
        return patch_model, slide_model

    def extract_slide_embeddings(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Feature extraction flow is intentionally left empty until slide/patch "
            "input contracts are finalized for your local TRIDENT snapshot."
        )
