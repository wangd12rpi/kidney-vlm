from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def install_visualization_utils_fallback(torch_module: Any) -> None:
    if "sam3.visualization_utils" in sys.modules:
        return

    fallback_module = types.ModuleType("sam3.visualization_utils")

    def normalize_bbox(boxes: Any, img_w: int, img_h: int):
        boxes_tensor = torch_module.as_tensor(boxes, dtype=torch_module.float32)
        scale = torch_module.tensor(
            [float(img_w), float(img_h), float(img_w), float(img_h)],
            dtype=boxes_tensor.dtype,
            device=boxes_tensor.device,
        )
        return boxes_tensor / scale

    fallback_module.normalize_bbox = normalize_bbox
    sys.modules["sam3.visualization_utils"] = fallback_module


def resolve_sam3_bpe_path(sam3_root: Path) -> Path:
    candidates = [
        sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find SAM3 tokenizer vocab file. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


@contextmanager
def cpu_safe_cuda_tensor_factories(torch_module: Any, enabled: bool):
    if not enabled:
        yield
        return

    original_zeros = torch_module.zeros
    original_arange = torch_module.arange

    def _normalize_device(kwargs: dict[str, Any]) -> dict[str, Any]:
        if kwargs.get("device") == "cuda":
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
        return kwargs

    def _zeros_wrapper(*args, **kwargs):
        return original_zeros(*args, **_normalize_device(kwargs))

    def _arange_wrapper(*args, **kwargs):
        return original_arange(*args, **_normalize_device(kwargs))

    torch_module.zeros = _zeros_wrapper
    torch_module.arange = _arange_wrapper
    try:
        yield
    finally:
        torch_module.zeros = original_zeros
        torch_module.arange = original_arange


def load_sam3_module(
    *,
    module_path: Path,
    sam3_root: Path,
    torch_module: Any,
) -> Any:
    if not module_path.is_file():
        raise FileNotFoundError(f"Medical-SAM3 inference module not found: {module_path}")
    if not sam3_root.exists():
        raise FileNotFoundError(
            "SAM3 source tree was not found. Expected it at "
            f"{sam3_root}. Pass a valid sam3_root."
        )

    spec = importlib.util.spec_from_file_location("medical_sam3_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {module_path}")

    install_visualization_utils_fallback(torch_module)

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SAM3_ROOT = sam3_root.resolve()
    return module


def patch_sam3_module_for_device(
    *,
    sam3_module: Any,
    device: str,
    torch_module: Any,
) -> None:
    original_load_model = sam3_module.SAM3Model.load_model

    def patched_load_model(self):
        if self.model is not None:
            return

        if self.checkpoint_path:
            print(f"Loading SAM3 model from checkpoint: {self.checkpoint_path}")
        else:
            print("Loading SAM3 model from HuggingFace...")

        if device == "cuda":
            torch_module.backends.cuda.matmul.allow_tf32 = True
            torch_module.backends.cudnn.allow_tf32 = True
            torch_module.autocast("cuda", dtype=torch_module.bfloat16).__enter__()

        bpe_path = resolve_sam3_bpe_path(Path(sam3_module.SAM3_ROOT))
        build_kwargs = {
            "bpe_path": str(bpe_path),
            "checkpoint_path": None,
            "load_from_HF": False,
            "device": device,
        }

        with cpu_safe_cuda_tensor_factories(torch_module, enabled=device == "cpu"):
            self.model = sam3_module.build_sam3_image_model(**build_kwargs)
            self._load_custom_checkpoint(self.checkpoint_path)

        self.processor = sam3_module.Sam3Processor(
            self.model,
            device=device,
            confidence_threshold=self.confidence_threshold,
        )

        print(f"SAM3 model loaded successfully on {device}!")

    sam3_module.SAM3Model.load_model = patched_load_model
    sam3_module.SAM3Model._original_load_model = original_load_model


def resolve_device(requested_device: str, torch_module: Any) -> str:
    requested = str(requested_device).strip() or "cpu"
    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        return "cpu"
    return requested
