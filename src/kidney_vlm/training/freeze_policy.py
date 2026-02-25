from __future__ import annotations

from typing import Iterable

from torch import nn

DEFAULT_FREEZE_PREFIXES = ("pathology", "radiology", "segmentation")
DEFAULT_PROJECTOR_PREFIXES = ("projector", "projectors")


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.") or name.startswith(f"{prefix}_")


def should_freeze(name: str, prefixes: Iterable[str]) -> bool:
    return any(_matches_prefix(name, prefix) for prefix in prefixes)


def apply_freeze_policy(model: nn.Module, freeze_prefixes: Iterable[str] = DEFAULT_FREEZE_PREFIXES) -> nn.Module:
    prefixes = tuple(freeze_prefixes)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not should_freeze(name, prefixes)
    return model


def apply_training_stage(
    model: nn.Module,
    stage: str,
    always_frozen_prefixes: Iterable[str] = DEFAULT_FREEZE_PREFIXES,
    projector_prefixes: Iterable[str] = DEFAULT_PROJECTOR_PREFIXES,
    freeze_projectors_in_vlm_stage: bool = False,
) -> nn.Module:
    """Apply trainability policy for two-stage training.

    stage='projectors': only projector parameters are trainable.
    stage='vlm': all non-foundation/non-segmentation params are trainable
                 (optionally freezing projectors).
    """
    stage_name = str(stage).lower()
    always_frozen = tuple(always_frozen_prefixes)
    projectors = tuple(projector_prefixes)

    for name, parameter in model.named_parameters():
        if should_freeze(name, always_frozen):
            parameter.requires_grad = False
            continue

        if stage_name == "projectors":
            parameter.requires_grad = should_freeze(name, projectors)
            continue

        if stage_name == "vlm":
            if freeze_projectors_in_vlm_stage and should_freeze(name, projectors):
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True
            continue

        raise ValueError("Unknown training stage. Expected one of: 'projectors', 'vlm'.")

    return model


def list_trainable_parameters(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
