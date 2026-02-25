from __future__ import annotations

import pytest


def test_freeze_policy_vlm_only_trainable() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    from kidney_vlm.training.freeze_policy import apply_freeze_policy

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pathology_encoder = nn.Linear(4, 4)
            self.segmentation_head = nn.Linear(4, 4)
            self.radiology_encoder = nn.Linear(4, 4)
            self.vlm = nn.Linear(4, 4)

    model = DummyModel()
    apply_freeze_policy(model, ["pathology", "radiology", "segmentation"])

    for name, parameter in model.named_parameters():
        if name.startswith("vlm"):
            assert parameter.requires_grad is True
        else:
            assert parameter.requires_grad is False


def test_stage_projectors_only_trainable() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    from kidney_vlm.training.freeze_policy import apply_training_stage

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pathology_encoder = nn.Linear(4, 4)
            self.radiology_encoder = nn.Linear(4, 4)
            self.segmentation_head = nn.Linear(4, 4)
            self.projectors = nn.Linear(4, 4)
            self.vlm = nn.Linear(4, 4)

    model = DummyModel()
    apply_training_stage(
        model,
        stage="projectors",
        always_frozen_prefixes=["pathology", "radiology", "segmentation"],
        projector_prefixes=["projector", "projectors"],
    )

    for name, parameter in model.named_parameters():
        if name.startswith("projectors"):
            assert parameter.requires_grad is True
        else:
            assert parameter.requires_grad is False


def test_stage_vlm_trainable_and_foundations_frozen() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    from kidney_vlm.training.freeze_policy import apply_training_stage

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pathology_encoder = nn.Linear(4, 4)
            self.radiology_encoder = nn.Linear(4, 4)
            self.segmentation_head = nn.Linear(4, 4)
            self.projectors = nn.Linear(4, 4)
            self.vlm = nn.Linear(4, 4)

    model = DummyModel()
    apply_training_stage(
        model,
        stage="vlm",
        always_frozen_prefixes=["pathology", "radiology", "segmentation"],
        projector_prefixes=["projector", "projectors"],
        freeze_projectors_in_vlm_stage=False,
    )

    for name, parameter in model.named_parameters():
        if (
            name.startswith("pathology_encoder")
            or name.startswith("radiology_encoder")
            or name.startswith("segmentation_head")
        ):
            assert parameter.requires_grad is False
        else:
            assert parameter.requires_grad is True
