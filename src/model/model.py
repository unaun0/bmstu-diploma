from __future__ import annotations
from pathlib import Path
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.model.deeplab_ca_dsf import DeepLabV3PlusDSFCA
from src.model.settings import (
    ACTIVATION,
    ENCODER,
    ENCODER_WEIGHTS,
    IN_CHANNELS,
    OUT_CHANNELS,
)


def build_model(pretrained: bool = True) -> nn.Module:
    weights = ENCODER_WEIGHTS if pretrained else None
    backbone = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=weights,
        in_channels=IN_CHANNELS,
        classes=OUT_CHANNELS,
        activation=ACTIVATION,
    )
    return DeepLabV3PlusDSFCA(backbone)


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)
    return target
