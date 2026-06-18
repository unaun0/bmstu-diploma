from __future__ import annotations
from pathlib import Path
import numpy as np
from src.detection.engine import DefectSegmentation
from src.ui.settings import UI_CHECKPOINT_PATH

_engine = DefectSegmentation(checkpoint=UI_CHECKPOINT_PATH)


def init_model(checkpoint: str | Path | None = None) -> None:
    global _engine
    _engine = DefectSegmentation(checkpoint=checkpoint or UI_CHECKPOINT_PATH)


def get_logits(image: np.ndarray) -> np.ndarray:
    return _engine.get_logits(image)


def mask_from_logits(logits: np.ndarray, confidence_threshold: float) -> np.ndarray:
    return _engine.logits_to_mask(logits, confidence_threshold)


def get_segmentation_mask(
    image: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    return mask_from_logits(get_logits(image), confidence_threshold)
