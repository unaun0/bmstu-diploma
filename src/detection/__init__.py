from src.detection.engine import (
    DefectSegmentation,
    get_default_engine,
    get_ui_engine,
)
from src.detection.engine import (
    MODEL_NAME,
    UI_CHECKPOINT_FILENAME,
    UI_CHECKPOINT_PATH,
)


def get_logits(image):
    return get_default_engine().get_logits(image)


def logits_to_mask(logits, threshold):
    return get_default_engine().logits_to_mask(logits, threshold)


def segment(image, threshold):
    return get_default_engine().segment(image, threshold)

__all__ = [
    "DefectSegmentation",
    "MODEL_NAME",
    "UI_CHECKPOINT_FILENAME",
    "UI_CHECKPOINT_PATH",
    "get_default_engine",
    "get_logits",
    "get_ui_engine",
    "logits_to_mask",
    "segment",
]
