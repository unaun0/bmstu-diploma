from __future__ import annotations
from pathlib import Path
import numpy as np
from src.detection.utils import logits_to_mask, predict_logits
from src.model import ensure_model, load_model
from src.model.settings import MODEL_NAME, WEIGHTS_DIR

UI_CHECKPOINT_FILENAME = "mn3_ca_dsf.pt"
UI_CHECKPOINT_PATH = WEIGHTS_DIR / UI_CHECKPOINT_FILENAME


class DefectSegmentation:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        img_size: int | None = None,
        threshold: float | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._img_size = img_size
        self._threshold = threshold
        if checkpoint is not None:
            load_model(
                checkpoint,
                img_size=img_size,
                threshold=threshold,
            )

    def _ensure_session(self):
        return ensure_model(
            self._checkpoint,
            img_size=self._img_size,
            threshold=self._threshold,
        )

    def get_logits(self, image: np.ndarray) -> np.ndarray:
        s = self._ensure_session()
        return predict_logits(
            s.model,
            image,
            img_size=s.img_size,
            device=s.device,
        )

    def logits_to_mask(self, logits: np.ndarray, threshold: float) -> np.ndarray:
        return logits_to_mask(logits, threshold)

    def segment(self, image: np.ndarray, threshold: float) -> np.ndarray:
        logits = self.get_logits(image)
        return logits_to_mask(logits, threshold)


_ENGINE = DefectSegmentation()


def get_default_engine() -> DefectSegmentation:
    return _ENGINE


def get_ui_engine() -> DefectSegmentation:
    return DefectSegmentation(checkpoint=UI_CHECKPOINT_PATH)
