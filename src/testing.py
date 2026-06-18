from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common import DEFAULT_WEIGHTS_DIR, IMAGE_EXTENSIONS, load_image_rgb
from src.detection.utils import logits_to_mask, predict_logits
from src.model.settings import MODEL_NAME
from src.model.utils import create_model_session
from src.training.utils import iou_defect, load_mask_binary

VAL_DIR = _PROJECT_ROOT / "data" / "val"
CHECKPOINT = DEFAULT_WEIGHTS_DIR / "mn3_ca_dsf.pt"
PASS_IOU = 0.6
THRESHOLD = 0.5

_samples: list[tuple[np.ndarray, np.ndarray]] = []


def _load_samples() -> list[tuple[np.ndarray, np.ndarray]]:
    session = create_model_session(CHECKPOINT)
    images_dir = VAL_DIR / "images"
    masks_dir = VAL_DIR / "masks"
    samples: list[tuple[np.ndarray, np.ndarray]] = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            continue
        image = load_image_rgb(image_path)
        gt = load_mask_binary(mask_path).astype(np.float32)
        if gt.shape != image.shape[:2]:
            gt = cv2.resize(
                gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        gt = (gt > 0.5).astype(np.float32)
        logits = predict_logits(
            session.model, image, img_size=session.img_size, device=session.device
        )
        samples.append((logits, gt))

    if not samples:
        raise FileNotFoundError(f"Нет данных в {VAL_DIR}")
    return samples


def load_samples() -> list[tuple[np.ndarray, np.ndarray]]:
    global _samples
    _samples = _load_samples()
    return _samples


def test_defect_localization() -> bool:
    scores = [
        iou_defect(logits_to_mask(logits, THRESHOLD), gt)
        for logits, gt in _samples
    ]
    mean_iou = float(np.mean(scores))
    passed = mean_iou > PASS_IOU
    status = "УСПЕХ" if passed else "НЕУСПЕХ"
    print(f"IoU дефект: {mean_iou:.4f}  (порог > {PASS_IOU:.1f})")
    print(f"Результат:  {status}", flush=True)
    return passed


def main() -> int:
    print("Тестирование выявления и локализации дефектов")
    print(f"Выборка:  {VAL_DIR}")
    print(f"Модель:   {MODEL_NAME}")
    print(f"Чекпоинт: {CHECKPOINT.name}")
    load_samples()
    print(f"Кадров:   {len(_samples)}")
    print(f"Порог:    {THRESHOLD:.2f}")
    passed = test_defect_localization()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
