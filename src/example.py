from __future__ import annotations

import sys
from dataclasses import dataclass
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

TEST_DIR = _PROJECT_ROOT / "test"
CHECKPOINT = DEFAULT_WEIGHTS_DIR / "mn3_ca_dsf.pt"
THRESHOLD_STEP = 0.05

COLOR_PRED = (0, 0, 255)
COLOR_GT = (255, 0, 0)
COLOR_OVERLAP = (255, 0, 255)


@dataclass
class Sample:
    test_folder: Path
    image_name: str
    stem: str
    logits: np.ndarray
    gt: np.ndarray


def _mask_path_for_image(masks_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        path = masks_dir / f"{stem}{ext}"
        if path.is_file():
            return path
    return None


def _align_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return (mask > 0.5).astype(np.float32)
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.float32)


def _make_intersection_image(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> np.ndarray:
    pred = pred_mask > 0.5
    gt = gt_mask > 0.5
    image = np.zeros((*pred.shape, 3), dtype=np.uint8)
    image[pred & ~gt] = COLOR_PRED
    image[~pred & gt] = COLOR_GT
    image[pred & gt] = COLOR_OVERLAP
    return image


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    binary = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), binary):
        raise OSError(f"Не удалось сохранить маску: {path}")


def _iter_test_folders() -> list[Path]:
    return sorted(
        path for path in TEST_DIR.iterdir()
        if path.is_dir() and path.name.startswith("test_")
    )


def load_all_samples(session) -> list[Sample]:
    samples: list[Sample] = []

    for test_folder in _iter_test_folders():
        images_dir = test_folder / "images"
        masks_dir = test_folder / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            gt_path = _mask_path_for_image(masks_dir, image_path.stem)
            if gt_path is None:
                continue

            image = load_image_rgb(image_path)
            gt = _align_mask(
                load_mask_binary(gt_path), image.shape[0], image.shape[1]
            )
            logits = predict_logits(
                session.model,
                image,
                img_size=session.img_size,
                device=session.device,
            )
            samples.append(
                Sample(
                    test_folder=test_folder,
                    image_name=image_path.name,
                    stem=image_path.stem,
                    logits=logits,
                    gt=gt,
                )
            )

    if not samples:
        raise FileNotFoundError(f"Нет пар изображение/маска в {TEST_DIR}/test_*")
    return samples


def find_best_threshold(samples: list[Sample]) -> float:
    best_threshold = 0.0
    best_score = -1.0
    threshold = 0.0

    while threshold <= 1.0 + 1e-9:
        scores = [
            iou_defect(logits_to_mask(sample.logits, threshold), sample.gt)
            for sample in samples
        ]
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_threshold = round(threshold, 2)
        threshold += THRESHOLD_STEP

    return best_threshold


def save_results(samples: list[Sample], threshold: float) -> None:
    by_folder: dict[Path, list[Sample]] = {}
    for sample in samples:
        by_folder.setdefault(sample.test_folder, []).append(sample)

    for test_folder, folder_samples in by_folder.items():
        results_dir = test_folder / "results"
        inters_dir = test_folder / "inters"
        results_dir.mkdir(parents=True, exist_ok=True)
        inters_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{test_folder.name}")
        print("-" * len(test_folder.name))

        for sample in folder_samples:
            pred = logits_to_mask(sample.logits, threshold)
            iou = iou_defect(pred, sample.gt)

            result_path = results_dir / f"{sample.stem}_mask.png"
            inter_path = inters_dir / f"{sample.stem}_inter.png"

            _save_mask_png(result_path, pred)
            if not cv2.imwrite(str(inter_path), _make_intersection_image(pred, sample.gt)):
                raise OSError(f"Не удалось сохранить пересечение: {inter_path}")

            print(f"  {sample.image_name}: IoU = {iou:.4f}")


def main() -> int:
    if not TEST_DIR.is_dir():
        print(f"Папка не найдена: {TEST_DIR}")
        return 1
    if not CHECKPOINT.is_file():
        print(f"Чекпоинт не найден: {CHECKPOINT}")
        return 1
    if not _iter_test_folders():
        print(f"В {TEST_DIR} нет папок test_*")
        return 1

    print("Пример сегментации на тестовых типах ткани")
    print(f"Модель:   {MODEL_NAME}")
    print(f"Чекпоинт: {CHECKPOINT.name}")

    session = create_model_session(CHECKPOINT)
    samples = load_all_samples(session)

    print(f"Изображений: {len(samples)}")
    print("Подбор порога сегментации...")

    threshold = find_best_threshold(samples)
    print(f"Порог:    {threshold:.2f}")

    save_results(samples, threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
