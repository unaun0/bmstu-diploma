from __future__ import annotations

import json
import random
import sys
from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from src.common.settings import EXPECTED_CONFIG, IMAGE_MEAN, IMAGE_SIDE, IMAGE_STD

SCALE_JITTER_LIMIT = 0.25
SCALE_JITTER_PROB = 0.5


def load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"Файл не найден: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Ошибка чтения JSON в {path}: {e}")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_config(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def parse_set_args(set_args: list[str] | None) -> dict:
    if not set_args:
        return {}
    result = {}
    for item in set_args:
        if "=" not in item:
            sys.exit(
                f"fatalError: Некорректный формат --set: {item!r}. Ожидается key=value"
            )
        k, v = item.split("=", 1)
        v_low = v.lower()
        if v_low == "true":
            v = True
        elif v_low == "false":
            v = False
        elif v.isdigit():
            v = int(v)
        else:
            try:
                v = float(v)
            except ValueError:
                pass
        result[k] = v
    return result


def validate_config(config: dict) -> None:
    expected_keys = [item[0] for item in EXPECTED_CONFIG]
    missing = [k for k in expected_keys if k not in config]
    if missing:
        sys.exit(f"fatalError: Отсутствуют ключи: {missing}")

    for key, exp_type, constraints in EXPECTED_CONFIG:
        value = config[key]
        if not isinstance(value, exp_type):
            sys.exit(
                f"fatalError: Ключ '{key}' должен быть {exp_type.__name__}, "
                f"получен {type(value).__name__} ({value})"
            )
        for constr in constraints:
            if constr[0] == "positive" and value <= 0:
                sys.exit(f"fatalError: {key} = {value} должен быть > 0")
            elif constr[0] == "non_empty" and not value.strip():
                sys.exit(f"fatalError: {key} не может быть пустой строкой")
            elif constr[0] == "range":
                lo, hi = constr[1], constr[2]
                if not (lo <= value <= hi):
                    sys.exit(
                        f"fatalError: {key} = {value} должен быть в диапазоне [{lo}, {hi}]"
                    )


def validate_data(data_path: Path | str) -> None:
    data_path = Path(data_path)
    if not data_path.exists():
        sys.exit(f"fatalError: Папка с данными не найдена: {data_path}")

    for split in ("train", "val"):
        split_dir = data_path / split
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"

        if not images_dir.is_dir():
            sys.exit(f"fatalError: Отсутствует папка {images_dir}")
        if not masks_dir.is_dir():
            sys.exit(f"fatalError: Отсутствует папка {masks_dir}")

        image_files = list(images_dir.glob("*"))
        if not image_files:
            sys.exit(f"fatalError: В {images_dir} нет изображений")

        if not any(list(masks_dir.glob(f"{img.stem}.*")) for img in image_files):
            sys.exit(f"fatalError: В {split_dir} нет ни одной пары изображение-маска")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_center_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    top, left, side = center_square_crop_bounds(h, w)
    return image[top : top + side, left : left + side]


def crop_center_square_and_resize(image: np.ndarray, size: int = IMAGE_SIDE) -> np.ndarray:
    cropped = crop_center_square(image)
    return cv2.resize(
        cropped,
        (size, size),
        interpolation=cv2.INTER_LANCZOS4,
    )


def center_square_crop_bounds(height: int, width: int) -> tuple[int, int, int]:
    """Координаты центрального квадратного кропа: top, left, side."""
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return top, left, side


def map_model_output_to_image(
    values: np.ndarray,
    height: int,
    width: int,
    *,
    model_size: int,
) -> np.ndarray:
    """
    Разложить выход model_size×model_size на полное изображение height×width
    после center-crop препроцесса (области вне кропа — нули).
    """
    if values.shape != (model_size, model_size):
        values = cv2.resize(
            values,
            (model_size, model_size),
            interpolation=cv2.INTER_LINEAR,
        )
    top, left, side = center_square_crop_bounds(height, width)
    patch = cv2.resize(
        values,
        (side, side),
        interpolation=cv2.INTER_LINEAR,
    )
    out = np.zeros((height, width), dtype=values.dtype)
    out[top : top + side, left : left + side] = patch
    return out


def preprocess_image(image: np.ndarray, img_size: int = IMAGE_SIDE) -> torch.Tensor:
    resized = crop_center_square_and_resize(image, img_size)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    mean = torch.as_tensor(IMAGE_MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGE_STD, dtype=tensor.dtype).view(1, 3, 1, 1)
    
    return (tensor - mean) / std


def build_train_transforms(img_size: int = IMAGE_SIDE, augmentation_prob: float = 0.5) -> A.Compose:
    transforms: list[A.BasicTransform] = [
        A.RandomScale(scale_limit=SCALE_JITTER_LIMIT, p=SCALE_JITTER_PROB),
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=augmentation_prob),
        A.VerticalFlip(p=augmentation_prob),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=augmentation_prob * 0.8,
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=augmentation_prob * 0.8,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=augmentation_prob * 0.5),
        A.GaussNoise(var_limit=(5.0, 25.0), mean=0, p=augmentation_prob * 0.5),
        A.Normalize(mean=IMAGE_MEAN.tolist(), std=IMAGE_STD.tolist()),
        ToTensorV2(),
    ]
    return A.Compose(transforms, additional_targets={"mask": "mask"})


def build_val_transforms(img_size: int = IMAGE_SIDE) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGE_MEAN.tolist(), std=IMAGE_STD.tolist()),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )
