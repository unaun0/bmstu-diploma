from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

from src.common.utils import crop_center_square, load_image_rgb
from src.ui.settings import (
    MAX_IMAGE_FILE_BYTES,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    OVERLAY_COLOR,
)


def format_bytes(size: int) -> str:
    if size < 0:
        size = 0
    units = ("Б", "КБ", "МБ", "ГБ")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == "ГБ":
            if unit == "Б":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}".replace(".0 ", " ")
        value /= 1024
    return f"{value:.1f} ГБ"


def format_duration(seconds: float) -> str:
    if seconds < 0.001:
        return "<1 мс"
    if seconds < 1:
        return f"{seconds * 1000:.0f} мс"
    if seconds < 60:
        return f"{seconds:.2f} с"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes} мин {secs:.1f} с"


def format_percent(value: int) -> str:
    return f"{value}%"


def resolution_text(width: int, height: int) -> str:
    return f"{width}×{height}"


def estimate_rgb_memory(width: int, height: int) -> int:
    return width * height * 3


def estimate_pipeline_memory(image: np.ndarray, mask: np.ndarray) -> int:
    return int(image.nbytes + mask.nbytes)


def validate_file_size(path: Path) -> str | None:
    size = path.stat().st_size
    if size > MAX_IMAGE_FILE_BYTES:
        return (
            f"размер файла ({format_bytes(size)}) больше максимума "
            f"{format_bytes(MAX_IMAGE_FILE_BYTES)}"
        )
    return None


def validate_image_size(width: int, height: int) -> str | None:
    for name, val in (("ширина", width), ("высота", height)):
        if val < MIN_IMAGE_SIDE:
            return f"{name} ({val}px) меньше минимума {MIN_IMAGE_SIDE}px"
        if val > MAX_IMAGE_SIDE:
            return f"{name} ({val}px) больше максимума {MAX_IMAGE_SIDE}px"
    return None


def read_image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def load_image_from_path(path: Path) -> np.ndarray:
    return crop_center_square(load_image_rgb(path))


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), mask_to_uint8(mask)):
        raise OSError(f"Cannot write mask: {path}")


def encode_mask_png(mask: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", mask_to_uint8(mask))
    if not ok:
        raise OSError("Failed to encode mask PNG")
    return buffer.tobytes()


def apply_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    color: tuple[int, int, int] = OVERLAY_COLOR,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    m = np.clip(mask.astype(np.float32), 0, 1)
    if m.shape[:2] != image.shape[:2]:
        m = cv2.resize(
            m,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if m.max() <= 0:
        return image.copy()

    overlay = np.zeros_like(image, dtype=np.float32)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    blend = (m * alpha)[..., np.newaxis]
    out = image.astype(np.float32) * (1 - blend) + overlay * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        return mask
    return (np.clip(mask, 0, 1) * 255).astype(np.uint8)
