from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from src.common import get_device
from src.common.utils import map_model_output_to_image, preprocess_image


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    image: np.ndarray,
    *,
    img_size: int,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = get_device()

    h, w = image.shape[:2]
    tensor = preprocess_image(image, img_size).to(device)

    model.eval()
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        logits = model(tensor)

    logits_np = logits.squeeze().float().cpu().numpy()
    if logits_np.shape != (h, w):
        logits_np = map_model_output_to_image(
            logits_np, h, w, model_size=img_size,
        )

    return logits_np.astype(np.float32, copy=False)


@torch.no_grad()
def predict_prob_map(
    model: nn.Module,
    image: np.ndarray,
    *,
    img_size: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Полный инференс одного изображения.

    Препроцессинг → forward модели → logits → sigmoid → карта вероятностей
    того же размера, что и входное RGB-изображение.

    Returns
    -------
    np.ndarray
        float32 (H, W), значения в [0, 1].
    """
    logits = predict_logits(
        model,
        image,
        img_size=img_size,
        device=device,
    )
    return logits_to_prob_map(logits)


def logits_to_prob_map(logits: np.ndarray) -> np.ndarray:
    """
    Sigmoid: logits (H, W) -> вероятности [0, 1].
    """
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def logits_to_mask(logits: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0.0:
        return np.ones_like(logits, dtype=np.float32)
    if threshold >= 1.0:
        return np.zeros_like(logits, dtype=np.float32)
    logit_threshold = np.float32(np.log(threshold / (1.0 - threshold)))
    return (logits >= logit_threshold).astype(np.float32)


def detect_defects(
    model: nn.Module,
    image: np.ndarray,
    threshold: float,
    *,
    img_size: int,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prob_map = predict_prob_map(
        model,
        image,
        img_size=img_size,
        device=device,
    )
    mask = (prob_map >= threshold).astype(np.float32)
    return prob_map, mask


@torch.no_grad()
def find_defects(
    image: torch.Tensor,
    model: nn.Module,
    device: torch.device
) -> torch.Tensor:
    model.to(device)
    model.eval()

    tensor = image.to(device)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        logits = model(tensor)

    logits = logits.squeeze().float()

    return torch.sigmoid(logits)
