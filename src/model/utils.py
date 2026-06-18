from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn

from src.common import DEFAULT_CHECKPOINTS_DIR, DEFAULT_CONFIG_PATH, get_device, load_json
from src.model.model import build_model, save_checkpoint
from src.model.settings import (
    CHECKPOINT_PREFIXES,
    CHECKPOINT_PREFIX_ORDER,
    CHECKPOINT_STEMS,
    DEFAULT_MODEL,
    WEIGHTS_DIR,
)

_SESSION: "ModelSession | None" = None


@dataclass
class ModelSession:
    model: nn.Module
    device: torch.device
    img_size: int
    checkpoint_path: Path
    model_name: str
    default_threshold: float = 0.5


def resolve_model_name(checkpoint: Path) -> str:
    stem = checkpoint.stem.lower()
    if stem in CHECKPOINT_STEMS:
        return CHECKPOINT_STEMS[stem]

    for prefix in CHECKPOINT_PREFIX_ORDER:
        if stem.startswith(prefix):
            return CHECKPOINT_PREFIXES[prefix]

    return DEFAULT_MODEL


def load_checkpoint_state(checkpoint: Path, device: torch.device) -> dict:
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Не удалось прочитать веса: {checkpoint}")


def find_default_checkpoint() -> Path:
    if WEIGHTS_DIR.is_dir():
        for prefix in CHECKPOINT_PREFIX_ORDER:
            matches = sorted(WEIGHTS_DIR.glob(f"{prefix}*.pt"))
            if matches:
                return matches[0]

    for candidate in (
        DEFAULT_CHECKPOINTS_DIR / "best.pt",
        DEFAULT_CHECKPOINTS_DIR / "last.pt",
    ):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Не найден чекпоинт. Укажите путь к .pt в src/model/weights/ или checkpoints/."
    )


def create_model_session(
    checkpoint: str | Path,
    *,
    img_size: int | None = None,
    threshold: float | None = None,
) -> ModelSession:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    config = load_json(DEFAULT_CONFIG_PATH)
    resolved_name = resolve_model_name(checkpoint_path)
    resolved_size = img_size or config["img_size"]
    resolved_threshold = (
        threshold if threshold is not None else config.get("threshold", 0.5)
    )

    device = get_device()
    model = build_model(pretrained=False)
    model.load_state_dict(load_checkpoint_state(checkpoint_path, device))
    model.to(device)

    return ModelSession(
        model=model,
        device=device,
        img_size=resolved_size,
        checkpoint_path=checkpoint_path,
        model_name=resolved_name,
        default_threshold=resolved_threshold,
    )


def load_model(
    checkpoint: str | Path,
    *,
    img_size: int | None = None,
    threshold: float | None = None,
) -> ModelSession:
    global _SESSION
    _SESSION = create_model_session(
        checkpoint,
        img_size=img_size,
        threshold=threshold,
    )
    return _SESSION


def get_model_session() -> ModelSession:
    if _SESSION is None:
        raise RuntimeError("Модель не загружена. Вызовите load_model().")
    return _SESSION


def ensure_model(
    checkpoint: str | Path | None = None,
    *,
    img_size: int | None = None,
    threshold: float | None = None,
) -> ModelSession:
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    ckpt = Path(checkpoint) if checkpoint else find_default_checkpoint()
    _SESSION = create_model_session(
        ckpt,
        img_size=img_size,
        threshold=threshold,
    )
    return _SESSION


def save_model(model: nn.Module, path: str | Path) -> Path:
    return save_checkpoint(model, path)
