from __future__ import annotations
import logging
import sys
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from src.common import setup_logger
from src.common.settings import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH
from src.common.utils import load_json, merge_config, save_json, validate_config, validate_data


def prepare_training(args) -> tuple[dict, logging.Logger]:
    if not DEFAULT_CONFIG_PATH.is_file():
        sys.exit(f"fatalError: Базовый конфиг не найден: {DEFAULT_CONFIG_PATH}")

    config = load_json(DEFAULT_CONFIG_PATH)
    if args.config:
        config = merge_config(config, load_json(args.config))

    if args.workers is not None:
        config["num_workers"] = args.workers
    if args.data_path is not None:
        config["data_path"] = args.data_path
    if args.img_size is not None:
        config["img_size"] = args.img_size
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.aug_prob is not None:
        config["aug_prob"] = args.aug_prob
    if args.patience is not None:
        config["patience"] = args.patience
    if args.checkpoint_dir is not None:
        config["checkpoint_path"] = args.checkpoint_dir
    if args.log_dir is not None:
        config["log_path"] = args.log_dir

    validate_config(config)
    validate_data(config["data_path"])

    Path(config["checkpoint_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["log_path"]).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(Path(config["log_path"]), f"log_{timestamp}.log")
    
    return config, logger


def load_mask_binary(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 127).astype(np.uint8)


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}ч. {minutes}мин. {secs}сек."
    if minutes:
        return f"{minutes}мин. {secs}сек."
    return f"{secs}s"


def miou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Размер pred {pred_mask.shape} не совпадает с gt {gt_mask.shape}"
        )

    pred = (pred_mask > 0.5).astype(bool)
    gt = (gt_mask > 0.5).astype(bool)

    def _class_iou(a: np.ndarray, b: np.ndarray) -> float:
        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 1.0
        return float((intersection + eps) / (union + eps))

    return (_class_iou(pred, gt) + _class_iou(~pred, ~gt)) / 2.0


def iou_defect(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Размер pred {pred_mask.shape} не совпадает с gt {gt_mask.shape}"
        )

    pred = (pred_mask > 0.5).astype(bool)
    gt = (gt_mask > 0.5).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def _pixel_counts(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[int, int, int, int]:
    pred = (pred_mask > 0.5).astype(bool)
    gt = (gt_mask > 0.5).astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    return tp, fn, tn, fp


def pa_defect(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    tp, fn, _, _ = _pixel_counts(pred_mask, gt_mask)
    return float((tp + eps) / (tp + fn + eps))


def pa_background(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    _, _, tn, fp = _pixel_counts(pred_mask, gt_mask)
    return float((tn + eps) / (tn + fp + eps))


def mpa(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    return (pa_defect(pred_mask, gt_mask, eps=eps) + pa_background(pred_mask, gt_mask, eps=eps)) / 2.0


def find_best_threshold_for_gt(
    logits: np.ndarray,
    gt_mask: np.ndarray,
    *,
    t_min: float = 0.05,
    t_max: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    from src.detection.utils import logits_to_mask

    best_threshold = t_min
    best_score = -1.0
    for threshold in np.arange(t_min, t_max + step / 2, step):
        pred = logits_to_mask(logits, float(threshold))
        score = miou(pred, gt_mask)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score
