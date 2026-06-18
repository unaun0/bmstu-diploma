from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model.model import save_checkpoint
from src.training.settings import (
    CHECKPOINT_FILENAME_TEMPLATE,
    SCHEDULER_FACTOR,
    SCHEDULER_MODE,
    SCHEDULER_PATIENCE,
)
from src.training.utils import format_duration


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        bce_weight: float,
        dice_weight: float,
        lr: float,
        patience: int,
        threshold: float,
        num_epochs: int,
        checkpoint_path: str | None,
        logger: logging.Logger,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_path or "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.model = model.to(device)
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.threshold = threshold
        self.patience = patience

        if hasattr(torch, "GradScaler"):
            self.scaler = torch.GradScaler(enabled=device.type == "cuda")
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        self.criterion = SegmentationLoss(bce_weight=bce_weight, dice_weight=dice_weight)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=SCHEDULER_MODE,
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )

        self.best_iou = -1.0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def train(self) -> None:
        training_start = time.perf_counter()
        val_metrics = EpochMetrics(0.0, 0.0, 0.0)

        for epoch in range(self.num_epochs):
            epoch_start = time.perf_counter()
            train_metrics, val_metrics = self._run_epoch(epoch)
            epoch_time = time.perf_counter() - epoch_start

            self.scheduler.step(val_metrics.loss)
            self.logger.info(
                f"\nЭпоха {epoch:03d}/{self.num_epochs} | "
                f"LR={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Длительность={format_duration(epoch_time)}\n"
                f"Обучение  | Loss={train_metrics.loss:.4f} | "
                f"IoU={train_metrics.iou:.4f} | Dice={train_metrics.dice:.4f}\n"
                f"Валидация | Loss={val_metrics.loss:.4f} | "
                f"IoU={val_metrics.iou:.4f} | Dice={val_metrics.dice:.4f}"
            )

            if val_metrics.iou > self.best_iou:
                self.best_iou = val_metrics.iou
                save_checkpoint(
                    self.model,
                    self.checkpoint_dir
                    / CHECKPOINT_FILENAME_TEMPLATE.format(epoch=epoch, val_iou=val_metrics.iou),
                )
                self.logger.info(f"Сохранена лучшая модель (вал. IoU={val_metrics.iou:.4f})")

            if val_metrics.loss < self.best_val_loss:
                self.best_val_loss = val_metrics.loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Ранняя остановка после {epoch} эпох(и).")
                break

        save_checkpoint(
            self.model,
            self.checkpoint_dir
            / CHECKPOINT_FILENAME_TEMPLATE.format(epoch=epoch, val_iou=val_metrics.iou),
        )
        self.logger.info(
            f"Обучение завершено — длительность={format_duration(time.perf_counter() - training_start)}"
        )

    def _run_epoch(self, epoch: int) -> tuple[EpochMetrics, EpochMetrics]:
        train_metrics = self._run_loader_phase(
            self.train_loader, train=True, desc=f"Эпоха {epoch} [обуч.]"
        )
        val_metrics = self._run_loader_phase(
            self.val_loader, train=False, desc=f"Эпоха {epoch} [вал.]"
        )
        return train_metrics, val_metrics

    def _run_loader_phase(
        self,
        loader: DataLoader,
        *,
        train: bool,
        desc: str,
    ) -> EpochMetrics:
        self.model.train(train)
        running_loss = running_iou = running_dice = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc, leave=True):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                if train:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=self.device.type == "cuda",
                    ):
                        logits = self.model(images)
                        loss = self.criterion(logits, masks)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            iou = batch_iou(logits, masks, threshold=self.threshold)
            dice = batch_dice(logits, masks, threshold=self.threshold)
            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            num_batches += 1

        n = max(num_batches, 1)
        return EpochMetrics(running_loss / n, running_iou / n, running_dice / n)



class SegmentationLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


@torch.no_grad()
def batch_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets = targets > 0.5
    intersection = (preds & targets).sum(dim=(1, 2, 3)).float()
    union = (preds | targets).sum(dim=(1, 2, 3)).float()
    return ((intersection + eps) / (union + eps)).mean().item()


@torch.no_grad()
def batch_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets = targets > 0.5
    intersection = (preds & targets).sum(dim=(1, 2, 3)).float()
    dice = (2.0 * intersection + eps) / (
        preds.sum(dim=(1, 2, 3)).float()
        + targets.sum(dim=(1, 2, 3)).float()
        + eps
    )
    return dice.mean().item()


@dataclass
class EpochMetrics:
    loss: float
    iou: float
    dice: float