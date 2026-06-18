from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.utils import build_train_transforms, build_val_transforms, get_device, set_seed
from src.model import build_model
from src.training.dataset import FabricDefectDataset
from src.training.settings import PREFETCH_FACTOR, TRAIN_DIR_NAME, VAL_DIR_NAME
from src.training.trainer import Trainer
from src.training.utils import prepare_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение модели сегментации дефектов ткани")
    parser.add_argument(
        "--config", 
        type=Path, 
        default=None, 
        help="Путь к конфигурации"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        help="Количество воркеров"
    )
    parser.add_argument(
        "--data-path", 
        dest="data_path", 
        type=str, 
        help="Путь к данным"
    )
    parser.add_argument(
        "--img-size", 
        dest="img_size", 
        type=int, 
        help="Размер изображений"
    )
    parser.add_argument(
        "--batch-size", 
        dest="batch_size", 
        type=int, 
        help="Размер пакета"
    )
    parser.add_argument(
        "--aug-prob", 
        dest="aug_prob", 
        type=float, 
        help="Вероятность применения аугментаций"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        help="Количество эпох без улучшения для ранней остановки"
    )
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        type=str,
        help="Путь для хранения весов модели",
    )
    parser.add_argument(
        "--log-dir", 
        dest="log_dir", 
        type=str, 
        help="Путь для сохранения лога"
    )

    config, logger = prepare_training(parser.parse_args())
    
    run_training(config, logger)


def run_training(config: dict, logger: logging.Logger) -> None:
    set_seed(config["seed"])
    device = get_device()
    logger.info(f"Используемое устройство: {device}")

    data_root = Path(config["data_path"])
    train_ds = FabricDefectDataset(
        data_root / TRAIN_DIR_NAME,
        build_train_transforms(config["img_size"], config["aug_prob"]),
    )
    val_ds = FabricDefectDataset(
        data_root / VAL_DIR_NAME,
        build_val_transforms(config["img_size"]),
    )
    logger.info(f"Данные | Обучение — {len(train_ds)} | Валидация — {len(val_ds)}")

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": device.type == "cuda",
        "persistent_workers": config["num_workers"] > 0,
        "prefetch_factor": PREFETCH_FACTOR,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=True, **loader_kwargs)

    model = build_model()
    Trainer(
        model,
        device,
        train_loader,
        val_loader,
        bce_weight=config["bce_weight"],
        dice_weight=config["dice_weight"],
        lr=config["learning_rate"],
        patience=config["patience"],
        threshold=config["threshold"],
        num_epochs=config["num_epochs"],
        checkpoint_path=config["checkpoint_path"],
        logger=logger,
    ).train()


if __name__ == "__main__":
    main()