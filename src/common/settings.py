from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "configs"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "default.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).parent.parent / "model" / "weights"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

EXPECTED_CONFIG = [
    ("num_workers", int, [("positive",)]),
    ("img_size", int, [("positive",)]),
    ("batch_size", int, [("positive",)]),
    ("aug_prob", float, [("range", 0.0, 1.0)]),
    ("threshold", float, [("range", 0.0, 1.0)]),
    ("num_epochs", int, [("positive",)]),
    ("learning_rate", float, [("positive",)]),
    ("patience", int, [("positive",)]),
    ("bce_weight", float, [("range", 0.0, 1.0)]),
    ("dice_weight", float, [("range", 0.0, 1.0)]),
    ("seed", int, [("positive",)]),
    ("data_path", str, []),
    ("checkpoint_path", str, []),
    ("log_path", str, []),
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
IMAGE_SIDE = 512

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
