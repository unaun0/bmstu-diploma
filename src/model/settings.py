from __future__ import annotations
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = _MODEL_DIR / "weights"

IN_CHANNELS = 3
OUT_CHANNELS = 1
ACTIVATION = None
ENCODER_WEIGHTS = "imagenet"

MODEL_NAME = "deeplabv3plus_mobilenetv3_ca_dsf"
DEFAULT_MODEL = MODEL_NAME
ENCODER = "tu-tf_mobilenetv3_large_100"

CHECKPOINT_STEMS: dict[str, str] = {
    "mn3_ca_dsf": MODEL_NAME,
}

CHECKPOINT_PREFIXES: dict[str, str] = {
    "dl_mn3_ca_dsf": MODEL_NAME,
}

CHECKPOINT_PREFIX_ORDER = tuple(
    sorted(CHECKPOINT_PREFIXES, key=len, reverse=True)
)
