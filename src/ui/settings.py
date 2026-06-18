from __future__ import annotations
from src.model.settings import WEIGHTS_DIR

UI_CHECKPOINT_FILENAME = "mn3_ca_dsf.pt"
UI_CHECKPOINT_PATH = WEIGHTS_DIR / UI_CHECKPOINT_FILENAME

# Окна
MAIN_WINDOW_TITLE = "Выявление и локализация дефектов на изображениях текстильного полотна"
MAIN_MIN_SIZE = (780, 520)
MAIN_DEFAULT_SIZE = (920, 620)
DETAIL_MIN_SIZE = (900, 700)
DETAIL_DEFAULT_SIZE = (1100, 800)

# Изображения
MIN_IMAGE_SIDE = 256
MAX_IMAGE_SIDE = 1024
MAX_IMAGE_FILE_BYTES = 2 * 1024 * 1024  # 2 МБ
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MASK_ALPHA = 0.45
OVERLAY_COLOR = (0, 220, 180)

IMAGE_FILE_FILTER = (
    "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;Все файлы (*)"
)
ZIP_DEFAULT_NAME = "segmentation_masks.zip"

# Масштаб просмотра
ZOOM_MIN = 0.25
ZOOM_MAX = 8.0
ZOOM_STEP = 1.15

# Список файлов
COLOR_PROCESSED = "#00c9a7"
COLOR_PENDING = "#9aa0a6"
COLOR_META_PROCESSED = "#6eb8a3"
COLOR_META_PENDING = "#6e7681"
LIST_ITEM_MIN_HEIGHT = 44

SLIDER_TRACK_ACTIVE = "#00c9a7"
SLIDER_TRACK_INACTIVE = "#2d3142"

APP_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #0f1117;
}

QWidget {
    color: #e8eaed;
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}

QLabel#title {
    font-size: 22px;
    font-weight: 600;
    color: #ffffff;
}

QLabel#subtitle {
    font-size: 12px;
    color: #9aa0a6;
}

QPushButton {
    background-color: #2d3142;
    border: 1px solid #3d4459;
    border-radius: 8px;
    padding: 10px 18px;
    color: #e8eaed;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #3d4459;
    border-color: #5c6370;
}

QPushButton:pressed {
    background-color: #252836;
}

QPushButton#primary {
    background-color: #00c9a7;
    border: none;
    color: #0f1117;
    font-weight: 600;
}

QPushButton#primary:hover {
    background-color: #00e0bb;
}

QPushButton#primary:disabled {
    background-color: #1a3d36;
    color: #5a6b66;
}

QPushButton#zoomBtn {
    padding: 2px 4px;
    min-width: 20px;
    font-size: 12px;
    font-weight: 600;
}

QPushButton#zoomResetBtn {
    padding: 4px 10px;
    font-size: 12px;
}

QPushButton#danger {
    background-color: #3a1c22;
    border: 1px solid #8b3a48;
    color: #ff9a9a;
}

QPushButton#danger:hover {
    background-color: #4d242c;
    border-color: #b84a5a;
    color: #ffb8b8;
}

QPushButton#danger:pressed {
    background-color: #2a1418;
}

QListWidget {
    background-color: #161922;
    border: 1px solid #2d3142;
    border-radius: 10px;
    padding: 6px;
    outline: none;
}

QListWidget::item {
    border-radius: 8px;
    padding: 0px;
    margin: 2px 0;
}

QListWidget::item:selected {
    background-color: #1e3a34;
}

QListWidget::item:hover {
    background-color: #1f2433;
}

QScrollBar:vertical {
    background: #161922;
    width: 10px;
    border-radius: 5px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #3d4459;
    border-radius: 5px;
    min-height: 24px;
}

QScrollBar:horizontal {
    background: #161922;
    height: 10px;
    border-radius: 5px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #3d4459;
    border-radius: 5px;
    min-width: 24px;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
    height: 0;
    background: transparent;
    border: none;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: transparent;
}

QSlider:focus {
    outline: none;
    border: none;
}

QLabel:disabled {
    color: #6e7681;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
}

QCheckBox::indicator:unchecked {
    border-radius: 4px;
    border: 1px solid #3d4459;
    background: #161922;
}

QCheckBox::indicator:checked {
    border: none;
    background: transparent;
    image: url(data:image/svg+xml;charset=UTF-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 18 18'%3E%3Crect x='0.5' y='0.5' width='17' height='17' rx='4' fill='%2300c9a7' stroke='%2300c9a7'/%3E%3Cpath d='M4.5 9.2 L7.5 12.2 L13.5 5.8' fill='none' stroke='%230f1117' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E);
}

QLineEdit {
    background-color: #161922;
    border: 1px solid #2d3142;
    border-radius: 8px;
    padding: 8px 12px;
    selection-background-color: #00c9a7;
}

QFrame#card {
    background-color: #161922;
    border: 1px solid #2d3142;
    border-radius: 12px;
}
"""

SLIDER_STYLESHEET = f"""
QSlider#detailSlider {{
    background: transparent;
    border: none;
    padding: 0px;
    min-height: 30px;
    max-height: 30px;
}}

QSlider#detailSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: transparent;
    border-radius: 3px;
    margin: 12px 0;
}}

QSlider#detailSlider::sub-page:horizontal {{
    border: none;
    height: 6px;
    background: {SLIDER_TRACK_ACTIVE};
    border-radius: 3px;
    margin: 12px 0;
}}

QSlider#detailSlider::add-page:horizontal {{
    border: none;
    height: 6px;
    background: {SLIDER_TRACK_INACTIVE};
    border-radius: 3px;
    margin: 12px 0;
}}

QSlider#detailSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -4px 0;
    background: {SLIDER_TRACK_ACTIVE};
    border: none;
    border-radius: 7px;
}}
"""

IMAGE_VIEWER_STYLESHEET = (
    "QScrollArea { background-color: #0a0c10; border-radius: 10px; "
    "border: 1px solid #2d3142; }"
    "QScrollArea QScrollBar:horizontal {"
    "  background: #161922; height: 10px; border-radius: 5px;"
    "}"
    "QScrollArea QScrollBar::handle:horizontal {"
    "  background: #3d4459; border-radius: 5px; min-width: 24px;"
    "}"
    "QScrollArea QScrollBar:vertical {"
    "  background: #161922; width: 10px; border-radius: 5px;"
    "}"
    "QScrollArea QScrollBar::handle:vertical {"
    "  background: #3d4459; border-radius: 5px; min-height: 24px;"
    "}"
)

EMPTY_LIST_LABEL_STYLE = (
    "color: #9aa0a6; font-size: 14px; background: transparent;"
)
