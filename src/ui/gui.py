from __future__ import annotations
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor, QFontMetrics, QImage, QPixmap, QWheelEvent
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.ui import service
from src.ui.settings import (
    COLOR_META_PENDING,
    COLOR_META_PROCESSED,
    COLOR_PENDING,
    COLOR_PROCESSED,
    DEFAULT_CONFIDENCE,
    DEFAULT_MASK_ALPHA,
    DETAIL_DEFAULT_SIZE,
    DETAIL_MIN_SIZE,
    EMPTY_LIST_LABEL_STYLE,
    IMAGE_FILE_FILTER,
    IMAGE_VIEWER_STYLESHEET,
    LIST_ITEM_MIN_HEIGHT,
    MAIN_DEFAULT_SIZE,
    MAIN_MIN_SIZE,
    MAIN_WINDOW_TITLE,
    SLIDER_STYLESHEET,
    ZIP_DEFAULT_NAME,
    ZOOM_MAX,
    ZOOM_MIN,
    ZOOM_STEP,
)
from src.ui.utils import (
    apply_mask_overlay,
    encode_mask_png,
    estimate_pipeline_memory,
    estimate_rgb_memory,
    format_bytes,
    format_duration,
    format_percent,
    load_image_from_path,
    mask_to_uint8,
    read_image_size,
    resolution_text,
    save_mask_png,
    validate_file_size,
    validate_image_size,
)


@dataclass
class PipelineItem:
    path: Path
    image: np.ndarray
    mask: np.ndarray
    logits: np.ndarray
    elapsed_sec: float
    has_mask: bool = True


@dataclass
class FileEntry:
    path: Path
    width: int
    height: int
    pipeline: PipelineItem | None = None


def _make_slider(min_v: int, max_v: int, value: int) -> QSlider:
    slider = QSlider(Qt.Horizontal)
    slider.setObjectName("detailSlider")
    slider.setFocusPolicy(Qt.NoFocus)
    slider.setAutoFillBackground(False)
    slider.setMinimumHeight(30)
    slider.setStyleSheet(SLIDER_STYLESHEET)
    slider.setRange(min_v, max_v)
    slider.setValue(value)
    return slider


class ImageViewer(QWidget):
    zoom_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._zoom = 1.0
        self._pixmap: QPixmap | None = None
        self._panning = False
        self._pan_last_global: QPoint | None = None

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet(IMAGE_VIEWER_STYLESHEET)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background-color: #0a0c10;")
        self._scroll.setWidget(self._label)

        viewport = self._scroll.viewport()
        viewport.setContextMenuPolicy(Qt.NoContextMenu)
        viewport.setMouseTracking(True)
        viewport.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll)
        self.setFocusPolicy(Qt.WheelFocus)

    def zoom_percent(self) -> int:
        return int(round(self._zoom * 100))

    def zoom_in(self) -> None:
        self._zoom_by_factor(ZOOM_STEP, self._cursor_anchor())

    def zoom_out(self) -> None:
        self._zoom_by_factor(1.0 / ZOOM_STEP, self._cursor_anchor())

    def reset_zoom(self) -> None:
        self._set_zoom(1.0, anchor=None)

    def set_image(self, rgb: np.ndarray | None, *, keep_zoom: bool = False) -> None:
        if rgb is None:
            self._pixmap = None
            self._label.clear()
            return
        h, w, ch = rgb.shape
        if ch != 3:
            raise ValueError("Expected RGB image")
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        if keep_zoom:
            self._update_display()
        else:
            self._set_zoom(1.0, anchor=None)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is not self._scroll.viewport():
            return super().eventFilter(obj, event)
        if event.type() == QEvent.Wheel:
            self._handle_wheel(event)
            return True
        if event.type() == QEvent.MouseButtonPress and self._pan_start(event):
            return True
        if event.type() == QEvent.MouseMove and self._pan_move(event):
            return True
        if event.type() == QEvent.MouseButtonRelease and self._pan_end(event):
            return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        self._handle_wheel(event)
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_display()

    def _cursor_anchor(self) -> QPointF:
        viewport = self._scroll.viewport()
        local = viewport.mapFromGlobal(QCursor.pos())
        if viewport.rect().contains(local):
            return QPointF(local)
        return QPointF(viewport.rect().center())

    def _set_zoom(self, value: float, *, anchor: QPointF | None) -> None:
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, value))
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        if anchor is None:
            self._zoom = new_zoom
            self._update_display()
            self.zoom_changed.emit()
            return
        self._zoom_by_factor(new_zoom / self._zoom, anchor)

    def _zoom_by_factor(self, factor: float, anchor: QPointF) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-6:
            return

        viewport = self._scroll.viewport()
        anchor_pt = QPoint(anchor.toPoint())
        label_pt = self._label.mapFrom(viewport, anchor_pt)
        old_w, old_h = self._label.width(), self._label.height()

        self._zoom = new_zoom
        self._update_display()

        new_w, new_h = self._label.width(), self._label.height()
        if old_w < 1 or old_h < 1:
            self.zoom_changed.emit()
            return

        new_label_pt = QPoint(
            int(label_pt.x() * new_w / old_w),
            int(label_pt.y() * new_h / old_h),
        )
        label_origin = self._label.mapTo(viewport, QPoint(0, 0))
        desired = QPoint(anchor_pt.x() - new_label_pt.x(), anchor_pt.y() - new_label_pt.y())
        h_bar = self._scroll.horizontalScrollBar()
        v_bar = self._scroll.verticalScrollBar()
        h_bar.setValue(h_bar.value() + label_origin.x() - desired.x())
        v_bar.setValue(v_bar.value() + label_origin.y() - desired.y())
        self.zoom_changed.emit()

    def _handle_wheel(self, event: QWheelEvent) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        anchor = QPointF(self._scroll.viewport().mapFromGlobal(event.globalPos()))
        angle = event.angleDelta().y()
        if angle != 0:
            self._zoom_by_factor(ZOOM_STEP ** (angle / 120.0), anchor)
            return
        pixel = event.pixelDelta().y()
        if pixel != 0:
            self._zoom_by_factor(ZOOM_STEP ** (-pixel / 120.0), anchor)

    def _pan_start(self, event) -> bool:
        if (
            event.button() == Qt.RightButton
            and self._pixmap is not None
            and not self._pixmap.isNull()
        ):
            self._panning = True
            self._pan_last_global = event.globalPos()
            self._scroll.viewport().setCursor(Qt.ClosedHandCursor)
            self._scroll.viewport().grabMouse()
            return True
        return False

    def _pan_move(self, event) -> bool:
        if not self._panning or self._pan_last_global is None:
            return False
        delta = event.globalPos() - self._pan_last_global
        self._pan_last_global = event.globalPos()
        h_bar = self._scroll.horizontalScrollBar()
        v_bar = self._scroll.verticalScrollBar()
        h_bar.setValue(h_bar.value() - delta.x())
        v_bar.setValue(v_bar.value() - delta.y())
        return True

    def _pan_end(self, event) -> bool:
        if event.button() != Qt.RightButton or not self._panning:
            return False
        self._panning = False
        self._pan_last_global = None
        viewport = self._scroll.viewport()
        if viewport.mouseGrabber() is viewport:
            viewport.releaseMouse()
        viewport.unsetCursor()
        return True

    def _fit_scale(self) -> float:
        if self._pixmap is None or self._pixmap.isNull():
            return 1.0
        viewport = self._scroll.viewport().size()
        if viewport.width() < 10 or viewport.height() < 10:
            return 1.0
        sx = viewport.width() / self._pixmap.width()
        sy = viewport.height() / self._pixmap.height()
        return min(sx, sy)

    def _update_display(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scale = self._fit_scale() * self._zoom
        w = max(1, int(self._pixmap.width() * scale))
        h = max(1, int(self._pixmap.height() * scale))
        scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())


class FileListRow(QWidget):
    def __init__(self, name: str, meta: str, processed: bool, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)

        prefix = "✓  " if processed else "○  "
        self._full_name = prefix + name
        self._name = QLabel(self._full_name)
        self._name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._name.setMinimumWidth(120)

        self._meta = QLabel(meta)
        self._meta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._meta.setWordWrap(False)

        name_color = COLOR_PROCESSED if processed else COLOR_PENDING
        meta_color = COLOR_META_PROCESSED if processed else COLOR_META_PENDING
        self._name.setStyleSheet(f"color: {name_color}; font-weight: 500;")
        self._meta.setStyleSheet(f"color: {meta_color}; font-size: 12px;")

        layout.addWidget(self._name, stretch=3)
        layout.addWidget(self._meta, stretch=2)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        metrics = QFontMetrics(self._name.font())
        available = max(self._name.width(), 80)
        self._name.setText(metrics.elidedText(self._full_name, Qt.ElideMiddle, available))

    def sizeHint(self):  # noqa: N802
        hint = super().sizeHint()
        return QSize(hint.width(), max(hint.height(), 40))


class DetailWindow(QDialog):
    def __init__(
        self,
        file_path: Path,
        image: np.ndarray,
        parent: QWidget | None = None,
        initial_mask: np.ndarray | None = None,
        initial_logits: np.ndarray | None = None,
        on_reprocessed: Callable[[np.ndarray, np.ndarray, float], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._path = file_path
        self._image = image
        self._on_reprocessed = on_reprocessed
        self._confidence = DEFAULT_CONFIDENCE
        self._mask_alpha = DEFAULT_MASK_ALPHA
        self._show_mask = True
        self._logits = initial_logits.copy() if initial_logits is not None else None
        self._raw_mask = initial_mask.copy() if initial_mask is not None else None

        self.setWindowTitle(file_path.name)
        self.setMinimumSize(*DETAIL_MIN_SIZE)
        self.resize(*DETAIL_DEFAULT_SIZE)

        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(20, 20, 20, 20)

        path_label = QLabel(f"<b>Путь к файлу:</b> {file_path.resolve()}")
        path_label.setObjectName("subtitle")
        path_label.setTextFormat(Qt.RichText)
        path_label.setWordWrap(True)
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(path_label)

        self._viewer = ImageViewer()
        self._viewer.zoom_changed.connect(self._update_zoom_label)
        root.addWidget(self._viewer, stretch=1)

        controls = QFrame()
        controls.setObjectName("card")
        ctrl = QVBoxLayout(controls)
        ctrl.setSpacing(14)
        ctrl.setContentsMargins(16, 16, 16, 16)

        self._mask_check = QCheckBox("Показать маску")
        self._mask_check.setChecked(True)
        self._mask_check.toggled.connect(self._on_mask_toggle)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setObjectName("subtitle")
        self._zoom_label.setMinimumWidth(40)
        self._zoom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        zoom_reset = QPushButton("По размеру окна")
        zoom_reset.setObjectName("zoomResetBtn")
        zoom_reset.clicked.connect(self._on_zoom_reset)
        zoom_out = QPushButton("−")
        zoom_out.setObjectName("zoomBtn")
        zoom_out.setFixedSize(28, 26)
        zoom_out.clicked.connect(self._on_zoom_out)
        zoom_in = QPushButton("+")
        zoom_in.setObjectName("zoomBtn")
        zoom_in.setFixedSize(28, 26)
        zoom_in.clicked.connect(self._on_zoom_in)

        mask_zoom = QHBoxLayout()
        mask_zoom.addWidget(self._mask_check)
        mask_zoom.addStretch()
        mask_zoom.addWidget(self._zoom_label)
        mask_zoom.addWidget(zoom_reset)
        mask_zoom.addWidget(zoom_out)
        mask_zoom.addWidget(zoom_in)

        _, self._conf_slider = self._add_slider(
            ctrl, "Порог уверенности", 50, self._on_confidence_changed
        )
        self._alpha_title, self._alpha_value_label, self._alpha_slider = self._add_slider(
            ctrl, "Непрозрачность маски", 45, self._on_alpha_changed, with_title=True
        )

        actions = QHBoxLayout()
        reprocess_btn = QPushButton("Повторно обработать")
        reprocess_btn.clicked.connect(self._reprocess_image)
        save_btn = QPushButton("Сохранить")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._save_mask)
        actions.addWidget(reprocess_btn)
        actions.addWidget(save_btn)
        actions.addStretch()

        ctrl.addLayout(mask_zoom)
        ctrl.addLayout(actions)

        root.addWidget(controls)

        if self._logits is None:
            self._ensure_logits()
        if self._raw_mask is None:
            self._apply_threshold()
        self._update_alpha_controls()
        self._refresh_view()

    def _add_slider(
        self,
        parent_layout: QVBoxLayout,
        title: str,
        initial: int,
        on_change: Callable[[int], None],
        *,
        with_title: bool = False,
    ):
        row = QVBoxLayout()
        header = QHBoxLayout()
        title_label = QLabel(title)
        value_label = QLabel(format_percent(initial))
        header.addWidget(title_label)
        header.addStretch()
        header.addWidget(value_label)
        slider = _make_slider(0, 100, initial)

        def _changed(v: int) -> None:
            value_label.setText(format_percent(v))
            on_change(v)

        slider.valueChanged.connect(_changed)
        row.addLayout(header)
        row.setContentsMargins(0, 0, 0, 4)
        row.addWidget(slider)
        parent_layout.addLayout(row)

        if with_title:
            return title_label, value_label, slider
        return value_label, slider

    def _update_alpha_controls(self) -> None:
        enabled = self._mask_check.isChecked()
        self._alpha_title.setEnabled(enabled)
        self._alpha_value_label.setEnabled(enabled)

    def _on_mask_toggle(self, checked: bool) -> None:
        self._show_mask = checked
        self._update_alpha_controls()
        self._refresh_view()

    def _on_confidence_changed(self, v: int) -> None:
        self._confidence = v / 100.0
        self._apply_threshold()
        self._refresh_view()

    def _on_alpha_changed(self, v: int) -> None:
        if not self._mask_check.isChecked():
            return
        self._mask_alpha = v / 100.0
        self._refresh_view()

    def _ensure_logits(self) -> None:
        if self._logits is None:
            self._logits = service.get_logits(self._image)

    def _apply_threshold(self) -> None:
        self._ensure_logits()
        self._raw_mask = service.mask_from_logits(self._logits, self._confidence)

    def _refresh_view(self) -> None:
        if self._raw_mask is None:
            self._viewer.set_image(self._image, keep_zoom=True)
            return
        display = (
            apply_mask_overlay(self._image, self._raw_mask, self._mask_alpha)
            if self._show_mask
            else self._image
        )
        self._viewer.set_image(display, keep_zoom=True)
        self._update_zoom_label()

    def _update_zoom_label(self) -> None:
        self._zoom_label.setText(f"{self._viewer.zoom_percent()}%")

    def _on_zoom_in(self) -> None:
        self._viewer.zoom_in()
        self._update_zoom_label()

    def _on_zoom_out(self) -> None:
        self._viewer.zoom_out()
        self._update_zoom_label()

    def _on_zoom_reset(self) -> None:
        self._viewer.reset_zoom()
        self._update_zoom_label()

    def _reprocess_image(self) -> None:
        t0 = time.perf_counter()
        self._logits = service.get_logits(self._image)
        self._apply_threshold()
        elapsed = time.perf_counter() - t0
        self._refresh_view()
        if self._on_reprocessed is not None:
            self._on_reprocessed(self._raw_mask, self._logits, elapsed)
        QMessageBox.information(
            self,
            "Готово",
            f"Изображение обработано заново.\nВремя: {format_duration(elapsed)}",
        )

    def _save_mask(self) -> None:
        if self._raw_mask is None:
            QMessageBox.warning(self, "Ошибка", "Маска ещё не вычислена.")
            return
        default = self._path.with_name(f"{self._path.stem}_mask.png")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить маску",
            str(default),
            "PNG (*.png);;Все файлы (*)",
        )
        if not path:
            return
        out = Path(path)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            save_mask_png(out, self._raw_mask)
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{exc}")
            return
        QMessageBox.information(self, "Готово", f"Маска сохранена:\n{out}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(MAIN_WINDOW_TITLE)
        self.setMinimumSize(*MAIN_MIN_SIZE)
        self.resize(*MAIN_DEFAULT_SIZE)

        self._entries: list[FileEntry] = []
        self._default_confidence = DEFAULT_CONFIDENCE

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(28, 28, 28, 28)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(12)
        card_layout.setContentsMargins(16, 16, 16, 16)

        hint = QLabel(
            "Чтобы открыть окно детального просмотра, дважды нажмите левой кнопкой мыши "
            "по обработанному файлу в списке."
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        hint.setContentsMargins(0, 0, 0, 4)
        card_layout.addWidget(hint)

        self._list = QListWidget()
        self._list.setSpacing(4)
        self._list.itemDoubleClicked.connect(self._open_detail)
        self._empty_label = QLabel("Нет файлов", self._list.viewport())
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(EMPTY_LIST_LABEL_STYLE)
        self._empty_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._empty_label.hide()
        self._list.viewport().installEventFilter(self)
        card_layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Добавить файлы…")
        self._add_btn.clicked.connect(self._add_files)
        self._clear_btn = QPushButton("Очистить")
        self._clear_btn.setObjectName("danger")
        self._clear_btn.clicked.connect(self._clear_list)
        self._zip_btn = QPushButton("Сохранить как архив (.zip)")
        self._zip_btn.clicked.connect(self._save_zip)
        self._zip_btn.setVisible(False)
        self._process_btn = QPushButton("Обработать")
        self._process_btn.setObjectName("primary")
        self._process_btn.clicked.connect(self._process_all)
        self._process_btn.setEnabled(False)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._zip_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._process_btn)
        card_layout.addLayout(btn_row)

        layout.addWidget(card, stretch=1)
        self._refresh_list()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._sync_list_sizes()
        self._position_empty_label()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self._list.viewport() and event.type() == QEvent.Resize:
            self._position_empty_label()
        return super().eventFilter(obj, event)

    def _position_empty_label(self) -> None:
        if not self._empty_label.isVisible():
            return
        vp = self._list.viewport()
        self._empty_label.setGeometry(0, 0, vp.width(), vp.height())
        self._empty_label.raise_()

    def _sync_list_sizes(self) -> None:
        width = self._list.viewport().width()
        if width <= 0:
            return
        for i in range(self._list.count()):
            item = self._list.item(i)
            widget = self._list.itemWidget(item)
            if item is None or widget is None:
                continue
            widget.setFixedWidth(width)
            widget.adjustSize()
            height = max(widget.sizeHint().height(), LIST_ITEM_MIN_HEIGHT)
            item.setSizeHint(QSize(width, height))

    def _pending(self) -> list[FileEntry]:
        return [e for e in self._entries if e.pipeline is None]

    def _processed(self) -> list[FileEntry]:
        return [e for e in self._entries if e.pipeline is not None]

    @staticmethod
    def _meta(entry: FileEntry) -> str:
        res = resolution_text(entry.width, entry.height)
        if entry.pipeline is None:
            mem = estimate_rgb_memory(entry.width, entry.height)
            return f"{res} · {format_bytes(mem)} · Ожидает"
        p = entry.pipeline
        mem = estimate_pipeline_memory(p.image, p.mask)
        return (
            f"{res} · "
            f"{format_bytes(mem)} · {format_duration(p.elapsed_sec)} · Обработано"
        )

    def _refresh_list(self) -> None:
        self._list.clear()
        if not self._entries:
            self._empty_label.show()
            QTimer.singleShot(0, self._position_empty_label)
            self._process_btn.setText("Обработать")
            self._process_btn.setEnabled(False)
            self._zip_btn.setVisible(False)
            return

        self._empty_label.hide()
        for entry in self._entries:
            processed = entry.pipeline is not None
            item = QListWidgetItem()
            item.setData(Qt.UserRole, entry)
            row = FileListRow(entry.path.name, self._meta(entry), processed)
            self._list.addItem(item)
            self._list.setItemWidget(item, row)

        QTimer.singleShot(0, self._sync_list_sizes)

        pending = self._pending()
        processed = self._processed()
        self._process_btn.setEnabled(len(pending) > 0)
        if pending:
            self._process_btn.setText(
                f"Обработать ({len(pending)})" if processed else "Обработать"
            )
        else:
            self._process_btn.setText("Обработать")
        self._zip_btn.setVisible(len(processed) > 0)

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображения", "", IMAGE_FILE_FILTER
        )
        if not paths:
            return

        known = {e.path for e in self._entries}
        errors: list[str] = []

        for raw in paths:
            path = Path(raw)
            if path in known:
                continue
            try:
                err = validate_file_size(path)
                if err:
                    errors.append(f"{path.name}: {err}")
                    continue
                size = read_image_size(path)
                if size is None:
                    errors.append(f"{path.name}: не удалось открыть файл")
                    continue
                w, h = size
                err = validate_image_size(w, h)
                if err:
                    errors.append(f"{path.name}: {err}")
                    continue
                side = min(w, h)
                self._entries.append(FileEntry(path=path, width=side, height=side))
            except OSError:
                errors.append(f"{path.name}: не удалось открыть файл")

        self._refresh_list()
        if errors:
            QMessageBox.warning(
                self,
                "Часть файлов пропущена",
                "\n".join(errors[:12]) + ("\n…" if len(errors) > 12 else ""),
            )

    def _clear_list(self) -> None:
        self._entries.clear()
        self._refresh_list()

    def _process_all(self) -> None:
        pending = self._pending()
        if not pending:
            return

        errors: list[str] = []
        processed_count = 0

        for entry in pending:
            path = entry.path
            try:
                rgb = load_image_from_path(path)
                h, w = rgb.shape[:2]
                entry.width = w
                entry.height = h
                t0 = time.perf_counter()
                logits = service.get_logits(rgb)
                mask = service.mask_from_logits(logits, self._default_confidence)
                elapsed = time.perf_counter() - t0
                entry.pipeline = PipelineItem(
                    path=path,
                    image=rgb,
                    mask=mask,
                    logits=logits,
                    elapsed_sec=elapsed,
                    has_mask=mask is not None and np.any(mask > 0),
                )
                processed_count += 1
            except OSError:
                errors.append(f"{path.name}: ошибка чтения")

        self._refresh_list()
        if errors:
            QMessageBox.warning(self, "Ошибка обработки", "\n".join(errors))
        if processed_count:
            QMessageBox.information(
                self, "Готово", f"Изображений обработано: {processed_count}."
            )

    def _save_zip(self) -> None:
        processed = self._processed()
        if not processed:
            QMessageBox.warning(self, "Архив", "Нет обработанных изображений.")
            return
        zip_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить архив", ZIP_DEFAULT_NAME, "ZIP (*.zip)"
        )
        if not zip_path:
            return
        out = Path(zip_path)
        if out.suffix.lower() != ".zip":
            out = out.with_suffix(".zip")

        try:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for entry in processed:
                    p = entry.pipeline
                    assert p is not None
                    zf.writestr(f"{entry.path.stem}_mask.png", encode_mask_png(p.mask))
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось создать архив:\n{exc}")
            return

        QMessageBox.information(
            self, "Готово", f"Архив сохранён:\n{out}\n\nФайлов: {len(processed)}"
        )

    def _open_detail(self, item: QListWidgetItem) -> None:
        entry = item.data(Qt.UserRole)
        if not isinstance(entry, FileEntry):
            return
        if entry.pipeline is None:
            QMessageBox.information(
                self,
                "Сначала обработайте",
                "Нажмите «Обработать», чтобы получить маски сегментации.",
            )
            return

        p = entry.pipeline

        def _on_reprocessed(mask: np.ndarray, logits: np.ndarray, elapsed: float) -> None:
            p.mask = mask
            p.logits = logits
            p.elapsed_sec = elapsed
            p.has_mask = mask is not None and np.any(mask > 0)
            self._refresh_list()

        DetailWindow(
            entry.path,
            p.image,
            self,
            initial_mask=p.mask,
            initial_logits=p.logits,
            on_reprocessed=_on_reprocessed,
        ).exec_()
