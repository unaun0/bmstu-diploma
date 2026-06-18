from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PyQt5.QtWidgets import QApplication
from src.ui.gui import MainWindow
from src.ui.settings import APP_STYLESHEET, UI_CHECKPOINT_PATH


def main() -> int:
    if not UI_CHECKPOINT_PATH.is_file():
        print(
            f"Чекпоинт для UI не найден: {UI_CHECKPOINT_PATH}\n"
            f"Укажите UI_CHECKPOINT_FILENAME в src/ui/settings.py",
            file=sys.stderr,
        )
        return 1

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
