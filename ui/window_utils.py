
"""
Easy Quadrat v2.0.0
Shared window positioning and branding utilities.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QWidget

from app_config import APP_ICON_PATH, APP_LOGO_PATH


def apply_application_icon(window: QWidget) -> None:
    """Apply the application icon with a PNG fallback."""

    for icon_path in (APP_ICON_PATH, APP_LOGO_PATH):
        if not icon_path.is_file():
            continue

        icon = QIcon(str(icon_path))

        if not icon.isNull():
            window.setWindowIcon(icon)
            return


def center_on_screen(
    window: QWidget,
    parent: QWidget | None = None,
) -> None:
    """
    Center a window on the available screen area.

    Child windows use the screen containing their parent.
    """

    screen = None

    if parent is not None:
        screen = parent.screen()

    if screen is None:
        screen = window.screen()

    if screen is None:
        screen = QApplication.primaryScreen()

    if screen is None:
        return

    available = screen.availableGeometry()

    # Prevent the window from being larger than the available area.
    new_width = min(window.width(), available.width())
    new_height = min(window.height(), available.height())

    if (
        new_width != window.width()
        or new_height != window.height()
    ):
        window.resize(new_width, new_height)

    frame = window.frameGeometry()
    frame.moveCenter(available.center())
    window.move(frame.topLeft())


def configure_window(
    window: QWidget,
    parent: QWidget | None = None,
    keep_on_top: bool = True,
) -> None:
    """
    Set shared window branding and positioning behavior.

    Call this before displaying the window.
    """

    apply_application_icon(window)

    window.setWindowFlag(
        Qt.WindowType.WindowStaysOnTopHint,
        keep_on_top,
    )

    # The geometry is finalized after the window enters the event loop.
    QTimer.singleShot(
        0,
        lambda: center_on_screen(window, parent),
    )