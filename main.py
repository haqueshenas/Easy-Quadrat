"""
Easy Quadrat v2.0.0
Application entry point and startup splash screen.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from app_config import (
    APP_LOGO_PATH,
    APP_NAME,
    APP_VERSION,
)
from ui.theme import STYLESHEET


# ---------------------------------------------------------------------
# Application resources
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
APP_ICON_PATH = PROJECT_ROOT / "assets" / "EQ.ico"


# ---------------------------------------------------------------------
# Startup splash screen
# ---------------------------------------------------------------------

def create_startup_splash(
    app: QApplication,
) -> QSplashScreen | None:
    """
    Create and display the startup splash using the official app logo.

    The splash follows the size and display constraints used by
    Canopy GSM for Python: up to 420 x 280 pixels, constrained to
    40% of the available screen dimensions, with the logo's aspect
    ratio preserved.
    """
    if not APP_LOGO_PATH.is_file():
        return None

    pixmap = QPixmap(str(APP_LOGO_PATH))

    if pixmap.isNull():
        return None

    screen = app.primaryScreen()

    if screen is not None:
        available = screen.availableGeometry()

        max_width = min(
            420,
            int(available.width() * 0.40),
        )
        max_height = min(
            280,
            int(available.height() * 0.40),
        )

        if max_width > 0 and max_height > 0:
            pixmap = pixmap.scaled(
                max_width,
                max_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

    splash = QSplashScreen(pixmap)

    splash.setWindowFlags(
        Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.FramelessWindowHint
    )

    splash.show()
    app.processEvents()

    return splash


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

def main() -> int:
    """Initialize and execute the Easy Quadrat application."""
    app = QApplication(sys.argv)

    # Preserve the original application metadata.
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("Easy-Phenotyping Lab")

    # Preserve the original Qt appearance.
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLESHEET)

    # Use the official Windows application icon.
    if APP_ICON_PATH.is_file():
        icon = QIcon(str(APP_ICON_PATH))

        if not icon.isNull():
            app.setWindowIcon(icon)

    # Show the logo before importing the main window and its methods.
    splash = create_startup_splash(app)

    # Delay importing the main GUI until the splash is visible.
    from ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()

    # Ensure the main window is presented before closing the splash.
    app.processEvents()

    if splash is not None:
        splash.finish(window)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())