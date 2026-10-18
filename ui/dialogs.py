
"""
Easy Quadrat v2.0.0
Standard application message dialogs.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from PySide6.QtWidgets import QMessageBox, QWidget

from ui.window_utils import configure_window


def show_message(
    parent: QWidget,
    title: str,
    message: str,
    icon: QMessageBox.Icon,
) -> None:
    """Display a branded, centered application message."""

    dialog = QMessageBox(
        icon,
        title,
        message,
        QMessageBox.StandardButton.Ok,
        parent,
    )

    configure_window(
        dialog,
        parent=parent,
        keep_on_top=True,
    )

    dialog.exec()


def show_information(
    parent: QWidget,
    title: str,
    message: str,
) -> None:
    show_message(
        parent,
        title,
        message,
        QMessageBox.Icon.Information,
    )


def show_warning(
    parent: QWidget,
    title: str,
    message: str,
) -> None:
    show_message(
        parent,
        title,
        message,
        QMessageBox.Icon.Warning,
    )


def show_error(
    parent: QWidget,
    title: str,
    message: str,
) -> None:
    show_message(
        parent,
        title,
        message,
        QMessageBox.Icon.Critical,
    )