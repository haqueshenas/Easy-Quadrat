"""
Easy Quadrat v2.0.0
Main application window.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app_config import (
    APP_LOGO_PATH,
    APP_NAME,
    APP_SHORT_TAGLINE,
    APP_VERSION,
)
from auto_classic_i_method import AutoClassicIDialog
from auto_classic_ii_method import AutoClassicIIDialog
from auto_ai_method import AutoDeepLearningDialog
from fixed_frame_method import FixedFrameDialog
from manual_method import ManualDialog
from ui.about_dialog import AboutDialog
from ui.window_utils import configure_window


# ---------------------------------------------------------------------
# Method card
# ---------------------------------------------------------------------

class MethodCard(QFrame):
    """
    Interactive card representing an image-processing method.

    Available cards respond to mouse clicks and keyboard activation.
    Methods that have not yet been integrated are visibly marked as
    being in development and do not pretend to start processing.
    """

    clicked = Signal()

    def __init__(
        self,
        number: str,
        title: str,
        description: str,
        available: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._available = available
        self._pressed = False
        self._mouse_pressed = False
        self._keyboard_pressed = False

        self.setObjectName(
            "workflowCard" if available else "workflowCardUnavailable"
        )

        self.setMinimumHeight(172)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.setAccessibleName(f"{title} method")
        self.setAccessibleDescription(
            description
            if available
            else f"{description} This method is still in development."
        )

        if available:
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip(f"Click to open {title}")
            self.setStatusTip(f"Open the {title} method")
        else:
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setToolTip(
                f"{title} is not available yet. "
                "Its implementation has not been integrated."
            )

        # Card shadow and hover animation.
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(13)
        self._shadow.setOffset(0, 3)
        self._shadow.setColor(QColor(30, 45, 25, 22))
        self.setGraphicsEffect(self._shadow)

        self._shadow_animation = QParallelAnimationGroup(self)

        self._blur_animation = QPropertyAnimation(
            self._shadow,
            b"blurRadius",
            self._shadow_animation,
        )
        self._blur_animation.setDuration(150)
        self._blur_animation.setEasingCurve(
            QEasingCurve.Type.OutCubic
        )

        self._offset_animation = QPropertyAnimation(
            self._shadow,
            b"yOffset",
            self._shadow_animation,
        )
        self._offset_animation.setDuration(150)
        self._offset_animation.setEasingCurve(
            QEasingCurve.Type.OutCubic
        )

        self._shadow_animation.addAnimation(self._blur_animation)
        self._shadow_animation.addAnimation(self._offset_animation)

        # Card content.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(17, 15, 17, 14)
        layout.setSpacing(8)

        number_label = QLabel(number)
        number_label.setObjectName("cardNumber")
        number_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setWordWrap(True)
        title_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
        )
        title_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )

        description_label = QLabel(description)
        description_label.setObjectName("cardText")
        description_label.setWordWrap(True)
        description_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
        )
        description_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        description_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )

        layout.addWidget(number_label)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addStretch(1)

        status_label = QLabel(
            "OPEN METHOD  ↗"
            if available
            else "IN DEVELOPMENT"
        )
        status_label.setObjectName("cardStatus")
        status_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        layout.addWidget(status_label)

        # Card appearance.
        self.setStyleSheet(
            """
            QFrame#workflowCard {
                background-color: #FFFFFF;
                border: 1px solid #DDE5DA;
                border-radius: 9px;
            }

            QFrame#workflowCard:hover {
                background-color: #F9FCF7;
                border: 1px solid #8EAF80;
            }

            QFrame#workflowCard[pressed="true"] {
                background-color: #EDF5E8;
                border: 1px solid #57863C;
            }

            QFrame#workflowCardUnavailable {
                background-color: #F6F7F5;
                border: 1px solid #E4E7E1;
                border-radius: 9px;
            }

            QLabel#cardNumber {
                color: #57863C;
                font-size: 9pt;
                font-weight: 700;
                background: transparent;
                border: none;
            }

            QLabel#cardTitle {
                color: #243124;
                font-size: 15pt;
                font-weight: 600;
                background: transparent;
                border: none;
            }

            QLabel#cardText {
                color: #596358;
                font-size: 10pt;
                background: transparent;
                border: none;
            }

            QLabel#cardStatus {
                color: #57863C;
                font-size: 9pt;
                font-weight: 600;
                background: transparent;
                border: none;
            }

            QFrame#workflowCardUnavailable QLabel#cardNumber {
                color: #8A9188;
            }

            QFrame#workflowCardUnavailable QLabel#cardTitle {
                color: #737A70;
            }

            QFrame#workflowCardUnavailable QLabel#cardText {
                color: #858B82;
            }

            QFrame#workflowCardUnavailable QLabel#cardStatus {
                color: #878D84;
            }
            """
        )

        self.setProperty("pressed", False)

    # -----------------------------------------------------------------
    # Visual feedback
    # -----------------------------------------------------------------

    def _animate_shadow(self, hovered: bool) -> None:
        """Animate the card shadow without changing its geometry."""
        if not self._available:
            return

        self._shadow_animation.stop()

        self._blur_animation.setStartValue(
            float(self._shadow.blurRadius())
        )
        self._blur_animation.setEndValue(
            21.0 if hovered else 13.0
        )

        self._offset_animation.setStartValue(
            float(self._shadow.yOffset())
        )
        self._offset_animation.setEndValue(
            5.0 if hovered else 3.0
        )

        self._shadow_animation.start()

    def _set_pressed(self, pressed: bool) -> None:
        """Update the pressed visual state."""
        self._pressed = pressed
        self.setProperty("pressed", pressed)

        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def enterEvent(self, event) -> None:
        if self._available:
            self._animate_shadow(True)

            if self._mouse_pressed and not self._keyboard_pressed:
                self._set_pressed(True)

        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._available:
            self._animate_shadow(False)

            if self._mouse_pressed and not self._keyboard_pressed:
                self._set_pressed(False)

        super().leaveEvent(event)

    # -----------------------------------------------------------------
    # Mouse interaction
    # -----------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            self._available
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._mouse_pressed = True
            self._set_pressed(True)

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            self._available
            and event.button() == Qt.MouseButton.LeftButton
        ):
            was_mouse_pressed = self._mouse_pressed
            self._mouse_pressed = False
            self._set_pressed(False)

            released_inside = self.rect().contains(
                event.position().toPoint()
            )

            if was_mouse_pressed and released_inside:
                self.clicked.emit()

            event.accept()
            return

        super().mouseReleaseEvent(event)

    # -----------------------------------------------------------------
    # Keyboard interaction
    # -----------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        activation_keys = (
            Qt.Key.Key_Space,
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        )

        if (
            self._available
            and event.key() in activation_keys
        ):
            if not event.isAutoRepeat():
                self._keyboard_pressed = True
                self._set_pressed(True)

            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        activation_keys = (
            Qt.Key.Key_Space,
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        )

        if (
            self._available
            and event.key() in activation_keys
            and self._keyboard_pressed
        ):
            self._keyboard_pressed = False
            self._set_pressed(False)
            self.clicked.emit()

            event.accept()
            return

        super().keyReleaseEvent(event)


# ---------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main application window for Easy Quadrat."""

    def __init__(self) -> None:
        super().__init__()

        self.setObjectName("mainWindow")
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setMinimumSize(980, 680)
        self.resize(1120, 760)

        self._active_method_dialog: QDialog | None = None

        configure_window(
            self,
            keep_on_top=True,
        )

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(30, 25, 30, 19)
        layout.setSpacing(15)

        # -----------------------------------------------------------------
        # Header
        # -----------------------------------------------------------------

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(15)

        logo_label = QLabel()
        logo_label.setObjectName("appLogo")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setFixedSize(60, 60)

        if APP_LOGO_PATH.is_file():
            pixmap = QPixmap(str(APP_LOGO_PATH))

            if not pixmap.isNull():
                logo_label.setPixmap(
                    pixmap.scaled(
                        60,
                        60,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                logo_label.setText("EQ")
                logo_label.setObjectName("logoFallback")
        else:
            logo_label.setText("EQ")
            logo_label.setObjectName("logoFallback")

        header.addWidget(
            logo_label,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(4)

        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")

        tagline = QLabel(APP_SHORT_TAGLINE)
        tagline.setObjectName("subtitle")
        tagline.setWordWrap(True)

        title_column.addWidget(title)
        title_column.addWidget(tagline)

        header.addLayout(title_column, 1)

        # The public application window identifies the release version,
        # rather than displaying a temporary development-build status.
        version_label = QLabel(
            f"VERSION\n{APP_VERSION}"
        )
        version_label.setObjectName("versionPill")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header.addWidget(
            version_label,
            0,
            Qt.AlignmentFlag.AlignTop,
        )

        layout.addLayout(header)

        # -----------------------------------------------------------------
        # Header separator
        # -----------------------------------------------------------------

        header_rule = QFrame()
        header_rule.setObjectName("headerRule")
        header_rule.setFixedHeight(1)

        layout.addWidget(header_rule)

        # -----------------------------------------------------------------
        # Processing-method heading
        # -----------------------------------------------------------------

        section_title = QLabel("Processing methods")
        section_title.setObjectName("sectionTitle")

        section_description = QLabel(
            "Choose a method to identify, crop, and segment quadrat "
            "regions in plant images. Inspect automated outputs "
            "before using them for measurements."
        )
        section_description.setObjectName("sectionDescription")
        section_description.setWordWrap(True)

        layout.addWidget(section_title)
        layout.addWidget(section_description)

        # -----------------------------------------------------------------
        # Method cards
        # -----------------------------------------------------------------

        methods_grid = QGridLayout()
        methods_grid.setContentsMargins(0, 2, 0, 2)
        methods_grid.setHorizontalSpacing(12)
        methods_grid.setVerticalSpacing(12)

        for column in range(6):
            methods_grid.setColumnStretch(column, 1)

        methods_grid.setRowStretch(0, 1)
        methods_grid.setRowStretch(1, 1)

        methods = [
            {
                "number": "METHOD 01",
                "title": "Manual",
                "description": (
                    "Select the quadrat region manually, then save "
                    "the cropped and segmented images with polygon labels."
                ),
                "available": True,
                "action": self.open_manual,
            },
            {
                "number": "METHOD 02",
                "title": "Fixed Frame",
                "description": (
                    "Set percentage boundaries and crop a consistent "
                    "region across a collection of images."
                ),
                "available": True,
                "action": self.open_fixed_frame,
            },
            {
                "number": "METHOD 03",
                "title": "Auto · Classic I",
                "description": (
                    "Uses low-saturation thresholding, isotropic "
                    "morphology, largest-background-component selection, "
                    "and border clearing."
                ),
                "available": True,
                "action": self.open_classic_i,
            },
            {
                "number": "METHOD 04",
                "title": "Auto · Classic II",
                "description": (
                    "Uses saturation-based preprocessing, then compares "
                    "directional rectangular-opening candidates refined "
                    "with convex hulls and selected by area."
                ),
                "available": True,
                "action": self.open_classic_ii,
            },
            {
                "number": "METHOD 05",
                "title": "Auto · Deep Learning",
                "description": (
                    "Uses trained YOLO instance-segmentation models to "
                    "detect and segment white quadrats, with standard, "
                    "Frameless, and custom-model options."
                ),
                "available": True,
                "action": self.open_deep_learning,
            },
        ]

        cards: list[MethodCard] = []

        for method in methods:
            card = MethodCard(
                number=method["number"],
                title=method["title"],
                description=method["description"],
                available=method["available"],
            )

            action = method["action"]

            if action is not None:
                card.clicked.connect(action)

            cards.append(card)

        # First row: the two non-automated methods, centered.
        # Each card spans two columns; the outer columns remain empty
        # to center the pair horizontally.
        methods_grid.addWidget(
            cards[0],
            0,
            1,
            1,
            2,
        )
        methods_grid.addWidget(
            cards[1],
            0,
            3,
            1,
            2,
        )

        # Second row: all three automated methods, side by side.
        methods_grid.addWidget(
            cards[2],
            1,
            0,
            1,
            2,
        )
        methods_grid.addWidget(
            cards[3],
            1,
            2,
            1,
            2,
        )
        methods_grid.addWidget(
            cards[4],
            1,
            4,
            1,
            2,
        )

        layout.addLayout(methods_grid, 1)

        # -----------------------------------------------------------------
        # Footer
        # -----------------------------------------------------------------

        footer_line = QFrame()
        footer_line.setObjectName("footerLine")
        footer_line.setFixedHeight(1)

        layout.addWidget(footer_line)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(10)

        footer_label = QLabel(
            "Easy-Phenotyping Lab (EPL) · "
            "Open and accessible phenotyping tools · "
            "© 2026 Abbas Haghshenas"
        )
        footer_label.setObjectName("footerText")
        footer_label.setWordWrap(True)

        footer.addWidget(footer_label, 1)

        # Method-launch buttons have deliberately been removed.
        # Method selection happens directly through the cards.
        about_button = QPushButton("About")
        about_button.setObjectName("aboutButton")
        about_button.setFixedHeight(31)
        about_button.setMinimumWidth(76)
        about_button.clicked.connect(self.open_about)

        footer.addWidget(about_button)
        layout.addLayout(footer)

    # -----------------------------------------------------------------
    # Method-window lifecycle
    # -----------------------------------------------------------------

    def _bring_active_method_forward(self) -> bool:
        """Raise the existing method window, if one is already open."""
        dialog = self._active_method_dialog

        if dialog is None:
            return False

        try:
            if dialog.isMinimized():
                dialog.showNormal()

            if not dialog.isVisible():
                dialog.show()

            dialog.raise_()
            dialog.activateWindow()
            return True

        except RuntimeError:
            # The Qt object has already been deleted.
            self._active_method_dialog = None
            return False

    def _register_method_dialog(
        self,
        dialog: QDialog,
    ) -> None:
        """Show one modeless method window and track its lifecycle."""
        if self._bring_active_method_forward():
            # Do not allow two method windows to run simultaneously.
            dialog.deleteLater()
            return

        self._active_method_dialog = dialog

        dialog.setAttribute(
            Qt.WidgetAttribute.WA_DeleteOnClose,
            True,
        )

        dialog.finished.connect(
            lambda _result, owner=dialog:
            self._release_method_dialog(owner)
        )

        dialog.destroyed.connect(
            lambda _object=None, owner=dialog:
            self._release_method_dialog(owner)
        )

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _release_method_dialog(
        self,
        dialog: QDialog,
    ) -> None:
        """Clear the method-window reference after it closes."""
        if self._active_method_dialog is dialog:
            self._active_method_dialog = None

    # -----------------------------------------------------------------
    # Processing methods
    # -----------------------------------------------------------------

    def open_manual(self) -> None:
        """Open Manual from its main-window card."""
        if self._bring_active_method_forward():
            return

        dialog = ManualDialog(self)
        self._register_method_dialog(dialog)

    def open_fixed_frame(self) -> None:
        """Open Fixed Frame from its main-window card."""
        if self._bring_active_method_forward():
            return

        dialog = FixedFrameDialog(self)
        self._register_method_dialog(dialog)

    def open_classic_i(self) -> None:
        """Open Auto Classic I from its main-window card."""
        if self._bring_active_method_forward():
            return

        dialog = AutoClassicIDialog(self)
        self._register_method_dialog(dialog)

    def open_classic_ii(self) -> None:
        """Open Auto Classic II from its main-window card."""
        if self._bring_active_method_forward():
            return

        dialog = AutoClassicIIDialog(self)
        self._register_method_dialog(dialog)

    def open_deep_learning(self) -> None:
        """Open Auto Deep Learning from its main-window card."""
        if self._bring_active_method_forward():
            return

        dialog = AutoDeepLearningDialog(self)
        self._register_method_dialog(dialog)

    def open_about(self) -> None:
        """Open the application information dialog."""
        dialog = AboutDialog(self)
        dialog.exec()