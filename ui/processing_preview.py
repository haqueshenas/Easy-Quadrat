"""
Easy Quadrat v2.0.0
Shared visual component for optional image-processing previews.

Preview rendering is presentation-only. It must not modify the source
arrays used by the scientific algorithms or output-saving functions.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


PREVIEW_MAX_SIZE = (800, 560)


def rgb_array_to_qimage(rgb_array: np.ndarray) -> QImage:
    """Convert an independent uint8 RGB array to a detached QImage."""

    array = np.asarray(rgb_array)

    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
    ):
        raise ValueError("Preview images must be uint8 RGB arrays.")

    contiguous = np.ascontiguousarray(array)
    height, width = contiguous.shape[:2]

    return QImage(
        contiguous.data,
        width,
        height,
        int(contiguous.strides[0]),
        QImage.Format.Format_RGB888,
    ).copy()


def make_preview_base(
    image_rgb: np.ndarray,
    max_width: int = PREVIEW_MAX_SIZE[0],
    max_height: int = PREVIEW_MAX_SIZE[1],
) -> np.ndarray:
    """
    Make an independent, reduced display copy of the source image.

    The original array is neither resized nor modified.
    """

    array = np.asarray(image_rgb)

    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
    ):
        raise ValueError(
            "The preview source must be an 8-bit RGB image."
        )

    if max_width < 1 or max_height < 1:
        raise ValueError("Preview dimensions must be positive.")

    with Image.fromarray(
        np.ascontiguousarray(array)
    ) as image:
        image.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS,
        )

        return np.asarray(
            image,
            dtype=np.uint8,
        ).copy()


def _resize_preview_mask(
    mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Resize a binary mask for display only.

    The resized mask must never be used as input to the detection algorithm
    or as a source for saved output images.
    """

    binary = np.asarray(mask, dtype=bool)

    if binary.ndim != 2:
        raise ValueError("Preview masks must be two-dimensional.")

    if binary.shape == (height, width):
        return binary.copy()

    mask_pixels = binary.astype(np.uint8) * 255

    with Image.fromarray(mask_pixels) as mask_image:
        with mask_image.resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        ) as resized:
            return (
                np.asarray(
                    resized,
                    dtype=np.uint8,
                ).copy()
                > 0
            )


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    """Return boundary pixels from a two-dimensional binary mask."""

    binary = np.asarray(mask, dtype=bool)

    if binary.ndim != 2:
        raise ValueError(
            "A mask outline requires a two-dimensional mask."
        )

    height, width = binary.shape

    if height < 3 or width < 3:
        return binary.copy()

    interior = np.zeros_like(
        binary,
        dtype=bool,
    )

    interior[1:-1, 1:-1] = (
        binary[1:-1, 1:-1]
        & binary[:-2, 1:-1]
        & binary[2:, 1:-1]
        & binary[1:-1, :-2]
        & binary[1:-1, 2:]
    )

    return binary & ~interior


def render_processing_preview(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    stage_name: str,
    background_rgb: tuple[int, int, int] = (0, 0, 0),
) -> QImage:
    """
    Render one intermediate or final mask as an independent QImage.

    Intermediate masks use a translucent red overlay.

    The final mask is shown as a visual segmentation mock-up with the
    selected background color, a green outline and a yellow crop box.

    All resizing, tinting and annotation happen on a new display array.
    They never modify the source pixels or the actual detection mask.
    """

    base = np.asarray(base_rgb)

    if (
        base.dtype != np.uint8
        or base.ndim != 3
        or base.shape[2] != 3
    ):
        raise ValueError(
            "The preview base must be a uint8 RGB image."
        )

    rendered = np.ascontiguousarray(base).copy()
    height, width = rendered.shape[:2]

    small_mask = _resize_preview_mask(
        mask,
        width,
        height,
    )

    final_stage = (
        "final quadrat mask" in stage_name.casefold()
    )

    if final_stage:
        background = np.asarray(
            background_rgb,
            dtype=np.uint8,
        )

        # Display-only representation of the segmented result.
        rendered[~small_mask] = background

        if small_mask.any():
            selected_pixels = rendered[small_mask].astype(
                np.float32
            )

            green = np.array(
                (73, 170, 76),
                dtype=np.float32,
            )

            rendered[small_mask] = np.clip(
                selected_pixels * 0.86 + green * 0.14,
                0,
                255,
            ).astype(np.uint8)

            rendered[_mask_outline(small_mask)] = (
                70, 255, 95
            )

            ys, xs = np.nonzero(small_mask)

            left = int(xs.min())
            right = int(xs.max())
            top = int(ys.min())
            bottom = int(ys.max())

            yellow = (255, 225, 75)
            thickness = 2

            for offset in range(thickness):
                y_top = top + offset
                y_bottom = bottom - offset
                x_left = left + offset
                x_right = right - offset

                if y_top <= y_bottom:
                    rendered[
                        y_top,
                        max(0, x_left):min(width, x_right + 1),
                    ] = yellow

                    rendered[
                        y_bottom,
                        max(0, x_left):min(width, x_right + 1),
                    ] = yellow

                if x_left <= x_right:
                    rendered[
                        max(0, y_top):min(height, y_bottom + 1),
                        x_left,
                    ] = yellow

                    rendered[
                        max(0, y_top):min(height, y_bottom + 1),
                        x_right,
                    ] = yellow

    else:
        selected_pixels = rendered[small_mask].astype(
            np.float32
        )

        red = np.array(
            (245, 48, 48),
            dtype=np.float32,
        )

        rendered[small_mask] = np.clip(
            selected_pixels * 0.55 + red * 0.45,
            0,
            255,
        ).astype(np.uint8)

        rendered[_mask_outline(small_mask)] = (
            255, 65, 65
        )

    return rgb_array_to_qimage(rendered)


class ProcessingPreviewPanel(QFrame):
    """
    Shared image preview panel for automated methods.

    The Show preview checkbox is deliberately outside this widget.
    This allows the user to disable the entire panel and enable it again.
    """

    test_requested = Signal()
    previous_stage_requested = Signal()
    next_stage_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        enable_stage_navigation: bool = False,
    ) -> None:
        super().__init__(parent)

        # Stage navigation is opt-in so other methods using this shared
        # panel retain their existing appearance and behavior.
        self._stage_navigation_enabled = bool(enable_stage_navigation)

        self.setObjectName("processingPreviewPanel")
        self.setMinimumWidth(280)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 14)
        layout.setSpacing(9)

        heading = QLabel("Processing preview")
        heading.setObjectName("previewHeading")
        layout.addWidget(heading)

        description = QLabel(
            "Visual feedback from the actual detection stages"
        )
        description.setObjectName("previewDescription")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.image_label = QLabel()
        self.image_label.setObjectName("previewImage")
        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.image_label.setMinimumSize(250, 200)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.image_label.setWordWrap(True)
        self.image_label.setText(
            "Select an input folder to preview its first image."
        )
        layout.addWidget(
            self.image_label,
            1,
        )

        self.stage_label = QLabel(
            "No processing stage yet"
        )
        self.stage_label.setObjectName("previewStage")
        self.stage_label.setWordWrap(True)

        # One compact row: green stage name on the left, navigation
        # triangles on the right. They remain hidden in other methods.
        stage_row = QHBoxLayout()
        stage_row.setContentsMargins(0, 0, 0, 0)
        stage_row.setSpacing(4)
        stage_row.addWidget(self.stage_label, 1)

        self.previous_stage_button = QPushButton("◀")
        self.previous_stage_button.setFixedSize(23, 22)
        self.previous_stage_button.setToolTip("Show the previous test stage")
        self.previous_stage_button.setAccessibleName("Previous test stage")
        self.previous_stage_button.setAccessibleDescription(
            "Display the preceding saved stage from the current method's latest test."
        )
        self.previous_stage_button.clicked.connect(
            self.previous_stage_requested.emit
        )

        self.next_stage_button = QPushButton("▶")
        self.next_stage_button.setFixedSize(23, 22)
        self.next_stage_button.setToolTip("Show the next test stage")
        self.next_stage_button.setAccessibleName("Next test stage")
        self.next_stage_button.setAccessibleDescription(
            "Display the following saved stage from the current method's latest test."
        )
        self.next_stage_button.clicked.connect(
            self.next_stage_requested.emit
        )

        navigation_button_style = """
            QPushButton {
                color: #57863C;
                background-color: #F4F7F1;
                border: 1px solid #DDE5DA;
                border-radius: 4px;
                padding: 0px;
                font-size: 9px;
                font-weight: 700;
            }
            QPushButton:enabled:hover {
                background-color: #E8F1E3;
                border-color: #AFC7A5;
            }
            QPushButton:disabled {
                color: #C5CCC1;
                background-color: #F7F8F6;
                border-color: #EBEEE8;
            }
        """
        self.previous_stage_button.setStyleSheet(navigation_button_style)
        self.next_stage_button.setStyleSheet(navigation_button_style)

        for navigation_button in (
            self.previous_stage_button,
            self.next_stage_button,
        ):
            navigation_button.setVisible(self._stage_navigation_enabled)
            navigation_button.setEnabled(False)
            stage_row.addWidget(navigation_button, 0, Qt.AlignmentFlag.AlignVCenter)

        layout.addLayout(stage_row)

        self.status_label = QLabel(
            "Preview does not save output files."
        )
        self.status_label.setObjectName("previewStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addStretch(1)

        self.test_button = QPushButton("Run a test")
        self.test_button.setObjectName("primaryButton")
        self.test_button.setMinimumWidth(105)
        self.test_button.setToolTip(
            "Run the actual detection algorithm on the first input image "
            "only. The test does not create output images, output folders "
            "or a processing log."
        )
        self.test_button.clicked.connect(
            self.test_requested.emit
        )
        controls.addWidget(self.test_button)

        layout.addLayout(controls)

        self.setStyleSheet(
            """
            QFrame#processingPreviewPanel {
                background-color: #FFFFFF;
                border: 1px solid #DDE5DA;
                border-radius: 10px;
            }

            QLabel#previewHeading {
                color: #243124;
                font-size: 13pt;
                font-weight: 700;
                background: transparent;
                border: none;
            }

            QLabel#previewDescription {
                color: #687366;
                font-size: 9pt;
                background: transparent;
                border: none;
            }

            QLabel#previewImage {
                background-color: #F2F4F0;
                border: 1px solid #DDE5DA;
                border-radius: 5px;
                padding: 5px;
                color: #687366;
            }

            QLabel#previewStage {
                color: #365D2B;
                font-size: 10pt;
                font-weight: 600;
                background: transparent;
                border: none;
            }

            QLabel#previewStatus {
                color: #687366;
                font-size: 9pt;
                background: transparent;
                border: none;
            }
            """
        )

    def set_frame(
        self,
        image: QImage,
        stage_name: str,
    ) -> None:
        """Display a detached QImage in the GUI thread."""

        if image.isNull():
            return

        self._pixmap = QPixmap.fromImage(image)
        self.stage_label.setText(stage_name)
        self._refresh_pixmap()

    def set_placeholder(
        self,
        text: str,
    ) -> None:
        """Clear the displayed image and show text instead."""

        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText(text)
        self.stage_label.setText(
            "Preview not available"
        )
        self.set_stage_navigation_state(
            previous_enabled=False,
            next_enabled=False,
        )

    def set_stage_navigation_state(
        self,
        *,
        previous_enabled: bool,
        next_enabled: bool,
    ) -> None:
        """Enable only the valid directions for an available test history."""

        if not self._stage_navigation_enabled:
            return

        self.previous_stage_button.setEnabled(bool(previous_enabled))
        self.next_stage_button.setEnabled(bool(next_enabled))

    def set_status(
        self,
        text: str,
        status: str = "neutral",
    ) -> None:
        """Display a concise status message."""

        colors = {
            "neutral": "#687366",
            "success": "#35752D",
            "error": "#A52B2B",
            "working": "#496F32",
        }

        color = colors.get(
            status,
            colors["neutral"],
        )

        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"""
            QLabel#previewStatus {{
                color: {color};
                font-size: 9pt;
                background: transparent;
                border: none;
            }}
            """
        )

    def _refresh_pixmap(self) -> None:
        """Fit the image into the current preview area."""

        if self._pixmap is None or self._pixmap.isNull():
            return

        available_size = self.image_label.size()

        if (
            available_size.width() < 2
            or available_size.height() < 2
        ):
            return

        scaled = self._pixmap.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.image_label.setText("")
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._refresh_pixmap()
