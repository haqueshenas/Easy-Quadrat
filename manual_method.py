"""
Easy Quadrat v2.0.0
Manual image-cropping and labelling method.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw

from PySide6.QtCore import Qt, QPoint, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from core.image_output import (
    SUPPORTED_EXTENSIONS as OUTPUT_SUPPORTED_EXTENSIONS,
    inspect_image as inspect_output_image,
    supported_image_files,
    normalize_policy,
    plan_output,
    save_planned_output,
    find_jpegtran,
)
from output_manager import (
    ProcessingMethod,
    prepare_output_layout,
)


# ---------------------------------------------------------------------
# Shared image support and application constants
# ---------------------------------------------------------------------

SUPPORTED_EXTENSIONS = set(OUTPUT_SUPPORTED_EXTENSIONS)

MIN_CUSTOM_VERTICES = 3
MAX_CUSTOM_VERTICES = 1000
DEFAULT_CUSTOM_VERTICES = 21

EXIF_ORIENTATION_TOOLTIP = (
    "Applies the image's EXIF Orientation metadata before display and "
    "quadrat selection. When disabled, the metadata is ignored and the "
    "stored pixel orientation is used. The original input image is not modified."
)

PALETTE_COLOURS = [
    ("Red", "#ff0000"),
    ("Green", "#00a000"),
    ("Blue", "#0000ff"),
    ("Yellow", "#ffff00"),
    ("Magenta", "#ff00ff"),
    ("Purple", "#800080"),
    ("Black", "#000000"),
    ("White", "#ffffff"),
]


# ---------------------------------------------------------------------
# Image discovery and metadata
# ---------------------------------------------------------------------

def list_manual_images(folder: Path) -> list[Path]:
    """Use the central image-output module's supported-format policy."""

    return supported_image_files(folder)


def get_frame_count(image: Image.Image) -> int:
    """Return the number of frames/pages reported by Pillow."""

    try:
        return max(1, int(getattr(image, "n_frames", 1)))
    except (TypeError, ValueError, EOFError, AttributeError):
        return 1


def inspect_image(path: Path) -> dict:
    """Use central source-format and metadata inspection."""

    return inspect_output_image(path)


def load_first_frame(
    path: Path,
    handle_orientation: bool,
) -> Image.Image:
    """
    Load a separate working copy of the primary image.

    If EXIF Orientation is enabled, normalize pixel orientation before
    displaying the image and selecting the quadrat. The source file is
    never modified.
    """

    with Image.open(path) as source:
        source.seek(0)
        source.load()

        if handle_orientation:
            normalized = ImageOps.exif_transpose(source)

            try:
                result = normalized.copy()
                result.info.update(normalized.info)
                return result
            finally:
                if normalized is not source:
                    normalized.close()

        result = source.copy()
        result.info.update(source.info)
        return result


def _segmented_output_mode(path: Path) -> str:
    """
    Predict the output pixel mode for a segmented image.

    Palette-based images are expanded for segmentation so that the selected
    background color can be represented exactly.
    """

    with Image.open(path) as source:
        source.seek(0)

        if source.mode == "P":
            if "transparency" in source.info:
                return "RGBA"
            return "RGB"

        return source.mode


def _write_json(path: Path, content: dict) -> None:
    """Write UTF-8 JSON with readable indentation."""

    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Polygon geometry
# ---------------------------------------------------------------------

def polygon_crop_box(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """
    Return a clamped integer rectangle enclosing the polygon vertices.

    Pillow's right and bottom crop coordinates are exclusive.
    """

    if len(points) < 3:
        raise ValueError(
            "At least three polygon vertices are required."
        )

    left = max(0, math.floor(min(point[0] for point in points)))
    top = max(0, math.floor(min(point[1] for point in points)))

    right = min(
        width,
        math.ceil(max(point[0] for point in points)) + 1,
    )
    bottom = min(
        height,
        math.ceil(max(point[1] for point in points)) + 1,
    )

    if left >= right or top >= bottom:
        raise ValueError(
            "The selected quadrat has an empty crop rectangle."
        )

    return left, top, right, bottom


def square_from_diagonal(
    first: tuple[float, float],
    second: tuple[float, float],
) -> list[tuple[float, float]]:
    """Construct a square from two opposite vertices of its diagonal."""

    x1, y1 = first
    x2, y2 = second

    dx = x2 - x1
    dy = y2 - y1

    if math.hypot(dx, dy) < 1.0:
        raise ValueError(
            "The two diagonal points are too close together."
        )

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    perpendicular_x = -dy / 2.0
    perpendicular_y = dx / 2.0

    third = (
        center_x + perpendicular_x,
        center_y + perpendicular_y,
    )

    fourth = (
        center_x - perpendicular_x,
        center_y - perpendicular_y,
    )

    return [first, third, second, fourth]


def polygon_area(
    points: list[tuple[float, float]],
) -> float:
    """Calculate polygon area using the shoelace formula."""

    if len(points) < 3:
        return 0.0

    total = 0.0

    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]

        total += (
            point[0] * next_point[1]
            - next_point[0] * point[1]
        )

    return abs(total) / 2.0


# ---------------------------------------------------------------------
# Segmented image construction
# ---------------------------------------------------------------------

def _background_value_for_mode(
    image: Image.Image,
    rgb: tuple[int, int, int],
):
    """
    Represent the selected background in the current pixel mode.

    Unsupported modes are rejected rather than silently converted.
    """

    red, green, blue = rgb

    grey = round(
        0.299 * red + 0.587 * green + 0.114 * blue
    )

    mode = image.mode

    if mode == "RGB":
        return rgb

    if mode == "RGBA":
        return red, green, blue, 255

    if mode == "L":
        return grey

    if mode == "LA":
        return grey, 255

    if mode == "1":
        return 1 if grey >= 128 else 0

    if mode in {"CMYK", "YCbCr"}:
        sample = Image.new("RGB", (1, 1), rgb)

        try:
            return sample.convert(mode).getpixel((0, 0))
        finally:
            sample.close()

    if mode == "I":
        return grey

    if mode in {"I;16", "I;16L", "I;16B"}:
        return round(grey * 65535 / 255)

    if mode == "F":
        return grey / 255.0

    if mode == "P":
        palette = image.getpalette()

        if not palette:
            raise ValueError(
                "This palette image has no readable palette. "
                "Its background color cannot be represented reliably."
            )

        best_index = 0
        best_distance = float("inf")

        for index in range(len(palette) // 3):
            pr = palette[index * 3]
            pg = palette[index * 3 + 1]
            pb = palette[index * 3 + 2]

            distance = (
                (pr - red) ** 2
                + (pg - green) ** 2
                + (pb - blue) ** 2
            )

            if distance < best_distance:
                best_distance = distance
                best_index = index

        return best_index

    raise ValueError(
        f"Cannot safely create the selected background in image mode "
        f"{mode!r}. The pixel mode will not be silently converted."
    )


def make_background_image(
    source: Image.Image,
    rgb: tuple[int, int, int],
) -> Image.Image:
    """Create a solid background without silently reducing pixel precision."""

    value = _background_value_for_mode(source, rgb)

    if source.mode == "P":
        background = Image.new(
            "P",
            source.size,
            value,
        )

        palette = source.getpalette()

        if palette:
            background.putpalette(palette)

        background.info.update(source.info)
        return background

    background = Image.new(
        source.mode,
        source.size,
        value,
    )

    background.info.update(source.info)
    return background


def make_segmented_image(
    source: Image.Image,
    polygon: list[tuple[float, float]],
    background_rgb: tuple[int, int, int],
) -> Image.Image:
    """
    Recolour pixels outside the polygon.

    The source is never modified. Palette images are converted to RGB or
    RGBA for the segmented derivative only.
    """

    working_source = source
    converted_source = None

    if source.mode == "P":
        output_mode = (
            "RGBA"
            if "transparency" in source.info
            else "RGB"
        )

        converted_source = source.convert(output_mode)
        converted_source.info.update(source.info)
        converted_source.info.pop("transparency", None)

        working_source = converted_source

    mask = Image.new(
        "L",
        working_source.size,
        0,
    )

    try:
        draw = ImageDraw.Draw(mask)

        integer_points = [
            (
                max(0, min(working_source.width - 1, round(x))),
                max(0, min(working_source.height - 1, round(y))),
            )
            for x, y in polygon
        ]

        draw.polygon(
            integer_points,
            fill=255,
        )

        background = make_background_image(
            working_source,
            background_rgb,
        )

        try:
            segmented = Image.composite(
                working_source,
                background,
                mask,
            )

            segmented.info.update(working_source.info)

            if segmented.mode in {"RGB", "RGBA"}:
                segmented.info.pop("transparency", None)

            return segmented

        finally:
            background.close()

    finally:
        mask.close()

        if converted_source is not None:
            converted_source.close()


# ---------------------------------------------------------------------
# Interactive image canvas
# ---------------------------------------------------------------------

class ManualImageView(QGraphicsView):
    """Pixel-coordinate canvas with zoom and right-button panning."""

    imageClicked = Signal(float, float)
    zoomChanged = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.scene_object = QGraphicsScene(self)
        self.setScene(self.scene_object)

        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.overlay_items = []

        self.zoom_value = 1.0
        self.panning = False
        self.last_pan_position = None

        self.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            True,
        )
        self.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            True,
        )

        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter
        )

        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.setBackgroundBrush(QColor("#f3f4f1"))
        self.setMinimumSize(460, 360)

    def clear_image(self) -> None:
        self.scene_object.clear()
        self.pixmap_item = None
        self.overlay_items = []
        self.zoom_value = 1.0
        self.resetTransform()

    def set_image(self, image: Image.Image) -> None:
        """Display an RGB preview; the saved raster remains the PIL image."""

        preview = image.convert("RGB")

        try:
            qimage = QImage(
                preview.tobytes(),
                preview.width,
                preview.height,
                preview.width * 3,
                QImage.Format.Format_RGB888,
            ).copy()

            pixmap = QPixmap.fromImage(qimage)

        finally:
            preview.close()

        self.scene_object.clear()
        self.overlay_items = []

        self.pixmap_item = self.scene_object.addPixmap(pixmap)
        self.pixmap_item.setOffset(0, 0)

        self.scene_object.setSceneRect(
            QRectF(0, 0, image.width, image.height)
        )

        self.zoom_value = 1.0
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        """Set the fit-to-window baseline and apply the selected zoom."""

        if self.pixmap_item is None:
            return

        bounds = self.scene_object.sceneRect()

        self.resetTransform()

        self.fitInView(
            bounds,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        if self.zoom_value > 1.0:
            self.scale(
                self.zoom_value,
                self.zoom_value,
            )

    def set_zoom(
        self,
        value: float,
        anchor_global_position: QPoint | None = None,
    ) -> None:
        """Set zoom between 1× and 10×."""

        new_zoom = max(
            1.0,
            min(10.0, float(value)),
        )

        if self.pixmap_item is None:
            self.zoom_value = new_zoom
            self.zoomChanged.emit(self.zoom_value)
            return

        if (
            anchor_global_position is None
            or new_zoom <= 1.0
            or math.isclose(
                new_zoom,
                self.zoom_value,
                abs_tol=1e-9,
            )
        ):
            self.zoom_value = new_zoom
            self._apply_zoom()
            self.zoomChanged.emit(self.zoom_value)
            return

        old_zoom = self.zoom_value

        cursor_before = self.viewport().mapFromGlobal(
            anchor_global_position
        )
        anchor_scene = self.mapToScene(cursor_before)

        factor = new_zoom / old_zoom
        original_anchor = self.transformationAnchor()

        self.zoom_value = new_zoom
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.NoAnchor
        )

        try:
            self.scale(factor, factor)
        finally:
            self.setTransformationAnchor(original_anchor)

        cursor_after = self.viewport().mapFromGlobal(
            anchor_global_position
        )
        anchor_after = self.mapFromScene(anchor_scene)
        displacement = anchor_after - cursor_after

        horizontal = self.horizontalScrollBar()
        vertical = self.verticalScrollBar()

        horizontal.setValue(
            horizontal.value() + displacement.x()
        )
        vertical.setValue(
            vertical.value() + displacement.y()
        )

        self.zoomChanged.emit(self.zoom_value)

    def show_polygon(
        self,
        points: list[tuple[float, float]],
        outline_colour: str,
        complete: bool,
    ) -> None:
        for item in self.overlay_items:
            self.scene_object.removeItem(item)

        self.overlay_items = []

        if not points:
            return

        pen = QPen(
            QColor(outline_colour),
            3,
        )
        pen.setCosmetic(True)

        if complete and len(points) > 2:
            polygon_item = self.scene_object.addPolygon(
                QPolygonF(
                    [QPointF(x, y) for x, y in points]
                ),
                pen,
                QBrush(Qt.BrushStyle.NoBrush),
            )
            self.overlay_items.append(polygon_item)

        else:
            for index in range(len(points) - 1):
                x1, y1 = points[index]
                x2, y2 = points[index + 1]

                item = self.scene_object.addLine(
                    x1,
                    y1,
                    x2,
                    y2,
                    pen,
                )
                self.overlay_items.append(item)

        for x, y in points:
            marker = self.scene_object.addEllipse(
                QRectF(x - 4, y - 4, 8, 8),
                QPen(QColor(outline_colour), 2),
                QBrush(Qt.BrushStyle.NoBrush),
            )

            marker.setZValue(10)
            self.overlay_items.append(marker)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.panning = True
            self.last_pan_position = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.pixmap_item is None:
                event.accept()
                return

            position = self.mapToScene(
                event.position().toPoint()
            )

            if self.scene_object.sceneRect().contains(position):
                self.imageClicked.emit(
                    position.x(),
                    position.y(),
                )

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.panning and self.last_pan_position is not None:
            current = event.position().toPoint()
            delta = current - self.last_pan_position

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

            self.last_pan_position = current
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.panning = False
            self.last_pan_position = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            wheel_delta = event.angleDelta().y()

            if wheel_delta == 0:
                event.accept()
                return

            increment = 0.1 if wheel_delta > 0 else -0.1

            self.set_zoom(
                round(self.zoom_value + increment, 1),
                anchor_global_position=event.globalPosition().toPoint(),
            )

            event.accept()
            return

        super().wheelEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        if self.zoom_value == 1.0:
            self._apply_zoom()


# ---------------------------------------------------------------------
# Manual workflow
# ---------------------------------------------------------------------

class ManualDialog(QDialog):
    """Manual quadrat selection, cropping, segmentation and label export."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Easy Quadrat — Manual")

        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)

        self._parent_content: QWidget | None = None
        self._parent_content_enabled = True

        self.input_files: list[Path] = []
        self.file_metadata: dict[Path, dict] = {}
        self.current_index = 0

        self.current_path: Path | None = None
        self.current_format = ""
        self.current_pages = 1

        self.reference_image: Image.Image | None = None
        self.points: list[tuple[float, float]] = []
        self.polygon_complete = False

        self.custom_points_count = DEFAULT_CUSTOM_VERTICES
        self._last_point_selection = "2-Sq."

        self.output_policy: str | None = None
        self.jpegtran = find_jpegtran()
        self.log_started = False

        self._completion_notified = False
        self._processed_count = 0
        self._skipped_count = 0

        # None means the user has not yet answered the MPO question during
        # this input-folder session. True/False is the remembered decision.
        self._mpo_session_decision: bool | None = None

        self._build_ui()
        self._set_responsive_geometry()
        self._lock_parent_content()
        self._set_image_controls_enabled(False)

    # -----------------------------------------------------------------
    # Responsive window and parent locking
    # -----------------------------------------------------------------

    def _set_responsive_geometry(self) -> None:
        screen = (
            self.parentWidget().screen()
            if self.parentWidget() is not None
            else None
        )

        if screen is None:
            screen = QApplication.primaryScreen()

        if screen is None:
            self.setMinimumSize(720, 500)
            self.resize(1180, 780)
            return

        area = screen.availableGeometry()

        min_width = min(850, max(650, area.width() - 40))
        min_height = min(620, max(460, area.height() - 40))

        width = min(1400, max(min_width, area.width() - 40))
        height = min(900, max(min_height, area.height() - 40))

        self.setMinimumSize(min_width, min_height)
        self.resize(width, height)

        self.move(
            area.x() + (area.width() - width) // 2,
            area.y() + (area.height() - height) // 2,
        )

    def _lock_parent_content(self) -> None:
        parent = self.parentWidget()

        if parent is None:
            return

        central_method = getattr(parent, "centralWidget", None)

        if callable(central_method):
            central = central_method()

            if isinstance(central, QWidget):
                self._parent_content = central
                self._parent_content_enabled = central.isEnabled()
                central.setEnabled(False)

    def _unlock_parent_content(self) -> None:
        if self._parent_content is not None:
            self._parent_content.setEnabled(
                self._parent_content_enabled
            )
            self._parent_content = None

    # -----------------------------------------------------------------
    # User interface
    # -----------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(10)

        heading = QLabel("Manual")
        heading.setObjectName("dialogTitle")
        heading.setStyleSheet(
            "font-size: 22px; "
            "font-weight: 700; "
            "color: #57863c;"
        )

        outer.addWidget(heading)

        folders_row = QHBoxLayout()
        folders_row.setSpacing(8)

        input_button = QPushButton("Input folder…")
        input_button.clicked.connect(self._choose_input_folder)

        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("Input image folder path")

        output_button = QPushButton("Output folder…")
        output_button.clicked.connect(self._choose_output_folder)

        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Output folder path")

        folders_row.addWidget(input_button)
        folders_row.addWidget(self.input_edit, 2)
        folders_row.addWidget(output_button)
        folders_row.addWidget(self.output_edit, 2)

        outer.addLayout(folders_row)

        # Keep image information available, but place it beside the action
        # buttons instead of consuming a separate layout row.
        self.image_info = QLabel("Choose an input folder.")
        self.image_info.setObjectName("sectionDescription")
        self.image_info.setWordWrap(True)
        self.image_info.setMinimumWidth(180)
        self.image_info.setToolTip(
            "Image information includes the actual decoded format, "
            "which may differ from the filename extension."
        )

        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_points)

        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self._skip_image)

        self.jump_button = QPushButton("Jump")
        self.jump_button.clicked.connect(self._jump_to_image)

        self.crop_button = QPushButton("Crop")
        self.crop_button.setObjectName("primaryButton")
        self.crop_button.clicked.connect(self._crop_current_image)

        action_row.addWidget(self.reset_button)
        action_row.addWidget(self.skip_button)
        action_row.addWidget(self.jump_button)
        action_row.addWidget(self.crop_button)

        # A clear gap separates Crop from the image-information text.
        action_row.addSpacing(32)
        action_row.addWidget(self.image_info, 1)

        outer.addLayout(action_row)

        options_row = QHBoxLayout()
        options_row.setSpacing(10)

        options_row.addWidget(QLabel("Points"))

        self.point_combo = QComboBox()
        self.point_combo.addItems(
            ["2-Sq."]
            + [str(number) for number in range(3, 21)]
            + ["Custom"]
        )
        self.point_combo.setCurrentIndex(0)
        self.point_combo.setToolTip(
            "Choose 2-Sq. to define a square using its diagonal, "
            "select 3–20 vertices, or choose Custom to enter "
            "a number from 3 to 1000."
        )
        self.point_combo.currentTextChanged.connect(
            self._points_mode_changed
        )

        options_row.addWidget(self.point_combo)

        options_row.addWidget(QLabel("Line"))

        self.line_colour_combo = QComboBox()
        self._populate_colours(self.line_colour_combo)
        self.line_colour_combo.setCurrentText("Red")
        self.line_colour_combo.currentTextChanged.connect(
            self._redraw_points
        )

        options_row.addWidget(self.line_colour_combo)

        options_row.addWidget(QLabel("Background"))

        self.background_colour_combo = QComboBox()
        self._populate_colours(self.background_colour_combo)
        self.background_colour_combo.setCurrentText("Black")

        options_row.addWidget(self.background_colour_combo)

        self.exif_orientation_checkbox = QCheckBox("EXIF Orientation")
        self.exif_orientation_checkbox.setChecked(False)
        self.exif_orientation_checkbox.setToolTip(
            EXIF_ORIENTATION_TOOLTIP
        )
        self.exif_orientation_checkbox.toggled.connect(
            self._toggle_exif_orientation
        )

        options_row.addWidget(self.exif_orientation_checkbox)

        options_row.addWidget(QLabel("Zoom"))

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 100)
        self.zoom_slider.setSingleStep(1)
        self.zoom_slider.setPageStep(10)
        self.zoom_slider.setValue(10)
        self.zoom_slider.setMinimumWidth(64)
        self.zoom_slider.setToolTip(
            "Zoom from 1× to 10×. You can also hold Ctrl and use "
            "the mouse wheel over the image."
        )
        self.zoom_slider.valueChanged.connect(
            self._zoom_slider_changed
        )

        options_row.addWidget(self.zoom_slider, 1)
        outer.addLayout(options_row)

        # The space previously used by the separate image-info row and
        # footer is now available to the image canvas.
        self.image_view = ManualImageView()
        self.image_view.imageClicked.connect(self._add_point)
        self.image_view.zoomChanged.connect(self._sync_zoom_slider)

        outer.addWidget(self.image_view, 1)

        self.status_label = QLabel(
            "Click on the image to define the quadrat vertices."
        )
        self.status_label.setObjectName("sectionDescription")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        # The Back to main menu button and its footer row have been removed.
        # Users can close this standard window using its title-bar close button.

        self.reset_button.setEnabled(False)
        self.crop_button.setEnabled(False)
        self.skip_button.setEnabled(False)
        self.jump_button.setEnabled(False)

    @staticmethod
    def _populate_colours(combo: QComboBox) -> None:
        for name, colour in PALETTE_COLOURS:
            combo.addItem(name, colour)

    def _zoom_slider_changed(self, value: int) -> None:
        self.image_view.set_zoom(value / 10.0)

    def _sync_zoom_slider(self, value: float) -> None:
        """Synchronize the slider without triggering another zoom operation."""

        target = max(10, min(100, round(value * 10)))

        if self.zoom_slider.value() == target:
            return

        previous_state = self.zoom_slider.blockSignals(True)

        try:
            self.zoom_slider.setValue(target)
        finally:
            self.zoom_slider.blockSignals(previous_state)

    def _set_image_controls_enabled(self, enabled: bool) -> None:
        has_files = bool(self.input_files)
        has_points = bool(self.points)

        mpo_declined = (
            self.current_format == "MPO"
            and self._mpo_session_decision is False
        )

        # Once MPO processing is declined for this session, do not let the
        # user waste time annotating a file that cannot be processed.
        selection_enabled = enabled and not mpo_declined

        self.reset_button.setEnabled(
            selection_enabled and has_points
        )
        self.skip_button.setEnabled(enabled and has_files)
        self.jump_button.setEnabled(enabled and has_files)

        self.point_combo.setEnabled(selection_enabled)
        self.line_colour_combo.setEnabled(selection_enabled)
        self.background_colour_combo.setEnabled(selection_enabled)
        self.exif_orientation_checkbox.setEnabled(selection_enabled)
        self.zoom_slider.setEnabled(enabled)

        self.crop_button.setEnabled(
            selection_enabled and self.polygon_complete
        )

    # -----------------------------------------------------------------
    # Folder selection
    # -----------------------------------------------------------------

    def _choose_input_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select input image folder",
            self.input_edit.text() or str(Path.home()),
        )

        if folder:
            self.input_edit.setText(folder)
            self._refresh_images()

    def _choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_edit.text() or str(Path.home()),
        )

        if folder:
            self.output_edit.setText(folder)
            self.log_started = False

    def _refresh_images(self) -> None:
        self._close_reference_image()

        self.input_files = []
        self.file_metadata = {}
        self.current_index = 0
        self.current_path = None
        self.current_format = ""
        self.current_pages = 1

        self.output_policy = None
        self.log_started = False
        self._completion_notified = False
        self._processed_count = 0
        self._skipped_count = 0

        # A new folder selection starts a new MPO-decision session.
        self._mpo_session_decision = None

        self.points = []
        self.polygon_complete = False
        self.image_view.clear_image()

        folder_text = self.input_edit.text().strip()

        if not folder_text:
            self.image_info.clear()
            self.status_label.setText(
                "No input folder has been selected."
            )
            self._set_image_controls_enabled(False)
            return

        folder = Path(folder_text)

        try:
            candidates = list_manual_images(folder)
        except (OSError, ValueError) as exc:
            self.image_info.setText(str(exc))
            self.status_label.setText("The input folder is not valid.")
            self._set_image_controls_enabled(False)

            self._show_message(
                "Invalid input folder",
                str(exc),
                QMessageBox.Icon.Warning,
            )
            return

        if not candidates:
            self.image_info.setText("No supported images found.")
            self.status_label.setText(
                "The selected folder contains no supported image files."
            )
            self._set_image_controls_enabled(False)

            self._show_message(
                "No images found",
                "The selected folder contains no supported image files.",
                QMessageBox.Icon.Information,
            )
            return

        failed: list[str] = []

        for path in candidates:
            try:
                meta = inspect_image(path)
                self.file_metadata[path] = meta
                self.input_files.append(path)
            except Exception:
                failed.append(path.name)

        if not self.input_files:
            self.image_info.setText("No readable images found.")
            self.status_label.setText(
                "The supported image files could not be read."
            )
            self._set_image_controls_enabled(False)

            self._show_message(
                "No readable images",
                "The selected folder contains image files, "
                "but none could be read.",
                QMessageBox.Icon.Warning,
                details="\n".join(failed),
            )
            return

        self.current_index = 0
        self._open_current_image()

        if failed:
            self.status_label.setText(
                f"{len(self.input_files)} readable image(s) found. "
                f"{len(failed)} file(s) could not be read and were excluded."
            )

    def _close_reference_image(self) -> None:
        if self.reference_image is not None:
            self.reference_image.close()
            self.reference_image = None

    # -----------------------------------------------------------------
    # Current image and completion
    # -----------------------------------------------------------------

    def _open_current_image(self) -> None:
        self._close_reference_image()

        self.points = []
        self.polygon_complete = False

        if not self.input_files:
            self.image_view.clear_image()
            self.image_info.setText("No images are available.")
            self.status_label.setText(
                "Choose an input folder containing readable images."
            )
            self._set_image_controls_enabled(False)
            return

        if self.current_index >= len(self.input_files):
            self.image_view.clear_image()
            self.image_info.setText("Completed — no more images.")
            self.status_label.setText(
                "All images in the input list have been considered."
            )
            self._set_image_controls_enabled(False)
            self._notify_completion()
            return

        self.current_path = self.input_files[self.current_index]
        meta = self.file_metadata[self.current_path]

        self.current_format = meta["format"]
        self.current_pages = meta["frames"]

        try:
            self.reference_image = load_first_frame(
                self.current_path,
                self.exif_orientation_checkbox.isChecked(),
            )

            self.image_view.set_image(self.reference_image)
            self._sync_zoom_slider(1.0)

            info_text = (
                f"Image {self.current_index + 1} / {len(self.input_files)}: "
                f"{self.current_path.name} · "
                f"{self.reference_image.width} × "
                f"{self.reference_image.height} · "
                f"Format: {self.current_format}"
            )

            if self.current_format == "MPO":
                info_text += (
                    f" · {self.current_pages} pictures; "
                    "primary picture only will be exported"
                )
                self.image_info.setToolTip(
                    "This file is an MPO multi-picture container even if "
                    "its filename ends in .JPG. The current annotation is "
                    "for the primary picture. The original file is not "
                    "modified; secondary pictures remain in the source file."
                )
            else:
                self.image_info.setToolTip(
                    "Image dimensions and the actual decoded format."
                )

            self.image_info.setText(info_text)

            if self.exif_orientation_checkbox.isChecked():
                status_text = (
                    "EXIF Orientation is enabled: image orientation is "
                    "corrected before display, cropping, segmentation "
                    "and label creation."
                )
            else:
                status_text = (
                    "Click on the image to define the quadrat vertices. "
                    "Right-drag to pan; Ctrl+mouse wheel to zoom."
                )

            if self.current_format == "MPO":
                status_text += (
                    " This MPO contains multiple pictures. The primary "
                    "picture is the one shown and annotated; the original "
                    "MPO remains unchanged."
                )
            elif self.current_pages > 1:
                status_text = (
                    f"This file contains {self.current_pages} frames/pages. "
                    "The selected polygon will be applied to each frame, "
                    "scaled proportionally if dimensions differ."
                )

            if (
                self.current_format == "MPO"
                and self._mpo_session_decision is False
            ):
                status_text = (
                    "MPO processing was declined for this input-folder "
                    "session. This image will not be processed. Use Skip "
                    "to continue to the next image. The question will not "
                    "be shown again during this session."
                )

            self.status_label.setText(status_text)

            self._set_image_controls_enabled(True)
            self._update_selection_status()

        except Exception as exc:
            self.image_view.clear_image()
            self._set_image_controls_enabled(True)
            self.crop_button.setEnabled(False)
            self.reset_button.setEnabled(False)

            self.image_info.setText(self.current_path.name)
            self.status_label.setText(
                f"Could not open this image: "
                f"{type(exc).__name__}: {exc}. Use Skip to continue."
            )

    def _notify_completion(self) -> None:
        """Show one completion notification per input-folder session."""

        if self._completion_notified:
            return

        self._completion_notified = True

        message = (
            "Completed! No more images remain in the input list.\n\n"
            f"Successfully processed: {self._processed_count}\n"
            f"Skipped: {self._skipped_count}"
        )

        self._show_message(
            "Completed!",
            message,
            QMessageBox.Icon.Information,
        )

    # -----------------------------------------------------------------
    # Point and polygon selection
    # -----------------------------------------------------------------

    def _points_required(self) -> int:
        selection = self.point_combo.currentText()

        if selection == "2-Sq.":
            return 2

        if selection == "Custom":
            return self.custom_points_count

        return int(selection)

    def _add_point(self, x: float, y: float) -> None:
        if (
            self.reference_image is None
            or self.polygon_complete
            or (
                self.current_format == "MPO"
                and self._mpo_session_decision is False
            )
        ):
            return

        width = self.reference_image.width
        height = self.reference_image.height

        if not (0 <= x < width and 0 <= y < height):
            return

        self.points.append((float(x), float(y)))

        if (
            self.point_combo.currentText() == "2-Sq."
            and len(self.points) == 2
        ):
            try:
                generated_points = square_from_diagonal(
                    self.points[0],
                    self.points[1],
                )
            except ValueError as exc:
                self.points = []
                self._redraw_points()
                self.status_label.setText(
                    f"{exc} Please select the diagonal points again."
                )
                return

            if any(
                px < 0 or py < 0 or px >= width or py >= height
                for px, py in generated_points
            ):
                self.points = []
                self._redraw_points()
                self.status_label.setText(
                    "The square derived from that diagonal extends beyond "
                    "the image. Reset and choose a shorter diagonal, or "
                    "select 3 or more points."
                )
                return

            self.points = generated_points
            self.polygon_complete = True

        elif len(self.points) >= self._points_required():
            self.polygon_complete = True

        self._redraw_points()
        self._update_selection_status()

    def _points_mode_changed(self, selection: str) -> None:
        """Reset selection when the point mode changes."""

        if (
            self.current_format == "MPO"
            and self._mpo_session_decision is False
        ):
            return

        if selection == "Custom":
            selected_count, accepted = QInputDialog.getInt(
                self,
                "Custom polygon",
                (
                    "Number of vertices "
                    f"({MIN_CUSTOM_VERTICES}–{MAX_CUSTOM_VERTICES}):"
                ),
                self.custom_points_count,
                MIN_CUSTOM_VERTICES,
                MAX_CUSTOM_VERTICES,
                1,
            )

            if not accepted:
                self.point_combo.blockSignals(True)
                self.point_combo.setCurrentText(
                    self._last_point_selection
                )
                self.point_combo.blockSignals(False)
                return

            self.custom_points_count = selected_count

        self._last_point_selection = selection
        self._reset_points()

    def _toggle_exif_orientation(self, checked: bool) -> None:
        """Reload the image so selection never mixes coordinate systems."""

        self._open_current_image()

    def _redraw_points(self, *args) -> None:
        colour = self.line_colour_combo.currentData() or "#ff0000"

        self.image_view.show_polygon(
            self.points,
            colour,
            self.polygon_complete,
        )

    def _update_selection_status(self) -> None:
        """Update status text while respecting the remembered MPO decision."""

        if (
            self.current_format == "MPO"
            and self._mpo_session_decision is False
        ):
            self.status_label.setText(
                "MPO processing was declined for this input-folder session. "
                "This image cannot be processed. Use Skip to continue."
            )
            self.reset_button.setEnabled(False)
            self.crop_button.setEnabled(False)
            return

        required = self._points_required()
        selection = self.point_combo.currentText()

        if self.polygon_complete:
            self.status_label.setText(
                f"Quadrat defined with {len(self.points)} vertices. "
                "Select Crop to save the rectangular crop, segmented image "
                "and COCO/YOLO labels."
            )
        else:
            remaining = max(0, required - len(self.points))

            if selection == "2-Sq.":
                instruction = (
                    "Select the two opposite vertices of the square's diagonal."
                )
            elif selection == "Custom":
                instruction = (
                    f"Custom polygon: select {required} vertices."
                )
            else:
                instruction = f"Select {required} polygon vertices."

            self.status_label.setText(
                f"{instruction} Selected {len(self.points)} of {required} "
                f"required point(s); {remaining} more click(s) needed."
            )

        self.reset_button.setEnabled(bool(self.points))
        self.crop_button.setEnabled(self.polygon_complete)

    def _reset_points(self) -> None:
        if (
            self.current_format == "MPO"
            and self._mpo_session_decision is False
        ):
            return

        self.points = []
        self.polygon_complete = False

        self.image_view.show_polygon(
            [],
            self.line_colour_combo.currentData() or "#ff0000",
            False,
        )

        self.reset_button.setEnabled(False)
        self.crop_button.setEnabled(False)

        if self.reference_image is not None:
            selection = self.point_combo.currentText()

            if selection == "2-Sq.":
                message = (
                    "Selection reset. Select two opposite vertices "
                    "of the square's diagonal."
                )
            elif selection == "Custom":
                message = (
                    f"Selection reset. Select {self.custom_points_count} "
                    "vertices for the custom polygon."
                )
            else:
                message = (
                    "Selection reset. Click on the image to define "
                    "the quadrat vertices."
                )

            self.status_label.setText(message)

    def _skip_image(self) -> None:
        if (
            not self.input_files
            or self.current_index >= len(self.input_files)
        ):
            return

        if self.current_path is not None:
            self._log_event(
                f"Skipped image: {self.current_path.name}"
            )

        self._skipped_count += 1
        self.current_index += 1
        self._open_current_image()

    def _skip_declined_mpo(self) -> None:
        """
        Skip the current MPO after the user declines MPO processing.

        The decision is session-wide; this helper does not reset it and
        therefore cannot cause the confirmation to appear again.
        """

        if self.current_path is not None:
            self._log_event(
                f"Skipped MPO image: {self.current_path.name}; "
                "MPO processing was declined for this input-folder session."
            )

        self._skipped_count += 1
        self.current_index += 1
        self._open_current_image()

    def _jump_to_image(self) -> None:
        if not self.input_files:
            return

        selected, accepted = QInputDialog.getInt(
            self,
            "Jump to image",
            f"Image number (1–{len(self.input_files)}):",
            self.current_index + 1,
            1,
            len(self.input_files),
            1,
        )

        if accepted:
            self.current_index = selected - 1
            self._open_current_image()

    # -----------------------------------------------------------------
    # Centralized output policy
    # -----------------------------------------------------------------

    def _choose_output_policy(self) -> str:
        """
        Ask once per session when JPEG/MPO sources are present.

        Other input formats use lossless output. Keep JPEG preserves JPEG
        output for compatible JPEG/MPO-derived single-frame outputs, while
        eligible operations may use coefficient-domain JPEG processing
        without re-encoding. Other source formats remain lossless.
        """

        if self.output_policy is not None:
            return self.output_policy

        jpeg_output_available = any(
            meta["format"] in {"JPEG", "MPO"}
            for meta in self.file_metadata.values()
        )

        if not jpeg_output_available:
            self.output_policy = "lossless"
            return self.output_policy

        message = (
            "Choose the output policy for this Manual session.\n\n"
            "Lossless output (recommended): the processed crop and "
            "segmented raster are saved with a lossless encoder and their "
            "decoded pixels are verified before the final files are "
            "replaced. JPEG crops use coefficient cropping only when "
            "the central module can verify that path; otherwise, a "
            "lossless format such as PNG or TIFF is used.\n\n"
            "Keep JPEG: JPEG/MPO-derived single-frame outputs remain JPEG "
            "when their pixel mode supports it. When JPEG re-encoding is "
            "required, the central saver reuses the source quantization "
            "tables and chroma subsampling when supported. Re-encoding "
            "remains lossy, so decoded pixel values may change. Eligible "
            "JPEG operations may instead use coefficient-domain processing "
            "without JPEG re-encoding. Other source formats remain lossless.\n\n"
            "For MPO, the annotation applies only to the primary picture. "
            "Secondary pictures remain untouched in the original file.\n\n"
            f"Input collection: {len(self.input_files)} readable image(s). "
            "This selection is remembered for this Manual session."
        )

        box = QMessageBox(
            QMessageBox.Icon.Warning,
            "Choose image output policy",
            message,
            QMessageBox.StandardButton.NoButton,
            self,
        )

        lossless_button = box.addButton(
            "Lossless output (recommended)",
            QMessageBox.ButtonRole.AcceptRole,
        )

        jpeg_button = box.addButton(
            "Keep JPEG (lossy if re-encoded)",
            QMessageBox.ButtonRole.ActionRole,
        )

        cancel_button = box.addButton(
            "Cancel",
            QMessageBox.ButtonRole.RejectRole,
        )

        box.setDefaultButton(lossless_button)
        box.setEscapeButton(cancel_button)
        box.setWindowModality(Qt.WindowModality.WindowModal)
        box.exec()

        clicked = box.clickedButton()

        if clicked is lossless_button:
            self.output_policy = "lossless"
        elif clicked is jpeg_button:
            # Use the same canonical policy name as Fixed Frame and the
            # central output module, not the former Manual-only alias.
            self.output_policy = "keep_jpeg"
        else:
            self.output_policy = "cancel"

        return self.output_policy

    def _label_stem(self, path: Path) -> str:
        """Disambiguate labels where input images share a filename stem."""

        matching = [
            item for item in self.input_files
            if item.stem.casefold() == path.stem.casefold()
        ]

        if len(matching) > 1:
            return (
                f"{path.stem}_from_"
                f"{path.suffix.lstrip('.').casefold()}"
            )

        return path.stem

    # -----------------------------------------------------------------
    # Shared output planning
    # -----------------------------------------------------------------

    def _create_output_plan(
        self,
        path: Path,
        crop_box: tuple[int, int, int, int],
        output_frames: int,
    ) -> dict:
        """Delegate output format, filename and strategy centrally."""

        output_root = Path(
            self.output_edit.text().strip()
        ).expanduser()

        layout = prepare_output_layout(
            output_root,
            ProcessingMethod.MANUAL,
        )

        if (
            layout.cropped_images is None
            or layout.segmented_images is None
            or layout.labels is None
        ):
            raise RuntimeError(
                "The Manual output folders could not be prepared."
            )

        orientation_normalized = (
            self.exif_orientation_checkbox.isChecked()
        )

        crop_mode = self.reference_image.mode
        segmented_mode = _segmented_output_mode(path)

        crop_plan = plan_output(
            source_path=path,
            output_directory=layout.cropped_images,
            prefix="C",
            policy=normalize_policy(self.output_policy),
            operation="crop",
            output_mode=crop_mode,
            output_frames=output_frames,
            crop_box=crop_box,
            orientation_normalized=orientation_normalized,
            jpegtran=self.jpegtran,
        )

        segmented_plan = plan_output(
            source_path=path,
            output_directory=layout.segmented_images,
            prefix="S",
            policy=normalize_policy(self.output_policy),
            operation="segmented",
            output_mode=segmented_mode,
            output_frames=output_frames,
            orientation_normalized=orientation_normalized,
            jpegtran=self.jpegtran,
        )

        return {
            "crop_plan": crop_plan,
            "seg_plan": segmented_plan,
            "labels_folder": layout.labels,
        }

    # -----------------------------------------------------------------
    # COCO and YOLO labels
    # -----------------------------------------------------------------

    def _write_labels(
        self,
        path: Path,
        points: list[tuple[float, float]],
        width: int,
        height: int,
        labels_folder: Path,
    ) -> tuple[Path, Path]:
        """Write COCO JSON and YOLO polygon labels in selection coordinates."""

        if len(points) < 3:
            raise ValueError(
                "At least three vertices are required for polygon labels."
            )

        labels_folder.mkdir(parents=True, exist_ok=True)

        stem = self._label_stem(path)
        json_path = labels_folder / f"{stem}.json"
        yolo_path = labels_folder / f"{stem}.txt"

        clean_points = [
            (
                max(0.0, min(float(width - 1), float(x))),
                max(0.0, min(float(height - 1), float(y))),
            )
            for x, y in points
        ]

        xs = [point[0] for point in clean_points]
        ys = [point[1] for point in clean_points]

        left = min(xs)
        top = min(ys)
        right = max(xs)
        bottom = max(ys)

        flat_segmentation = [
            coordinate
            for point in clean_points
            for coordinate in point
        ]

        coco_data = {
            "images": [
                {
                    "file_name": path.name,
                    "id": self.current_index + 1,
                    "width": width,
                    "height": height,
                }
            ],
            "annotations": [
                {
                    "segmentation": [flat_segmentation],
                    "iscrowd": 0,
                    "image_id": self.current_index + 1,
                    "bbox": [
                        left,
                        top,
                        right - left,
                        bottom - top,
                    ],
                    "area": polygon_area(clean_points),
                    "category_id": 1,
                    "id": self.current_index + 1,
                }
            ],
            "categories": [
                {
                    "supercategory": "Quadrat",
                    "id": 1,
                    "name": "Quadrat",
                }
            ],
        }

        _write_json(json_path, coco_data)

        yolo_values = []

        for x, y in clean_points:
            yolo_values.append(x / max(1, width))
            yolo_values.append(y / max(1, height))

        yolo_path.write_text(
            "0 "
            + " ".join(f"{value:.8f}" for value in yolo_values)
            + "\n",
            encoding="utf-8",
        )

        return json_path, yolo_path

    # -----------------------------------------------------------------
    # MPO confirmation: once per input-folder session
    # -----------------------------------------------------------------

    def _confirm_mpo_policy_once(self) -> bool:
        """
        Ask once whether MPO images may be processed.

        The answer is retained for every MPO image in the current input
        folder session. Yes permits processing later MPO images without
        another dialog. No prevents processing any later MPO image without
        repeating the dialog.
        """

        if self._mpo_session_decision is not None:
            return self._mpo_session_decision

        answer = QMessageBox.question(
            self,
            "MPO contains multiple pictures",
            (
                "This file contains multiple pictures, despite its "
                "filename extension possibly being .JPG.\n\n"
                "The Manual canvas displays the primary picture. Only "
                "that picture will be included in the derived Cropped "
                "images and Segmented images outputs.\n\n"
                "The original MPO file will not be modified. Its "
                "secondary pictures remain in the original file.\n\n"
                "Your answer will apply to all MPO files in this input-folder "
                "session. You will not be asked again unless you select a "
                "new input folder.\n\n"
                "Continue?"
            ),
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        self._mpo_session_decision = (
            answer == QMessageBox.StandardButton.Yes
        )

        self._log_event(
            "MPO session decision: "
            + (
                "Yes — process the primary picture of MPO files."
                if self._mpo_session_decision
                else "No — do not process MPO files during this session."
            )
        )

        return self._mpo_session_decision

    # -----------------------------------------------------------------
    # Crop and save the current image using the central module
    # -----------------------------------------------------------------

    def _crop_current_image(self) -> None:
        if (
            self.current_path is None
            or self.reference_image is None
            or not self.polygon_complete
        ):
            return

        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()

        if not input_text:
            self._show_message(
                "Input folder required",
                "Select an input folder before cropping.",
                QMessageBox.Icon.Warning,
            )
            return

        if not output_text:
            self._show_message(
                "Output folder required",
                "Select an output folder before cropping.",
                QMessageBox.Icon.Warning,
            )
            return

        input_folder = Path(input_text).expanduser().resolve()
        output_folder = Path(output_text).expanduser().resolve()

        if input_folder == output_folder:
            self._show_message(
                "Separate folders required",
                "The input and output folders must be different.",
                QMessageBox.Icon.Warning,
            )
            return

        # MPO confirmation is session-wide, not per path.
        if self.current_format == "MPO":
            if not self._confirm_mpo_policy_once():
                self._skip_declined_mpo()
                return

        policy = self._choose_output_policy()

        if policy == "cancel":
            self.output_policy = None
            return

        try:
            crop_box = polygon_crop_box(
                self.points,
                self.reference_image.width,
                self.reference_image.height,
            )

            output_directory = Path(
                output_text
            ).expanduser()

            layout = prepare_output_layout(
                output_directory,
                ProcessingMethod.MANUAL,
            )

            source_metadata = self.file_metadata[self.current_path]
            source_format = source_metadata["format"]

            # MPO annotations refer to the primary picture only. Other
            # supported multi-frame containers retain the existing workflow.
            frames_to_process = (
                1
                if source_format == "MPO"
                else self.current_pages
            )

            plan = self._create_output_plan(
                self.current_path,
                crop_box,
                frames_to_process,
            )

        except Exception as exc:
            self._show_message(
                "Output planning error",
                f"{self.current_path.name}\n\n"
                f"{type(exc).__name__}: {exc}",
                QMessageBox.Icon.Critical,
            )
            return

        labels_folder = plan["labels_folder"]
        label_stem = self._label_stem(self.current_path)

        candidate_outputs = [
            plan["crop_plan"].output_path,
            plan["seg_plan"].output_path,
            labels_folder / f"{label_stem}.json",
            labels_folder / f"{label_stem}.txt",
        ]

        resolved_outputs = [
            str(path.resolve()).casefold()
            for path in candidate_outputs
        ]

        if len(set(resolved_outputs)) != len(resolved_outputs):
            self._show_message(
                "Output filename collision",
                "The crop, segmented image, and label outputs do not have "
                "unique destination paths. Rename the affected input file "
                "or choose a different output arrangement.",
                QMessageBox.Icon.Critical,
            )
            return

        existing = [
            path for path in candidate_outputs if path.exists()
        ]

        if existing:
            answer = QMessageBox.question(
                self,
                "Overwrite existing outputs?",
                f"{len(existing)} output file(s) for this image "
                "already exist. Replace them?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if answer != QMessageBox.StandardButton.Yes:
                return

        crops: list[Image.Image] = []
        segmented_frames: list[Image.Image] = []
        metadata: dict = {}

        reference_width = self.reference_image.width
        reference_height = self.reference_image.height
        orientation_enabled = (
            self.exif_orientation_checkbox.isChecked()
        )

        try:
            with Image.open(self.current_path) as source:
                actual_format = (source.format or "").upper()

                if actual_format != source_format:
                    raise RuntimeError(
                        "The decoded source format changed after the "
                        "input folder was inspected."
                    )

                pages = get_frame_count(source)
                frames_to_process = (
                    1 if actual_format == "MPO" else pages
                )

                if frames_to_process != plan["crop_plan"].output_frames:
                    raise RuntimeError(
                        "The source frame count changed after output planning. "
                        "Please reload the input folder and try again."
                    )

                for frame_index in range(frames_to_process):
                    source.seek(frame_index)
                    raw_frame = source.copy()

                    try:
                        if orientation_enabled:
                            normalized = ImageOps.exif_transpose(raw_frame)

                            try:
                                working_frame = normalized.copy()
                                working_frame.info.update(normalized.info)
                            finally:
                                if normalized is not raw_frame:
                                    normalized.close()
                        else:
                            working_frame = raw_frame.copy()
                            working_frame.info.update(raw_frame.info)

                        try:
                            width = working_frame.width
                            height = working_frame.height

                            scaled_points = [
                                (
                                    x * width / reference_width,
                                    y * height / reference_height,
                                )
                                for x, y in self.points
                            ]

                            frame_box = polygon_crop_box(
                                scaled_points,
                                width,
                                height,
                            )

                            cropped = working_frame.crop(frame_box)
                            cropped.info.update(working_frame.info)
                            crops.append(cropped)

                            background_rgb = QColor(
                                self.background_colour_combo.currentData()
                            ).getRgb()[:3]

                            segmented = make_segmented_image(
                                working_frame,
                                scaled_points,
                                background_rgb,
                            )

                            segmented_frames.append(segmented)

                            if frame_index == 0:
                                metadata = dict(working_frame.info)

                        finally:
                            working_frame.close()

                    finally:
                        raw_frame.close()

            if not crops or not segmented_frames:
                raise RuntimeError(
                    "No image frames were prepared for output."
                )

            # ---------------------------------------------------------
            # Save crop through the central output module.
            # ---------------------------------------------------------

            crop_plan = plan["crop_plan"]

            try:
                crop_result = save_planned_output(
                    crops,
                    crop_plan,
                    metadata=metadata,
                    source_path=self.current_path,
                    source_mode=source_metadata["mode"],
                    orientation_normalized=orientation_enabled,
                    jpegtran=self.jpegtran,
                )

            except Exception as exc:
                if crop_plan.strategy != "jpegtran":
                    raise

                # If the coefficient route fails, re-plan instead of
                # silently re-encoding a JPEG under the Lossless policy.
                fallback_plan = plan_output(
                    source_path=self.current_path,
                    output_directory=crop_plan.output_path.parent,
                    prefix="C",
                    policy=normalize_policy(self.output_policy),
                    operation="crop",
                    output_mode=crops[0].mode,
                    output_frames=len(crops),
                    crop_box=None,
                    orientation_normalized=orientation_enabled,
                    jpegtran=self.jpegtran,
                )

                fallback_path = fallback_plan.output_path

                if (
                    fallback_path != crop_plan.output_path
                    and fallback_path.exists()
                ):
                    answer = QMessageBox.question(
                        self,
                        "Overwrite fallback crop?",
                        (
                            "The preferred lossless JPEG-cropping route "
                            "could not be verified.\n\n"
                            "A different output format is required, and "
                            "this fallback output already exists:\n"
                            f"{fallback_path.name}\n\n"
                            "Replace it?"
                        ),
                        QMessageBox.StandardButton.Yes
                        | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )

                    if answer != QMessageBox.StandardButton.Yes:
                        return

                crop_result = save_planned_output(
                    crops,
                    fallback_plan,
                    metadata=metadata,
                    source_path=self.current_path,
                    source_mode=source_metadata["mode"],
                    orientation_normalized=orientation_enabled,
                    jpegtran=self.jpegtran,
                )

                plan["crop_plan"] = fallback_plan

                self._log_event(
                    f"JPEG coefficient crop fallback: "
                    f"{self.current_path.name}; "
                    f"reason={type(exc).__name__}: {exc}; "
                    f"fallback={fallback_path.name}"
                )

            # ---------------------------------------------------------
            # Save segmented output through the central output module.
            # ---------------------------------------------------------

            seg_result = save_planned_output(
                segmented_frames,
                plan["seg_plan"],
                metadata=metadata,
                source_path=self.current_path,
                source_mode=source_metadata["mode"],
                orientation_normalized=orientation_enabled,
                jpegtran=self.jpegtran,
            )

            # ---------------------------------------------------------
            # Save COCO/YOLO polygon labels.
            # ---------------------------------------------------------

            self._write_labels(
                self.current_path,
                self.points,
                reference_width,
                reference_height,
                labels_folder,
            )

            mpo_note = (
                "; MPO primary picture only"
                if source_format == "MPO"
                else ""
            )

            self._log_event(
                f"Processed: {self.current_path.name}; "
                f"decoded_format={source_format}; "
                f"crop={crop_result.path.name}; "
                f"crop_strategy={crop_result.strategy}; "
                f"crop_pixels_verified={crop_result.pixels_verified}; "
                f"segmented={seg_result.path.name}; "
                f"segmented_strategy={seg_result.strategy}; "
                f"segmented_pixels_verified={seg_result.pixels_verified}; "
                f"policy={self.output_policy}; "
                f"EXIF Orientation="
                f"{'On' if orientation_enabled else 'Off'}"
                f"{mpo_note}"
            )

            for warning in crop_result.warnings:
                self._log_event(
                    f"Output warning for {crop_result.path.name}: "
                    f"{self._format_manual_output_warning(warning)}"
                )

            for warning in seg_result.warnings:
                self._log_event(
                    f"Output warning for {seg_result.path.name}: "
                    f"{self._format_manual_output_warning(warning)}"
                )

            self._processed_count += 1
            self.current_index += 1

            self._open_current_image()

        except Exception as exc:
            self._log_event(
                f"Error processing {self.current_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )

            self._show_message(
                "Manual cropping error",
                f"{self.current_path.name}\n\n"
                f"{type(exc).__name__}: {exc}",
                QMessageBox.Icon.Critical,
            )

        finally:
            for image in crops:
                image.close()

            for image in segmented_frames:
                image.close()

    # -----------------------------------------------------------------
    # Session log
    # -----------------------------------------------------------------

    @staticmethod
    def _format_manual_output_warning(warning: str) -> str:
        """
        Clarify the central MPO metadata warning in the Manual log.

        The central warning text is left unchanged, so this is a
        presentation-only adjustment for Manual. Other warnings are passed
        through unchanged.
        """

        mpo_metadata_marker = (
            "Container-sensitive EXIF/XMP metadata was omitted "
            "from this standalone MPO-derived image"
        )

        if mpo_metadata_marker not in warning:
            return warning

        omitted_fields = ""
        fields_marker = "Metadata fields omitted:"

        if fields_marker in warning:
            fields_text = warning.split(fields_marker, 1)[1]
            fields_text = fields_text.split(".", 1)[0].strip()

            if fields_text:
                omitted_fields = (
                    f" Fields omitted: {fields_text}."
                )

        return (
            "MPO-derived EXIF/XMP metadata was omitted from this standalone "
            "output because its references may depend on the original MPO "
            "container. The original MPO file was not modified, and "
            "metadata filtering did not alter the raster pixels supplied "
            "to the output saver."
            f"{omitted_fields}"
        )

    def _log_event(self, message: str) -> None:
        output_text = self.output_edit.text().strip()

        if not output_text:
            return

        root = Path(output_text)
        root.mkdir(parents=True, exist_ok=True)

        log_path = root / "Log.txt"

        with log_path.open("a", encoding="utf-8") as log:
            if not self.log_started:
                log.write(
                    "\nEasy Quadrat v2.0.0 — Manual\n"
                    "----------------------------------------\n"
                    f"Session time: "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Input folder: {self.input_edit.text().strip()}\n"
                    f"Output folder: {output_text}\n"
                    f"Output policy: {self.output_policy or 'not selected'}\n"
                    f"EXIF Orientation: "
                    f"{'On' if self.exif_orientation_checkbox.isChecked() else 'Off'}\n\n"
                )

                self.log_started = True

            log.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{message}\n"
            )

    # -----------------------------------------------------------------
    # Dialog helpers and close lifecycle
    # -----------------------------------------------------------------

    def _show_message(
        self,
        title: str,
        message: str,
        icon: QMessageBox.Icon,
        details: str | None = None,
    ) -> None:
        box = QMessageBox(
            icon,
            title,
            message,
            QMessageBox.StandardButton.Ok,
            self,
        )

        if details:
            box.setDetailedText(details)

        box.setWindowModality(Qt.WindowModality.WindowModal)
        box.exec()

    def reject(self) -> None:
        self._close_reference_image()
        self._unlock_parent_content()
        super().reject()

    def done(self, result: int) -> None:
        self._close_reference_image()
        self._unlock_parent_content()
        super().done(result)

    def closeEvent(self, event) -> None:
        self._close_reference_image()
        self._unlock_parent_content()
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._reset_points()
            event.accept()
            return

        if event.key() in {
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        }:
            self._crop_current_image()
            event.accept()
            return

        if event.key() == Qt.Key.Key_Space:
            self._skip_image()
            event.accept()
            return

        super().keyPressEvent(event)