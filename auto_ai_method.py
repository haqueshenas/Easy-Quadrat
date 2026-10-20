"""
Easy Quadrat v2.0.0 — Auto (Deep Learning) image-processing method.

Historical method name in Easy Quadrat v1.0:
    Auto (AI assisted)

The original method uses YOLO instance-segmentation masks to detect
white quadrats, create one cropped image for each detected quadrat, and
create one segmented image for each input image.

The test preview is presentation-only. It stores reduced, detached preview
frames for manual review and never saves output files or alters production
inference/output behavior.

Original implementation:
    https://github.com/haqueshenas/Easy-Quadrat/blob/v1.0/main.py

Historical model guidance:
    Easy Quadrat v1.0 release README.md

Image encoding and output verification are delegated to core.image_output.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Copyright (c) 2026 Abbas Haghshenas
License: MIT License for this source file; see LICENSE.
Third-party components and model weights have separate licensing terms.
See THIRD_PARTY_NOTICES.txt and the application About information.
AI coding assistance: OpenAI GPT-6
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from PySide6.QtCore import QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.image_catalog import list_image_files
from core.image_output import (
    OutputPolicy,
    find_jpegtran,
    inspect_image,
    plan_output,
    save_planned_output,
)
from output_manager import (
    ProcessingMethod,
    ProcessingRunLog,
    prepare_output_layout,
)
from ui.processing_preview import (
    ProcessingPreviewPanel,
    make_preview_base,
    rgb_array_to_qimage,
)
from ui.window_utils import configure_window


# ---------------------------------------------------------------------
# Built-in model definitions and defaults
# ---------------------------------------------------------------------

BUILT_IN_MODELS = {
    "eq1": {
        "label": "EQ1 — Standard (YOLOv8n-seg)",
        "filename": "easyquadrat.pt",
        "description": "Standard Easy Quadrat model based on YOLOv8n-seg.",
    },
    "eq2": {
        "label": "EQ2 — Standard (YOLOv8m-seg)",
        "filename": "easyquadrat2.pt",
        "description": (
            "Larger standard model based on YOLOv8m-seg. "
            "It may require more memory and processing time."
        ),
    },
    "frameless_1": {
        "label": "EQ Frameless 1",
        "filename": "EQ_Frameless_1.pt",
        "description": (
            "A frame-reduced-influence model variant. "
            "Performance depends on the image and quadrat."
        ),
    },
    "frameless_2": {
        "label": "EQ Frameless 2",
        "filename": "EQ_Frameless_2.pt",
        "description": (
            "A second frame-reduced-influence model variant. "
            "Performance depends on the image and quadrat."
        ),
    },
}

CUSTOM_MODEL_KEY = "custom"
MODEL_OPTIONS = (
    ("eq1", BUILT_IN_MODELS["eq1"]["label"]),
    ("eq2", BUILT_IN_MODELS["eq2"]["label"]),
    ("frameless_1", BUILT_IN_MODELS["frameless_1"]["label"]),
    ("frameless_2", BUILT_IN_MODELS["frameless_2"]["label"]),
    (CUSTOM_MODEL_KEY, "Custom model (.pt)"),
)

DEFAULT_CONFIDENCE = 0.25
DEFAULT_MAX_QUADRATS = 1
DEFAULT_BACKGROUND_NAME = "Black"
# Allow the GUI event loop to paint each test stage before the next is sent.
# This delay affects only the test-preview presentation, not model inference or batch processing.
TEST_PREVIEW_STAGE_INTERVAL_SECONDS = 0.55
BACKGROUND_RGB = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Purple": (128, 0, 128),
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
}

EXIF_ORIENTATIONS_THAT_TRANSFORM_PIXELS = frozenset({
    2, 3, 4, 5, 6, 7, 8,
})

_PREVIEW_MASK_COLORS = (
    # Detection overlays are display-only; red matches the other Auto previews.
    (230, 65, 65),
    (240, 166, 50),
    (65, 145, 235),
    (207, 76, 183),
    (42, 190, 190),
    (155, 105, 220),
    (180, 190, 55),
)


# ---------------------------------------------------------------------
# Application/model paths
# ---------------------------------------------------------------------

def _application_resource_root() -> Path:
    """Return the source-project or PyInstaller data root."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parent


def _built_in_model_path(model_key: str) -> Path:
    """Resolve a built-in model by its stable internal key."""

    specification = BUILT_IN_MODELS.get(model_key)
    if specification is None:
        raise ValueError(f"Unknown built-in model key: {model_key!r}.")
    return (
        _application_resource_root()
        / "models"
        / specification["filename"]
    )


def _ultralytics_is_available() -> bool:
    """Check whether the Ultralytics package is discoverable."""

    try:
        return importlib.util.find_spec("ultralytics") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _load_yolo_segmentation_model(model_path: Path):
    """Load a local YOLO model and reject an explicitly non-segmentation task."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The Ultralytics package is required for Auto Deep Learning but "
            "could not be imported. Install a compatible version in the active "
            "Python environment before running this method."
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(
            f"The selected model file does not exist: {model_path}"
        )

    model = YOLO(str(model_path))
    model_task = getattr(model, "task", None)
    if model_task is not None and str(model_task).casefold() != "segment":
        raise ValueError(
            "The selected YOLO model is not an instance-segmentation model. "
            f"Detected model task: {model_task!r}. Choose a segmentation "
            "model trained to produce masks."
        )
    return model


# ---------------------------------------------------------------------
# Input loading and preprocessing
# ---------------------------------------------------------------------

def _load_rgb_image(
    path: Path,
    source_metadata: dict,
) -> tuple[np.ndarray, dict, list[str], list[str]]:
    """
    Load the primary image as an independent 8-bit RGB array.

    EXIF Orientation is applied before inference. Alpha-bearing images are
    composited onto white. High-bit-depth images are rejected instead of
    silently down-converted. Metadata-omission warnings are deferred until
    an output is saved.
    """

    processing_warnings: list[str] = []
    metadata_warnings: list[str] = []

    with Image.open(path) as source:
        source.seek(0)
        source.load()
        original_mode = source.mode
        source_format = (source.format or "").upper()
        mode_upper = original_mode.upper()

        if (
            mode_upper in {"I", "F"}
            or mode_upper.startswith("I;16")
            or mode_upper.startswith("RGB;16")
            or mode_upper.startswith("RGBA;16")
        ):
            raise ValueError(
                f"High-bit-depth image mode {original_mode!r} is not "
                "supported by Auto Deep Learning. The image was not "
                "down-converted because doing so could silently lose "
                "pixel precision."
            )

        bits_per_sample = source_metadata.get("bits_per_sample")
        if (
            source_format == "TIFF"
            and original_mode in {"RGB", "RGBA"}
            and bits_per_sample
            and max(bits_per_sample) > 8
        ):
            raise ValueError(
                "High-bit-depth TIFF input is not supported by Auto Deep "
                f"Learning: {bits_per_sample} bits per sample."
            )

        try:
            exif_object = source.getexif()
            has_exif = bool(source.info.get("exif")) or len(exif_object) > 0
        except Exception:
            has_exif = bool(source.info.get("exif"))

        has_xmp = bool(
            source.info.get("xmp")
            or source.info.get("XML:com.adobe.xmp")
        )
        normalized = ImageOps.exif_transpose(source)

        try:
            normalized.load()
            metadata = {
                "icc_profile": normalized.info.get("icc_profile"),
                "dpi": normalized.info.get("dpi"),
            }
            has_alpha = (
                "A" in normalized.getbands()
                or "transparency" in normalized.info
            )

            if has_alpha:
                rgba = normalized.convert("RGBA")
                try:
                    alpha = rgba.getchannel("A")
                    try:
                        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
                        try:
                            rgb.paste(rgba, mask=alpha)
                            array = np.asarray(rgb, dtype=np.uint8).copy()
                        finally:
                            rgb.close()
                    finally:
                        alpha.close()
                finally:
                    rgba.close()

                processing_warnings.append(
                    "The source contains transparency. Transparent pixels were "
                    "composited onto white before inference; the processed "
                    "image is RGB."
                )
            else:
                rgb = normalized.convert("RGB")
                try:
                    array = np.asarray(rgb, dtype=np.uint8).copy()
                finally:
                    rgb.close()

                if normalized.mode != "RGB":
                    processing_warnings.append(
                        f"The source pixel mode {normalized.mode!r} was "
                        "converted to RGB for model inference."
                    )

            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(
                    "The source image could not be normalized to RGB."
                )
            if min(array.shape[:2]) < 2:
                raise ValueError("The image is too small to process.")
        finally:
            if normalized is not source:
                normalized.close()

    if source_format == "MPO":
        if has_exif or has_xmp:
            metadata_warnings.append(
                "MPO-derived EXIF/XMP metadata was omitted from standalone "
                "derivatives because its references may depend on the original "
                "MPO container. The original MPO file was not modified, and "
                "metadata filtering did not alter the raster pixels supplied "
                "to the output saver."
            )
    else:
        if has_exif:
            metadata_warnings.append(
                "Source EXIF metadata was not copied to AI-derived images. "
                "EXIF Orientation was applied to the decoded pixels where applicable."
            )
        if has_xmp:
            metadata_warnings.append(
                "Source XMP metadata was not copied to AI-derived images."
            )

    return array, metadata, processing_warnings, metadata_warnings


def _orientation_was_normalized(source_metadata: dict) -> bool:
    """Return whether a non-default EXIF Orientation transforms pixels."""

    try:
        orientation = int(source_metadata.get("exif_orientation", 1))
    except (TypeError, ValueError, OverflowError):
        orientation = 1
    return orientation in EXIF_ORIENTATIONS_THAT_TRANSFORM_PIXELS


# ---------------------------------------------------------------------
# YOLO mask extraction
# ---------------------------------------------------------------------

def _resize_binary_mask(
    mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Resize a discrete binary mask using nearest-neighbour interpolation."""

    binary = np.asarray(mask > 0.5, dtype=np.uint8)
    if binary.shape == (height, width):
        return binary.astype(bool, copy=False)

    mask_image = Image.fromarray(binary * 255)
    try:
        resized = mask_image.resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        )
        try:
            return np.asarray(resized, dtype=np.uint8) > 0
        finally:
            resized.close()
    finally:
        mask_image.close()


def _extract_instance_masks(
    results,
    image_shape: tuple[int, int],
) -> list[np.ndarray]:
    """Extract valid boolean instance masks aligned to the RGB image."""

    height, width = image_shape
    instance_masks: list[np.ndarray] = []
    if results is None:
        return instance_masks
    if not isinstance(results, (list, tuple)):
        results = [results]

    for result in results:
        result_masks = getattr(result, "masks", None)
        if result_masks is None:
            continue
        mask_data = getattr(result_masks, "data", None)
        if mask_data is None:
            continue
        if hasattr(mask_data, "detach"):
            mask_data = mask_data.detach()
        if hasattr(mask_data, "cpu"):
            mask_data = mask_data.cpu()
        if hasattr(mask_data, "numpy"):
            mask_data = mask_data.numpy()

        mask_array = np.asarray(mask_data)
        if mask_array.ndim == 2:
            mask_array = mask_array[np.newaxis, :, :]
        if mask_array.ndim != 3:
            continue

        for raw_mask in mask_array:
            if raw_mask.ndim != 2:
                continue
            normalized_mask = _resize_binary_mask(raw_mask, width, height)
            if normalized_mask.any():
                instance_masks.append(normalized_mask)
    return instance_masks


def _predict_instance_masks(
    model,
    image_rgb: np.ndarray,
    confidence: float,
    max_quadrats: int,
) -> list[np.ndarray]:
    """Predict masks from the same decoded raster used for output creation."""

    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    results = model.predict(
        source=image_bgr,
        conf=confidence,
        max_det=max_quadrats,
        save=False,
        retina_masks=True,
        show_labels=False,
        show_boxes=False,
        verbose=False,
    )
    masks = _extract_instance_masks(results, image_rgb.shape[:2])
    return masks[:max_quadrats]


# ---------------------------------------------------------------------
# YOLO-specific preview rendering (display-only arrays)
# ---------------------------------------------------------------------

def _mask_outline(mask: np.ndarray) -> np.ndarray:
    """Return the edge pixels of a 2-D binary mask."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("A preview mask must be two-dimensional.")
    height, width = binary.shape
    if height < 3 or width < 3:
        return binary.copy()

    interior = np.zeros_like(binary)
    interior[1:-1, 1:-1] = (
        binary[1:-1, 1:-1]
        & binary[:-2, 1:-1]
        & binary[2:, 1:-1]
        & binary[1:-1, :-2]
        & binary[1:-1, 2:]
    )
    return binary & ~interior


def _render_detection_overlay(
    preview_base_rgb: np.ndarray,
    masks: list[np.ndarray],
) -> QImage:
    """Render all detected instances with distinct colors and numbered bounds."""

    rendered = np.ascontiguousarray(preview_base_rgb).copy()
    height, width = rendered.shape[:2]
    boxes: list[tuple[int, int, int, int, tuple[int, int, int]]] = []

    for index, full_mask in enumerate(masks, start=1):
        mask = _resize_binary_mask(full_mask, width, height)
        if not mask.any():
            continue
        color = _PREVIEW_MASK_COLORS[(index - 1) % len(_PREVIEW_MASK_COLORS)]
        selected = rendered[mask].astype(np.float32)
        tint = np.asarray(color, dtype=np.float32)
        rendered[mask] = np.clip(
            selected * 0.55 + tint * 0.45,
            0,
            255,
        ).astype(np.uint8)
        rendered[_mask_outline(mask)] = color
        ys, xs = np.nonzero(mask)
        boxes.append((
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
            color,
        ))

    # Draw labels after the mask colors have been applied, so the annotations
    # are layered on top of the final overlay rather than on the source copy.
    with Image.fromarray(rendered) as image:
        draw = ImageDraw.Draw(image)
        for index, (left, top, right, bottom, color) in enumerate(boxes, start=1):
            for offset in range(2):
                x1, y1 = left + offset, top + offset
                x2, y2 = right - offset, bottom - offset
                if x1 <= x2 and y1 <= y2:
                    draw.rectangle((x1, y1, x2, y2), outline=color, width=1)
            badge = str(index)
            text_box = draw.textbbox((left, top), badge)
            text_w = max(12, text_box[2] - text_box[0] + 6)
            text_h = max(12, text_box[3] - text_box[1] + 4)
            badge_top = max(0, top - text_h)
            draw.rectangle(
                (left, badge_top, left + text_w, badge_top + text_h),
                fill=color,
            )
            draw.text((left + 3, badge_top + 1), badge, fill=(255, 255, 255))
        result_rgb = np.asarray(image, dtype=np.uint8).copy()
    return rgb_array_to_qimage(result_rgb)


def _render_combined_mask(
    preview_base_rgb: np.ndarray,
    masks: list[np.ndarray],
) -> QImage:
    """Render the union of all instance masks as a high-contrast binary image."""

    height, width = preview_base_rgb.shape[:2]
    combined = np.zeros((height, width), dtype=bool)
    for mask in masks:
        combined |= _resize_binary_mask(mask, width, height)
    rendered = np.zeros((height, width, 3), dtype=np.uint8)
    rendered[combined] = (245, 245, 245)
    rendered[_mask_outline(combined)] = (70, 255, 95)
    return rgb_array_to_qimage(rendered)


def _render_segmented_preview(
    preview_base_rgb: np.ndarray,
    masks: list[np.ndarray],
    background_rgb: tuple[int, int, int],
) -> QImage:
    """Render a reduced approximation of the segmented output for inspection."""

    rendered = np.ascontiguousarray(preview_base_rgb).copy()
    height, width = rendered.shape[:2]
    combined = np.zeros((height, width), dtype=bool)
    for mask in masks:
        combined |= _resize_binary_mask(mask, width, height)
    rendered[~combined] = np.asarray(background_rgb, dtype=np.uint8)
    rendered[_mask_outline(combined)] = (70, 255, 95)
    return rgb_array_to_qimage(rendered)


def _build_test_preview_frames(
    image_rgb: np.ndarray,
    image_path: Path,
    masks: list[np.ndarray],
    background_rgb: tuple[int, int, int],
) -> tuple[list[tuple[str, QImage]], tuple[int, int, int, int]]:
    """Build only the four review stages, ending at the final segmented image."""

    base = make_preview_base(image_rgb)
    combined_mask = np.zeros(image_rgb.shape[:2], dtype=bool)

    for mask in masks:
        combined_mask |= mask

    if not combined_mask.any():
        raise ValueError("YOLO returned masks, but none defined a valid segment.")

    ys, xs = np.nonzero(combined_mask)
    bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )

    raw_frames: list[tuple[str, QImage]] = [
        (f"Original image · {image_path.name}", rgb_array_to_qimage(base)),
        (
            f"YOLO detections · {len(masks)} quadrat(s)",
            _render_detection_overlay(base, masks),
        ),
        ("Combined instance mask", _render_combined_mask(base, masks)),
        ("Segmented preview", _render_segmented_preview(base, masks, background_rgb)),
    ]

    total = len(raw_frames)
    numbered_frames = [
        (f"{index} / {total} · {stage_name}", frame)
        for index, (stage_name, frame) in enumerate(raw_frames, start=1)
    ]
    return numbered_frames, bbox


# ---------------------------------------------------------------------
# Central output-plan helpers
# ---------------------------------------------------------------------

def _make_plan(
    source_path: Path,
    output_directory: Path,
    prefix: str,
    policy: OutputPolicy,
    operation: str,
    *,
    crop_box: tuple[int, int, int, int] | None,
    orientation_normalized: bool,
    jpegtran: str | None,
):
    """Create an output plan through the central image-output policy."""

    return plan_output(
        source_path=source_path,
        output_directory=output_directory,
        prefix=prefix,
        policy=policy,
        operation=operation,
        output_mode="RGB",
        output_frames=1,
        crop_box=crop_box,
        orientation_normalized=orientation_normalized,
        jpegtran=jpegtran,
    )


def _make_unique_plan(
    plan,
    source_path: Path,
    source_metadata: dict,
    *,
    detection_index: int | None = None,
    disambiguate_source_suffix: bool = False,
):
    """Produce a stable output name while retaining the central output plan."""

    output_path = plan.output_path
    output_stem = output_path.stem
    source_format = str(source_metadata.get("format", "")).upper()
    source_suffix = source_path.suffix.lstrip(".").casefold()
    registered_format = Image.registered_extensions().get(
        source_path.suffix.casefold(),
        "",
    )
    suffix_mismatch = (
        not registered_format
        or registered_format.upper() != source_format
    )
    format_changed = str(plan.output_format).upper() != source_format

    if suffix_mismatch or (format_changed and disambiguate_source_suffix):
        suffix_tag = f"_src-{source_suffix or 'image'}"
        if not output_stem.casefold().endswith(suffix_tag.casefold()):
            output_stem += suffix_tag

    if detection_index is not None:
        output_stem += f"_{detection_index:02d}"

    return replace(
        plan,
        output_path=output_path.with_name(output_stem + output_path.suffix),
    )


def _record_save_warnings(
    run_log: ProcessingRunLog,
    summary_warnings: list[str],
    input_filename: str,
    output_path: Path,
    warnings: tuple[str, ...] | list[str],
) -> None:
    """Record central saver warnings alongside the relevant output name."""

    for warning in warnings:
        summary_warnings.append(
            f"{input_filename} -> {output_path.name}: {warning}"
        )
        run_log.record_warning(
            f"{input_filename} -> {output_path.name}",
            warning,
        )


# ---------------------------------------------------------------------
# Preview/test worker — never saves output
# ---------------------------------------------------------------------

class DeepLearningPreviewWorker(QThread):
    """Run inference on one image and emit review frames without saving files."""

    stage_ready = Signal(int, str, object)
    completed = Signal(int, object)

    def __init__(
        self,
        request_id: int,
        image_path: Path,
        model_path: Path | None,
        confidence: float,
        max_quadrats: int,
        background_name: str,
        *,
        test_mode: bool,
        original_only: bool = False,
    ) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.image_path = Path(image_path)
        self.model_path = Path(model_path) if model_path is not None else None
        self.confidence = float(confidence)
        self.max_quadrats = int(max_quadrats)
        self.background_name = str(background_name)
        self.test_mode = bool(test_mode)
        self.original_only = bool(original_only)
        self.cancel_event = Event()

    def request_cancel(self) -> None:
        """Cancel the request when control returns from image/model inference."""

        self.cancel_event.set()

    def run(self) -> None:
        result = {
            "success": False,
            "cancelled": False,
            "error": None,
            "bbox": None,
            "detection_count": 0,
            "warnings": [],
            "test_mode": self.test_mode,
            "original_only": self.original_only,
        }

        try:
            if self.cancel_event.is_set():
                result["cancelled"] = True
                self.completed.emit(self.request_id, result)
                return

            source_metadata = inspect_image(self.image_path)
            image_rgb, _metadata, processing_warnings, _metadata_warnings = (
                _load_rgb_image(self.image_path, source_metadata)
            )
            result["warnings"] = list(processing_warnings)
            preview_base = make_preview_base(image_rgb)

            # This immediate image lets the user know which input is being tested.
            self.stage_ready.emit(
                self.request_id,
                f"Live image · {self.image_path.name}",
                rgb_array_to_qimage(preview_base),
            )

            if self.original_only:
                result["success"] = True
                self.completed.emit(self.request_id, result)
                return

            if self.cancel_event.is_set():
                result["cancelled"] = True
                self.completed.emit(self.request_id, result)
                return

            if self.model_path is None:
                raise ValueError("No model path was selected for the test.")

            model = _load_yolo_segmentation_model(self.model_path)
            if self.cancel_event.is_set():
                result["cancelled"] = True
                self.completed.emit(self.request_id, result)
                return

            masks = _predict_instance_masks(
                model,
                image_rgb,
                self.confidence,
                self.max_quadrats,
            )
            if not masks:
                raise ValueError(
                    "YOLO returned no valid quadrat masks at the selected "
                    f"confidence threshold ({self.confidence:.4f}). Try a "
                    "different model or review the confidence threshold. "
                    "No output files were saved."
                )

            if self.cancel_event.is_set():
                result["cancelled"] = True
                self.completed.emit(self.request_id, result)
                return

            frames, bbox = _build_test_preview_frames(
                image_rgb,
                self.image_path,
                masks,
                BACKGROUND_RGB[self.background_name],
            )

            # Frames are sent only after they are constructed successfully.
            # A short cancellable pause gives the GUI event loop time to paint
            # each intermediate stage. Without it, queued Qt signals can be
            # handled back-to-back and only the final stage becomes visible.
            # This is test-preview presentation only; inference and batch
            # processing are unchanged.
            for stage_index, (stage_name, frame) in enumerate(frames):
                if self.cancel_event.is_set():
                    result["cancelled"] = True
                    break

                self.stage_ready.emit(self.request_id, stage_name, frame)

                if stage_index < len(frames) - 1:
                    if self.cancel_event.wait(TEST_PREVIEW_STAGE_INTERVAL_SECONDS):
                        result["cancelled"] = True
                        break

            if not result["cancelled"]:
                result.update({
                    "success": True,
                    "bbox": bbox,
                    "detection_count": len(masks),
                })

        except Exception as exc:
            if self.cancel_event.is_set():
                result["cancelled"] = True
            else:
                result["error"] = f"{type(exc).__name__}: {exc}"

        self.completed.emit(self.request_id, result)


# ---------------------------------------------------------------------
# Background worker — batch processing and output saving
# ---------------------------------------------------------------------

class DeepLearningWorker(QThread):
    """Batch-processing worker for YOLO instance segmentation."""

    progress_changed = Signal(int, int, str)
    status_changed = Signal(str)
    batch_ready = Signal(int)
    preview_stage_ready = Signal(str, object)
    completed = Signal(object)

    def __init__(
        self,
        image_files: list[Path],
        output_layout,
        model_path: Path,
        model_label: str,
        confidence: float,
        max_quadrats: int,
        background_name: str,
        output_policy: OutputPolicy,
        jpegtran: str | None = None,
        preview_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.image_files = list(image_files)
        self.output_layout = output_layout
        self.model_path = Path(model_path)
        self.model_label = str(model_label)
        self.confidence = float(confidence)
        self.max_quadrats = int(max_quadrats)
        self.background_name = str(background_name)
        self.output_policy = output_policy
        self.jpegtran = jpegtran if jpegtran is not None else find_jpegtran()
        self.preview_enabled = bool(preview_enabled)
        self._preview_error_reported = False
        self.cancel_event = Event()

    def request_cancel(self) -> None:
        """Request cancellation after the currently processing image."""

        self.cancel_event.set()

    def _report_preview_error(self, stage_name: str, exc: Exception) -> None:
        """Report a display-only problem once without affecting image processing."""

        if self._preview_error_reported:
            return
        self._preview_error_reported = True
        self.status_changed.emit(
            "Live preview could not render one or more stages; batch processing "
            f"will continue. First preview issue at {stage_name}: "
            f"{type(exc).__name__}: {exc}"
        )

    def _emit_preview_stage(
        self,
        stage_name: str,
        render_frame: Callable[[], QImage],
    ) -> None:
        """Render and emit a preview defensively; display failure is non-fatal."""

        if not self.preview_enabled:
            return
        try:
            frame = render_frame()
            if isinstance(frame, QImage) and not frame.isNull():
                self.preview_stage_ready.emit(stage_name, frame)
        except Exception as exc:
            self._report_preview_error(stage_name, exc)

    def _record_input_warning(
        self,
        run_log: ProcessingRunLog,
        warnings: list[str],
        filename: str,
        warning: str,
    ) -> None:
        warnings.append(f"{filename}: {warning}")
        run_log.record_warning(filename, warning)

    def _load_model(self):
        """Load and validate the selected local segmentation model."""

        return _load_yolo_segmentation_model(self.model_path)

    def _save_crop(
        self,
        *,
        image_path: Path,
        cropped_image: Image.Image,
        source_metadata: dict,
        metadata: dict,
        crop_box: tuple[int, int, int, int],
        detection_index: int,
        orientation_normalized: bool,
        disambiguate_source_suffix: bool,
        run_log: ProcessingRunLog,
        summary_warnings: list[str],
    ):
        """Save one crop centrally, retaining the selected output policy."""

        crop_plan = _make_plan(
            image_path,
            self.output_layout.cropped_images,
            "C",
            self.output_policy,
            "crop",
            crop_box=crop_box,
            orientation_normalized=orientation_normalized,
            jpegtran=self.jpegtran,
        )
        crop_plan = _make_unique_plan(
            crop_plan,
            image_path,
            source_metadata,
            detection_index=detection_index,
            disambiguate_source_suffix=disambiguate_source_suffix,
        )

        try:
            crop_result = save_planned_output(
                [cropped_image],
                crop_plan,
                metadata=metadata,
                source_path=image_path,
                source_mode=source_metadata["mode"],
                orientation_normalized=orientation_normalized,
                jpegtran=self.jpegtran,
            )
        except Exception as exc:
            # Only the planned JPEG coefficient-domain route may fall back.
            if not str(crop_plan.strategy).startswith("jpegtran"):
                raise

            fallback_plan = _make_plan(
                image_path,
                self.output_layout.cropped_images,
                "C",
                self.output_policy,
                "crop",
                crop_box=None,
                orientation_normalized=orientation_normalized,
                jpegtran=self.jpegtran,
            )
            fallback_plan = _make_unique_plan(
                fallback_plan,
                image_path,
                source_metadata,
                detection_index=detection_index,
                disambiguate_source_suffix=disambiguate_source_suffix,
            )
            fallback_result = save_planned_output(
                [cropped_image],
                fallback_plan,
                metadata=metadata,
                source_path=image_path,
                source_mode=source_metadata["mode"],
                orientation_normalized=orientation_normalized,
                jpegtran=self.jpegtran,
            )
            fallback_warning = (
                "The planned JPEG coefficient-domain crop could not be "
                "completed or verified. The crop was re-planned under the "
                "selected output policy. "
                f"Original failure: {type(exc).__name__}: {exc}. "
                f"Fallback strategy: {fallback_result.strategy}."
            )
            summary_warnings.append(f"{image_path.name}: {fallback_warning}")
            run_log.record_warning(image_path.name, fallback_warning)
            crop_plan = fallback_plan
            crop_result = fallback_result

        _record_save_warnings(
            run_log,
            summary_warnings,
            image_path.name,
            crop_result.path,
            crop_result.warnings,
        )
        return crop_result, crop_plan

    def run(self) -> None:
        total = len(self.image_files)
        succeeded = 0
        attempted = 0
        errors: list[str] = []
        warnings: list[str] = []
        cancelled = False
        fatal_error = None
        run_log = None

        settings = (
            f"Model: {self.model_label}; "
            f"Model file: {self.model_path.name}; "
            f"Confidence: {self.confidence:.4f}; "
            f"Maximum quadrats: {self.max_quadrats}; "
            f"Background color: {self.background_name}; "
            f"Output policy: {self.output_policy.value}\n"
            f"Input directory: {self.image_files[0].parent.resolve()}\n"
            f"Output directory: {self.output_layout.root.resolve()}"
        )

        try:
            if (
                self.output_layout.log_file is None
                or self.output_layout.cropped_images is None
                or self.output_layout.segmented_images is None
            ):
                raise RuntimeError(
                    "The Deep Learning output folders or log file "
                    "were not prepared."
                )

            run_log = ProcessingRunLog(
                self.output_layout.log_file,
                "Automated (Deep Learning — YOLO Segmentation)",
                settings,
                total,
            )
            background_rgb = BACKGROUND_RGB[self.background_name]

            stem_counts = Counter(path.stem.casefold() for path in self.image_files)
            ambiguous_stems = {
                stem for stem, count in stem_counts.items() if count > 1
            }

            self.status_changed.emit(f"Loading model: {self.model_label}…")
            model = self._load_model()
            self.batch_ready.emit(total)
            self.status_changed.emit(f"Model loaded. Preparing {total} image(s)…")

            for index, image_path in enumerate(self.image_files, start=1):
                if self.cancel_event.is_set():
                    cancelled = True
                    break

                attempted = index
                saved_output_names: list[str] = []
                self.status_changed.emit(
                    f"Preparing image {index}/{total}: {image_path.name}"
                )

                try:
                    source_metadata = inspect_image(image_path)
                    (
                        image_rgb,
                        metadata,
                        input_warnings,
                        deferred_metadata_warnings,
                    ) = _load_rgb_image(image_path, source_metadata)

                    for warning in input_warnings:
                        self._record_input_warning(
                            run_log, warnings, image_path.name, warning
                        )

                    # Batch preview frames are display-only reduced copies.
                    # The full-resolution image and masks used by inference and
                    # output saving remain untouched.
                    preview_base = None
                    if self.preview_enabled:
                        try:
                            preview_base = make_preview_base(image_rgb)
                        except Exception as preview_exc:
                            self._report_preview_error(
                                f"original image · {image_path.name}",
                                preview_exc,
                            )
                    if preview_base is not None:
                        self._emit_preview_stage(
                            f"Image {index}/{total} · Original image · {image_path.name}",
                            lambda preview_base=preview_base: rgb_array_to_qimage(preview_base),
                        )

                    orientation_normalized = _orientation_was_normalized(
                        source_metadata
                    )
                    self.status_changed.emit(
                        f"Detecting quadrats {index}/{total}: {image_path.name}"
                    )
                    masks = _predict_instance_masks(
                        model,
                        image_rgb,
                        self.confidence,
                        self.max_quadrats,
                    )
                    if preview_base is not None:
                        self._emit_preview_stage(
                            f"Image {index}/{total} · YOLO detections · {len(masks)} quadrat(s) · {image_path.name}",
                            lambda preview_base=preview_base, masks=masks: _render_detection_overlay(
                                preview_base, masks
                            ),
                        )
                    if not masks:
                        raise ValueError(
                            "YOLO returned no valid quadrat masks at the "
                            f"selected confidence threshold ({self.confidence:.4f}). "
                            "No derivative was saved for this image."
                        )

                    height, width = image_rgb.shape[:2]
                    combined_mask = np.zeros((height, width), dtype=bool)
                    for instance_mask in masks:
                        combined_mask |= instance_mask

                    if preview_base is not None:
                        self._emit_preview_stage(
                            f"Image {index}/{total} · Combined instance mask · {image_path.name}",
                            lambda preview_base=preview_base, masks=masks: _render_combined_mask(
                                preview_base, masks
                            ),
                        )

                    segmented_rgb = image_rgb.copy()
                    segmented_rgb[~combined_mask] = background_rgb
                    if preview_base is not None:
                        self._emit_preview_stage(
                            f"Image {index}/{total} · Segmented preview · {image_path.name}",
                            lambda preview_base=preview_base, masks=masks, background_rgb=background_rgb: _render_segmented_preview(
                                preview_base,
                                masks,
                                background_rgb,
                            ),
                        )

                    crop_jobs: list[
                        tuple[int, tuple[int, int, int, int], object]
                    ] = []
                    disambiguate_source_suffix = (
                        image_path.stem.casefold() in ambiguous_stems
                    )

                    for detection_index, instance_mask in enumerate(masks, start=1):
                        ys, xs = np.nonzero(instance_mask)
                        if not xs.size or not ys.size:
                            continue
                        crop_box = (
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max()) + 1,
                            int(ys.max()) + 1,
                        )
                        crop_plan = _make_plan(
                            image_path,
                            self.output_layout.cropped_images,
                            "C",
                            self.output_policy,
                            "crop",
                            crop_box=crop_box,
                            orientation_normalized=orientation_normalized,
                            jpegtran=self.jpegtran,
                        )
                        crop_plan = _make_unique_plan(
                            crop_plan,
                            image_path,
                            source_metadata,
                            detection_index=detection_index,
                            disambiguate_source_suffix=disambiguate_source_suffix,
                        )
                        crop_jobs.append((detection_index, crop_box, crop_plan))

                    if not crop_jobs:
                        raise ValueError(
                            "YOLO produced masks, but none contained pixels "
                            "that could define a crop."
                        )

                    segmented_image = Image.fromarray(
                        np.ascontiguousarray(segmented_rgb)
                    )
                    deferred_warnings_recorded = False

                    def record_deferred_output_warnings() -> None:
                        nonlocal deferred_warnings_recorded
                        if deferred_warnings_recorded:
                            return
                        for warning in deferred_metadata_warnings:
                            self._record_input_warning(
                                run_log, warnings, image_path.name, warning
                            )
                        if source_metadata["frames"] > 1:
                            if source_metadata["format"] == "MPO":
                                frame_warning = (
                                    "The source is an MPO container. Only its "
                                    "primary image was processed; secondary "
                                    "pictures or thumbnails were excluded from "
                                    "the derivatives. The original MPO file "
                                    "was not modified."
                                )
                            else:
                                frame_warning = (
                                    f"The source contains {source_metadata['frames']} "
                                    "frames/pages. Only the first frame/page "
                                    "was processed."
                                )
                            self._record_input_warning(
                                run_log, warnings, image_path.name, frame_warning
                            )
                        deferred_warnings_recorded = True

                    try:
                        crop_summaries: list[str] = []
                        for crop_number, (
                            detection_index,
                            crop_box,
                            _initial_plan,
                        ) in enumerate(crop_jobs, start=1):
                            self.status_changed.emit(
                                f"Saving crop {crop_number}/{len(crop_jobs)} "
                                f"for image {index}/{total}: {image_path.name}"
                            )
                            left, top, right, bottom = crop_box
                            crop_array = image_rgb[top:bottom, left:right, :]
                            cropped_image = Image.fromarray(
                                np.ascontiguousarray(crop_array)
                            )
                            try:
                                crop_result, crop_plan = self._save_crop(
                                    image_path=image_path,
                                    cropped_image=cropped_image,
                                    source_metadata=source_metadata,
                                    metadata=metadata,
                                    crop_box=crop_box,
                                    detection_index=detection_index,
                                    orientation_normalized=orientation_normalized,
                                    disambiguate_source_suffix=disambiguate_source_suffix,
                                    run_log=run_log,
                                    summary_warnings=warnings,
                                )
                            finally:
                                cropped_image.close()

                            saved_output_names.append(crop_result.path.name)
                            record_deferred_output_warnings()
                            crop_summaries.append(
                                f"{crop_result.path.name}"
                                f"(strategy={crop_result.strategy}, "
                                f"pixels_verified={crop_result.pixels_verified})"
                            )

                        self.status_changed.emit(
                            f"Saving segmented image {index}/{total}: {image_path.name}"
                        )
                        segmented_plan = _make_plan(
                            image_path,
                            self.output_layout.segmented_images,
                            "S",
                            self.output_policy,
                            "segmented",
                            crop_box=None,
                            orientation_normalized=orientation_normalized,
                            jpegtran=self.jpegtran,
                        )
                        segmented_plan = _make_unique_plan(
                            segmented_plan,
                            image_path,
                            source_metadata,
                            disambiguate_source_suffix=disambiguate_source_suffix,
                        )
                        segmented_result = save_planned_output(
                            [segmented_image],
                            segmented_plan,
                            metadata=metadata,
                            source_path=image_path,
                            source_mode=source_metadata["mode"],
                            orientation_normalized=orientation_normalized,
                            jpegtran=self.jpegtran,
                        )
                        saved_output_names.append(segmented_result.path.name)
                        record_deferred_output_warnings()
                        _record_save_warnings(
                            run_log,
                            warnings,
                            image_path.name,
                            segmented_result.path,
                            segmented_result.warnings,
                        )
                    finally:
                        segmented_image.close()

                    run_log.record_success(
                        f"{image_path.name}; "
                        f"model={self.model_label}; "
                        f"detections={len(crop_summaries)}; "
                        f"crops={'; '.join(crop_summaries)}; "
                        f"segmented={segmented_result.path.name}; "
                        f"segmented_strategy={segmented_result.strategy}; "
                        f"segmented_pixels_verified={segmented_result.pixels_verified}; "
                        f"policy={self.output_policy.value}"
                    )
                    succeeded += 1
                    self.status_changed.emit(
                        f"Completed image {index}/{total}: {image_path.name}"
                    )

                except Exception as exc:
                    error_text = f"{image_path.name}: {type(exc).__name__}: {exc}"
                    errors.append(error_text)
                    run_log.record_error(image_path.name, exc)
                    self.status_changed.emit(
                        f"Error in image {index}/{total}: {image_path.name}"
                    )
                    if saved_output_names:
                        partial_warning = (
                            "One or more derivative files were saved before "
                            "this image encountered an error. Inspect these "
                            "files and Log.txt before reusing the results: "
                            + ", ".join(saved_output_names)
                        )
                        self._record_input_warning(
                            run_log, warnings, image_path.name, partial_warning
                        )
                finally:
                    self.progress_changed.emit(index, total, image_path.name)

            self.status_changed.emit("Finalizing the processing log…")
            if cancelled:
                run_log.close()
            else:
                run_log.finish()

        except Exception as exc:
            fatal_error = f"{type(exc).__name__}: {exc}"
            errors.append(f"Run error: {fatal_error}")
            self.status_changed.emit(f"Processing stopped: {fatal_error}")
            if run_log is not None:
                try:
                    run_log.close()
                except Exception:
                    pass

        self.completed.emit({
            "total": total,
            "attempted": attempted,
            "succeeded": succeeded,
            "errors": errors,
            "warnings": warnings,
            "cancelled": cancelled,
            "fatal_error": fatal_error,
            "output_directory": str(self.output_layout.root),
            "output_policy": self.output_policy.value,
            "model_label": self.model_label,
            "model_path": str(self.model_path),
        })


# ---------------------------------------------------------------------
# Settings dialog — shared Auto-method layout and optional test preview
# ---------------------------------------------------------------------

class AutoDeepLearningDialog(QDialog):
    """Settings, explicit YOLO test preview and batch-processing window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Easy Quadrat — Auto Deep Learning")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(980, 650)
        self.resize(1240, 790)

        self._worker: DeepLearningWorker | None = None
        self._preview_worker: DeepLearningPreviewWorker | None = None
        self._first_image_path: Path | None = None
        self._source_preview_frame: QImage | None = None
        self._preview_request_id = 0
        self._preview_pending_mode: str | None = None
        self._close_when_idle = False
        self._test_stage_history: list[tuple[str, QImage]] = []
        self._test_stage_history_ready = False
        self._test_stage_index = -1
        self._ui_ready = False

        configure_window(self, parent=parent, keep_on_top=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(13)

        title = QLabel("Auto · Deep Learning")
        title.setObjectName("dialogTitle")
        title.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #57863c;"
        )
        layout.addWidget(title)

        intro = QLabel(
            "Detect and segment quadrats using trained YOLO instance-segmentation "
            "models. Select Run a test to inspect YOLO detections, the combined "
            "mask, and the final segmented preview from the first image without "
            "saving outputs. Review detections before "
            "using them for measurements."
        )
        intro.setWordWrap(True)
        intro.setObjectName("sectionDescription")
        layout.addWidget(intro)

        # -------------------------------------------------------------
        # Two-column layout: compact controls and larger preview
        # -------------------------------------------------------------

        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        left_widget = QWidget()
        left_widget.setMinimumWidth(390)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(9)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.show_preview_checkbox = QCheckBox("Show preview")
        self.show_preview_checkbox.setChecked(True)
        self.show_preview_checkbox.setToolTip(
            "Turn off the preview panel and preview rendering. Model inference "
            "and batch output saving remain available."
        )
        self.show_preview_checkbox.toggled.connect(
            self._on_show_preview_toggled
        )
        right_layout.addWidget(self.show_preview_checkbox)

        self.preview_panel = ProcessingPreviewPanel(
            enable_stage_navigation=True,
        )
        self.preview_panel.test_button.setToolTip(
            "Run the selected YOLO segmentation model on the first input image "
            "only. The test does not create output images, output folders or a log."
        )
        self.preview_panel.test_requested.connect(self._run_test)
        self.preview_panel.previous_stage_requested.connect(
            self._show_previous_test_stage
        )
        self.preview_panel.next_stage_requested.connect(
            self._show_next_test_stage
        )
        right_layout.addWidget(self.preview_panel, 1)

        # -------------------------------------------------------------
        # Input/output folders — one compact row per path
        # -------------------------------------------------------------

        def add_folder_row(
            label_text: str,
            placeholder: str,
            browse_callback,
        ) -> tuple[QLineEdit, QPushButton]:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(7)

            label = QLabel(label_text)
            label.setFixedWidth(84)
            label.setWordWrap(False)
            label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )

            path_edit = QLineEdit()
            path_edit.setReadOnly(True)
            path_edit.setMinimumWidth(80)
            path_edit.setPlaceholderText(placeholder)
            path_edit.setToolTip(placeholder)

            browse_button = QPushButton("Browse…")
            browse_button.setFixedWidth(76)
            browse_button.clicked.connect(browse_callback)

            row.addWidget(label)
            row.addWidget(path_edit, 1)
            row.addWidget(browse_button)
            left_layout.addLayout(row)
            return path_edit, browse_button

        self.input_path_edit, self.input_browse_button = add_folder_row(
            "Input folder",
            "Choose input folder",
            self._browse_input,
        )
        self.output_path_edit, self.output_browse_button = add_folder_row(
            "Output folder",
            "Choose output folder",
            self._browse_output,
        )

        # -------------------------------------------------------------
        # Model and detection parameters — grouped compactly
        # -------------------------------------------------------------

        settings_container = QWidget()
        settings_container.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        settings = QGridLayout(settings_container)
        settings.setContentsMargins(0, 24, 0, 0)
        settings.setHorizontalSpacing(8)
        settings.setVerticalSpacing(5)
        settings.setColumnMinimumWidth(0, 154)
        settings.setColumnStretch(1, 1)
        settings.setRowMinimumHeight(5, 10)

        def add_setting_row(
            row: int,
            label_text: str,
            control: QWidget,
        ) -> None:
            label = QLabel(label_text)
            label.setMinimumWidth(154)
            label.setMaximumWidth(154)
            label.setWordWrap(False)
            label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            control.setMinimumWidth(170)
            control.setMaximumWidth(270)
            settings.addWidget(label, row, 0)
            settings.addWidget(
                control,
                row,
                1,
                1,
                1,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )

        self.model_combo = QComboBox()
        for key, label_text in MODEL_OPTIONS:
            self.model_combo.addItem(label_text, key)
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setMinimumWidth(205)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        add_setting_row(0, "Segmentation model", self.model_combo)

        custom_model_row = QHBoxLayout()
        custom_model_row.setContentsMargins(0, 0, 0, 0)
        custom_model_row.setSpacing(6)
        self.custom_model_path_edit = QLineEdit()
        self.custom_model_path_edit.setReadOnly(True)
        self.custom_model_path_edit.setMinimumWidth(80)
        self.custom_model_path_edit.setPlaceholderText("Choose a YOLO .pt model")
        self.custom_model_path_edit.setToolTip(
            "Select a compatible YOLO instance-segmentation model (.pt)."
        )
        self.custom_model_browse_button = QPushButton("Browse model…")
        self.custom_model_browse_button.setFixedWidth(100)
        self.custom_model_browse_button.clicked.connect(self._browse_custom_model)
        custom_model_row.addWidget(self.custom_model_path_edit, 1)
        custom_model_row.addWidget(self.custom_model_browse_button)
        custom_model_widget = QWidget()
        custom_model_widget.setLayout(custom_model_row)
        add_setting_row(1, "Custom model file", custom_model_widget)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setDecimals(4)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(DEFAULT_CONFIDENCE)
        self.confidence_spin.setToolTip(
            "Minimum YOLO confidence threshold. Increasing it can exclude "
            "potentially valid quadrats."
        )
        add_setting_row(2, "Confidence threshold (0–1)", self.confidence_spin)

        self.max_quadrats_spin = QSpinBox()
        self.max_quadrats_spin.setRange(1, 1000)
        self.max_quadrats_spin.setValue(DEFAULT_MAX_QUADRATS)
        self.max_quadrats_spin.setToolTip(
            "Maximum number of instances YOLO may return per image. This is "
            "an upper limit, not a guarantee that this many will be detected."
        )
        add_setting_row(3, "Maximum quadrats per image", self.max_quadrats_spin)

        self.background_combo = QComboBox()
        self.background_combo.addItems(list(BACKGROUND_RGB))
        self.background_combo.setCurrentText(DEFAULT_BACKGROUND_NAME)
        self.background_combo.setToolTip(
            "Background color used outside the union of detected instance masks."
        )
        add_setting_row(4, "Segmented background", self.background_combo)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedWidth(90)
        self.reset_button.setToolTip(
            "Restore the default model and all detection parameters."
        )
        self.reset_button.clicked.connect(self._reset_parameters)
        settings.addWidget(
            self.reset_button,
            6,
            1,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        left_layout.addWidget(
            settings_container,
            0,
            Qt.AlignmentFlag.AlignTop,
        )

        self.model_description = QLabel()
        self.model_description.setWordWrap(True)
        self.model_description.setMinimumWidth(0)
        self.model_description.setObjectName("cardText")
        left_layout.addWidget(self.model_description)

        note = QLabel(
            "Outputs: one C_<filename> crop per detected quadrat, one "
            "S_<filename> segmented image per input, and Log.txt. The "
            "segmented image combines all accepted masks. Lossless output is "
            "recommended when exact preservation of the processed raster is "
            "required. Custom weights must support YOLO instance segmentation."
        )
        note.setWordWrap(True)
        note.setMinimumWidth(0)
        note.setObjectName("cardText")
        left_layout.addWidget(note)

        self.progress_label = QLabel()
        self.progress_label.setWordWrap(True)
        self.progress_label.hide()
        left_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)

        # -------------------------------------------------------------
        # Actions — centered within the left column
        # -------------------------------------------------------------

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)

        self.run_button = QPushButton("Run Deep Learning")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumWidth(150)
        self.run_button.clicked.connect(self._run_processing)
        buttons.addWidget(self.run_button)

        self.cancel_button = QPushButton("Close")
        self.cancel_button.setMinimumWidth(90)
        self.cancel_button.clicked.connect(self._cancel_or_close)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        left_layout.addLayout(buttons)

        content_layout.addWidget(left_widget, 2)
        content_layout.addWidget(right_widget, 3)
        layout.addLayout(content_layout, 1)

        self.confidence_spin.valueChanged.connect(self._on_parameters_changed)
        self.max_quadrats_spin.valueChanged.connect(self._on_parameters_changed)
        self.background_combo.currentTextChanged.connect(
            self._on_parameters_changed
        )
        self._ui_ready = True
        self._on_model_changed()
        self._update_action_states()

    # -----------------------------------------------------------------
    # Input and model selection
    # -----------------------------------------------------------------

    @Slot()
    def _browse_input(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select input image folder",
            self.input_path_edit.text().strip() or str(Path.home()),
        )
        if folder:
            self.input_path_edit.setText(folder)
            self.input_path_edit.setToolTip(folder)
            self._refresh_first_input_preview()

    @Slot()
    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_path_edit.text().strip() or str(Path.home()),
        )
        if folder:
            self.output_path_edit.setText(folder)
            self.output_path_edit.setToolTip(folder)

    @Slot()
    def _on_model_changed(self, *_args) -> None:
        """Refresh model guidance and custom-model controls."""

        model_key = self.model_combo.currentData()
        is_custom = model_key == CUSTOM_MODEL_KEY
        self.custom_model_path_edit.setToolTip(
            self.custom_model_path_edit.text().strip()
            or "Select a compatible YOLO instance-segmentation model (.pt)."
        )

        if is_custom:
            description = (
                "Use a compatible custom YOLO instance-segmentation checkpoint. "
                "Detection-only weights do not produce the masks required by "
                "this method."
            )
        else:
            description = BUILT_IN_MODELS[model_key]["description"]
            model_path = _built_in_model_path(model_key)
            if not model_path.is_file():
                description += " Built-in model file not found in the current installation."
        self.model_description.setText(description)

        if self._ui_ready:
            self._invalidate_test_history()
            self._show_original_or_queue(
                "Model selection changed. Run a test to inspect detections with "
                "the selected model and current settings."
            )
            self._update_action_states()

    @Slot()
    def _browse_custom_model(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select a custom YOLO segmentation model",
            str(Path.home()),
            "PyTorch model weights (*.pt)",
        )
        if selected:
            selected_path = str(Path(selected).resolve())
            if selected_path == self.custom_model_path_edit.text().strip():
                return
            self.custom_model_path_edit.setText(selected_path)
            self.custom_model_path_edit.setToolTip(selected_path)
            self._on_parameters_changed()

    def _resolve_selected_model(self) -> tuple[Path, str]:
        """Return the selected local model path and visible label."""

        model_key = self.model_combo.currentData()
        visible_label = self.model_combo.currentText()
        if model_key == CUSTOM_MODEL_KEY:
            raw_path = self.custom_model_path_edit.text().strip()
            if not raw_path:
                raise ValueError("Please select a custom model file.")
            model_path = Path(raw_path).expanduser()
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"The custom model file was not found: {model_path}"
                )
            if model_path.suffix.casefold() != ".pt":
                raise ValueError(
                    "The selected custom model must use the .pt extension."
                )
            return model_path.resolve(), visible_label

        model_path = _built_in_model_path(model_key)
        if not model_path.is_file():
            raise FileNotFoundError(
                "The selected built-in model file is missing:\n"
                f"{model_path}\n\nCheck that the models folder is present "
                "beside the application or included in the PyInstaller bundle."
            )
        return model_path.resolve(), visible_label

    # -----------------------------------------------------------------
    # Preview request management and stage navigation
    # -----------------------------------------------------------------

    def _clear_test_stage_history(self) -> None:
        self._test_stage_history.clear()
        self._test_stage_history_ready = False
        self._test_stage_index = -1
        self.preview_panel.set_stage_navigation_state(
            previous_enabled=False,
            next_enabled=False,
        )

    def _invalidate_test_history(self) -> None:
        """Discard stages tied to old settings; never run inference automatically."""

        self._clear_test_stage_history()
        self._preview_request_id += 1
        self._preview_pending_mode = None
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.request_cancel()

    def _refresh_first_input_preview(self) -> None:
        """Find the first input image and load it for display without inference."""

        self._invalidate_test_history()
        self._source_preview_frame = None
        input_text = self.input_path_edit.text().strip()

        if not input_text:
            self._first_image_path = None
            self.preview_panel.set_placeholder("Select an input folder first.")
            self.preview_panel.set_status("No input image selected.")
            self._update_action_states()
            return

        try:
            image_files = list_image_files(Path(input_text).expanduser())
        except (OSError, ValueError) as exc:
            self._first_image_path = None
            self.preview_panel.set_placeholder(
                "The selected folder could not be read."
            )
            self.preview_panel.set_status(
                f"Input preview unavailable: {exc}",
                "error",
            )
            self._update_action_states()
            return

        if not image_files:
            self._first_image_path = None
            self.preview_panel.set_placeholder(
                "No supported images were found in this folder."
            )
            self.preview_panel.set_status(
                "Select a folder containing supported images.",
                "error",
            )
            self._update_action_states()
            return

        self._first_image_path = image_files[0]
        if not self.show_preview_checkbox.isChecked():
            self.preview_panel.set_placeholder("Preview is disabled.")
            self.preview_panel.set_status(
                "Enable Show preview to display the source image."
            )
            self._update_action_states()
            return
        self._queue_original_preview()

    def _queue_original_preview(self) -> None:
        if (
            not self.show_preview_checkbox.isChecked()
            or self._first_image_path is None
            or self._worker is not None
        ):
            return

        self._preview_request_id += 1
        self._preview_pending_mode = "original"
        self.preview_panel.set_placeholder(
            f"Loading original image:\n{self._first_image_path.name}"
        )
        self.preview_panel.set_status(
            "Loading the first source image. Model inference has not started.",
            "working",
        )

        if self._preview_worker is not None:
            if self._preview_worker.isRunning():
                self._preview_worker.request_cancel()
            else:
                self._start_pending_preview()
        else:
            self._start_pending_preview()
        self._update_action_states()

    def _show_original_or_queue(self, status_text: str) -> None:
        if not self.show_preview_checkbox.isChecked():
            self.preview_panel.set_status(
                "Preview is disabled. " + status_text
            )
            return
        if self._source_preview_frame is not None and self._first_image_path is not None:
            self.preview_panel.set_frame(
                self._source_preview_frame,
                f"Original image · {self._first_image_path.name}",
            )
            self.preview_panel.set_status(status_text)
        elif self._first_image_path is not None:
            self._queue_original_preview()
        self._update_action_states()

    @Slot(bool)
    def _on_show_preview_toggled(self, enabled: bool) -> None:
        """Enable or disable the preview panel, cancelling stale preview requests."""

        self._invalidate_test_history()
        if not enabled:
            self._preview_request_id += 1
            self._preview_pending_mode = None
            if self._preview_worker is not None and self._preview_worker.isRunning():
                self._preview_worker.request_cancel()
            self.preview_panel.set_placeholder("Preview is disabled.")
            self.preview_panel.set_status(
                "Preview rendering is off. Batch processing remains available."
            )
            self.preview_panel.setEnabled(False)
        else:
            self.preview_panel.setEnabled(True)
            if self._first_image_path is None:
                self.preview_panel.set_placeholder(
                    "Select an input folder to display its first image."
                )
                self.preview_panel.set_status(
                    "The source image will appear after a folder is selected."
                )
            elif self._source_preview_frame is not None:
                self.preview_panel.set_frame(
                    self._source_preview_frame,
                    f"Original image · {self._first_image_path.name}",
                )
                self.preview_panel.set_status(
                    "Original image displayed. Click Run a test to inspect YOLO detections."
                )
            else:
                self._queue_original_preview()
        self._update_action_states()

    def _on_parameters_changed(self, *_args) -> None:
        """Invalidate old stages after any inference/display parameter changes."""

        self._invalidate_test_history()
        self._show_original_or_queue(
            "Settings changed. Click Run a test to inspect the detections "
            "with the current settings."
        )
        self._update_action_states()

    @Slot()
    def _reset_parameters(self) -> None:
        """Restore the default model and detection settings without inference."""

        controls = (
            self.model_combo,
            self.confidence_spin,
            self.max_quadrats_spin,
            self.background_combo,
        )
        previous_states = [control.blockSignals(True) for control in controls]
        try:
            self.model_combo.setCurrentIndex(0)
            self.custom_model_path_edit.clear()
            self.custom_model_path_edit.setToolTip(
                "Select a compatible YOLO instance-segmentation model (.pt)."
            )
            self.confidence_spin.setValue(DEFAULT_CONFIDENCE)
            self.max_quadrats_spin.setValue(DEFAULT_MAX_QUADRATS)
            self.background_combo.setCurrentText(DEFAULT_BACKGROUND_NAME)
        finally:
            for control, previous_state in zip(controls, previous_states):
                control.blockSignals(previous_state)

        self._on_model_changed()
        self._invalidate_test_history()
        self._show_original_or_queue(
            "Default settings restored. Click Run a test to inspect detections."
        )
        self._update_action_states()

    def _start_pending_preview(self) -> None:
        """Start a queued original-image load or one-image model test."""

        if self._close_when_idle or not self.show_preview_checkbox.isChecked():
            self._preview_pending_mode = None
            self._update_action_states()
            return
        mode = self._preview_pending_mode
        if mode is None or self._first_image_path is None:
            self._preview_pending_mode = None
            self._update_action_states()
            return
        if self._preview_worker is not None:
            if self._preview_worker.isRunning():
                self._preview_worker.request_cancel()
            return

        model_path = None
        if mode == "test":
            try:
                model_path, _model_label = self._resolve_selected_model()
            except (OSError, ValueError, RuntimeError) as exc:
                self._preview_pending_mode = None
                self.preview_panel.set_status(
                    f"Model selection error: {exc}",
                    "error",
                )
                self._update_action_states()
                return
            if not _ultralytics_is_available():
                self._preview_pending_mode = None
                self.preview_panel.set_status(
                    "Ultralytics could not be found in the active Python "
                    "environment. Install a compatible version before testing.",
                    "error",
                )
                self._update_action_states()
                return

        self._preview_pending_mode = None
        self._start_preview_worker(mode=mode, model_path=model_path,
                                   request_id=self._preview_request_id)

    def _start_preview_worker(
        self,
        *,
        mode: str,
        model_path: Path | None,
        request_id: int,
    ) -> None:
        """Run a source display or an unsaved YOLO test in a background thread."""

        if (
            not self.show_preview_checkbox.isChecked()
            or self._first_image_path is None
            or self._preview_worker is not None
        ):
            self._update_action_states()
            return

        worker = DeepLearningPreviewWorker(
            request_id=request_id,
            image_path=self._first_image_path,
            model_path=model_path,
            confidence=self.confidence_spin.value(),
            max_quadrats=self.max_quadrats_spin.value(),
            background_name=self.background_combo.currentText(),
            test_mode=(mode == "test"),
            original_only=(mode == "original"),
        )
        worker.stage_ready.connect(self._on_preview_frame)
        worker.completed.connect(self._on_preview_completed)
        worker.finished.connect(
            lambda worker=worker: self._on_preview_worker_finished(worker)
        )
        self._preview_worker = worker

        if mode == "original":
            self.preview_panel.set_status(
                "Loading the original image. Model inference has not started.",
                "working",
            )
        else:
            self.preview_panel.set_status(
                "Loading the model and testing the first image. "
                "No output files will be saved.",
                "working",
            )
        self._update_action_states()
        worker.start()

    @Slot()
    def _run_test(self) -> None:
        """Test the chosen model on the first input image; save no output files."""

        if (
            not self.show_preview_checkbox.isChecked()
            or self._worker is not None
        ):
            return
        if self._first_image_path is None:
            self.preview_panel.set_status(
                "Please select an input folder first.",
                "error",
            )
            return
        try:
            self._resolve_selected_model()
        except (OSError, ValueError, RuntimeError) as exc:
            self.preview_panel.set_status(f"Model selection error: {exc}", "error")
            return
        if not _ultralytics_is_available():
            self.preview_panel.set_status(
                "Ultralytics could not be found in the active Python environment.",
                "error",
            )
            return

        self._clear_test_stage_history()
        self._preview_request_id += 1
        self._preview_pending_mode = "test"
        self.preview_panel.set_status(
            "Testing the first image with the selected YOLO model. "
            "No output files will be saved.",
            "working",
        )
        if self._preview_worker is not None:
            if self._preview_worker.isRunning():
                self._preview_worker.request_cancel()
            else:
                self._start_pending_preview()
        else:
            self._start_pending_preview()
        self._update_action_states()

    @Slot(int, str, object)
    def _on_preview_frame(self, request_id: int, stage_name: str, frame: object) -> None:
        """Ignore obsolete frames and keep detached frames for later review."""

        if request_id != self._preview_request_id:
            return
        if not self.show_preview_checkbox.isChecked():
            return
        if not isinstance(frame, QImage) or frame.isNull():
            return

        if stage_name.startswith("Live image"):
            self._source_preview_frame = frame.copy()
        elif (
            self._preview_worker is not None
            and self._preview_worker.request_id == request_id
            and self._preview_worker.test_mode
            and not self._preview_worker.original_only
        ):
            if "original image" in stage_name.casefold():
                self._source_preview_frame = frame.copy()
            self._test_stage_history.append((stage_name, frame.copy()))

        self.preview_panel.set_frame(frame, stage_name)

    def _show_test_stage(self, index: int) -> None:
        """Display one stored stage from the most recent completed test."""

        if not self._test_stage_history_ready:
            return
        if not (0 <= index < len(self._test_stage_history)):
            return
        if (
            self._worker is not None
            or self._preview_worker is not None
            or self._preview_pending_mode is not None
            or not self.show_preview_checkbox.isChecked()
        ):
            return

        self._test_stage_index = index
        stage_name, frame = self._test_stage_history[index]
        self.preview_panel.set_frame(frame, stage_name)
        self._update_action_states()

    @Slot()
    def _show_previous_test_stage(self) -> None:
        self._show_test_stage(self._test_stage_index - 1)

    @Slot()
    def _show_next_test_stage(self) -> None:
        self._show_test_stage(self._test_stage_index + 1)

    @Slot(int, object)
    def _on_preview_completed(self, request_id: int, summary: object) -> None:
        """Report test outcome and make completed stage history navigable."""

        if request_id != self._preview_request_id:
            return
        if not self.show_preview_checkbox.isChecked():
            return
        result = dict(summary)
        if result.get("cancelled"):
            return

        if result.get("original_only"):
            if result.get("success") and self._first_image_path is not None:
                self.preview_panel.set_status(
                    f"Original image loaded: {self._first_image_path.name}. "
                    "Click Run a test to inspect the selected model's detections. "
                    "No output files were created."
                )
            elif result.get("error"):
                self.preview_panel.set_placeholder(
                    "The source image could not be displayed. Check that the "
                    "file is readable and uses a supported image format."
                )
                self.preview_panel.set_status(
                    f"Could not display the source image: {result['error']}",
                    "error",
                )
            return

        self._test_stage_history_ready = bool(
            result.get("success") and self._test_stage_history
        )
        if self._test_stage_history_ready:
            # Show the final segmented stage first after testing. It is the last
            # review stage, so the next-stage arrow must remain disabled here.
            segmented_index = next(
                (
                    i for i, (name, _frame) in enumerate(self._test_stage_history)
                    if "segmented preview" in name.casefold()
                ),
                len(self._test_stage_history) - 1,
            )
            self._test_stage_index = segmented_index
            name, frame = self._test_stage_history[segmented_index]
            self.preview_panel.set_frame(frame, name)
            count = int(result.get("detection_count", 0))
            status_text = (
                f"Test complete: {count} quadrat(s) detected. "
                "No output files were saved. Use the arrows to review "
                "the original image, YOLO detections, combined mask, and final segment."
            )
            self.preview_panel.set_status(status_text, "success")
        else:
            self._clear_test_stage_history()
            error = result.get("error") or (
                "The selected model did not return a valid quadrat mask."
            )
            self.preview_panel.set_status(
                f"Test complete: no valid quadrat was detected. {error} "
                "No output files were saved.",
                "error",
            )
        self._update_action_states()

    def _on_preview_worker_finished(self, worker: DeepLearningPreviewWorker) -> None:
        """Release completed preview workers and service pending requests safely."""

        if self._preview_worker is not worker:
            return
        self._preview_worker = None
        if self._close_when_idle:
            QTimer.singleShot(0, self.close)
            return
        if self._preview_pending_mode is not None:
            self._start_pending_preview()
            return
        self._update_action_states()

    def _update_action_states(self) -> None:
        """Keep settings, test, batch and navigation controls consistent."""

        batch_busy = self._worker is not None
        preview_busy = (
            self._preview_worker is not None
            or self._preview_pending_mode is not None
        )
        controls = (
            self.input_browse_button,
            self.output_browse_button,
            self.model_combo,
            self.custom_model_path_edit,
            self.custom_model_browse_button,
            self.confidence_spin,
            self.max_quadrats_spin,
            self.background_combo,
            self.reset_button,
            self.run_button,
        )
        for control in controls:
            control.setEnabled(not batch_busy and not preview_busy)

        is_custom = self.model_combo.currentData() == CUSTOM_MODEL_KEY
        self.custom_model_path_edit.setEnabled(
            is_custom and not batch_busy and not preview_busy
        )
        self.custom_model_browse_button.setEnabled(
            is_custom and not batch_busy and not preview_busy
        )
        self.show_preview_checkbox.setEnabled(not batch_busy)
        self.preview_panel.test_button.setEnabled(
            self.show_preview_checkbox.isChecked()
            and not batch_busy
            and not preview_busy
            and self._first_image_path is not None
        )

        navigation_available = (
            self._test_stage_history_ready
            and bool(self._test_stage_history)
            and self.show_preview_checkbox.isChecked()
            and not batch_busy
            and not preview_busy
        )
        self.preview_panel.set_stage_navigation_state(
            previous_enabled=(
                navigation_available and self._test_stage_index > 0
            ),
            next_enabled=(
                navigation_available
                and self._test_stage_index < len(self._test_stage_history) - 1
            ),
        )

    # -----------------------------------------------------------------
    # Dialog messages and output policy
    # -----------------------------------------------------------------

    def _message(
        self,
        icon: QMessageBox.Icon,
        title: str,
        text: str,
        buttons: QMessageBox.StandardButton = QMessageBox.StandardButton.Ok,
        default_button: QMessageBox.StandardButton | None = None,
    ) -> QMessageBox.StandardButton:
        box = QMessageBox(icon, title, text, buttons, self)
        configure_window(box, parent=self, keep_on_top=True)
        if default_button is not None:
            box.setDefaultButton(default_button)
        return QMessageBox.StandardButton(box.exec())

    def _ask_overwrite(self, count: int) -> bool:
        answer = self._message(
            QMessageBox.Icon.Warning,
            "Existing outputs",
            f"The selected output folders contain {count} existing file(s), "
            "including possibly a previous Log.txt.\n\nFiles with matching "
            "output paths can be replaced. Unrelated files will not be deleted. "
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _choose_output_policy(
        self,
        image_files: list[Path],
    ) -> OutputPolicy | None:
        """Choose the central output policy once for this batch."""

        has_jpeg_source = False
        for image_path in image_files:
            try:
                source_metadata = inspect_image(image_path)
            except Exception:
                continue
            if source_metadata["format"] in {"JPEG", "MPO"}:
                has_jpeg_source = True
                break
        if not has_jpeg_source:
            return OutputPolicy.LOSSLESS

        box = QMessageBox(self)
        box.setWindowTitle("Choose output policy")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            "Choose how the cropped and segmented images should be saved."
        )
        box.setInformativeText(
            "Lossless output (recommended): preserves the processed raster "
            "pixels exactly using a suitable lossless format and verifies the "
            "saved pixels.\n\nKeep JPEG: JPEG/MPO-derived outputs remain JPEG "
            "when supported. Source JPEG quantization tables and chroma sampling "
            "are reused when safely available, but JPEG re-encoding remains lossy. "
            "Other source formats remain under the lossless policy."
        )
        lossless_button = box.addButton(
            "Lossless output (recommended)", QMessageBox.ButtonRole.AcceptRole
        )
        jpeg_button = box.addButton(
            "Keep JPEG (lossy re-encoding)", QMessageBox.ButtonRole.ActionRole
        )
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(lossless_button)
        configure_window(box, parent=self, keep_on_top=True)
        box.exec()
        clicked = box.clickedButton()
        if clicked is lossless_button:
            return OutputPolicy.LOSSLESS
        if clicked is jpeg_button:
            return OutputPolicy.KEEP_JPEG
        return None

    def _existing_output_count(self, output_layout) -> int:
        """Count existing files that merit an overwrite warning."""

        count = 0
        for directory in (
            output_layout.cropped_images,
            output_layout.segmented_images,
        ):
            if directory is None or not directory.is_dir():
                continue
            count += sum(1 for path in directory.iterdir() if path.is_file())
        if output_layout.log_file is not None and output_layout.log_file.is_file():
            count += 1
        return count

    # -----------------------------------------------------------------
    # Start batch processing
    # -----------------------------------------------------------------

    @Slot()
    def _run_processing(self) -> None:
        if (
            self._worker is not None
            or self._preview_worker is not None
            or self._preview_pending_mode is not None
        ):
            return

        input_text = self.input_path_edit.text().strip()
        output_text = self.output_path_edit.text().strip()
        if not input_text:
            self._message(
                QMessageBox.Icon.Warning,
                "Input folder required",
                "Please select the input image folder.",
            )
            return
        if not output_text:
            self._message(
                QMessageBox.Icon.Warning,
                "Output folder required",
                "Please select the output folder.",
            )
            return

        try:
            model_path, model_label = self._resolve_selected_model()
        except (OSError, ValueError, RuntimeError) as exc:
            self._message(
                QMessageBox.Icon.Warning,
                "Model selection error",
                str(exc),
            )
            return
        if not _ultralytics_is_available():
            self._message(
                QMessageBox.Icon.Critical,
                "Ultralytics is required",
                "The Ultralytics package could not be found in the active "
                "Python environment. Install a compatible version before "
                "running Auto Deep Learning.",
            )
            return

        input_directory = Path(input_text).expanduser()
        output_directory = Path(output_text).expanduser()
        try:
            image_files = list_image_files(input_directory)
        except (OSError, ValueError) as exc:
            self._message(
                QMessageBox.Icon.Critical,
                "Input folder error",
                str(exc),
            )
            return
        if not image_files:
            self._message(
                QMessageBox.Icon.Warning,
                "No images found",
                "The selected input folder contains no supported image files.",
            )
            return

        try:
            if input_directory.resolve() == output_directory.resolve():
                self._message(
                    QMessageBox.Icon.Warning,
                    "Separate folders required",
                    "Please choose a different output folder. This avoids "
                    "mixing the input images and generated files.",
                )
                return
        except OSError as exc:
            self._message(
                QMessageBox.Icon.Critical,
                "Folder validation error",
                str(exc),
            )
            return

        output_policy = self._choose_output_policy(image_files)
        if output_policy is None:
            return

        try:
            output_layout = prepare_output_layout(
                output_directory,
                ProcessingMethod.AI_ASSISTED,
            )
            if (
                output_layout.cropped_images is None
                or output_layout.segmented_images is None
                or output_layout.log_file is None
            ):
                raise RuntimeError(
                    "The Deep Learning output layout is incomplete."
                )
        except (OSError, ValueError, RuntimeError) as exc:
            self._message(
                QMessageBox.Icon.Critical,
                "Output folder error",
                str(exc),
            )
            return

        existing_count = self._existing_output_count(output_layout)
        if existing_count > 0 and not self._ask_overwrite(existing_count):
            return

        self._invalidate_test_history()
        self._preview_request_id += 1
        self._preview_pending_mode = None
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText(f"Loading model: {model_label}…")
        self.progress_label.show()
        self.progress_bar.show()
        if self.show_preview_checkbox.isChecked():
            self.preview_panel.set_status(
                "Batch processing is running. Test-stage navigation is disabled.",
                "working",
            )

        self._worker = DeepLearningWorker(
            image_files=image_files,
            output_layout=output_layout,
            model_path=model_path,
            model_label=model_label,
            confidence=self.confidence_spin.value(),
            max_quadrats=self.max_quadrats_spin.value(),
            background_name=self.background_combo.currentText(),
            output_policy=output_policy,
            jpegtran=find_jpegtran(),
            preview_enabled=self.show_preview_checkbox.isChecked(),
        )
        self._worker.status_changed.connect(self._on_status_changed)
        self._worker.preview_stage_ready.connect(self._on_batch_preview_frame)
        self._worker.batch_ready.connect(self._on_batch_ready)
        self._worker.progress_changed.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel processing")
        self._update_action_states()

    # -----------------------------------------------------------------
    # Batch progress and completion
    # -----------------------------------------------------------------

    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        self.progress_label.setText(status)

    @Slot(str, object)
    def _on_batch_preview_frame(self, stage_name: str, frame: object) -> None:
        """Display live, reduced batch stages without changing output pixels."""

        if (
            self._worker is None
            or not self.show_preview_checkbox.isChecked()
            or not isinstance(frame, QImage)
            or frame.isNull()
        ):
            return
        self.preview_panel.set_frame(frame, stage_name)

    @Slot(int)
    def _on_batch_ready(self, total: int) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            f"Model loaded. Preparing {total} image(s)…"
        )

    @Slot(int, int, str)
    def _on_progress(self, completed: int, total: int, filename: str) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(completed)
        self.progress_label.setText(
            f"Completed {completed} / {total}: {filename}"
        )

    @Slot(object)
    def _on_completed(self, summary: object) -> None:
        result = dict(summary)
        if self.show_preview_checkbox.isChecked():
            if result["cancelled"]:
                preview_status = "Batch processing was cancelled. You can run a new test when ready."
                preview_tone = "neutral"
            elif result["fatal_error"] or result["errors"]:
                preview_status = (
                    "Batch processing finished with errors. Inspect the report and Log.txt; "
                    "use Run a test to review a single image."
                )
                preview_tone = "error"
            elif result.get("warnings"):
                preview_status = (
                    "Batch processing finished with warnings. Review the report and Log.txt. "
                    "Use Run a test to inspect a single image."
                )
                preview_tone = "neutral"
            else:
                preview_status = (
                    "Batch processing finished. Use Run a test to inspect the current model "
                    "on the first image; the test does not save output files."
                )
                preview_tone = "success"
            self.preview_panel.set_status(preview_status, preview_tone)

        self.progress_bar.setRange(0, max(1, result["total"]))
        self.progress_bar.setValue(
            min(result["attempted"], max(1, result["total"]))
        )
        if result["cancelled"]:
            status = "Processing was cancelled."
        elif result["fatal_error"]:
            status = "Processing could not be completed."
        else:
            status = "Processing finished."

        errors = result["errors"]
        warnings = result.get("warnings", [])
        policy_label = (
            "Keep JPEG"
            if result.get("output_policy") == OutputPolicy.KEEP_JPEG.value
            else "Lossless"
        )
        self.progress_label.setText(
            f"{status} {result['succeeded']} of {result['total']} image(s) "
            "processed successfully."
        )
        report = (
            f"{status}\n\n"
            f"Model: {result.get('model_label', 'Unknown')}\n"
            f"Output policy: {policy_label}\n"
            f"Images found: {result['total']}\n"
            f"Images attempted: {result['attempted']}\n"
            f"Images processed successfully: {result['succeeded']}\n"
            f"Images with errors: {len(errors)}\n"
            f"Warnings: {len(warnings)}\n\n"
            f"Output folder:\n{result['output_directory']}"
        )
        if errors:
            report += "\n\nFirst errors:\n" + "\n".join(errors[:8])
            if len(errors) > 8:
                report += f"\n…and {len(errors) - 8} more error(s). See Log.txt."
        if warnings:
            report += "\n\nFirst warnings:\n" + "\n".join(warnings[:5])
            if len(warnings) > 5:
                report += f"\n…and {len(warnings) - 5} more warning(s). See Log.txt."

        if result["fatal_error"]:
            icon, title_text = QMessageBox.Icon.Critical, "Deep Learning error"
        elif errors:
            icon, title_text = QMessageBox.Icon.Warning, "Deep Learning finished with errors"
        elif warnings:
            icon, title_text = QMessageBox.Icon.Warning, "Deep Learning completed with warnings"
        else:
            icon, title_text = QMessageBox.Icon.Information, "Deep Learning completed"
        self._message(icon, title_text, report)

    @Slot()
    def _on_worker_finished(self) -> None:
        """Release the worker and restore controls after the thread returns."""

        self._worker = None
        self.progress_bar.hide()
        self.progress_label.hide()
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Close")
        self._update_action_states()

    # -----------------------------------------------------------------
    # Cancellation and closing
    # -----------------------------------------------------------------

    @Slot()
    def _cancel_or_close(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Cancelling…")
            self.progress_label.setText(
                "Cancellation requested; the current image will finish first."
            )
            return

        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._close_when_idle = True
            self._preview_pending_mode = None
            self._preview_worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Closing…")
            self.preview_panel.set_status(
                "Closing after the current model preview operation stops…",
                "working",
            )
            return
        self._preview_pending_mode = None
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._preview_pending_mode = None
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            event.ignore()
            return
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._close_when_idle = True
            self._preview_worker.request_cancel()
            event.ignore()
            return
        event.accept()