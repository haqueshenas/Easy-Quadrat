"""
Easy Quadrat v2.0.0 — Auto (Classic I) image-processing method.

The segmentation sequence preserves the original Classic I procedure:
low-saturation threshold, isotropic opening and closing, largest background
component, final isotropic opening, and border clearing.

Image encoding and output verification are delegated to core.image_output.

Scientific attribution:
The original Auto (Classic I) method was suggested by Peter Suter.
Original discussion:
https://discuss.python.org/t/how-can-i-detect-and-crop-the-rectangular-frame-in-the-image/32378/25

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from __future__ import annotations

from pathlib import Path
from threading import Event

import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageOps

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
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
from ui.window_utils import configure_window


# RGB values, not BGR. These preserve the eight legacy choices.
BACKGROUND_RGB = {
    "Black": (0, 0, 0),
    "Blue": (0, 0, 255),
    "Green": (0, 128, 0),
    "Magenta": (255, 0, 255),
    "Purple": (128, 0, 128),
    "Red": (255, 0, 0),
    "White": (255, 255, 255),
    "Yellow": (255, 255, 0),
}

_CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)

_EXIF_ORIENTATIONS_THAT_TRANSFORM_PIXELS = frozenset({
    2, 3, 4, 5, 6, 7, 8,
})


# ---------------------------------------------------------------------
# Image reading and preprocessing
# ---------------------------------------------------------------------

def _load_rgb_image(
    path: Path,
) -> tuple[np.ndarray, dict, list[str]]:
    """
    Load the first frame of an image as an independent 8-bit RGB array.

    EXIF Orientation is applied before the Classic I algorithm.

    Grayscale, palette, CMYK, and alpha-bearing 8-bit images are normalized
    to RGB. Transparent pixels are composited on white. High-bit-depth input
    is rejected instead of silently reducing scientific pixel precision.

    ICC profile and DPI metadata are retained when they remain suitable.
    Other EXIF/XMP metadata is not blindly copied to the derivative outputs.
    """

    warnings: list[str] = []

    with Image.open(path) as source:
        source.seek(0)

        original_mode = source.mode
        source_format = (source.format or "").upper()

        mode = original_mode.upper()

        if (
            mode in {"I", "F"}
            or mode.startswith("I;16")
            or mode.startswith("RGB;16")
            or mode.startswith("RGBA;16")
        ):
            raise ValueError(
                f"High-bit-depth image mode {original_mode!r} is not "
                "supported by this Classic I version. The image was "
                "not down-converted, to avoid silently losing pixel precision."
            )

        # Some TIFF decoders expose an RGB/RGBA mode even when the TIFF
        # declares more than eight bits per sample. Reject those files too.
        if (
            source_format == "TIFF"
            and original_mode in {"RGB", "RGBA"}
        ):
            try:
                raw_bits = source.tag_v2.get(258)

                if raw_bits is not None:
                    if isinstance(raw_bits, (tuple, list)):
                        bit_values = [
                            int(value) for value in raw_bits
                        ]
                    else:
                        bit_values = [int(raw_bits)]

                    if bit_values and max(bit_values) > 8:
                        raise ValueError(
                            f"High-bit-depth TIFF ({bit_values} bits per "
                            "sample) is not supported by this Classic I "
                            "processing path."
                        )

            except ValueError:
                raise
            except (
                AttributeError,
                TypeError,
                OverflowError,
            ):
                # If the bit-depth tag cannot be read, the explicit
                # mode checks above still apply.
                pass

        try:
            original_exif = source.info.get("exif")
            exif_object = source.getexif()
            has_exif = bool(original_exif) or len(exif_object) > 0
        except Exception:
            has_exif = False

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
                        rgb = Image.new(
                            "RGB",
                            rgba.size,
                            (255, 255, 255),
                        )

                        rgb.paste(rgba, mask=alpha)

                    finally:
                        alpha.close()

                finally:
                    rgba.close()

                warnings.append(
                    "The input contained alpha/transparency. Transparent "
                    "pixels were composited onto white before segmentation; "
                    "the derivative is RGB."
                )

            else:
                rgb = normalized.convert("RGB")

                if normalized.mode != "RGB":
                    warnings.append(
                        f"The source pixel mode {normalized.mode!r} was "
                        "converted to RGB for Classic I processing."
                    )

            try:
                array = np.asarray(
                    rgb,
                    dtype=np.uint8,
                ).copy()

            finally:
                rgb.close()

            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(
                    "The image could not be normalized to RGB."
                )

            if min(array.shape[:2]) < 2:
                raise ValueError(
                    "The image is too small to process."
                )

            if has_exif:
                warnings.append(
                    "Source EXIF metadata was not copied to the Classic I "
                    "derivatives. EXIF Orientation was applied to the "
                    "decoded pixels where applicable."
                )

            if has_xmp:
                warnings.append(
                    "Source XMP metadata was not copied to the Classic I "
                    "derivatives."
                )

            return array, metadata, warnings

        finally:
            if normalized is not source:
                normalized.close()


# ---------------------------------------------------------------------
# Isotropic morphology
# ---------------------------------------------------------------------

def _dilate(
    mask: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Euclidean-distance isotropic dilation."""

    if radius <= 0:
        return mask.copy()

    return ndi.distance_transform_edt(~mask) <= radius


def _erode(
    mask: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Euclidean-distance isotropic erosion."""

    if radius <= 0:
        return mask.copy()

    return ndi.distance_transform_edt(mask) > radius


def _opening(
    mask: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Isotropic opening: erosion followed by dilation."""

    if radius <= 0:
        return mask.copy()

    return _dilate(
        _erode(mask, radius),
        radius,
    )


def _closing(
    mask: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Isotropic closing: dilation followed by erosion."""

    if radius <= 0:
        return mask.copy()

    return _erode(
        _dilate(mask, radius),
        radius,
    )


# ---------------------------------------------------------------------
# Connected components and border clearing
# ---------------------------------------------------------------------

def _largest_component(
    mask: np.ndarray,
) -> np.ndarray:
    """Return the largest 8-connected foreground component."""

    labels, count = ndi.label(
        mask,
        structure=_CONNECTIVITY_8,
    )

    if count == 0:
        return np.asarray(mask, dtype=bool).copy()

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    return labels == int(sizes.argmax())


def _clear_border(
    mask: np.ndarray,
) -> np.ndarray:
    """Remove every 8-connected foreground component touching an image edge."""

    labels, count = ndi.label(
        mask,
        structure=_CONNECTIVITY_8,
    )

    if count == 0:
        return np.asarray(mask, dtype=bool).copy()

    edge_labels = np.unique(
        np.concatenate(
            (
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            )
        )
    )

    edge_labels = edge_labels[edge_labels != 0]
    result = np.asarray(mask, dtype=bool).copy()

    if edge_labels.size:
        result[np.isin(labels, edge_labels)] = False

    return result


# ---------------------------------------------------------------------
# Saturation threshold and Classic I segmentation
# ---------------------------------------------------------------------

def _low_saturation_mask(
    image_rgb: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Calculate the low-saturation threshold used by RGB-to-HSV conversion."""

    height = image_rgb.shape[0]
    result = np.empty(image_rgb.shape[:2], dtype=bool)
    chunk_rows = 512

    for start in range(0, height, chunk_rows):
        block = (
            image_rgb[start:start + chunk_rows].astype(np.float64)
            / 255.0
        )

        maximum = np.max(block, axis=2)
        minimum = np.min(block, axis=2)
        delta = maximum - minimum

        saturation = np.zeros(
            maximum.shape,
            dtype=np.float64,
        )

        np.divide(
            delta,
            maximum,
            out=saturation,
            where=maximum != 0.0,
        )

        saturation[delta == 0.0] = 0.0

        result[start:start + len(block)] = (
            saturation < threshold
        )

    return result


def detect_quadrat_mask(
    image_rgb: np.ndarray,
    saturation_threshold: float = 0.1,
    closing_radius: int = 40,
    opening_radius: int = 220,
) -> np.ndarray:
    """Detect the quadrat region using the original Classic I sequence."""

    if (
        image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
    ):
        raise ValueError(
            "Classic I requires an 8-bit RGB image."
        )

    if min(image_rgb.shape[:2]) < 2:
        raise ValueError(
            "The image is too small to process."
        )

    if not 0.0 <= saturation_threshold <= 1.0:
        raise ValueError(
            "Saturation threshold must be between 0 and 1."
        )

    if closing_radius < 0 or opening_radius < 0:
        raise ValueError(
            "Morphological radii cannot be negative."
        )

    # Step 1: identify pixels below the selected saturation threshold.
    seg = _low_saturation_mask(
        image_rgb,
        saturation_threshold,
    )

    # Step 2: remove small-scale foreground details.
    cleaned = _opening(seg, 1)

    # Step 3: close gaps in the candidate background.
    cleaned = _closing(
        cleaned,
        closing_radius,
    )

    # Steps 4 and 5: largest background component, then inversion.
    background = _largest_component(~cleaned)
    filled = ~background

    # Step 6: large-scale isotropic opening.
    mask = _opening(
        filled,
        opening_radius,
    )

    # Step 7: remove components touching the image boundary.
    mask = _clear_border(mask)

    if not mask.any():
        raise ValueError(
            "No quadrat region was detected. Adjust the saturation "
            "threshold or morphological radii, and inspect the input image."
        )

    return mask


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def _orientation_was_normalized(source_metadata: dict) -> bool:
    """Return whether ImageOps.exif_transpose should have changed orientation."""

    try:
        orientation = int(
            source_metadata.get("exif_orientation", 1)
        )
    except (TypeError, ValueError, OverflowError):
        orientation = 1

    return orientation in _EXIF_ORIENTATIONS_THAT_TRANSFORM_PIXELS


def _record_save_warnings(
    run_log: ProcessingRunLog,
    summary_warnings: list[str],
    input_filename: str,
    output_path: Path,
    warnings: tuple[str, ...] | list[str],
) -> None:
    """Record central output warnings with the actual output filename."""

    for warning in warnings:
        entry = f"{input_filename} -> {output_path.name}: {warning}"

        summary_warnings.append(entry)
        run_log.record_warning(
            f"{input_filename} -> {output_path.name}",
            warning,
        )


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


# ---------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------

class ClassicIWorker(QThread):
    progress_changed = Signal(int, int, str)
    completed = Signal(object)

    def __init__(
        self,
        image_files: list[Path],
        output_layout,
        saturation_threshold: float,
        closing_radius: int,
        opening_radius: int,
        background_name: str,
        output_policy: OutputPolicy,
        jpegtran: str | None = None,
    ) -> None:
        super().__init__()

        self.image_files = list(image_files)
        self.output_layout = output_layout

        self.saturation_threshold = float(saturation_threshold)
        self.closing_radius = int(closing_radius)
        self.opening_radius = int(opening_radius)
        self.background_name = background_name
        self.output_policy = output_policy
        self.jpegtran = (
            jpegtran
            if jpegtran is not None
            else find_jpegtran()
        )

        self.cancel_event = Event()

    def request_cancel(self) -> None:
        """Request cancellation after the current image has finished."""

        self.cancel_event.set()

    def _save_crop(
        self,
        *,
        image_path: Path,
        cropped_image: Image.Image,
        metadata: dict,
        source_metadata: dict,
        crop_box: tuple[int, int, int, int],
        orientation_normalized: bool,
        run_log: ProcessingRunLog,
        summary_warnings: list[str],
    ):
        """Save the crop centrally, with a policy-preserving JPEG fallback."""

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
            # The only automatic fallback is from the coefficient-domain
            # crop route. Never silently replace an explicit Lossless choice
            # with a re-encoded JPEG.
            if crop_plan.strategy != "jpegtran":
                raise

            fallback_plan = _make_plan(
                image_path,
                crop_plan.output_path.parent,
                "C",
                self.output_policy,
                "crop",
                crop_box=None,
                orientation_normalized=orientation_normalized,
                jpegtran=self.jpegtran,
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

            message = (
                "JPEG coefficient-domain crop could not be verified. "
                "The output was re-planned under the selected output policy. "
                f"Original failure: {type(exc).__name__}: {exc}. "
                f"Fallback strategy: {fallback_result.strategy}."
            )

            summary_warnings.append(
                f"{image_path.name}: {message}"
            )

            run_log.record_warning(
                image_path.name,
                message,
            )

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
            f"Saturation: {self.saturation_threshold:.6f}; "
            f"Closing radius: {self.closing_radius}; "
            f"Opening radius: {self.opening_radius}; "
            f"Background color: {self.background_name}; "
            f"Output policy: {self.output_policy.value}"
        )

        try:
            if (
                self.output_layout.log_file is None
                or self.output_layout.cropped_images is None
                or self.output_layout.segmented_images is None
            ):
                raise RuntimeError(
                    "Classic I output folders or its log file "
                    "were not prepared."
                )

            run_log = ProcessingRunLog(
                self.output_layout.log_file,
                "Automated (classic I)",
                settings,
                total,
            )

            background_rgb = BACKGROUND_RGB[self.background_name]

            for index, image_path in enumerate(
                self.image_files,
                start=1,
            ):
                if self.cancel_event.is_set():
                    cancelled = True
                    break

                attempted = index

                try:
                    source_metadata = inspect_image(image_path)

                    image_rgb, metadata, input_warnings = (
                        _load_rgb_image(image_path)
                    )

                    for warning in input_warnings:
                        warnings.append(
                            f"{image_path.name}: {warning}"
                        )

                        run_log.record_warning(
                            image_path.name,
                            warning,
                        )

                    orientation_normalized = (
                        _orientation_was_normalized(source_metadata)
                    )

                    mask = detect_quadrat_mask(
                        image_rgb,
                        self.saturation_threshold,
                        self.closing_radius,
                        self.opening_radius,
                    )

                    # Segmented output: preserve image pixels inside the mask.
                    # Replace pixels outside the detected region with the
                    # selected background color.
                    segmented = image_rgb.copy()
                    segmented[~mask] = background_rgb

                    ys, xs = np.nonzero(mask)

                    if not xs.size or not ys.size:
                        raise ValueError(
                            "The detected quadrat mask is empty."
                        )

                    left = int(xs.min())
                    right = int(xs.max()) + 1
                    top = int(ys.min())
                    bottom = int(ys.max()) + 1

                    # The last row and column are retained through +1.
                    cropped = image_rgb[top:bottom, left:right]

                    cropped_image = Image.fromarray(
                        np.ascontiguousarray(
                            cropped,
                            dtype=np.uint8,
                        )
                    )

                    segmented_image = Image.fromarray(
                        np.ascontiguousarray(
                            segmented,
                            dtype=np.uint8,
                        )
                    )

                    try:
                        crop_result, crop_plan = self._save_crop(
                            image_path=image_path,
                            cropped_image=cropped_image,
                            metadata=metadata,
                            source_metadata=source_metadata,
                            crop_box=(left, top, right, bottom),
                            orientation_normalized=orientation_normalized,
                            run_log=run_log,
                            summary_warnings=warnings,
                        )

                        seg_plan = _make_plan(
                            image_path,
                            self.output_layout.segmented_images,
                            "S",
                            self.output_policy,
                            "segmented",
                            crop_box=None,
                            orientation_normalized=orientation_normalized,
                            jpegtran=self.jpegtran,
                        )

                        seg_result = save_planned_output(
                            [segmented_image],
                            seg_plan,
                            metadata=metadata,
                            source_path=image_path,
                            source_mode=source_metadata["mode"],
                            orientation_normalized=orientation_normalized,
                            jpegtran=self.jpegtran,
                        )

                        _record_save_warnings(
                            run_log,
                            warnings,
                            image_path.name,
                            seg_result.path,
                            seg_result.warnings,
                        )

                    finally:
                        cropped_image.close()
                        segmented_image.close()

                    # Explicitly report first-frame-only handling for
                    # multi-frame formats, especially MPO.
                    if source_metadata["frames"] > 1:
                        if source_metadata["format"] == "MPO":
                            frame_warning = (
                                "The source is MPO. Only its primary image "
                                "was processed; secondary pictures or "
                                "thumbnails were excluded. The original "
                                "MPO file was not modified."
                            )
                        else:
                            frame_warning = (
                                f"The source contains {source_metadata['frames']} "
                                "frames/pages. Classic I processed only the "
                                "first frame/page."
                            )

                        warnings.append(
                            f"{image_path.name}: {frame_warning}"
                        )

                        run_log.record_warning(
                            image_path.name,
                            frame_warning,
                        )

                    run_log.record_success(
                        f"{image_path.name}; "
                        f"crop={crop_result.path.name}; "
                        f"crop_strategy={crop_result.strategy}; "
                        f"crop_pixels_verified={crop_result.pixels_verified}; "
                        f"segmented={seg_result.path.name}; "
                        f"segmented_strategy={seg_result.strategy}; "
                        f"segmented_pixels_verified={seg_result.pixels_verified}; "
                        f"policy={self.output_policy.value}"
                    )

                    succeeded += 1

                except Exception as exc:
                    error_text = (
                        f"{image_path.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    errors.append(error_text)

                    run_log.record_error(
                        image_path.name,
                        exc,
                    )

                self.progress_changed.emit(
                    index,
                    total,
                    image_path.name,
                )

            if cancelled:
                run_log.close()
            else:
                run_log.finish()

        except Exception as exc:
            fatal_error = f"{type(exc).__name__}: {exc}"
            errors.append(f"Run error: {fatal_error}")

            if run_log is not None:
                try:
                    run_log.close()
                except Exception:
                    pass

        self.completed.emit(
            {
                "total": total,
                "attempted": attempted,
                "succeeded": succeeded,
                "errors": errors,
                "warnings": warnings,
                "cancelled": cancelled,
                "fatal_error": fatal_error,
                "output_directory": str(
                    self.output_layout.root
                ),
                "output_policy": self.output_policy.value,
            }
        )


# ---------------------------------------------------------------------
# Classic I settings dialog
# ---------------------------------------------------------------------

class AutoClassicIDialog(QDialog):
    """Settings and batch-processing window for Auto (Classic I)."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(
            "Easy Quadrat — Auto (Classic I)"
        )
        self.setMinimumWidth(620)
        self.resize(700, 540)

        self._worker: ClassicIWorker | None = None

        configure_window(
            self,
            parent=parent,
            keep_on_top=True,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(13)

        title = QLabel("Auto · Classic I")
        title.setObjectName("dialogTitle")

        # Visual-only adjustment: make the method title more prominent
        # without changing the application's processing logic or controls.
        title.setStyleSheet(
            "font-size: 22px; "
            "font-weight: 700; "
            "color: #57863c;"
        )

        layout.addWidget(title)

        intro = QLabel(
            "Detect a quadrat using color saturation and morphological "
            "processing. This implementation follows the original Classic I "
            "sequence; inspect results before using them for measurements."
        )
        intro.setWordWrap(True)
        intro.setObjectName("sectionDescription")
        layout.addWidget(intro)

        # -------------------------------------------------------------
        # Input and output folders
        # -------------------------------------------------------------

        folders = QFormLayout()
        folders.setHorizontalSpacing(12)
        folders.setVerticalSpacing(9)

        input_row = QHBoxLayout()

        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        self.input_path_edit.setPlaceholderText(
            "Select a folder containing images"
        )

        self.input_browse_button = QPushButton("Browse…")
        self.input_browse_button.clicked.connect(
            self._browse_input
        )

        input_row.addWidget(self.input_path_edit, 1)
        input_row.addWidget(self.input_browse_button)

        folders.addRow("Input folder", input_row)

        output_row = QHBoxLayout()

        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setPlaceholderText(
            "Select an existing output folder"
        )

        self.output_browse_button = QPushButton("Browse…")
        self.output_browse_button.clicked.connect(
            self._browse_output
        )

        output_row.addWidget(self.output_path_edit, 1)
        output_row.addWidget(self.output_browse_button)

        folders.addRow("Output folder", output_row)

        layout.addLayout(folders)

        # -------------------------------------------------------------
        # Method parameters
        # -------------------------------------------------------------

        settings = QFormLayout()
        settings.setHorizontalSpacing(12)
        settings.setVerticalSpacing(10)

        self.saturation_spin = QDoubleSpinBox()
        self.saturation_spin.setRange(0.0, 1.0)
        self.saturation_spin.setDecimals(6)
        self.saturation_spin.setSingleStep(0.01)
        self.saturation_spin.setValue(0.1)
        self.saturation_spin.setToolTip(
            "Pixels whose HSV saturation is strictly below this "
            "value are candidates for background."
        )

        settings.addRow(
            "Saturation threshold (0–1)",
            self.saturation_spin,
        )

        self.closing_spin = QSpinBox()
        self.closing_spin.setRange(0, 5000)
        self.closing_spin.setValue(40)
        self.closing_spin.setSuffix(" px")

        settings.addRow(
            "Closing radius",
            self.closing_spin,
        )

        self.opening_spin = QSpinBox()
        self.opening_spin.setRange(0, 5000)
        self.opening_spin.setValue(220)
        self.opening_spin.setSuffix(" px")

        settings.addRow(
            "Opening radius",
            self.opening_spin,
        )

        self.background_combo = QComboBox()
        self.background_combo.addItems(
            list(BACKGROUND_RGB)
        )
        self.background_combo.setCurrentText("Black")

        settings.addRow(
            "Segmented-image background",
            self.background_combo,
        )

        layout.addLayout(settings)

        note = QLabel(
            "Outputs: Segmented images/S_<filename>, "
            "Cropped images/C_<filename>, and Log.txt. "
            "The output policy is selected at run time when JPEG/MPO "
            "inputs are present. Lossless output is recommended when "
            "exact preservation of the processed raster is required. "
            "High-bit-depth images are rejected rather than silently "
            "down-converted."
        )
        note.setWordWrap(True)
        note.setObjectName("cardText")
        layout.addWidget(note)

        self.progress_label = QLabel("Ready")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # -------------------------------------------------------------
        # Actions
        # -------------------------------------------------------------

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self.run_button = QPushButton("Run Classic I")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumWidth(125)
        self.run_button.clicked.connect(
            self._run_processing
        )
        buttons.addWidget(self.run_button)

        self.cancel_button = QPushButton("Close")
        self.cancel_button.setMinimumWidth(90)
        self.cancel_button.clicked.connect(
            self._cancel_or_close
        )
        buttons.addWidget(self.cancel_button)

        layout.addLayout(buttons)

    # -----------------------------------------------------------------
    # Folder selection
    # -----------------------------------------------------------------

    @Slot()
    def _browse_input(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select input image folder",
            self.input_path_edit.text().strip()
            or str(Path.home()),
        )

        if folder:
            self.input_path_edit.setText(folder)

    @Slot()
    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_path_edit.text().strip()
            or str(Path.home()),
        )

        if folder:
            self.output_path_edit.setText(folder)

    # -----------------------------------------------------------------
    # Messages
    # -----------------------------------------------------------------

    def _message(
        self,
        icon: QMessageBox.Icon,
        title: str,
        text: str,
        buttons: QMessageBox.StandardButton = (
            QMessageBox.StandardButton.Ok
        ),
        default_button: QMessageBox.StandardButton | None = None,
    ) -> QMessageBox.StandardButton:
        box = QMessageBox(
            icon,
            title,
            text,
            buttons,
            self,
        )

        configure_window(
            box,
            parent=self,
            keep_on_top=True,
        )

        if default_button is not None:
            box.setDefaultButton(default_button)

        return QMessageBox.StandardButton(box.exec())

    def _ask_overwrite(self, count: int) -> bool:
        answer = self._message(
            QMessageBox.Icon.Warning,
            "Overwrite existing outputs?",
            f"{count} output file(s) already exist.\n\n"
            "Files with the same names will be replaced. Continue?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return answer == QMessageBox.StandardButton.Yes

    def _choose_output_policy(
        self,
        image_files: list[Path],
    ) -> OutputPolicy | None:
        """
        Ask once for mixed/JPEG sessions when JPEG or MPO sources exist.

        Other source formats remain under the central lossless policy.
        Returns None if the user cancels.
        """

        has_jpeg_source = False

        for image_path in image_files:
            try:
                source_metadata = inspect_image(image_path)
            except Exception:
                # A malformed file will be reported by the worker.
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
            "Choose how Classic I should save the crop and segmented image."
        )
        box.setInformativeText(
            "Lossless output (recommended): preserves the processed raster "
            "pixels exactly using a suitable lossless format, then verifies "
            "the saved pixels.\n\n"
            "Keep JPEG: JPEG/MPO-derived outputs remain JPEG when supported. "
            "Source JPEG quantization tables and chroma sampling are reused "
            "when safely available, but re-encoding remains lossy. "
            "Other source formats remain under the lossless policy."
        )

        lossless_button = box.addButton(
            "Lossless output (recommended)",
            QMessageBox.ButtonRole.AcceptRole,
        )

        jpeg_button = box.addButton(
            "Keep JPEG (lossy re-encoding)",
            QMessageBox.ButtonRole.ActionRole,
        )

        cancel_button = box.addButton(
            "Cancel",
            QMessageBox.ButtonRole.RejectRole,
        )

        box.setDefaultButton(lossless_button)

        configure_window(
            box,
            parent=self,
            keep_on_top=True,
        )

        box.exec()
        clicked = box.clickedButton()

        if clicked is lossless_button:
            return OutputPolicy.LOSSLESS

        if clicked is jpeg_button:
            return OutputPolicy.KEEP_JPEG

        return None

    # -----------------------------------------------------------------
    # Start processing
    # -----------------------------------------------------------------

    @Slot()
    def _run_processing(self) -> None:
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

        input_directory = Path(input_text).expanduser()

        try:
            image_files = list_image_files(
                input_directory
            )
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
                "The selected input folder contains no supported "
                "image files.",
            )
            return

        output_directory = Path(
            output_text
        ).expanduser()

        try:
            if (
                input_directory.resolve()
                == output_directory.resolve()
            ):
                self._message(
                    QMessageBox.Icon.Warning,
                    "Separate folders required",
                    "Please choose a different output folder. This prevents "
                    "generated output files from being read again as inputs.",
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
                ProcessingMethod.CLASSIC_I,
            )

            if (
                output_layout.cropped_images is None
                or output_layout.segmented_images is None
                or output_layout.log_file is None
            ):
                raise RuntimeError(
                    "The Classic I output layout is incomplete."
                )

        except (OSError, ValueError, RuntimeError) as exc:
            self._message(
                QMessageBox.Icon.Critical,
                "Output folder error",
                str(exc),
            )
            return

        # Plan the expected paths before processing so overwrite confirmation
        # refers to the actual extensions chosen by the central output module.
        expected_outputs: list[Path] = []

        for image_path in image_files:
            try:
                source_metadata = inspect_image(image_path)
                orientation_normalized = (
                    _orientation_was_normalized(source_metadata)
                )

                crop_plan = _make_plan(
                    image_path,
                    output_layout.cropped_images,
                    "C",
                    output_policy,
                    "crop",
                    crop_box=None,
                    orientation_normalized=orientation_normalized,
                    jpegtran=find_jpegtran(),
                )

                segmented_plan = _make_plan(
                    image_path,
                    output_layout.segmented_images,
                    "S",
                    output_policy,
                    "segmented",
                    crop_box=None,
                    orientation_normalized=orientation_normalized,
                    jpegtran=find_jpegtran(),
                )

                expected_outputs.extend(
                    (
                        crop_plan.output_path,
                        segmented_plan.output_path,
                    )
                )

            except Exception:
                # The worker will report a per-image inspection or planning
                # error. A file that cannot be planned has no planned
                # destination to include in this overwrite check.
                continue

        # Deduplicate paths for reliable counting on case-insensitive Windows
        # file systems. Do not delete or alter pre-existing files.
        unique_output_paths: dict[str, Path] = {}

        for path in expected_outputs:
            key = str(path.resolve()).casefold()
            unique_output_paths.setdefault(key, path)

        existing_outputs = [
            path
            for path in unique_output_paths.values()
            if path.exists()
        ]

        if (
            existing_outputs
            and not self._ask_overwrite(len(existing_outputs))
        ):
            return

        controls = (
            self.input_browse_button,
            self.output_browse_button,
            self.saturation_spin,
            self.closing_spin,
            self.opening_spin,
            self.background_combo,
            self.run_button,
        )

        for control in controls:
            control.setEnabled(False)

        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel processing")

        self.progress_bar.setRange(
            0,
            len(image_files),
        )
        self.progress_bar.setValue(0)

        self.progress_label.setText(
            f"Starting: {len(image_files)} image(s)…"
        )

        self._worker = ClassicIWorker(
            image_files=image_files,
            output_layout=output_layout,
            saturation_threshold=self.saturation_spin.value(),
            closing_radius=self.closing_spin.value(),
            opening_radius=self.opening_spin.value(),
            background_name=self.background_combo.currentText(),
            output_policy=output_policy,
            jpegtran=find_jpegtran(),
        )

        self._worker.progress_changed.connect(
            self._on_progress
        )

        self._worker.completed.connect(
            self._on_completed
        )

        self._worker.finished.connect(
            self._on_worker_finished
        )

        self._worker.start()

    # -----------------------------------------------------------------
    # Progress and completion
    # -----------------------------------------------------------------

    @Slot(int, int, str)
    def _on_progress(
        self,
        completed: int,
        total: int,
        filename: str,
    ) -> None:
        self.progress_bar.setRange(
            0,
            max(1, total),
        )

        self.progress_bar.setValue(completed)

        self.progress_label.setText(
            f"Processing {completed} / {total}: {filename}"
        )

    @Slot(object)
    def _on_completed(self, summary: object) -> None:
        result = dict(summary)

        controls = (
            self.input_browse_button,
            self.output_browse_button,
            self.saturation_spin,
            self.closing_spin,
            self.opening_spin,
            self.background_combo,
            self.run_button,
            self.cancel_button,
        )

        for control in controls:
            control.setEnabled(True)

        self.cancel_button.setText("Close")

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
            f"{status} {result['succeeded']} of "
            f"{result['total']} image(s) processed successfully."
        )

        report = (
            f"{status}\n\n"
            f"Output policy: {policy_label}\n"
            f"Images found: {result['total']}\n"
            f"Images attempted: {result['attempted']}\n"
            f"Images processed successfully: {result['succeeded']}\n"
            f"Images with errors: {len(errors)}\n"
            f"Warnings: {len(warnings)}\n\n"
            f"Output folder:\n{result['output_directory']}"
        )

        if errors:
            report += (
                "\n\nFirst errors:\n"
                + "\n".join(errors[:8])
            )

            if len(errors) > 8:
                report += (
                    f"\n…and {len(errors) - 8} more error(s). "
                    "See Log.txt."
                )

        if warnings:
            report += (
                "\n\nFirst warnings:\n"
                + "\n".join(warnings[:5])
            )

            if len(warnings) > 5:
                report += (
                    f"\n…and {len(warnings) - 5} more warning(s). "
                    "See Log.txt."
                )

        if result["fatal_error"]:
            icon = QMessageBox.Icon.Critical
            title = "Classic I error"
        elif errors:
            icon = QMessageBox.Icon.Warning
            title = "Classic I finished with errors"
        elif warnings:
            icon = QMessageBox.Icon.Warning
            title = "Classic I completed with warnings"
        else:
            icon = QMessageBox.Icon.Information
            title = "Classic I completed"

        self._message(
            icon,
            title,
            report,
        )

    @Slot()
    def _on_worker_finished(self) -> None:
        # The worker has now fully returned from run().
        self._worker = None

    # -----------------------------------------------------------------
    # Cancellation and closing
    # -----------------------------------------------------------------

    @Slot()
    def _cancel_or_close(self) -> None:
        if (
            self._worker is not None
            and self._worker.isRunning()
        ):
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Cancelling…")

            self.progress_label.setText(
                "Cancellation requested; the current image will "
                "finish first."
            )
            return

        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if (
            self._worker is not None
            and self._worker.isRunning()
        ):
            self._worker.request_cancel()
            event.ignore()
        else:
            event.accept()