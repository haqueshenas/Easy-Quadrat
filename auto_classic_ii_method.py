"""
Easy Quadrat v2.0.0 — Auto (Classic II) image-processing method.

This implementation preserves the original Auto (Classic II) sequence:
low-saturation threshold, isotropic opening and closing, largest background
component and inversion, directional rectangular openings, border clearing,
convex-hull construction, and the legacy two-candidate area-selection rule.

Image encoding and output verification are delegated to core.image_output.

Original algorithm source:
https://github.com/haqueshenas/Easy-Quadrat/blob/v1.0/main.py

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import morphology
from skimage.morphology import convex_hull_image

from PySide6.QtCore import QThread, QTimer, Qt, Signal, Slot
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
    render_processing_preview,
    rgb_array_to_qimage,
)
from ui.window_utils import configure_window


# ---------------------------------------------------------------------
# Defaults and application constants
# ---------------------------------------------------------------------

DEFAULT_SATURATION_THRESHOLD = 0.1
DEFAULT_CLOSING_RADIUS = 40
DEFAULT_OPENING_FACTOR = 300
DEFAULT_BACKGROUND_NAME = "Black"

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

    EXIF Orientation is applied before the Classic II algorithm.

    Grayscale, palette, CMYK, and alpha-bearing 8-bit images are normalized
    to RGB. Transparent pixels are composited on white. High-bit-depth input
    is rejected instead of silently reducing scientific pixel precision.

    ICC profile and DPI metadata are retained when suitable.
    Other EXIF/XMP metadata is not blindly copied to derivative outputs.
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
                "supported by this Classic II version. The image was "
                "not down-converted, to avoid silently losing pixel precision."
            )

        # Some TIFF decoders expose an RGB/RGBA mode even when the TIFF
        # declares more than eight bits per sample.
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
                            "sample) is not supported by this Classic II "
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
                        "converted to RGB for Classic II processing."
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

            if source_format == "MPO":
                if has_exif or has_xmp:
                    warnings.append(
                        "MPO-derived EXIF/XMP metadata was omitted from "
                        "standalone derivatives because its references may "
                        "depend on the original MPO container. The original "
                        "MPO file was not modified, and metadata filtering "
                        "did not alter the raster pixels supplied to the "
                        "output saver."
                    )
            else:
                if has_exif:
                    warnings.append(
                        "Source EXIF metadata was not copied to Classic II "
                        "derivatives. EXIF Orientation was applied to the "
                        "decoded pixels where applicable."
                    )

                if has_xmp:
                    warnings.append(
                        "Source XMP metadata was not copied to Classic II "
                        "derivatives."
                    )

            return array, metadata, warnings

        finally:
            if normalized is not source:
                normalized.close()


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Directional rectangular opening — original Classic II footprints
# ---------------------------------------------------------------------

def _directional_opening(
    mask: np.ndarray,
    opening_factor: int,
    orientation: str,
) -> np.ndarray:
    """Apply the original vertical or horizontal rectangular opening."""

    if opening_factor < 0:
        raise ValueError("Opening factor size cannot be negative.")

    if opening_factor == 0:
        return np.asarray(mask, dtype=bool).copy()

    if orientation == "vertical":
        footprint = np.ones((opening_factor, 1), dtype=bool)
    elif orientation == "horizontal":
        footprint = np.ones((1, opening_factor), dtype=bool)
    else:
        raise ValueError(f"Unsupported opening orientation: {orientation!r}")

    # `morphology.opening` is the supported equivalent of the legacy
    # binary-opening function for boolean masks; the footprint is unchanged.
    return morphology.opening(
        np.asarray(mask, dtype=bool),
        footprint=footprint,
    )


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
# Optional preview callbacks
# ---------------------------------------------------------------------

class _PreviewCancelled(Exception):
    """Internal signal that a replaceable preview request was cancelled."""


def _check_preview_cancel(
    cancel_event: Event | None,
) -> None:
    """Stop a stale preview between actual processing stages."""

    if cancel_event is not None and cancel_event.is_set():
        raise _PreviewCancelled(
            "The preview request was replaced or cancelled."
        )


def _emit_preview_stage(
    stage_callback: Callable[[str, np.ndarray], None] | None,
    stage_name: str,
    mask: np.ndarray,
    cancel_event: Event | None,
) -> None:
    """
    Offer an independent mask copy to the display callback.

    Preview callback errors are ignored. A cancelled preview, however,
    is allowed to stop its own preview request.
    """

    _check_preview_cancel(cancel_event)

    if stage_callback is None:
        return

    try:
        stage_callback(
            stage_name,
            np.asarray(mask, dtype=bool).copy(),
        )
    except _PreviewCancelled:
        raise
    except Exception:
        # The visual component is optional and must not invalidate detection.
        pass


# ---------------------------------------------------------------------
# Saturation threshold and Classic II segmentation
# ---------------------------------------------------------------------

def _low_saturation_mask(
    image_rgb: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Calculate the low-saturation threshold used by RGB-to-HSV conversion."""

    height = image_rgb.shape[0]
    result = np.empty(
        image_rgb.shape[:2],
        dtype=bool,
    )
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
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
    closing_radius: int = DEFAULT_CLOSING_RADIUS,
    opening_factor: int = DEFAULT_OPENING_FACTOR,
    *,
    stage_callback: Callable[[str, np.ndarray], None] | None = None,
    cancel_event: Event | None = None,
) -> np.ndarray:
    """
    Detect the quadrat using the original Auto (Classic II) sequence.

    Preview callbacks are optional and presentation-only. The area-selection
    condition from the legacy implementation is intentionally preserved.
    """

    if (
        image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
    ):
        raise ValueError("Classic II requires an 8-bit RGB image.")

    if min(image_rgb.shape[:2]) < 2:
        raise ValueError("The image is too small to process.")

    if not 0.0 <= saturation_threshold <= 1.0:
        raise ValueError("Saturation threshold must be between 0 and 1.")

    if closing_radius < 0:
        raise ValueError("Closing radius cannot be negative.")

    if opening_factor < 0:
        raise ValueError("Opening factor size cannot be negative.")

    _check_preview_cancel(cancel_event)

    # Stage 1: low-saturation candidate mask.
    seg = _low_saturation_mask(image_rgb, saturation_threshold)
    _emit_preview_stage(
        stage_callback,
        "1 / 9 · Low-saturation candidates",
        seg,
        cancel_event,
    )

    # Stage 2: original small-scale isotropic opening.
    seg_cleaned = morphology.isotropic_opening(seg, 1)
    _emit_preview_stage(
        stage_callback,
        "2 / 9 · Initial isotropic opening",
        seg_cleaned,
        cancel_event,
    )

    # Stage 3: isotropic closing with the selected radius.
    seg_cleaned = morphology.isotropic_closing(
        seg_cleaned,
        closing_radius,
    )
    _emit_preview_stage(
        stage_callback,
        "3 / 9 · Isotropic closing",
        seg_cleaned,
        cancel_event,
    )

    # Stage 4: largest background component followed by inversion.
    background = _largest_component(~seg_cleaned)
    filled = ~background
    _emit_preview_stage(
        stage_callback,
        "4 / 9 · Largest background removed",
        filled,
        cancel_event,
    )

    # Stage 5: legacy candidate named mask_hor uses the vertical footprint.
    mask_hor = _directional_opening(
        filled,
        opening_factor,
        "vertical",
    )
    _emit_preview_stage(
        stage_callback,
        "5 / 9 · Candidate A — vertical opening",
        mask_hor,
        cancel_event,
    )

    # Stage 6: border clearing and convex hull for candidate A.
    cleared_hor = _clear_border(mask_hor)
    if cleared_hor.any():
        mask_hor = np.asarray(
            convex_hull_image(cleared_hor),
            dtype=bool,
        )
    else:
        mask_hor = np.zeros_like(cleared_hor, dtype=bool)
    _emit_preview_stage(
        stage_callback,
        "6 / 9 · Candidate A — border-cleared convex hull",
        mask_hor,
        cancel_event,
    )

    # Stage 7: legacy candidate named mask_ver uses the horizontal footprint.
    mask_ver = _directional_opening(
        filled,
        opening_factor,
        "horizontal",
    )
    _emit_preview_stage(
        stage_callback,
        "7 / 9 · Candidate B — horizontal opening",
        mask_ver,
        cancel_event,
    )

    # Stage 8: border clearing and convex hull for candidate B.
    cleared_ver = _clear_border(mask_ver)
    if cleared_ver.any():
        mask_ver = np.asarray(
            convex_hull_image(cleared_ver),
            dtype=bool,
        )
    else:
        mask_ver = np.zeros_like(cleared_ver, dtype=bool)
    _emit_preview_stage(
        stage_callback,
        "8 / 9 · Candidate B — border-cleared convex hull",
        mask_ver,
        cancel_event,
    )

    # Stage 9: preserve the original v1.0 decision exactly.
    area_hor = int(np.count_nonzero(mask_hor))
    area_ver = int(np.count_nonzero(mask_ver))

    if area_ver >= area_hor:
        mask = mask_hor
    else:
        mask = mask_ver

    if not mask.any():
        raise ValueError(
            "No quadrat region was detected. Inspect the image and adjust "
            "the saturation threshold or morphological parameters."
        )

    _emit_preview_stage(
        stage_callback,
        "9 / 9 · Final quadrat mask",
        mask,
        cancel_event,
    )

    return np.asarray(mask, dtype=bool)


# Output helpers
# ---------------------------------------------------------------------

def _orientation_was_normalized(
    source_metadata: dict,
) -> bool:
    """Return whether EXIF Orientation should have transformed decoded pixels."""

    try:
        orientation = int(
            source_metadata.get("exif_orientation", 1)
        )
    except (TypeError, ValueError, OverflowError):
        orientation = 1

    return (
        orientation
        in _EXIF_ORIENTATIONS_THAT_TRANSFORM_PIXELS
    )


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
# Preview/test worker — never saves output
# ---------------------------------------------------------------------

class ClassicIIPreviewWorker(QThread):
    """Compute a preview or test without creating any output files."""

    stage_ready = Signal(int, str, object)
    completed = Signal(int, object)

    def __init__(
        self,
        request_id: int,
        image_path: Path,
        saturation_threshold: float,
        closing_radius: int,
        opening_factor: int,
        background_name: str,
        *,
        test_mode: bool,
        original_only: bool = False,
    ) -> None:
        super().__init__()

        self.request_id = int(request_id)
        self.image_path = Path(image_path)

        self.saturation_threshold = float(saturation_threshold)
        self.closing_radius = int(closing_radius)
        self.opening_factor = int(opening_factor)
        self.background_name = str(background_name)
        self.test_mode = bool(test_mode)
        self.original_only = bool(original_only)

        self.cancel_event = Event()

    def request_cancel(self) -> None:
        """Cancel an obsolete preview between processing stages."""

        self.cancel_event.set()

    def run(self) -> None:
        result = {
            "success": False,
            "cancelled": False,
            "error": None,
            "bbox": None,
            "mask_pixels": 0,
            "warnings": [],
            "test_mode": self.test_mode,
            "original_only": self.original_only,
        }

        try:
            _check_preview_cancel(self.cancel_event)

            inspect_image(self.image_path)

            image_rgb, _metadata, input_warnings = _load_rgb_image(
                self.image_path
            )

            result["warnings"] = list(input_warnings)

            _check_preview_cancel(self.cancel_event)

            # Only the visual copy is reduced in size.
            preview_base = make_preview_base(image_rgb)

            self.stage_ready.emit(
                self.request_id,
                f"Original image · {self.image_path.name}",
                rgb_array_to_qimage(preview_base),
            )

            if self.original_only:
                # Folder selection should show the source image immediately.
                # Do not run segmentation for this lightweight display request.
                _check_preview_cancel(self.cancel_event)
                result["success"] = True
            else:
                background_rgb = BACKGROUND_RGB[
                    self.background_name
                ]

                def stage_callback(
                    stage_name: str,
                    stage_mask: np.ndarray,
                ) -> None:
                    _check_preview_cancel(self.cancel_event)

                    frame = render_processing_preview(
                        preview_base,
                        stage_mask,
                        stage_name,
                        background_rgb,
                    )

                    self.stage_ready.emit(
                        self.request_id,
                        stage_name,
                        frame,
                    )

                mask = detect_quadrat_mask(
                    image_rgb,
                    self.saturation_threshold,
                    self.closing_radius,
                    self.opening_factor,
                    stage_callback=stage_callback,
                    cancel_event=self.cancel_event,
                )

                _check_preview_cancel(self.cancel_event)

                ys, xs = np.nonzero(mask)

                if not xs.size or not ys.size:
                    raise ValueError(
                        "The detected quadrat mask is empty."
                    )

                left = int(xs.min())
                right = int(xs.max()) + 1
                top = int(ys.min())
                bottom = int(ys.max()) + 1

                result.update(
                    {
                        "success": True,
                        "bbox": (
                            left,
                            top,
                            right,
                            bottom,
                        ),
                        "mask_pixels": int(np.count_nonzero(mask)),
                    }
                )

        except _PreviewCancelled:
            result["cancelled"] = True

        except Exception as exc:
            result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        self.completed.emit(
            self.request_id,
            result,
        )


# ---------------------------------------------------------------------
# Background worker — batch processing and output saving
# ---------------------------------------------------------------------

class ClassicIIWorker(QThread):
    progress_changed = Signal(int, int, str)
    preview_frame = Signal(str, object)
    completed = Signal(object)

    def __init__(
        self,
        image_files: list[Path],
        output_layout,
        saturation_threshold: float,
        closing_radius: int,
        opening_factor: int,
        background_name: str,
        output_policy: OutputPolicy,
        jpegtran: str | None = None,
        preview_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.image_files = list(image_files)
        self.output_layout = output_layout

        self.saturation_threshold = float(saturation_threshold)
        self.closing_radius = int(closing_radius)
        self.opening_factor = int(opening_factor)
        self.background_name = background_name
        self.output_policy = output_policy
        self.preview_enabled = bool(preview_enabled)

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
            # Only fall back if the central planner explicitly selected
            # the JPEG coefficient-domain crop strategy.
            if not str(crop_plan.strategy).startswith("jpegtran"):
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
            f"Opening factor size: {self.opening_factor}; "
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
                    "Classic II output folders or its log file "
                    "were not prepared."
                )

            run_log = ProcessingRunLog(
                self.output_layout.log_file,
                "Automated (classic II)",
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

                    # Visual work is skipped entirely when Show preview
                    # is off. The analysis array is unaffected either way.
                    stage_callback = None

                    if self.preview_enabled:
                        try:
                            preview_base = make_preview_base(image_rgb)

                            self.preview_frame.emit(
                                f"{image_path.name} · Original image",
                                rgb_array_to_qimage(preview_base),
                            )

                            def batch_stage_callback(
                                stage_name: str,
                                stage_mask: np.ndarray,
                            ) -> None:
                                frame = render_processing_preview(
                                    preview_base,
                                    stage_mask,
                                    stage_name,
                                    background_rgb,
                                )

                                self.preview_frame.emit(
                                    f"{image_path.name} · {stage_name}",
                                    frame,
                                )

                            stage_callback = batch_stage_callback

                        except Exception:
                            # Preview setup failures must not affect detection.
                            stage_callback = None

                    mask = detect_quadrat_mask(
                        image_rgb,
                        self.saturation_threshold,
                        self.closing_radius,
                        self.opening_factor,
                        stage_callback=stage_callback,
                    )

                    # Preserve source pixels within the selected mask.
                    # Replace pixels outside the region with the chosen color.
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

                    cropped = image_rgb[
                        top:bottom,
                        left:right,
                    ]

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
                                f"The source contains "
                                f"{source_metadata['frames']} frames/pages. "
                                "Classic II processed only the first frame/page."
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
                        f"segmented_pixels_verified="
                        f"{seg_result.pixels_verified}; "
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
            errors.append(
                f"Run error: {fatal_error}"
            )

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
# Classic II settings dialog
# ---------------------------------------------------------------------

class AutoClassicIIDialog(QDialog):
    """Settings, explicit test preview and batch-processing window."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(
            "Easy Quadrat — Auto (Classic II)"
        )

        self.setWindowFlag(
            Qt.WindowType.WindowMaximizeButtonHint,
            True,
        )
        self.setWindowFlag(
            Qt.WindowType.WindowMinimizeButtonHint,
            True,
        )
        self.setSizeGripEnabled(True)

        self.setMinimumSize(880, 600)
        self.resize(1180, 760)

        self._worker: ClassicIIWorker | None = None
        self._preview_worker: ClassicIIPreviewWorker | None = None

        self._first_image_path: Path | None = None
        self._source_preview_frame = None
        self._preview_request_id = 0
        self._preview_pending_mode: str | None = None
        self._close_when_idle = False

        # Retain visual frames from the most recent explicit test only.
        # These QImage copies are display-only and never feed the detector.
        self._test_stage_history: list[tuple[str, object]] = []
        self._test_stage_index = -1
        self._test_stage_history_ready = False

        configure_window(
            self,
            parent=parent,
            keep_on_top=True,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(13)

        title = QLabel("Auto · Classic II")
        title.setObjectName("dialogTitle")
        title.setStyleSheet(
            "font-size: 22px; "
            "font-weight: 700; "
            "color: #57863c;"
        )
        layout.addWidget(title)

        intro = QLabel(
            "Detect the quadrat using color saturation, isotropic morphology, "
            "directional rectangular openings and convex-hull processing. "
            "Run a test to inspect all nine processing stages without saving "
            "outputs. Review the candidate masks before using results for "
            "measurements."
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
        left_widget.setMinimumWidth(360)

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
            "Turn off the preview panel and all preview rendering. "
            "Detection and output saving continue without display frames."
        )
        self.show_preview_checkbox.toggled.connect(
            self._on_show_preview_toggled
        )
        right_layout.addWidget(self.show_preview_checkbox)

        self.preview_panel = ProcessingPreviewPanel(
            enable_stage_navigation=True,
        )
        self.preview_panel.test_requested.connect(
            self._run_test
        )
        self.preview_panel.previous_stage_requested.connect(
            self._show_previous_test_stage
        )
        self.preview_panel.next_stage_requested.connect(
            self._show_next_test_stage
        )
        right_layout.addWidget(
            self.preview_panel,
            1,
        )

        content_layout.addWidget(
            left_widget,
            2,
        )
        content_layout.addWidget(
            right_widget,
            3,
        )

        # -------------------------------------------------------------
        # Input and output folders — label, path and button on one row
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
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter
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
        # Method parameters — aligned labels and equal-width controls
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
        settings.setColumnMinimumWidth(0, 174)
        settings.setColumnStretch(1, 0)

        # Keep a deliberate gap before Reset, without affecting
        # the spacing between the four parameter rows.
        settings.setRowMinimumHeight(4, 10)

        def add_setting_row(
            row: int,
            label_text: str,
            control: QWidget,
        ) -> None:
            label = QLabel(label_text)
            label.setWordWrap(False)
            label.setMinimumWidth(174)
            label.setMaximumWidth(174)
            label.setAlignment(
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter
            )

            control.setMinimumWidth(140)
            control.setMaximumWidth(145)
            control.setFixedHeight(26)

            settings.addWidget(label, row, 0)
            settings.addWidget(
                control,
                row,
                1,
                1,
                1,
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter,
            )

        self.saturation_spin = QDoubleSpinBox()
        self.saturation_spin.setRange(0.0, 1.0)
        self.saturation_spin.setDecimals(6)
        self.saturation_spin.setSingleStep(0.01)
        self.saturation_spin.setValue(
            DEFAULT_SATURATION_THRESHOLD
        )
        self.saturation_spin.setToolTip(
            "Pixels whose HSV saturation is strictly below this value "
            "are candidates for background. Range: 0–1."
        )
        add_setting_row(
            0,
            "Saturation threshold (0–1)",
            self.saturation_spin,
        )

        self.closing_spin = QSpinBox()
        self.closing_spin.setRange(0, 5000)
        self.closing_spin.setValue(DEFAULT_CLOSING_RADIUS)
        self.closing_spin.setSuffix(" px")
        self.closing_spin.setToolTip(
            "Radius used by the isotropic closing operation."
        )
        add_setting_row(
            1,
            "Closing radius",
            self.closing_spin,
        )

        self.opening_spin = QSpinBox()
        self.opening_spin.setRange(0, 5000)
        self.opening_spin.setValue(DEFAULT_OPENING_FACTOR)
        self.opening_spin.setSuffix(" px")
        self.opening_spin.setToolTip(
            "Size of the rectangular footprint used for both vertical and "
            "horizontal directional openings. Set to zero to disable them."
        )
        add_setting_row(
            2,
            "Opening factor size",
            self.opening_spin,
        )

        self.background_combo = QComboBox()
        self.background_combo.addItems(list(BACKGROUND_RGB))
        self.background_combo.setCurrentText(
            DEFAULT_BACKGROUND_NAME
        )
        add_setting_row(
            3,
            "Segmented background",
            self.background_combo,
        )

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedWidth(90)
        self.reset_button.setToolTip(
            "Restore all Classic II parameters to their default values."
        )
        self.reset_button.clicked.connect(
            self._reset_parameters
        )

        settings.addWidget(
            self.reset_button,
            5,
            1,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter,
        )

        left_layout.addWidget(
            settings_container,
            0,
            Qt.AlignmentFlag.AlignTop,
        )

        # -------------------------------------------------------------
        # Output information — constrained to the control column
        # -------------------------------------------------------------

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
        note.setMinimumWidth(0)
        note.setObjectName("cardText")
        left_layout.addWidget(note)

        # Progress controls remain hidden except during batch processing.
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
        # Actions
        # -------------------------------------------------------------

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)

        self.run_button = QPushButton("Run Classic II")
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

        buttons.addStretch(1)
        left_layout.addLayout(buttons)

        # Parameter changes do not start the detector automatically.
        self.saturation_spin.valueChanged.connect(
            self._on_parameters_changed
        )
        self.closing_spin.valueChanged.connect(
            self._on_parameters_changed
        )
        self.opening_spin.valueChanged.connect(
            self._on_parameters_changed
        )
        self.background_combo.currentTextChanged.connect(
            self._on_parameters_changed
        )

        layout.addLayout(content_layout, 1)

        self._update_action_states()

    # -----------------------------------------------------------------
    # Folder selection and initial preview
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
            self.input_path_edit.setToolTip(folder)
            self._refresh_first_input_preview()

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
            self.output_path_edit.setToolTip(folder)

    def _clear_test_stage_history(self) -> None:
        """Discard stale test stages and disable manual stage navigation."""

        self._test_stage_history.clear()
        self._test_stage_index = -1
        self._test_stage_history_ready = False
        self.preview_panel.set_stage_navigation_state(
            previous_enabled=False,
            next_enabled=False,
        )

    def _invalidate_pending_preview(self) -> None:
        """Invalidate outdated preview requests and cancel active preview work."""

        self._clear_test_stage_history()
        self._preview_request_id += 1
        self._preview_pending_mode = None

        if (
            self._preview_worker is not None
            and self._preview_worker.isRunning()
        ):
            self._preview_worker.request_cancel()

    def _refresh_first_input_preview(self) -> None:
        """Find the first input image and display it without segmentation."""

        self._invalidate_pending_preview()
        self._source_preview_frame = None

        input_text = self.input_path_edit.text().strip()

        if not input_text:
            self._first_image_path = None
            self.preview_panel.set_placeholder(
                "Select an input folder first."
            )
            self.preview_panel.set_status(
                "No input image selected."
            )
            self._update_action_states()
            return

        try:
            image_files = list_image_files(
                Path(input_text).expanduser()
            )
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
            self.preview_panel.set_placeholder(
                "Preview is disabled."
            )
            self.preview_panel.set_status(
                "Enable Show preview to display the source image."
            )
            self._update_action_states()
            return

        self._queue_original_preview()

    def _queue_original_preview(self) -> None:
        """Load and display the first source image without running detection."""

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
            "Loading the first source image. Detection has not started.",
            "working",
        )

        if self._preview_worker is not None:
            if self._preview_worker.isRunning():
                self._preview_worker.request_cancel()
        else:
            self._start_pending_preview()

        self._update_action_states()

    # -----------------------------------------------------------------
    # Show preview toggle
    # -----------------------------------------------------------------

    @Slot(bool)
    def _on_show_preview_toggled(
        self,
        enabled: bool,
    ) -> None:
        """Enable or disable preview display without launching a test."""

        self._invalidate_pending_preview()

        if not enabled:
            self.preview_panel.set_placeholder(
                "Preview is disabled."
            )
            self.preview_panel.set_status(
                "Preview rendering is off. Batch processing and output "
                "saving remain available."
            )
            self.preview_panel.setEnabled(False)

        else:
            self.preview_panel.setEnabled(True)

            if self._first_image_path is None:
                self.preview_panel.set_placeholder(
                    "Select an input folder to display its first image."
                )
                self.preview_panel.set_status(
                    "The source image will appear automatically after selection."
                )
            else:
                if self._source_preview_frame is not None:
                    self.preview_panel.set_frame(
                        self._source_preview_frame,
                        f"Original image · {self._first_image_path.name}",
                    )
                    self.preview_panel.set_status(
                        "Original image displayed. Click Run a test to inspect "
                        "the processing stages."
                    )
                    self._update_action_states()
                else:
                    self._queue_original_preview()
                return

        self._update_action_states()

    # -----------------------------------------------------------------
    # Parameter changes and reset
    # -----------------------------------------------------------------

    @Slot()
    def _on_parameters_changed(
        self,
        *_args,
    ) -> None:
        """Mark the preview as outdated; do not launch automatic processing."""

        if not self.show_preview_checkbox.isChecked():
            return

        if self._first_image_path is None:
            return

        self._invalidate_pending_preview()

        if self._source_preview_frame is not None:
            self.preview_panel.set_frame(
                self._source_preview_frame,
                f"Original image · {self._first_image_path.name}",
            )
            self.preview_panel.set_status(
                "Settings changed. Click Run a test to inspect the processing "
                "stages with the current settings."
            )
        else:
            self._queue_original_preview()
            return

        self._update_action_states()

    @Slot()
    def _reset_parameters(self) -> None:
        """Restore defaults without launching an automatic test."""

        self._invalidate_pending_preview()

        controls = (
            self.saturation_spin,
            self.closing_spin,
            self.opening_spin,
            self.background_combo,
        )

        previous_states = [
            control.blockSignals(True)
            for control in controls
        ]

        try:
            self.saturation_spin.setValue(
                DEFAULT_SATURATION_THRESHOLD
            )
            self.closing_spin.setValue(
                DEFAULT_CLOSING_RADIUS
            )
            self.opening_spin.setValue(
                DEFAULT_OPENING_FACTOR
            )
            self.background_combo.setCurrentText(
                DEFAULT_BACKGROUND_NAME
            )
        finally:
            for control, previous_state in zip(
                controls,
                previous_states,
            ):
                control.blockSignals(previous_state)

        if self.show_preview_checkbox.isChecked():
            if self._first_image_path is not None:
                if self._source_preview_frame is not None:
                    self.preview_panel.set_frame(
                        self._source_preview_frame,
                        f"Original image · {self._first_image_path.name}",
                    )
                    self.preview_panel.set_status(
                        "Default parameters restored. Click Run a test to "
                        "inspect the processing stages."
                    )
                else:
                    self._queue_original_preview()
            else:
                self.preview_panel.set_placeholder(
                    "Defaults restored. Select an input folder to display its first image."
                )
                self.preview_panel.set_status(
                    "Default parameters restored. No processing was started."
                )
        else:
            self.preview_panel.set_status(
                "Default parameters restored. Preview remains disabled."
            )

        self._update_action_states()

    # -----------------------------------------------------------------
    # Explicit preview/test request management
    # -----------------------------------------------------------------

    def _start_pending_preview(self) -> None:
        """Start a requested preview when no older worker is active."""

        if self._close_when_idle:
            return

        if not self.show_preview_checkbox.isChecked():
            self._preview_pending_mode = None
            self._update_action_states()
            return

        mode = self._preview_pending_mode

        if mode is None:
            return

        if self._first_image_path is None:
            self._preview_pending_mode = None
            self._update_action_states()
            return

        if self._preview_worker is not None:
            if self._preview_worker.isRunning():
                self._preview_worker.request_cancel()
            return

        self._preview_pending_mode = None

        self._start_preview_worker(
            mode=mode,
            request_id=self._preview_request_id,
        )

    def _start_preview_worker(
        self,
        *,
        mode: str,
        request_id: int,
    ) -> None:
        """Run a source-image display or test without saving outputs."""

        if not self.show_preview_checkbox.isChecked():
            self._update_action_states()
            return

        if self._first_image_path is None:
            self.preview_panel.set_status(
                "Please select an input folder first.",
                "error",
            )
            self._update_action_states()
            return

        if self._preview_worker is not None:
            return

        worker = ClassicIIPreviewWorker(
            request_id=request_id,
            image_path=self._first_image_path,
            saturation_threshold=self.saturation_spin.value(),
            closing_radius=self.closing_spin.value(),
            opening_factor=self.opening_spin.value(),
            background_name=self.background_combo.currentText(),
            test_mode=(mode == "test"),
            original_only=(mode == "original"),
        )

        worker.stage_ready.connect(
            self._on_preview_frame
        )
        worker.completed.connect(
            self._on_preview_completed
        )
        worker.finished.connect(
            lambda worker=worker:
            self._on_preview_worker_finished(worker)
        )

        self._preview_worker = worker

        if mode == "original":
            self.preview_panel.set_status(
                "Loading the original image. No detection is running.",
                "working",
            )
        else:
            self.preview_panel.set_status(
                "Testing the first image with the actual detector. "
                "No output files will be saved.",
                "working",
            )

        self._update_action_states()
        worker.start()

    @Slot()
    def _run_test(self) -> None:
        """Test the first image using current settings; save no output files."""

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

        self._clear_test_stage_history()
        self._preview_request_id += 1
        self._preview_pending_mode = "test"

        self.preview_panel.set_status(
            "Testing the first image with the actual detector. "
            "No output files will be saved.",
            "working",
        )

        if self._preview_worker is not None:
            if self._preview_worker.isRunning():
                self._preview_worker.request_cancel()
        else:
            self._start_pending_preview()

        self._update_action_states()

    @Slot(int, str, object)
    def _on_preview_frame(
        self,
        request_id: int,
        stage_name: str,
        frame: object,
    ) -> None:
        """Ignore frames belonging to superseded or disabled preview requests."""

        if request_id != self._preview_request_id:
            return

        if not self.show_preview_checkbox.isChecked():
            return

        if stage_name.startswith("Original image"):
            self._source_preview_frame = frame
        elif (
            self._preview_worker is not None
            and self._preview_worker.request_id == request_id
            and self._preview_worker.test_mode
            and not self._preview_worker.original_only
        ):
            # Capture a detached frame for later manual review. The worker
            # emits QImage objects; copy them so history is independent.
            self._test_stage_history.append(
                (stage_name, frame.copy())
            )

        self.preview_panel.set_frame(
            frame,
            stage_name,
        )

    def _show_test_stage(self, index: int) -> None:
        """Display a stored stage from the most recently completed test."""

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
        """Move backward through the most recent test's saved stages."""

        self._show_test_stage(self._test_stage_index - 1)

    @Slot()
    def _show_next_test_stage(self) -> None:
        """Move forward through the most recent test's saved stages."""

        self._show_test_stage(self._test_stage_index + 1)

    @Slot(int, object)
    def _on_preview_completed(
        self,
        request_id: int,
        summary: object,
    ) -> None:
        """Report the test result without creating files."""

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
                    "Click Run a test to inspect the actual processing stages. "
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

        if result.get("test_mode"):
            self._test_stage_history_ready = bool(
                self._test_stage_history
            )
            self._test_stage_index = (
                len(self._test_stage_history) - 1
                if self._test_stage_history_ready
                else -1
            )

        if result.get("success"):
            left, top, right, bottom = result["bbox"]

            self.preview_panel.set_status(
                "Test complete: quadrat detected. "
                f"Bounding box: left={left}, top={top}, "
                f"width={right - left}, height={bottom - top}. "
                "No output files were saved.",
                "success",
            )

        else:
            error = result.get("error") or (
                "The detector did not return a valid quadrat mask."
            )

            self.preview_panel.set_status(
                "Test complete: no valid quadrat was detected. "
                f"{error} No output files were saved.",
                "error",
            )

    def _on_preview_worker_finished(
        self,
        worker: ClassicIIPreviewWorker,
    ) -> None:
        """Release a finished worker and start a queued test if necessary."""

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

    @Slot(str, object)
    def _on_batch_preview_frame(
        self,
        stage_name: str,
        frame: object,
    ) -> None:
        """Display actual intermediate frames from the batch worker."""

        if self._worker is None:
            return

        if not self.show_preview_checkbox.isChecked():
            return

        self.preview_panel.set_frame(
            frame,
            stage_name,
        )
        self.preview_panel.set_status(
            "Showing an actual intermediate processing stage. "
            "The saved result is generated independently.",
            "working",
        )

    def _update_action_states(self) -> None:
        """Keep settings, test and batch actions consistent with worker state."""

        batch_busy = self._worker is not None

        preview_busy = (
            self._preview_worker is not None
            or self._preview_pending_mode is not None
        )

        for control in (
            self.input_browse_button,
            self.output_browse_button,
            self.saturation_spin,
            self.closing_spin,
            self.opening_spin,
            self.background_combo,
            self.reset_button,
        ):
            control.setEnabled(
                not batch_busy and not preview_busy
            )

        # Keep the checkbox operable during a preview test so rendering
        # can be disabled immediately. The checkbox is outside the panel.
        self.show_preview_checkbox.setEnabled(
            not batch_busy
        )

        self.run_button.setEnabled(
            not batch_busy and not preview_busy
        )

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
                navigation_available
                and self._test_stage_index > 0
            ),
            next_enabled=(
                navigation_available
                and self._test_stage_index
                < len(self._test_stage_history) - 1
            ),
        )

    # -----------------------------------------------------------------
    # Messages and overwrite confirmation
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

        return QMessageBox.StandardButton(
            box.exec()
        )

    def _ask_overwrite(
        self,
        count: int,
    ) -> bool:
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
                # Malformed files are reported by the processing worker.
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
            "Choose how Classic II should save the crop and segmented image."
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
        box.addButton(
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

        input_directory = Path(input_text).expanduser()

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
                "The selected input folder contains no supported "
                "image files.",
            )
            return

        output_directory = Path(output_text).expanduser()

        try:
            if input_directory.resolve() == output_directory.resolve():
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
                ProcessingMethod.CLASSIC_II,
            )

            if (
                output_layout.cropped_images is None
                or output_layout.segmented_images is None
                or output_layout.log_file is None
            ):
                raise RuntimeError(
                    "The Classic II output layout is incomplete."
                )

        except (OSError, ValueError, RuntimeError) as exc:
            self._message(
                QMessageBox.Icon.Critical,
                "Output folder error",
                str(exc),
            )
            return

        # Plan expected paths before processing, so the overwrite warning
        # follows the extensions chosen by the central output policy.
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
                # The worker will report per-image inspection/planning errors.
                continue

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

        # Invalidate old preview results and test-stage navigation before
        # starting a batch run. Batch preview frames are live-only.
        self._clear_test_stage_history()
        self._preview_pending_mode = None
        self._preview_request_id += 1

        preview_enabled = (
            self.show_preview_checkbox.isChecked()
        )

        self.progress_bar.setRange(
            0,
            len(image_files),
        )
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            f"Starting: {len(image_files)} image(s)…"
        )

        self.progress_label.show()
        self.progress_bar.show()

        if preview_enabled:
            self.preview_panel.set_status(
                "Live processing display is enabled.",
                "working",
            )

        self._worker = ClassicIIWorker(
            image_files=image_files,
            output_layout=output_layout,
            saturation_threshold=self.saturation_spin.value(),
            closing_radius=self.closing_spin.value(),
            opening_factor=self.opening_spin.value(),
            background_name=self.background_combo.currentText(),
            output_policy=output_policy,
            jpegtran=find_jpegtran(),
            preview_enabled=preview_enabled,
        )

        self._worker.progress_changed.connect(
            self._on_progress
        )
        self._worker.preview_frame.connect(
            self._on_batch_preview_frame
        )
        self._worker.completed.connect(
            self._on_completed
        )
        self._worker.finished.connect(
            self._on_worker_finished
        )

        self._worker.start()

        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel processing")
        self._update_action_states()

    # -----------------------------------------------------------------
    # Batch progress and completion
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
    def _on_completed(
        self,
        summary: object,
    ) -> None:
        result = dict(summary)

        self.progress_bar.hide()
        self.progress_label.hide()

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
            title = "Classic II error"
        elif errors:
            icon = QMessageBox.Icon.Warning
            title = "Classic II finished with errors"
        elif warnings:
            icon = QMessageBox.Icon.Warning
            title = "Classic II completed with warnings"
        else:
            icon = QMessageBox.Icon.Information
            title = "Classic II completed"

        self._message(
            icon,
            title,
            report,
        )

    @Slot()
    def _on_worker_finished(self) -> None:
        """Release the finished worker and restore UI controls."""

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
        if (
            self._worker is not None
            and self._worker.isRunning()
        ):
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Cancelling…")

            self.progress_label.setText(
                "Cancellation requested; the current image will finish first."
            )
            return

        if (
            self._preview_worker is not None
            and self._preview_worker.isRunning()
        ):
            self._close_when_idle = True
            self._preview_pending_mode = None
            self._preview_worker.request_cancel()

            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Closing…")

            self.preview_panel.set_status(
                "Closing after the current preview operation stops…",
                "working",
            )
            return

        self._preview_pending_mode = None
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._preview_pending_mode = None

        if (
            self._worker is not None
            and self._worker.isRunning()
        ):
            self._worker.request_cancel()
            event.ignore()
            return

        if (
            self._preview_worker is not None
            and self._preview_worker.isRunning()
        ):
            self._close_when_idle = True
            self._preview_worker.request_cancel()
            event.ignore()
            return

        event.accept()
