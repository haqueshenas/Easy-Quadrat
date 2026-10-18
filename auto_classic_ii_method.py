
"""
Easy Quadrat v2.0.0 — Auto (Classic II) image-processing method.

This implementation ports the original Auto (Classic II) sequence from
Easy Quadrat v1.0:

1. Low-saturation threshold.
2. Isotropic opening.
3. Isotropic closing.
4. Largest connected background component and inversion.
5. Directional rectangular openings in both axes.
6. Border clearing and convex-hull construction.
7. Selection between the two hull masks using the original decision rule.

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

from pathlib import Path
from threading import Event

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import morphology, segmentation
from skimage.morphology import convex_hull_image

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

CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)

EXIF_ORIENTATIONS_THAT_TRANSFORM_PIXELS = frozenset({
    2, 3, 4, 5, 6, 7, 8,
})


# ---------------------------------------------------------------------
# Image reading and preprocessing
# ---------------------------------------------------------------------

def _load_rgb_image(
    path: Path,
) -> tuple[np.ndarray, dict, list[str]]:
    """
    Load the primary frame as an independent 8-bit RGB array.

    EXIF Orientation is applied before Classic II processing.

    Grayscale, palette, CMYK, and alpha-bearing 8-bit images are normalized
    to RGB. Transparent pixels are composited onto white.

    High-bit-depth images are rejected instead of silently reducing their
    pixel precision. ICC profile and DPI metadata are retained when available.
    EXIF/XMP metadata is not blindly copied to derivative images.
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
                "supported by Classic II. The image was not down-converted "
                "because that could silently lose pixel precision."
            )

        # Some TIFF decoders report RGB/RGBA even for files whose samples
        # were stored at more than eight bits per channel.
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
                            "High-bit-depth TIFF input is not supported "
                            f"by Classic II: {bit_values} bits per sample."
                        )

            except ValueError:
                raise
            except (
                AttributeError,
                TypeError,
                OverflowError,
            ):
                # If the TIFF bit-depth tag is unreadable, retain the
                # explicit image-mode checks above.
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
                    "The source contains transparency. Transparent pixels "
                    "were composited onto white before Classic II processing; "
                    "the processed image is RGB."
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
                    "The source image could not be normalized to RGB."
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
# Low-saturation threshold
# ---------------------------------------------------------------------

def _low_saturation_mask(
    image_rgb: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Calculate HSV saturation from normalized 8-bit RGB pixels.

    The formula matches the saturation channel used by RGB-to-HSV
    conversion for RGB data normalized to the interval [0, 1].

    Rows are processed in chunks to avoid creating an additional full-size
    float64 image for high-resolution photographs.
    """

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


# ---------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------

def _largest_component(
    mask: np.ndarray,
) -> np.ndarray:
    """Return the largest 8-connected foreground component."""

    labels, count = ndi.label(
        mask,
        structure=CONNECTIVITY_8,
    )

    if count == 0:
        return np.asarray(mask, dtype=bool).copy()

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    return labels == int(sizes.argmax())


def _clear_border(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Remove 8-connected components touching any image boundary.

    This matches the 2-D connectivity used by the original Classic II method.
    """

    labels, count = ndi.label(
        mask,
        structure=CONNECTIVITY_8,
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
# Directional rectangular opening and convex hull
# ---------------------------------------------------------------------

def _directional_opening(
    mask: np.ndarray,
    opening_factor: int,
    orientation: str,
) -> np.ndarray:
    """
    Apply the legacy rectangular opening in the specified image axis.

    The legacy morphology.rectangle(height, width) footprint is represented
    by a boolean array containing the same number of rows and columns.

    opening_factor == 0 disables this operation; this avoids creating
    an invalid zero-sized structuring element.
    """

    if opening_factor < 0:
        raise ValueError(
            "Opening factor size cannot be negative."
        )

    if opening_factor == 0:
        return np.asarray(mask, dtype=bool).copy()

    if orientation == "vertical":
        footprint = np.ones(
            (opening_factor, 1),
            dtype=bool,
        )
    elif orientation == "horizontal":
        footprint = np.ones(
            (1, opening_factor),
            dtype=bool,
        )
    else:
        raise ValueError(
            f"Unsupported opening orientation: {orientation!r}"
        )

    # Use scikit-image's binary opening to retain the original method's
    # morphology behavior as closely as possible.
    return morphology.binary_opening(
        mask,
        footprint,
    )


def _convex_hull_after_border_clearing(
    mask: np.ndarray,
) -> np.ndarray:
    """Clear border components and construct the convex hull if nonempty."""

    cleared = _clear_border(
        np.asarray(mask, dtype=bool)
    )

    if not cleared.any():
        return np.zeros_like(
            cleared,
            dtype=bool,
        )

    hull = convex_hull_image(cleared)

    return np.asarray(
        hull,
        dtype=bool,
    )


# ---------------------------------------------------------------------
# Classic II segmentation
# ---------------------------------------------------------------------

def detect_quadrat_mask(
    image_rgb: np.ndarray,
    saturation_threshold: float = 0.1,
    closing_radius: int = 40,
    opening_factor: int = 300,
) -> np.ndarray:
    """
    Detect the quadrat using the original Auto (Classic II) sequence.

    The two directional candidates are constructed independently from the
    filled mask. Each candidate receives border clearing and a convex hull.
    The legacy area-selection condition is preserved unchanged.
    """

    if (
        image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
    ):
        raise ValueError(
            "Classic II requires an 8-bit RGB image."
        )

    if min(image_rgb.shape[:2]) < 2:
        raise ValueError(
            "The image is too small to process."
        )

    if not 0.0 <= saturation_threshold <= 1.0:
        raise ValueError(
            "Saturation threshold must be between 0 and 1."
        )

    if closing_radius < 0:
        raise ValueError(
            "Closing radius cannot be negative."
        )

    if opening_factor < 0:
        raise ValueError(
            "Opening factor size cannot be negative."
        )

    # Step 1: pixels with saturation below the chosen threshold.
    seg = _low_saturation_mask(
        image_rgb,
        saturation_threshold,
    )

    # Step 2: small-scale isotropic opening.
    seg_cleaned = morphology.isotropic_opening(
        seg,
        1,
    )

    # Step 3: isotropic closing with the user-selected radius.
    seg_cleaned = morphology.isotropic_closing(
        seg_cleaned,
        closing_radius,
    )

    # Steps 4 and 5: identify the largest background component and invert it.
    background = _largest_component(~seg_cleaned)
    filled = ~background

    # Step 6: directional rectangular openings.
    #
    # The mapping of the two candidate names to their footprints follows
    # the original v1.0 implementation exactly.
    mask_hor = _directional_opening(
        filled,
        opening_factor,
        "vertical",
    )

    mask_hor = _convex_hull_after_border_clearing(
        mask_hor
    )

    mask_ver = _directional_opening(
        filled,
        opening_factor,
        "horizontal",
    )

    mask_ver = _convex_hull_after_border_clearing(
        mask_ver
    )

    # Step 7: preserve the original v1.0 area-selection decision.
    #
    # Do not reverse this condition without comparing the results against
    # representative legacy outputs and confirming the intended behavior.
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

    return np.asarray(
        mask,
        dtype=bool,
    )


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def _orientation_was_normalized(
    source_metadata: dict,
) -> bool:
    """Return whether EXIF Orientation requires transforming the decoded pixels."""

    try:
        orientation = int(
            source_metadata.get("exif_orientation", 1)
        )
    except (TypeError, ValueError, OverflowError):
        orientation = 1

    return (
        orientation
        in EXIF_ORIENTATIONS_THAT_TRANSFORM_PIXELS
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
    """Create an output plan through the central output policy."""

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


def _record_save_warnings(
    run_log: ProcessingRunLog,
    summary_warnings: list[str],
    input_filename: str,
    output_path: Path,
    warnings: tuple[str, ...] | list[str],
) -> None:
    """Record central save warnings together with the output filename."""

    for warning in warnings:
        summary_warnings.append(
            f"{input_filename} -> {output_path.name}: {warning}"
        )

        run_log.record_warning(
            f"{input_filename} -> {output_path.name}",
            warning,
        )


# ---------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------

class ClassicIIWorker(QThread):
    progress_changed = Signal(int, int, str)
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
    ) -> None:
        super().__init__()

        self.image_files = list(image_files)
        self.output_layout = output_layout

        self.saturation_threshold = float(saturation_threshold)
        self.closing_radius = int(closing_radius)
        self.opening_factor = int(opening_factor)
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
        """Save the detected crop using the central output module."""

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
            # Only an explicitly planned JPEG coefficient-domain route may
            # use this fallback. The selected output policy is kept intact.
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

            fallback_warning = (
                "The JPEG coefficient-domain crop could not be verified. "
                "The crop was re-planned under the selected output policy. "
                f"Original failure: {type(exc).__name__}: {exc}. "
                f"Fallback strategy: {fallback_result.strategy}."
            )

            summary_warnings.append(
                f"{image_path.name}: {fallback_warning}"
            )

            run_log.record_warning(
                image_path.name,
                fallback_warning,
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

                    mask = detect_quadrat_mask(
                        image_rgb,
                        self.saturation_threshold,
                        self.closing_radius,
                        self.opening_factor,
                    )

                    ys, xs = np.nonzero(mask)

                    if not xs.size or not ys.size:
                        raise ValueError(
                            "The detected quadrat mask is empty."
                        )

                    # The bottom and right crop coordinates are exclusive.
                    # Add one to include the last selected pixel.
                    left = int(xs.min())
                    right = int(xs.max()) + 1
                    top = int(ys.min())
                    bottom = int(ys.max()) + 1

                    crop_box = (
                        left,
                        top,
                        right,
                        bottom,
                    )

                    # The crop retains the original RGB pixels inside the
                    # detected bounding rectangle. Segmentation separately
                    # replaces pixels outside the selected mask.
                    cropped = image_rgb[
                        top:bottom,
                        left:right,
                        :,
                    ]

                    segmented = image_rgb.copy()
                    segmented[~mask] = background_rgb

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
                            crop_box=crop_box,
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
                                "frames/pages. Classic II processed only the "
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
    """Settings and batch-processing window for Auto (Classic II)."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(
            "Easy Quadrat — Auto (Classic II)"
        )
        self.setMinimumWidth(620)
        self.resize(700, 560)

        self._worker: ClassicIIWorker | None = None

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
            "Detect the quadrat using color saturation, isotropic "
            "morphology, directional rectangular openings and convex-hull "
            "processing. This follows the original Classic II sequence. "
            "Inspect the results before using them for measurements."
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
            "Select an output folder"
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
        # Method parameters — the v1.0 defaults are retained
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
        self.closing_spin.setToolTip(
            "Radius of the isotropic closing operation."
        )

        settings.addRow(
            "Closing radius",
            self.closing_spin,
        )

        self.opening_spin = QSpinBox()
        self.opening_spin.setRange(0, 5000)
        self.opening_spin.setValue(300)
        self.opening_spin.setSuffix(" px")
        self.opening_spin.setToolTip(
            "Legacy rectangular opening footprint size. The method "
            "evaluates both vertical and horizontal rectangular footprints."
        )

        settings.addRow(
            "Opening factor size",
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
        Ask once for JPEG/MPO sessions.

        Other source formats use the central lossless policy. Returns None
        if the user cancels the policy selection.
        """

        has_jpeg_source = False

        for image_path in image_files:
            try:
                source_metadata = inspect_image(image_path)
            except Exception:
                # The worker will report inspection errors for this image.
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
            "when safely available, but JPEG re-encoding remains lossy. "
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
        output_directory = Path(output_text).expanduser()

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

        try:
            resolved_input = input_directory.resolve()
            resolved_output = output_directory.resolve()

            if resolved_input == resolved_output:
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

        output_policy = self._choose_output_policy(
            image_files
        )

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

        # Plan expected output paths before processing so that overwrite
        # confirmation uses the central output policy and file extensions.
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
                # The worker will report an individual inspection/planning
                # failure; no expected path is available for this image.
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
            and not self._ask_overwrite(
                len(existing_outputs)
            )
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

        self._worker = ClassicIIWorker(
            image_files=image_files,
            output_layout=output_layout,
            saturation_threshold=self.saturation_spin.value(),
            closing_radius=self.closing_spin.value(),
            opening_factor=self.opening_spin.value(),
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
    def _on_completed(
        self,
        summary: object,
    ) -> None:
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
        # The QThread has fully returned from run().
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