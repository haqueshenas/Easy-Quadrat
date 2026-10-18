"""
Easy Quadrat v2.0.0 — Auto (Deep Learning) image-processing method.

Historical method name in Easy Quadrat v1.0:
    Auto (AI assisted)

The original method uses YOLO instance-segmentation masks to detect
white quadrats, create one cropped image for each detected quadrat, and
create one segmented image for each input image.

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
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
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


# ---------------------------------------------------------------------
# Built-in model definitions
# ---------------------------------------------------------------------

# Keep the filenames used by the previous release for the two standard
# models. The Frameless filenames match the weights in the models folder.
BUILT_IN_MODELS = {
    "eq1": {
        "label": "EQ1 — Standard (YOLOv8n-seg)",
        "filename": "easyquadrat.pt",
        "description": (
            "Standard Easy Quadrat model based on YOLOv8n-seg."
        ),
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


# RGB values, not OpenCV BGR values.
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


# ---------------------------------------------------------------------
# Application/model paths
# ---------------------------------------------------------------------

def _application_resource_root() -> Path:
    """
    Return the source-project or PyInstaller data root.

    In a PyInstaller build, the models folder must be included in the
    application data directory, below the resolved resource root.
    """

    bundle_root = getattr(sys, "_MEIPASS", None)

    if bundle_root:
        return Path(bundle_root).resolve()

    return Path(__file__).resolve().parent


def _built_in_model_path(model_key: str) -> Path:
    """Resolve a built-in model by its stable internal key."""

    specification = BUILT_IN_MODELS.get(model_key)

    if specification is None:
        raise ValueError(
            f"Unknown built-in model key: {model_key!r}."
        )

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


# ---------------------------------------------------------------------
# Input loading and preprocessing
# ---------------------------------------------------------------------

def _load_rgb_image(
    path: Path,
    source_metadata: dict,
) -> tuple[np.ndarray, dict, list[str], list[str]]:
    """
    Load the primary image as an independent 8-bit RGB array.

    EXIF Orientation is applied before inference. Alpha-bearing images
    are composited onto white. Other ordinary modes are converted to RGB.

    High-bit-depth images are rejected instead of silently down-converted.

    Returns:
        RGB pixels,
        metadata safe to offer to the central saver,
        processing warnings,
        metadata-omission warnings.

    Metadata-omission warnings are deferred until an output is saved.
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
                "High-bit-depth TIFF input is not supported by "
                "Auto Deep Learning: "
                f"{bits_per_sample} bits per sample."
            )

        try:
            exif_object = source.getexif()
            has_exif = (
                bool(source.info.get("exif"))
                or len(exif_object) > 0
            )
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
                        rgb = Image.new(
                            "RGB",
                            rgba.size,
                            (255, 255, 255),
                        )

                        try:
                            rgb.paste(rgba, mask=alpha)

                            array = np.asarray(
                                rgb,
                                dtype=np.uint8,
                            ).copy()

                        finally:
                            rgb.close()

                    finally:
                        alpha.close()

                finally:
                    rgba.close()

                processing_warnings.append(
                    "The source contains transparency. Transparent pixels "
                    "were composited onto white before inference; the "
                    "processed image is RGB."
                )

            else:
                rgb = normalized.convert("RGB")

                try:
                    array = np.asarray(
                        rgb,
                        dtype=np.uint8,
                    ).copy()

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
                raise ValueError(
                    "The image is too small to process."
                )

        finally:
            if normalized is not source:
                normalized.close()

    if source_format == "MPO":
        if has_exif or has_xmp:
            metadata_warnings.append(
                "MPO-derived EXIF/XMP metadata was omitted from standalone "
                "derivatives because its references may depend on the "
                "original MPO container. The original MPO file was not "
                "modified, and metadata filtering did not alter the raster "
                "pixels supplied to the output saver."
            )

    else:
        if has_exif:
            metadata_warnings.append(
                "Source EXIF metadata was not copied to AI-derived images. "
                "EXIF Orientation was applied to the decoded pixels where "
                "applicable."
            )

        if has_xmp:
            metadata_warnings.append(
                "Source XMP metadata was not copied to AI-derived images."
            )

    return (
        array,
        metadata,
        processing_warnings,
        metadata_warnings,
    )


def _orientation_was_normalized(
    source_metadata: dict,
) -> bool:
    """Return whether a non-default EXIF Orientation transforms pixels."""

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


# ---------------------------------------------------------------------
# YOLO mask extraction
# ---------------------------------------------------------------------

def _resize_binary_mask(
    mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Return a binary mask at the decoded image dimensions.

    Nearest-neighbour interpolation is used because a segmentation mask
    is a discrete image, not a continuous-valued raster.
    """

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
    """
    Extract valid instance masks from Ultralytics Results objects.

    Returned masks are boolean arrays aligned to the decoded RGB image.
    Empty or malformed masks are ignored.
    """

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

            normalized_mask = _resize_binary_mask(
                raw_mask,
                width,
                height,
            )

            if normalized_mask.any():
                instance_masks.append(normalized_mask)

    return instance_masks


def _predict_instance_masks(
    model,
    image_rgb: np.ndarray,
    confidence: float,
    max_quadrats: int,
) -> list[np.ndarray]:
    """
    Predict masks from the same decoded raster used for output creation.

    Ultralytics accepts a BGR NumPy image, so the RGB array is reversed
    for inference without re-reading or independently decoding the file.
    """

    image_bgr = np.ascontiguousarray(
        image_rgb[:, :, ::-1]
    )

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

    masks = _extract_instance_masks(
        results,
        image_rgb.shape[:2],
    )

    return masks[:max_quadrats]


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
    """
    Produce a stable output name while retaining the central output plan.

    The central planner already adds '_from_<format>' where the destination
    format differs from the source format. Do not redundantly add a source
    suffix for ordinary JPEG files whose extension matches their contents.

    If the source's extension does not match its actual decoded format,
    retain the suffix as provenance, which is especially useful for MPO
    files carried in a .JPG container.

    If two inputs share the same stem and a format change would otherwise
    make their destination names collide, include their source suffixes.
    """

    output_path = plan.output_path
    output_stem = output_path.stem

    source_format = str(
        source_metadata.get("format", "")
    ).upper()

    source_suffix = (
        source_path.suffix.lstrip(".").casefold()
    )

    registered_format = Image.registered_extensions().get(
        source_path.suffix.casefold(),
        "",
    )

    suffix_mismatch = (
        not registered_format
        or registered_format.upper() != source_format
    )

    format_changed = (
        str(plan.output_format).upper() != source_format
    )

    if suffix_mismatch or (
        format_changed and disambiguate_source_suffix
    ):
        suffix_label = source_suffix or "image"
        suffix_tag = f"_src-{suffix_label}"

        if not output_stem.casefold().endswith(
            suffix_tag.casefold()
        ):
            output_stem += suffix_tag

    if detection_index is not None:
        output_stem += f"_{detection_index:02d}"

    unique_path = output_path.with_name(
        output_stem + output_path.suffix
    )

    return replace(
        plan,
        output_path=unique_path,
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
# Background worker
# ---------------------------------------------------------------------

class DeepLearningWorker(QThread):
    """Batch-processing worker for YOLO instance segmentation."""

    progress_changed = Signal(int, int, str)
    status_changed = Signal(str)
    batch_ready = Signal(int)
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
        self.jpegtran = (
            jpegtran
            if jpegtran is not None
            else find_jpegtran()
        )

        self.cancel_event = Event()

    def request_cancel(self) -> None:
        """Request cancellation after the currently processing image."""

        self.cancel_event.set()

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
        """
        Load the selected local segmentation model.

        Ultralytics is imported inside the worker so that the main
        application can open without importing the model framework.
        """

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The Ultralytics package is required for Auto Deep "
                "Learning but could not be imported. Install a compatible "
                "Ultralytics version in the active Python environment "
                "before running this method."
            ) from exc

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"The selected model file does not exist: {self.model_path}"
            )

        model = YOLO(str(self.model_path))
        model_task = getattr(model, "task", None)

        if (
            model_task is not None
            and str(model_task).casefold() != "segment"
        ):
            raise ValueError(
                "The selected YOLO model is not an instance-segmentation "
                f"model. Detected model task: {model_task!r}. "
                "Choose a segmentation model trained to produce masks."
            )

        return model

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
        """
        Save one crop through the central output policy.

        If an explicitly planned jpegtran crop fails, re-plan the same
        derivative under the selected output policy and try the standard
        saver. Do not silently change the user's policy.
        """

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
            # A fallback is appropriate only when the central plan
            # explicitly selected the JPEG coefficient-domain route.
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
            f"Model: {self.model_label}; "
            f"Model file: {self.model_path.name}; "
            f"Confidence: {self.confidence:.4f}; "
            f"Maximum quadrats: {self.max_quadrats}; "
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

            stem_counts = Counter(
                path.stem.casefold()
                for path in self.image_files
            )

            ambiguous_stems = {
                stem
                for stem, count in stem_counts.items()
                if count > 1
            }

            self.status_changed.emit(
                f"Loading model: {self.model_label}…"
            )

            model = self._load_model()

            # Switch the progress bar from busy mode to image-count mode.
            self.batch_ready.emit(total)

            self.status_changed.emit(
                f"Model loaded. Preparing {total} image(s)…"
            )

            for index, image_path in enumerate(
                self.image_files,
                start=1,
            ):
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
                    ) = _load_rgb_image(
                        image_path,
                        source_metadata,
                    )

                    for warning in input_warnings:
                        self._record_input_warning(
                            run_log,
                            warnings,
                            image_path.name,
                            warning,
                        )

                    orientation_normalized = (
                        _orientation_was_normalized(source_metadata)
                    )

                    self.status_changed.emit(
                        f"Detecting quadrats {index}/{total}: "
                        f"{image_path.name}"
                    )

                    masks = _predict_instance_masks(
                        model,
                        image_rgb,
                        self.confidence,
                        self.max_quadrats,
                    )

                    if not masks:
                        raise ValueError(
                            "YOLO returned no valid quadrat masks at the "
                            f"selected confidence threshold "
                            f"({self.confidence:.4f}). No derivative was "
                            "saved for this image."
                        )

                    height, width = image_rgb.shape[:2]

                    combined_mask = np.zeros(
                        (height, width),
                        dtype=bool,
                    )

                    for instance_mask in masks:
                        combined_mask |= instance_mask

                    segmented_rgb = image_rgb.copy()
                    segmented_rgb[~combined_mask] = background_rgb

                    crop_jobs: list[
                        tuple[
                            int,
                            tuple[int, int, int, int],
                            object,
                        ]
                    ] = []

                    disambiguate_source_suffix = (
                        image_path.stem.casefold() in ambiguous_stems
                    )

                    for detection_index, instance_mask in enumerate(
                        masks,
                        start=1,
                    ):
                        ys, xs = np.nonzero(instance_mask)

                        if not xs.size or not ys.size:
                            continue

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

                        crop_plan = _make_plan(
                            image_path,
                            self.output_layout.cropped_images,
                            "C",
                            self.output_policy,
                            "crop",
                            crop_box=crop_box,
                            orientation_normalized=(
                                orientation_normalized
                            ),
                            jpegtran=self.jpegtran,
                        )

                        crop_plan = _make_unique_plan(
                            crop_plan,
                            image_path,
                            source_metadata,
                            detection_index=detection_index,
                            disambiguate_source_suffix=(
                                disambiguate_source_suffix
                            ),
                        )

                        crop_jobs.append(
                            (
                                detection_index,
                                crop_box,
                                crop_plan,
                            )
                        )

                    if not crop_jobs:
                        raise ValueError(
                            "YOLO produced masks, but none contained "
                            "pixels that could define a crop."
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
                                run_log,
                                warnings,
                                image_path.name,
                                warning,
                            )

                        if source_metadata["frames"] > 1:
                            if source_metadata["format"] == "MPO":
                                frame_warning = (
                                    "The source is an MPO container. Only "
                                    "its primary image was processed; "
                                    "secondary pictures or thumbnails were "
                                    "excluded from the derivatives. The "
                                    "original MPO file was not modified."
                                )
                            else:
                                frame_warning = (
                                    f"The source contains "
                                    f"{source_metadata['frames']} frames/"
                                    "pages. Only the first frame/page "
                                    "was processed."
                                )

                            self._record_input_warning(
                                run_log,
                                warnings,
                                image_path.name,
                                frame_warning,
                            )

                        deferred_warnings_recorded = True

                    try:
                        crop_summaries: list[str] = []

                        for crop_number, (
                            detection_index,
                            crop_box,
                            _initial_plan,
                        ) in enumerate(
                            crop_jobs,
                            start=1,
                        ):
                            self.status_changed.emit(
                                f"Saving crop {crop_number}/{len(crop_jobs)} "
                                f"for image {index}/{total}: "
                                f"{image_path.name}"
                            )

                            left, top, right, bottom = crop_box

                            crop_array = image_rgb[
                                top:bottom,
                                left:right,
                                :,
                            ]

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
                                    orientation_normalized=(
                                        orientation_normalized
                                    ),
                                    disambiguate_source_suffix=(
                                        disambiguate_source_suffix
                                    ),
                                    run_log=run_log,
                                    summary_warnings=warnings,
                                )
                            finally:
                                cropped_image.close()

                            saved_output_names.append(
                                crop_result.path.name
                            )

                            record_deferred_output_warnings()

                            crop_summaries.append(
                                f"{crop_result.path.name}"
                                f"(strategy={crop_result.strategy}, "
                                f"pixels_verified="
                                f"{crop_result.pixels_verified})"
                            )

                        self.status_changed.emit(
                            f"Saving segmented image {index}/{total}: "
                            f"{image_path.name}"
                        )

                        segmented_plan = _make_plan(
                            image_path,
                            self.output_layout.segmented_images,
                            "S",
                            self.output_policy,
                            "segmented",
                            crop_box=None,
                            orientation_normalized=(
                                orientation_normalized
                            ),
                            jpegtran=self.jpegtran,
                        )

                        segmented_plan = _make_unique_plan(
                            segmented_plan,
                            image_path,
                            source_metadata,
                            disambiguate_source_suffix=(
                                disambiguate_source_suffix
                            ),
                        )

                        segmented_result = save_planned_output(
                            [segmented_image],
                            segmented_plan,
                            metadata=metadata,
                            source_path=image_path,
                            source_mode=source_metadata["mode"],
                            orientation_normalized=(
                                orientation_normalized
                            ),
                            jpegtran=self.jpegtran,
                        )

                        saved_output_names.append(
                            segmented_result.path.name
                        )

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
                        f"segmented_pixels_verified="
                        f"{segmented_result.pixels_verified}; "
                        f"policy={self.output_policy.value}"
                    )

                    succeeded += 1

                    self.status_changed.emit(
                        f"Completed image {index}/{total}: "
                        f"{image_path.name}"
                    )

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

                    self.status_changed.emit(
                        f"Error in image {index}/{total}: "
                        f"{image_path.name}"
                    )

                    if saved_output_names:
                        partial_warning = (
                            "One or more derivative files were saved before "
                            "this image encountered an error. Inspect these "
                            "files and Log.txt before reusing the results: "
                            + ", ".join(saved_output_names)
                        )

                        self._record_input_warning(
                            run_log,
                            warnings,
                            image_path.name,
                            partial_warning,
                        )

                finally:
                    self.progress_changed.emit(
                        index,
                        total,
                        image_path.name,
                    )

            self.status_changed.emit("Finalizing the processing log…")

            if cancelled:
                run_log.close()
            else:
                run_log.finish()

        except Exception as exc:
            fatal_error = f"{type(exc).__name__}: {exc}"

            errors.append(
                f"Run error: {fatal_error}"
            )

            self.status_changed.emit(
                f"Processing stopped: {fatal_error}"
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
                "model_label": self.model_label,
                "model_path": str(self.model_path),
            }
        )


# ---------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------

class AutoDeepLearningDialog(QDialog):
    """Settings and batch-processing window for YOLO segmentation."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(
            "Easy Quadrat — Auto Deep Learning"
        )
        self.setMinimumWidth(680)
        self.resize(760, 720)

        self._worker: DeepLearningWorker | None = None

        configure_window(
            self,
            parent=parent,
            keep_on_top=True,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Auto · Deep Learning")
        title.setObjectName("dialogTitle")
        title.setStyleSheet(
            "font-size: 22px; "
            "font-weight: 700; "
            "color: #57863c;"
        )
        layout.addWidget(title)

        intro = QLabel(
            "Detect and segment white quadrats using trained YOLO "
            "instance-segmentation models. The standard and Frameless "
            "models are provided for different image conditions. "
            "Inspect the outputs before using them for measurements."
        )
        intro.setWordWrap(True)
        intro.setObjectName("sectionDescription")
        layout.addWidget(intro)

        # -------------------------------------------------------------
        # Input/output folders
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
        # Model selection
        # -------------------------------------------------------------

        settings = QFormLayout()
        settings.setHorizontalSpacing(12)
        settings.setVerticalSpacing(10)

        self.model_combo = QComboBox()

        for key, label in MODEL_OPTIONS:
            self.model_combo.addItem(label, key)

        self.model_combo.setCurrentIndex(0)
        self.model_combo.currentIndexChanged.connect(
            self._on_model_changed
        )

        settings.addRow(
            "Segmentation model",
            self.model_combo,
        )

        custom_model_row = QHBoxLayout()

        self.custom_model_path_edit = QLineEdit()
        self.custom_model_path_edit.setReadOnly(True)
        self.custom_model_path_edit.setPlaceholderText(
            "Choose a custom YOLO segmentation .pt model"
        )

        self.custom_model_browse_button = QPushButton(
            "Browse model…"
        )
        self.custom_model_browse_button.clicked.connect(
            self._browse_custom_model
        )

        custom_model_row.addWidget(
            self.custom_model_path_edit,
            1,
        )
        custom_model_row.addWidget(
            self.custom_model_browse_button,
        )

        settings.addRow(
            "Custom model file",
            custom_model_row,
        )

        self.model_description = QLabel()
        self.model_description.setWordWrap(True)
        self.model_description.setObjectName("cardText")
        layout.addWidget(self.model_description)

        # -------------------------------------------------------------
        # Detection settings
        # -------------------------------------------------------------

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setDecimals(4)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.25)
        self.confidence_spin.setToolTip(
            "Minimum YOLO confidence threshold. Increasing it can "
            "reduce detections, including potentially valid quadrats."
        )

        settings.addRow(
            "Confidence threshold (0–1)",
            self.confidence_spin,
        )

        self.max_quadrats_spin = QSpinBox()
        self.max_quadrats_spin.setRange(1, 1000)
        self.max_quadrats_spin.setValue(1)
        self.max_quadrats_spin.setToolTip(
            "Maximum number of instances YOLO may return per image. "
            "This is an upper limit, not a guarantee that this many "
            "quadrats will be detected."
        )

        settings.addRow(
            "Maximum quadrats per image",
            self.max_quadrats_spin,
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
            "Outputs: Cropped images/C_…; Segmented images/S_…; "
            "and Log.txt. Each detected quadrat receives its own crop; "
            "the segmented image combines the accepted masks. "
            "Lossless output is recommended when exact preservation "
            "of the processed raster is required. Custom weights must "
            "be compatible YOLO instance-segmentation models."
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

        self.run_button = QPushButton("Run Deep Learning")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumWidth(150)
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

        self._on_model_changed()

    # -----------------------------------------------------------------
    # Folder and model selection
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

    @Slot()
    def _on_model_changed(self, *_args) -> None:
        """Enable custom-model controls only for the custom option."""

        is_custom = (
            self.model_combo.currentData()
            == CUSTOM_MODEL_KEY
        )

        self.custom_model_path_edit.setEnabled(is_custom)
        self.custom_model_browse_button.setEnabled(is_custom)

        model_key = self.model_combo.currentData()

        if model_key == CUSTOM_MODEL_KEY:
            description = (
                "Use a compatible custom YOLO instance-segmentation "
                "checkpoint. Detection-only weights do not produce "
                "the masks required by this method."
            )
        else:
            description = BUILT_IN_MODELS[model_key]["description"]

        self.model_description.setText(description)

    @Slot()
    def _browse_custom_model(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select a custom YOLO segmentation model",
            str(Path.home()),
            "PyTorch model weights (*.pt)",
        )

        if selected:
            self.custom_model_path_edit.setText(
                str(Path(selected).resolve())
            )

    def _resolve_selected_model(
        self,
    ) -> tuple[Path | None, str]:
        """Return the selected local model path and display label."""

        model_key = self.model_combo.currentData()
        visible_label = self.model_combo.currentText()

        if model_key == CUSTOM_MODEL_KEY:
            raw_path = self.custom_model_path_edit.text().strip()

            if not raw_path:
                raise ValueError(
                    "Please select a custom model file."
                )

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
                f"{model_path}\n\n"
                "Check that the models folder is present beside the "
                "application or included in the PyInstaller bundle."
            )

        return model_path.resolve(), visible_label

    # -----------------------------------------------------------------
    # Dialog messages
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
            "Existing outputs",
            f"The selected output folders contain {count} existing "
            "file(s), including possibly a previous Log.txt.\n\n"
            "Files with matching output paths can be replaced. "
            "Unrelated files will not be deleted. Continue?",
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
        Select the central output policy once per batch.

        Keep JPEG is offered when the input batch contains JPEG or MPO.
        Other source formats still use the central lossless policy.
        """

        has_jpeg_source = False

        for image_path in image_files:
            try:
                source_metadata = inspect_image(image_path)
            except Exception:
                # The worker reports individual inspection errors.
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
            "pixels exactly using a suitable lossless format and verifies "
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

    def _existing_output_count(
        self,
        output_layout,
    ) -> int:
        """Count existing files that merit an overwrite warning."""

        count = 0

        for directory in (
            output_layout.cropped_images,
            output_layout.segmented_images,
        ):
            if directory is None or not directory.is_dir():
                continue

            count += sum(
                1
                for path in directory.iterdir()
                if path.is_file()
            )

        if (
            output_layout.log_file is not None
            and output_layout.log_file.is_file()
        ):
            count += 1

        return count

    # -----------------------------------------------------------------
    # Start processing
    # -----------------------------------------------------------------

    @Slot()
    def _run_processing(self) -> None:
        if (
            self._worker is not None
            and self._worker.isRunning()
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

        if model_path is None:
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
                "The selected input folder contains no supported image files.",
            )
            return

        try:
            resolved_input = input_directory.resolve()
            resolved_output = output_directory.resolve()

            if resolved_input == resolved_output:
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

        output_policy = self._choose_output_policy(
            image_files
        )

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

        existing_count = self._existing_output_count(
            output_layout
        )

        if (
            existing_count > 0
            and not self._ask_overwrite(existing_count)
        ):
            return

        controls = (
            self.input_browse_button,
            self.output_browse_button,
            self.model_combo,
            self.custom_model_path_edit,
            self.custom_model_browse_button,
            self.confidence_spin,
            self.max_quadrats_spin,
            self.background_combo,
            self.run_button,
        )

        for control in controls:
            control.setEnabled(False)

        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel processing")

        # Use a busy indicator while the model is loading. Once the
        # worker reports that the model is ready, switch to image count.
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText(
            f"Loading model: {model_label}…"
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
        )

        self._worker.status_changed.connect(
            self._on_status_changed
        )

        self._worker.batch_ready.connect(
            self._on_batch_ready
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

    @Slot(str)
    def _on_status_changed(
        self,
        status: str,
    ) -> None:
        """Show the current processing stage in the dialog."""

        self.progress_label.setText(status)

    @Slot(int)
    def _on_batch_ready(
        self,
        total: int,
    ) -> None:
        """Switch from the startup busy indicator to image-count progress."""

        self.progress_bar.setRange(
            0,
            max(1, total),
        )
        self.progress_bar.setValue(0)

        self.progress_label.setText(
            f"Model loaded. Preparing {total} image(s)…"
        )

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
            f"Completed {completed} / {total}: {filename}"
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
            self.model_combo,
            self.confidence_spin,
            self.max_quadrats_spin,
            self.background_combo,
            self.run_button,
        )

        for control in controls:
            control.setEnabled(True)

        self._on_model_changed()

        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Close")

        # If startup failed before model loading completed, the progress
        # bar may still be in busy mode. Always reset it on completion.
        self.progress_bar.setRange(
            0,
            max(1, result["total"]),
        )
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
            f"{status} {result['succeeded']} of "
            f"{result['total']} image(s) processed successfully."
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
            title = "Deep Learning error"
        elif errors:
            icon = QMessageBox.Icon.Warning
            title = "Deep Learning finished with errors"
        elif warnings:
            icon = QMessageBox.Icon.Warning
            title = "Deep Learning completed with warnings"
        else:
            icon = QMessageBox.Icon.Information
            title = "Deep Learning completed"

        self._message(
            icon,
            title,
            report,
        )

    @Slot()
    def _on_worker_finished(self) -> None:
        """Release the worker reference after its thread has returned."""

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
                "Cancellation requested; the current image will finish first."
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