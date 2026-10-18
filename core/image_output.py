"""
Easy Quadrat v2.0.0
Central image-output policy.

This module centralizes:
- Output format and filename planning.
- Lossless output encoding and pixel verification.
- JPEG re-encoding using source quantization and sampling settings when
  available, with an explicit and logged fallback.
- Verified lossless JPEG coefficient cropping and rotation.
- Metadata filtering and EXIF orientation handling.
- Atomic file replacement.
- Exact-byte copies for true no-op operations.

A lossless raster save means exact preservation of the processed raster pixels.
It does not undo intentional operations such as cropping, orientation
normalization, or recolouring outside a quadrat.

A JPEG coefficient-domain rotation does not re-encode the JPEG coefficients.
Its decoded raster is not claimed to be byte-identical to a separate
Pillow-transposed raster unless such a comparison has actually been made.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
License: MIT License; see LICENSE
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from PIL import Image, JpegImagePlugin


# ---------------------------------------------------------------------
# Shared format and pixel-mode definitions
# ---------------------------------------------------------------------

SUPPORTED_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".jpe", ".jfif",
    ".png",
    ".tif", ".tiff",
    ".bmp", ".dib",
    ".webp",
    ".mpo", ".gif",
    ".pbm", ".pgm", ".ppm", ".pnm",
    ".pcx", ".tga", ".ico",
})

JPEG_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".jpe", ".jfif",
})

JPEG_MODES = frozenset({"L", "RGB", "CMYK"})

PNG_MODES = frozenset({
    "1", "L", "LA", "P", "RGB", "RGBA",
    "I;16", "I;16L", "I;16B",
})

BMP_MODES = frozenset({"1", "L", "RGB"})

WEBP_MODES = frozenset({"RGB", "RGBA"})

_FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tif",
    "BMP": ".bmp",
    "WEBP": ".webp",
    "GIF": ".gif",
    "MPO": ".mpo",
    "PCX": ".pcx",
    "TGA": ".tga",
    "ICO": ".ico",
    "PPM": ".ppm",
    "PGM": ".pgm",
    "PBM": ".pbm",
    "PNM": ".pnm",
}

# JPEG settings used only when the source JPEG's quantization tables cannot
# safely be reused. Values above 95, especially 100, can generate very large
# files without a corresponding practical gain.
JPEG_FALLBACK_QUALITY = 92


# ---------------------------------------------------------------------
# Shared policy and result types
# ---------------------------------------------------------------------

class OutputPolicy(str, Enum):
    """Output policy shared by all processing methods."""

    LOSSLESS = "lossless"
    KEEP_JPEG = "keep_jpeg"


def normalize_policy(
    value: OutputPolicy | str | None,
) -> OutputPolicy:
    """Accept legacy UI labels while normalizing to one policy enum."""

    if isinstance(value, OutputPolicy):
        return value

    normalized = (
        str(value or "lossless")
        .strip()
        .casefold()
        .replace("-", " ")
        .replace("_", " ")
    )

    if normalized.startswith("keep jpeg") or normalized.startswith("jpeg"):
        return OutputPolicy.KEEP_JPEG

    if normalized in {
        "lossless",
        "lossless output",
        "lossless output (recommended)",
        "recommended",
    }:
        return OutputPolicy.LOSSLESS

    raise ValueError(f"Unknown output policy: {value!r}")


@dataclass(frozen=True, slots=True)
class OutputPlan:
    """Resolved destination and saving strategy for one derivative image."""

    source_path: Path
    output_path: Path

    source_format: str
    source_mode: str

    output_format: str
    output_mode: str
    output_frames: int

    operation: str
    strategy: str
    policy: OutputPolicy

    crop_box: tuple[int, int, int, int] | None = None
    reason: str = ""

    @property
    def needs_format_change(self) -> bool:
        return self.output_format.upper() != self.source_format.upper()


@dataclass(frozen=True, slots=True)
class SaveResult:
    """Result returned after an output has been verified and saved."""

    path: Path
    image_format: str
    strategy: str
    pixels_verified: bool
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------
# Image discovery and source inspection
# ---------------------------------------------------------------------

def supported_image_files(folder: str | Path) -> list[Path]:
    """List supported regular raster images in deterministic order."""

    root = Path(folder).expanduser()

    if not root.is_dir():
        raise ValueError(f"Input folder does not exist: {root}")

    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file()
            and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def inspect_image(path: str | Path) -> dict:
    """
    Read the actual decoder format, pixel mode, dimensions and frame count.

    The actual decoded format is used instead of blindly trusting the
    filename extension. TIFF BitsPerSample is also recorded where available.
    """

    path = Path(path)

    with Image.open(path) as image:
        image.seek(0)
        image.load()

        try:
            frames = max(1, int(getattr(image, "n_frames", 1)))
        except (AttributeError, TypeError, ValueError, EOFError):
            frames = 1

        fmt = (image.format or "").upper()

        try:
            orientation = int(image.getexif().get(274, 1))
        except (TypeError, ValueError, OverflowError):
            orientation = 1

        bits_per_sample = None

        if fmt == "TIFF":
            try:
                raw_bits = image.tag_v2.get(258)

                if raw_bits is not None:
                    if isinstance(raw_bits, (tuple, list)):
                        bits_per_sample = tuple(
                            int(value) for value in raw_bits
                        )
                    else:
                        bits_per_sample = (int(raw_bits),)

            except (
                AttributeError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                bits_per_sample = None

        return {
            "path": path,
            "format": fmt,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "frames": frames,
            "exif_orientation": orientation,
            "bits_per_sample": bits_per_sample,
        }


# ---------------------------------------------------------------------
# JPEG coefficient-transform support
# ---------------------------------------------------------------------

def find_jpegtran() -> str | None:
    """
    Find the bundled jpegtran first, then use an optional system installation.

    The bundled copy makes the application independent of the development
    computer's libjpeg-turbo installation path.
    """

    roots: list[Path] = []

    # Source project root, or the equivalent application-internal root
    # when this module is bundled by PyInstaller.
    try:
        roots.append(Path(__file__).resolve().parents[1])
    except (OSError, RuntimeError, IndexError):
        pass

    # PyInstaller's bundled data directory.
    bundle_root = getattr(sys, "_MEIPASS", None)

    if bundle_root:
        try:
            roots.append(Path(bundle_root).resolve())
        except (OSError, RuntimeError):
            pass

    # Directory containing the Python interpreter or packaged executable.
    try:
        roots.append(Path(sys.executable).resolve().parent)
    except (OSError, RuntimeError):
        pass

    # Current working directory is a final project-development location.
    try:
        roots.append(Path.cwd())
    except (OSError, RuntimeError):
        pass

    candidates = (
        # Preferred project-bundled location.
        Path("vendor/libjpeg-turbo/win-x64/jpegtran.exe"),
        Path("vendor/libjpeg-turbo/win-x64/jpegtran"),

        # Backward-compatible locations used by earlier development setups.
        Path("assets/jpegtran.exe"),
        Path("bin/jpegtran.exe"),
        Path("vendor/jpegtran.exe"),
        Path("jpegtran.exe"),
        Path("assets/jpegtran"),
        Path("bin/jpegtran"),
        Path("vendor/jpegtran"),
        Path("jpegtran"),
    )

    seen: set[Path] = set()

    for root in roots:
        for relative in candidates:
            try:
                candidate = (root / relative).resolve()

                if candidate in seen:
                    continue

                seen.add(candidate)

                if candidate.is_file():
                    return str(candidate)

            except (OSError, RuntimeError):
                continue

    # Optional fallback for developers who have jpegtran installed globally.
    # A bundled project copy always takes precedence.
    return shutil.which("jpegtran.exe") or shutil.which("jpegtran")


def jpeg_mcu_size(
    image: Image.Image,
) -> tuple[int, int] | None:
    """Return JPEG iMCU dimensions based on component sampling information."""

    if (image.format or "").upper() != "JPEG":
        return None

    layers = getattr(image, "layer", None)

    if layers:
        try:
            max_h = max(int(component[1]) for component in layers)
            max_v = max(int(component[2]) for component in layers)

            if max_h > 0 and max_v > 0:
                return 8 * max_h, 8 * max_v

        except (TypeError, ValueError, IndexError):
            pass

    try:
        sampling = JpegImagePlugin.get_sampling(image)
    except Exception:
        return None

    return {
        0: (8, 8),
        1: (16, 8),
        2: (16, 16),
    }.get(sampling)


def jpeg_crop_is_aligned(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
) -> bool:
    """
    Check whether JPEG coefficient cropping can represent the exact rectangle.

    An aligned rectangle is necessary, but final output is still verified
    after jpegtran has run.
    """

    mcu = jpeg_mcu_size(image)

    if mcu is None:
        return False

    left, top, right, bottom = map(int, crop_box)
    width, height = image.size

    if not (
        0 <= left < right <= width
        and 0 <= top < bottom <= height
    ):
        return False

    mcu_width, mcu_height = mcu

    return (
        left % mcu_width == 0
        and top % mcu_height == 0
        and (right == width or right % mcu_width == 0)
        and (bottom == height or bottom % mcu_height == 0)
    )


def jpeg_rotate_270_is_perfect(
    image: Image.Image,
) -> bool:
    """
    Check the relevant iMCU boundary for a perfect 270-degree JPEG rotation.

    jpegtran's 270-degree rotation is equivalent to a 90-degree
    counter-clockwise rotation. A perfect transform requires the source
    width to be divisible by its iMCU width. The saver also supplies
    jpegtran's -perfect switch, so the tool itself rejects an imperfect
    transform rather than trimming or altering an edge.
    """

    mcu = jpeg_mcu_size(image)

    if mcu is None:
        return False

    mcu_width, _mcu_height = mcu

    if mcu_width <= 0:
        return False

    return image.width % mcu_width == 0


# ---------------------------------------------------------------------
# Lossless output-format selection
# ---------------------------------------------------------------------

def _supported_lossless_format(
    source_format: str,
    image_mode: str,
    frame_count: int,
) -> str:
    """
    Select a conservative lossless output container.

    Source format is retained when it can safely represent the processed
    pixels. Multi-frame derived images use TIFF pages so raster frames are
    not silently collapsed into a single image.
    """

    source_format = (source_format or "").upper()

    if frame_count > 1:
        return "TIFF"

    if source_format == "TIFF":
        return "TIFF"

    if source_format == "PNG" and image_mode in PNG_MODES:
        return "PNG"

    if source_format == "BMP" and image_mode in BMP_MODES:
        return "BMP"

    if source_format == "WEBP" and image_mode in WEBP_MODES:
        return "WEBP"

    if image_mode in PNG_MODES:
        return "PNG"

    return "TIFF"


def _source_extension_matches(
    path: Path,
    image_format: str,
) -> bool:
    """Check the extension against Pillow's registered format mapping."""

    registered = Image.registered_extensions()

    return (
        registered.get(path.suffix.casefold(), "").upper()
        == image_format.upper()
    )


def _planned_path(
    source_path: Path,
    output_directory: Path,
    prefix: str,
    source_format: str,
    output_format: str,
) -> Path:
    """Create a deterministic path with an extension matching the output."""

    source_format = source_format.upper()
    output_format = output_format.upper()
    prefix = prefix.strip().rstrip("_")

    if not prefix:
        raise ValueError("An output filename prefix is required.")

    if (
        source_format == output_format
        and _source_extension_matches(source_path, source_format)
    ):
        return output_directory / f"{prefix}_{source_path.name}"

    extension = _FORMAT_EXTENSIONS.get(output_format)

    if extension is None:
        extension = next(
            (
                ext
                for ext, fmt in Image.registered_extensions().items()
                if fmt.upper() == output_format
            ),
            None,
        )

    if extension is None:
        raise ValueError(
            f"No reliable filename extension is registered for "
            f"{output_format!r}."
        )

    source_label = (
        source_format.casefold()
        or source_path.suffix.lstrip(".").casefold()
        or "image"
    )

    return (
        output_directory
        / f"{prefix}_{source_path.stem}_from_{source_label}{extension}"
    )


# ---------------------------------------------------------------------
# Output planning
# ---------------------------------------------------------------------

def plan_output(
    source_path: str | Path,
    output_directory: str | Path,
    prefix: str,
    policy: OutputPolicy | str | None,
    operation: str,
    *,
    output_mode: str | None = None,
    output_frames: int = 1,
    crop_box: tuple[int, int, int, int] | None = None,
    orientation_normalized: bool = False,
    jpegtran: str | None = None,
) -> OutputPlan:
    """
    Choose the destination format, path and encoder strategy.

    Parameters
    ----------
    source_path:
        Original image path; the actual decoded format is inspected.
    output_directory:
        Destination folder for this derivative.
    prefix:
        Usually C for cropped images or S for segmented images.
    policy:
        Lossless by default, or explicit Keep JPEG.
    operation:
        Either "crop" or "segmented".
    output_mode:
        Expected PIL mode of the processed derivative.
    output_frames:
        Number of raster frames/pages the derivative contains. This is not
        necessarily the same as the source container's frame count.
    crop_box:
        Integer coordinates used for a rectangular crop, when applicable.
    orientation_normalized:
        True when pixels have been spatially normalized. For a JPEG full-frame
        quarter-turn, this can identify a pure orientation transform, but the
        lossless raster policy still uses a pixel-preserving output container.
    jpegtran:
        Optional explicit jpegtran path. If omitted, a compatible executable
        is searched for automatically.
    """

    source_path = Path(source_path).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()

    if operation not in {"crop", "segmented"}:
        raise ValueError("operation must be 'crop' or 'segmented'.")

    if output_frames < 1:
        raise ValueError("output_frames must be at least 1.")

    selected_policy = normalize_policy(policy)
    meta = inspect_image(source_path)

    source_format = meta["format"]
    source_mode = meta["mode"]
    final_mode = output_mode or source_mode

    if not source_format:
        raise ValueError(
            f"Could not determine the actual format of {source_path.name}."
        )

    # Pillow may expose a high-bit-depth, multichannel TIFF as an ordinary
    # RGB/RGBA mode. Do not silently publish a lower-precision derivative.
    bits = meta.get("bits_per_sample")

    if (
        source_format == "TIFF"
        and source_mode in {"RGB", "RGBA"}
        and bits
        and max(bits) > 8
    ):
        raise ValueError(
            f"{source_path.name} is a high-bit-depth {source_mode} TIFF "
            f"({bits} bits per sample). This Pillow-based processing path "
            "cannot guarantee preservation of its native channel precision; "
            "the file was not assigned an output plan."
        )

    jpegtran = jpegtran if jpegtran is not None else find_jpegtran()

    can_jpegtran_crop = False

    if (
        operation == "crop"
        and output_frames == 1
        and crop_box is not None
        and not orientation_normalized
        and source_format == "JPEG"
        and jpegtran is not None
        and meta["exif_orientation"] == 1
    ):
        try:
            with Image.open(source_path) as source:
                source.seek(0)
                can_jpegtran_crop = jpeg_crop_is_aligned(
                    source,
                    crop_box,
                )
        except Exception:
            can_jpegtran_crop = False

    # Identify a full-image quarter-turn requested by Fixed Frame.
    #
    # Fixed Frame passes a full-frame crop rectangle measured in the
    # post-rotation coordinate system. For a 90-degree rotation, its width
    # and height match the source height and width, respectively. The
    # orientation tag must be 1 so that jpegtran's physical rotation will
    # not conflict with a non-default EXIF orientation.
    rotation_only = (
        operation == "crop"
        and output_frames == 1
        and crop_box is not None
        and orientation_normalized
        and source_format == "JPEG"
        and meta["exif_orientation"] == 1
        and tuple(map(int, crop_box))
        == (
            0,
            0,
            int(meta["height"]),
            int(meta["width"]),
        )
    )

    can_jpegtran_rotate = False

    if (
        rotation_only
        and jpegtran is not None
        and final_mode in JPEG_MODES
    ):
        try:
            with Image.open(source_path) as source:
                source.seek(0)
                can_jpegtran_rotate = jpeg_rotate_270_is_perfect(source)
        except Exception:
            can_jpegtran_rotate = False

    # Keep JPEG means JPEG is retained when the source itself is JPEG or MPO
    # and the output has one compatible raster frame. A lossless coefficient
    # rotation is preferred to re-encoding when the user selects Keep JPEG
    # and the transform is demonstrably perfect.
    jpeg_source = source_format in {"JPEG", "MPO"}

    if (
        rotation_only
        and selected_policy is OutputPolicy.KEEP_JPEG
        and can_jpegtran_rotate
    ):
        output_format = "JPEG"
        strategy = "jpegtran_rotate"
        reason = (
            "Perfect 90-degree counter-clockwise JPEG coefficient rotation "
            "is available. jpegtran -perfect will reject an imperfect "
            "transform; the JPEG coefficients are not re-encoded."
        )

    elif (
        jpeg_source
        and output_frames == 1
        and final_mode in JPEG_MODES
    ):
        if operation == "crop" and can_jpegtran_crop:
            output_format = "JPEG"
            strategy = "jpegtran"
            reason = (
                "Exact JPEG coefficient crop; "
                "no JPEG re-encoding is required."
            )

        elif selected_policy is OutputPolicy.KEEP_JPEG:
            output_format = "JPEG"
            strategy = "jpeg_quality"
            reason = (
                "JPEG is retained by explicit policy; "
                "re-encoding remains lossy. The source quantization tables "
                "and chroma sampling will be retained when safely available."
            )

        else:
            output_format = _supported_lossless_format(
                "JPEG",
                final_mode,
                1,
            )
            strategy = "lossless"

            if rotation_only:
                reason = (
                    "Lossless raster rotation uses a non-JPEG container "
                    "to preserve the processed pixel values exactly. "
                    "JPEG coefficient rotation is not used for this policy "
                    "because this policy verifies the decoded raster."
                )
            else:
                reason = (
                    "Lossless output requires a non-JPEG container "
                    "for this derivative."
                )

    else:
        output_format = _supported_lossless_format(
            source_format,
            final_mode,
            output_frames,
        )
        strategy = "lossless"

        if (
            selected_policy is OutputPolicy.KEEP_JPEG
            and not jpeg_source
        ):
            reason = (
                "Keep JPEG applies only to JPEG/MPO-derived outputs; "
                "this source remains lossless."
            )

        elif output_format != source_format:
            reason = (
                f"{source_format} cannot safely represent this derivative "
                "under the selected lossless policy."
            )

        else:
            reason = (
                "The selected output format can preserve "
                "the processed raster pixels."
            )

    output_path = _planned_path(
        source_path,
        output_directory,
        prefix,
        source_format,
        output_format,
    )

    return OutputPlan(
        source_path=source_path,
        output_path=output_path,
        source_format=source_format,
        source_mode=source_mode,
        output_format=output_format,
        output_mode=final_mode,
        output_frames=output_frames,
        operation=operation,
        strategy=strategy,
        policy=selected_policy,
        crop_box=crop_box,
        reason=reason,
    )


def plan_exact_copy(
    source_path: str | Path,
    output_directory: str | Path,
    prefix: str = "C",
) -> tuple[Path, str]:
    """
    Plan an exact-byte no-op output.

    The original suffix is kept when it is a valid source suffix for the
    actual format. MPO is a special case: MPO data may be stored in files
    using the conventional .JPG/.JPEG suffix. For a true no-op copy, keep
    that original suffix instead of changing it to .mpo. No conversion is
    performed by this decision.
    """

    source = Path(source_path).expanduser().resolve()
    output_dir = Path(output_directory).expanduser().resolve()

    meta = inspect_image(source)
    image_format = meta["format"]

    if not image_format:
        raise ValueError(
            f"Could not determine the actual format of {source.name}."
        )

    suffix = source.suffix.casefold()

    mpo_with_jpeg_suffix = (
        image_format == "MPO"
        and suffix in JPEG_EXTENSIONS
    )

    if (
        _source_extension_matches(source, image_format)
        or mpo_with_jpeg_suffix
    ):
        return (
            output_dir
            / f"{prefix.strip().rstrip('_')}_{source.name}",
            image_format,
        )

    return (
        _planned_path(
            source,
            output_dir,
            prefix,
            image_format,
            image_format,
        ),
        image_format,
    )


# ---------------------------------------------------------------------
# Metadata safety
# ---------------------------------------------------------------------

def _colour_model(mode: str) -> str:
    if mode in {
        "1", "L", "LA", "I", "F",
        "I;16", "I;16L", "I;16B",
    }:
        return "GRAY"

    if mode in {"RGB", "RGBA"}:
        return "RGB"

    if mode == "P":
        return "PALETTE"

    if mode == "CMYK":
        return "CMYK"

    if mode == "YCbCr":
        return "YCBCR"

    return mode.upper()


def _exif_without_orientation(
    exif_data: bytes | bytearray | None,
) -> bytes | None:
    """Remove EXIF Orientation after pixels have actually been normalized."""

    if not exif_data:
        return None

    try:
        exif = Image.Exif()
        exif.load(exif_data)

        if 274 in exif:
            del exif[274]

        return exif.tobytes()

    except Exception:
        # A contradictory orientation tag is worse than omitting EXIF.
        return None


# NEW: MPO metadata filtering is deliberately narrow. It does not change
# metadata handling for JPEG, PNG, TIFF, BMP, WebP, or other non-MPO sources.
_MPO_CONTAINER_SENSITIVE_METADATA_KEYS = frozenset({
    "exif",
    "xmp",
    "xml:com.adobe.xmp",
})


def _is_mpo_container_sensitive_metadata_key(key: object) -> bool:
    """
    Identify metadata fields that must not be copied blindly from an MPO
    container to a standalone derivative.

    This is intentionally limited to EXIF/XMP fields. ICC profiles, DPI,
    and other independently meaningful metadata are not excluded by this
    helper.
    """

    return (
        str(key).strip().casefold()
        in _MPO_CONTAINER_SENSITIVE_METADATA_KEYS
    )


def _metadata_options(
    metadata: dict | None,
    output_format: str,
    source_mode: str,
    output_mode: str,
    orientation_normalized: bool,
) -> tuple[dict, list[str]]:
    """
    Pass only metadata options suitable for the destination encoder.

    Arbitrary TIFF tag dictionaries are not copied directly. Some tags may
    be unsupported by the installed TIFF encoder, and spatial tags may no
    longer describe the image correctly after cropping.

    JPEG compression parameters are deliberately NOT fixed here. The JPEG
    saver obtains the source JPEG's quantization tables and sampling settings
    separately, because a processed Pillow image is usually no longer a
    JpegImageFile and cannot safely use quality="keep" directly.
    """

    metadata = dict(metadata or {})
    fmt = output_format.upper()
    warnings: list[str] = []

    icc_profile = metadata.get("icc_profile")

    if (
        icc_profile
        and _colour_model(source_mode) != _colour_model(output_mode)
    ):
        metadata.pop("icc_profile", None)
        warnings.append(
            "The source ICC profile was not copied because the output "
            "pixel colour model changed."
        )

    if orientation_normalized:
        cleaned_exif = _exif_without_orientation(
            metadata.get("exif")
        )

        if metadata.get("exif") and not cleaned_exif:
            warnings.append(
                "EXIF metadata was omitted because its Orientation tag "
                "could not be safely removed."
            )

        if cleaned_exif:
            metadata["exif"] = cleaned_exif
        else:
            metadata.pop("exif", None)

    if fmt == "TIFF" and metadata.get("tiffinfo") is not None:
        # Do not pass an arbitrary source TIFF tag dictionary to libtiff.
        # Unsupported tag values can make the encoder fail, and crop-related
        # offsets/georeferencing should not be copied unchanged.
        metadata.pop("tiffinfo", None)
        warnings.append(
            "The original TIFF-specific tag dictionary was not copied. "
            "Some tags may be unsupported by the destination encoder or "
            "may no longer be valid after cropping."
        )

    allowed_options = {
        "JPEG": {"icc_profile", "dpi", "exif", "xmp"},
        "PNG": {
            "icc_profile",
            "dpi",
            "exif",
            "transparency",
            "xmp",
            "pnginfo",
        },
        "TIFF": {
            "icc_profile",
            "dpi",
            "exif",
            "description",
            "software",
            "date_time",
            "artist",
            "copyright",
        },
        "WEBP": {"icc_profile", "exif", "xmp"},
        "BMP": {"dpi"},
    }.get(fmt, set())

    options = {
        key: metadata[key]
        for key in allowed_options
        if metadata.get(key) is not None
    }

    if fmt == "JPEG":
        # Do not set quality=100 or force 4:4:4 here. The saving path will
        # reuse source JPEG coding parameters where safely available.
        options.update({
            "optimize": True,
            "progressive": False,
        })

    elif fmt == "PNG":
        options.setdefault("compress_level", 6)

        # Animated timing/loop properties are not ordinary raster pixels.
        # Multi-frame derivatives are planned as TIFF pages.
        for key in ("duration", "loop", "default_image"):
            options.pop(key, None)

    elif fmt == "WEBP":
        options.update({
            "lossless": True,
            "quality": 100,
            "method": 6,
            "exact": True,
        })

    return options, warnings


# ---------------------------------------------------------------------
# Source JPEG encoding-parameter recovery
# ---------------------------------------------------------------------

def _normalise_quantization_tables(
    tables: object,
) -> dict[int, list[int]] | None:
    """
    Return a validated, contiguous mapping of JPEG quantization tables.

    The returned tables can be passed to Pillow's JPEG encoder without
    specifying a quality value that would rescale those tables.

    A missing, malformed, non-contiguous, or out-of-range table set is
    rejected rather than silently replaced with an assumed source setting.
    """

    if not isinstance(tables, dict) or not tables:
        return None

    normalized: dict[int, list[int]] = {}

    try:
        for raw_table_id, raw_values in tables.items():
            table_id = int(raw_table_id)
            values = [int(value) for value in raw_values]

            if table_id < 0:
                return None

            if len(values) != 64:
                return None

            # JPEG quantization values must be positive. Pillow's JPEG
            # encoder transports quantization entries as unsigned 16-bit
            # values, so avoid accepting values outside that range.
            if any(value < 1 or value > 65535 for value in values):
                return None

            normalized[table_id] = values

    except (TypeError, ValueError, OverflowError):
        return None

    if not 1 <= len(normalized) <= 4:
        return None

    # Require table IDs 0..N-1. Pillow accepts indexed sequences of tables;
    # rejecting gaps prevents a silent change in table ordering.
    if set(normalized) != set(range(len(normalized))):
        return None

    return {
        table_id: normalized[table_id]
        for table_id in sorted(normalized)
    }


def _quantization_signature(
    tables: object,
) -> tuple[tuple[int, tuple[int, ...]], ...] | None:
    """Make quantization-table comparisons independent of list identity."""

    normalized = _normalise_quantization_tables(tables)

    if normalized is None:
        return None

    return tuple(
        (table_id, tuple(values))
        for table_id, values in sorted(normalized.items())
    )


def _sampling_name(sampling: int | None) -> str:
    """Return an explicit human-readable JPEG chroma-sampling label."""

    return {
        0: "4:4:4",
        1: "4:2:2",
        2: "4:2:0",
    }.get(sampling, "unknown")


def _jpeg_encoding_options_from_source(
    source_path: str | Path,
    output_mode: str,
) -> tuple[dict, list[str]]:
    """
    Construct JPEG encoder options using the actual source JPEG/MPO.

    For compatible source and output color models, preserve the source
    quantization tables. For RGB, also preserve the source chroma sampling
    when it can be identified.

    These settings reduce avoidable changes to the JPEG encoding policy.
    They do NOT make arbitrary cropped/re-encoded JPEG output lossless:
    decoded pixel values can still change when the derivative is encoded.

    If the source tables cannot be safely reused, use a moderate fallback
    quality setting rather than quality=100, and make that fallback explicit
    in the warnings returned to the caller.
    """

    source_path = Path(source_path)

    options: dict = {
        "optimize": True,
        "progressive": False,
    }

    warnings: list[str] = []

    source_format = ""
    source_mode = ""
    source_tables: dict[int, list[int]] | None = None
    source_sampling: int | None = None

    try:
        with Image.open(source_path) as source:
            source.seek(0)
            source.load()

            source_format = (source.format or "").upper()
            source_mode = source.mode

            if source_format in {"JPEG", "MPO"}:
                source_tables = _normalise_quantization_tables(
                    getattr(source, "quantization", None)
                )

                try:
                    sampling_value = int(
                        JpegImagePlugin.get_sampling(source)
                    )

                    if sampling_value in {0, 1, 2}:
                        source_sampling = sampling_value

                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                ):
                    source_sampling = None

    except Exception as exc:
        warnings.append(
            "Could not read the source JPEG encoding parameters "
            f"({type(exc).__name__}: {exc}). "
            "A documented fallback JPEG encoding will be used."
        )

    source_is_jpeg = source_format in {"JPEG", "MPO"}

    same_colour_model = (
        bool(source_mode)
        and _colour_model(source_mode) == _colour_model(output_mode)
    )

    tables_were_reused = (
        source_is_jpeg
        and same_colour_model
        and source_tables is not None
    )

    if tables_were_reused:
        # Do not add "quality" when using explicit qtables. Pillow's encoder
        # can otherwise rescale the supplied tables according to quality.
        options["qtables"] = source_tables

    else:
        options["quality"] = JPEG_FALLBACK_QUALITY

        if not source_is_jpeg:
            warnings.append(
                "The actual source container could not provide JPEG "
                "quantization tables; fallback JPEG quality="
                f"{JPEG_FALLBACK_QUALITY} was used."
            )

        elif not same_colour_model:
            warnings.append(
                "The source JPEG color model differs from the output "
                "color model, so source quantization tables were not reused. "
                "Fallback JPEG quality="
                f"{JPEG_FALLBACK_QUALITY} was used."
            )

        else:
            warnings.append(
                "Usable source JPEG quantization tables were unavailable. "
                "Fallback JPEG quality="
                f"{JPEG_FALLBACK_QUALITY} was used."
            )

    if output_mode == "RGB":
        if source_is_jpeg and source_sampling in {0, 1, 2}:
            options["subsampling"] = source_sampling
        else:
            # Avoid introducing additional chroma subsampling when the
            # source sampling cannot be identified. This fallback is
            # intentionally visible in the returned warnings.
            options["subsampling"] = 0

            warnings.append(
                "The source chroma-sampling mode could not be recovered. "
                "RGB output will use 4:4:4 sampling to avoid additional "
                "chroma subsampling."
            )

    if tables_were_reused:
        if output_mode == "RGB":
            sampling_description = _sampling_name(
                source_sampling if source_sampling in {0, 1, 2} else None
            )

            if source_sampling in {0, 1, 2}:
                warnings.append(
                    "JPEG re-encoding reused the source quantization "
                    f"tables and {sampling_description} chroma sampling. "
                    "The JPEG bitstream is still newly encoded, so decoded "
                    "pixel values may change."
                )
            else:
                warnings.append(
                    "JPEG re-encoding reused the source quantization "
                    "tables, but the source chroma-sampling mode could not "
                    "be recovered; 4:4:4 sampling was used. The JPEG "
                    "bitstream is newly encoded, so decoded pixel values "
                    "may change."
                )
        else:
            warnings.append(
                "JPEG re-encoding reused the source quantization tables. "
                "The JPEG bitstream is still newly encoded, so decoded "
                "pixel values may change."
            )

    return options, warnings


# ---------------------------------------------------------------------
# Lossless verification
# ---------------------------------------------------------------------

def _frame_count(image: Image.Image) -> int:
    try:
        return max(1, int(getattr(image, "n_frames", 1)))
    except (AttributeError, TypeError, ValueError, EOFError):
        return 1


def _palette_signature(image: Image.Image):
    if image.mode != "P":
        return None

    return (
        image.getpalette(),
        image.info.get("transparency"),
    )


def verify_lossless_output(
    path: str | Path,
    expected_frames: Sequence[Image.Image],
) -> None:
    """
    Raise unless frame count, size, pixel mode and decoded pixel values match.

    Indexed-palette images are checked for palette and transparency equality,
    not only equality of their integer palette indices.
    """

    path = Path(path)

    if not expected_frames:
        raise ValueError("There are no expected frames to verify.")

    with Image.open(path) as saved:
        actual_count = _frame_count(saved)

        if actual_count != len(expected_frames):
            raise RuntimeError(
                "Lossless verification failed: expected "
                f"{len(expected_frames)} frame(s), found {actual_count}."
            )

        for index, expected in enumerate(expected_frames):
            saved.seek(index)
            saved.load()

            if saved.size != expected.size:
                raise RuntimeError(
                    "Lossless verification failed: dimensions differ "
                    f"in frame {index + 1}."
                )

            if saved.mode != expected.mode:
                raise RuntimeError(
                    "Lossless verification failed: mode changed from "
                    f"{expected.mode!r} to {saved.mode!r} "
                    f"in frame {index + 1}."
                )

            if saved.tobytes() != expected.tobytes():
                raise RuntimeError(
                    "Lossless verification failed: decoded pixel bytes "
                    f"differ in frame {index + 1}."
                )

            if (
                _palette_signature(saved)
                != _palette_signature(expected)
            ):
                raise RuntimeError(
                    "Lossless verification failed: palette/transparency "
                    f"differs in frame {index + 1}."
                )


# ---------------------------------------------------------------------
# Atomic lossless saving
# ---------------------------------------------------------------------

def _save_lossless_temp(
    frames: Sequence[Image.Image],
    temporary: Path,
    output_format: str,
    options: dict,
) -> list[str]:
    """
    Encode frames without lossy compression.

    For TIFF, prevent inherited source EXIF/TIFF metadata from being silently
    merged into Pillow's encoder settings. Try lossless compressed TIFF output
    first and progressively reduce optional metadata only when necessary.

    The caller must verify all decoded output pixels before final replacement.
    """

    fmt = output_format.upper()

    if not frames:
        raise ValueError("There are no image frames to save.")

    if fmt == "JPEG":
        raise ValueError("JPEG is not a lossless output format.")

    if len(frames) > 1 and fmt != "TIFF":
        raise ValueError(
            "Multi-frame lossless outputs use TIFF to preserve raster pages "
            "reliably in this version."
        )

    if fmt == "BMP" and len(frames) != 1:
        raise ValueError("BMP cannot store multiple raster frames.")

    if fmt == "WEBP" and len(frames) != 1:
        raise ValueError(
            "This WebP lossless path supports one raster frame only."
        )

    if fmt not in {"PNG", "TIFF", "BMP", "WEBP"}:
        raise ValueError(
            f"Unsupported lossless output format: {fmt}."
        )

    if fmt != "TIFF":
        temporary.write_bytes(b"")

        frames[0].save(
            temporary,
            format=fmt,
            **dict(options),
        )

        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(
                f"The {fmt} encoder did not create a non-empty file."
            )

        return []

    # Snapshot the frames' original metadata so it can be restored even if
    # encoding fails. These are processed frames, not the source files.
    original_infos = [
        dict(frame.info)
        for frame in frames
    ]

    inherited_exif = (
        bool(options.get("exif"))
        or any(bool(info.get("exif")) for info in original_infos)
    )

    inherited_tiffinfo = (
        bool(options.get("tiffinfo"))
        or any(bool(info.get("tiffinfo")) for info in original_infos)
    )

    try:
        # Pillow may merge image.info into the encoder's parameters.
        # Clear inherited entries for the TIFF save, then pass only the
        # metadata selected explicitly below.
        for frame in frames:
            frame.info.clear()

        full_options = dict(options)

        # Raw JPEG/MPO EXIF is not blindly transferable into the new TIFF.
        # Likewise, don't pass an arbitrary TIFF tag dictionary through.
        full_options.pop("exif", None)
        full_options.pop("tiffinfo", None)

        metadata_variants: list[tuple[str, dict]] = []
        seen_signatures: set[tuple[str, ...]] = set()

        def add_metadata_variant(
            description: str,
            candidate: dict,
        ) -> None:
            signature = tuple(sorted(candidate.keys()))

            if signature in seen_signatures:
                return

            seen_signatures.add(signature)
            metadata_variants.append(
                (description, dict(candidate))
            )

        add_metadata_variant(
            "selected safe metadata",
            full_options,
        )

        # If a remaining optional field is incompatible with the TIFF encoder,
        # retry without that field while preserving the others.
        for key in (
            "description",
            "software",
            "date_time",
            "artist",
            "copyright",
            "icc_profile",
            "dpi",
        ):
            if key not in full_options:
                continue

            candidate = dict(full_options)
            candidate.pop(key, None)

            add_metadata_variant(
                f"{key} omitted",
                candidate,
            )

        minimal_metadata = {
            key: full_options[key]
            for key in ("icc_profile", "dpi")
            if full_options.get(key) is not None
        }

        add_metadata_variant(
            "ICC profile and DPI only",
            minimal_metadata,
        )
        add_metadata_variant(
            "no optional metadata",
            {},
        )

        compression_methods = [
            "tiff_adobe_deflate",
            "tiff_lzw",
            "packbits",
            None,
        ]

        failures: list[str] = []
        last_error: Exception | None = None

        for metadata_description, attempt_options in metadata_variants:
            for compression in compression_methods:
                save_options = dict(attempt_options)

                if len(frames) > 1:
                    save_options["save_all"] = True
                    save_options["append_images"] = list(frames[1:])

                # A failed attempt must not leave bytes that could be
                # mistaken for a completed output.
                temporary.write_bytes(b"")

                try:
                    if compression is None:
                        frames[0].save(
                            temporary,
                            format="TIFF",
                            **save_options,
                        )
                    else:
                        frames[0].save(
                            temporary,
                            format="TIFF",
                            compression=compression,
                            **save_options,
                        )

                    if (
                        not temporary.is_file()
                        or temporary.stat().st_size == 0
                    ):
                        raise RuntimeError(
                            "The TIFF encoder did not create a non-empty file."
                        )

                    warnings: list[str] = []

                    if inherited_exif:
                        warnings.append(
                            "Raw inherited EXIF metadata was omitted from "
                            "the TIFF output because the source EXIF IFD "
                            "offset was incompatible with safe TIFF encoding. "
                            "Raster pixel values were not changed."
                        )

                    if inherited_tiffinfo:
                        warnings.append(
                            "Inherited TIFF-specific tag dictionaries were "
                            "not copied because their offsets and values may "
                            "not remain valid in the new TIFF."
                        )

                    omitted_keys = sorted(
                        set(full_options) - set(attempt_options)
                    )

                    if omitted_keys:
                        warnings.append(
                            "Additional optional metadata fields were omitted: "
                            + ", ".join(omitted_keys)
                            + "."
                        )

                    if compression is None:
                        warnings.append(
                            "TIFF was saved using the default compression "
                            "after all explicit compressed methods failed. "
                            "The file may be larger."
                        )
                    elif compression != "tiff_adobe_deflate":
                        warnings.append(
                            f"TIFF was saved using {compression} compression "
                            "instead of the preferred Adobe Deflate method."
                        )

                    if failures:
                        warnings.append(
                            "Earlier TIFF encoder attempts failed: "
                            + " | ".join(failures)
                        )

                    if metadata_description != "selected safe metadata":
                        warnings.append(
                            "Successful TIFF metadata configuration: "
                            f"{metadata_description}."
                        )

                    return warnings

                except Exception as exc:
                    last_error = exc

                    failures.append(
                        f"compression={compression or 'default'}, "
                        f"metadata={metadata_description}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        raise RuntimeError(
            "TIFF saving failed after all compression and metadata "
            "fallbacks. "
            f"Last error: {type(last_error).__name__}: {last_error}. "
            "Attempt details: " + " | ".join(failures)
        ) from last_error

    finally:
        # Restore info for the caller, including when every encoder attempt
        # fails. The processed images remain reusable until the worker closes
        # them.
        for frame, original_info in zip(frames, original_infos):
            frame.info.clear()
            frame.info.update(original_info)


# ---------------------------------------------------------------------
# JPEG verification
# ---------------------------------------------------------------------

def _verify_jpeg_output(
    path: Path,
    expected: Image.Image,
    *,
    expected_quantization_tables: object = None,
    expected_subsampling: int | None = None,
) -> None:
    """
    Check JPEG format, dimensions, pixel mode, and applicable encoder settings.

    This is not a pixel-losslessness test. A JPEG re-encoding can change
    decoded pixel values even when its quantization tables and sampling match
    the source.
    """

    with Image.open(path) as actual:
        actual.load()

        if (actual.format or "").upper() != "JPEG":
            raise RuntimeError(
                "JPEG verification failed: output file is not JPEG."
            )

        if actual.size != expected.size:
            raise RuntimeError(
                "JPEG verification failed: output dimensions changed."
            )

        if actual.mode != expected.mode:
            raise RuntimeError(
                "JPEG verification failed: output mode changed from "
                f"{expected.mode!r} to {actual.mode!r}."
            )

        if expected_quantization_tables is not None:
            expected_signature = _quantization_signature(
                expected_quantization_tables
            )

            actual_signature = _quantization_signature(
                getattr(actual, "quantization", None)
            )

            if expected_signature is None:
                raise RuntimeError(
                    "JPEG verification could not validate the expected "
                    "quantization tables."
                )

            if actual_signature != expected_signature:
                raise RuntimeError(
                    "JPEG verification failed: output quantization tables "
                    "differ from the source tables selected for reuse."
                )

        if expected_subsampling is not None:
            try:
                actual_subsampling = int(
                    JpegImagePlugin.get_sampling(actual)
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                raise RuntimeError(
                    "JPEG verification failed: output chroma sampling "
                    "could not be determined."
                ) from exc

            if actual_subsampling != expected_subsampling:
                raise RuntimeError(
                    "JPEG verification failed: expected chroma sampling "
                    f"{_sampling_name(expected_subsampling)}, but found "
                    f"{_sampling_name(actual_subsampling)}."
                )


def _run_jpegtran_rotation(
    executable: str,
    source_path: Path,
    temporary: Path,
    expected: Image.Image,
) -> None:
    """
    Rotate a JPEG 90 degrees counter-clockwise through jpegtran.

    The transform is performed in the JPEG coefficient domain. No trim option
    is used. The -perfect option requires the requested transform to complete
    without discarding non-transformable edge blocks.
    """

    result = subprocess.run(
        [
            executable,
            "-copy", "all",
            "-perfect",
            "-rotate", "270",
            "-outfile", str(temporary),
            str(source_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    if (
        result.returncode != 0
        or not temporary.is_file()
        or temporary.stat().st_size == 0
    ):
        reason = (
            result.stderr
            or result.stdout
            or "jpegtran rotation failed."
        ).strip()

        raise RuntimeError(reason)

    with Image.open(temporary) as actual:
        actual.load()

        if (actual.format or "").upper() != "JPEG":
            raise RuntimeError(
                "JPEG rotation verification failed: output is not JPEG."
            )

        if actual.size != expected.size:
            raise RuntimeError(
                "JPEG rotation verification failed: "
                "output dimensions differ from the expected rotation."
            )

        if actual.mode != expected.mode:
            raise RuntimeError(
                "JPEG rotation verification failed: output mode changed "
                f"from {expected.mode!r} to {actual.mode!r}."
            )


# ---------------------------------------------------------------------
# Central save operation
# ---------------------------------------------------------------------

def save_planned_output(
    frames: Sequence[Image.Image],
    plan: OutputPlan,
    *,
    metadata: dict | None = None,
    source_path: str | Path | None = None,
    source_mode: str | None = None,
    orientation_normalized: bool = False,
    jpegtran: str | None = None,
) -> SaveResult:
    """
    Save according to a central plan, verify the temporary file, then replace
    the destination atomically.

    The source image itself is never modified. Existing destination files are
    not replaced unless the final encode and applicable verification succeed.

    MPO safety:
        EXIF/XMP container metadata is removed from MPO-derived standalone
        raster outputs. When a frame contains such metadata in Image.info, a
        sanitized copy is encoded so the working image's metadata is not
        mutated. Non-MPO saving and JPEG coefficient-transform paths retain
        their existing behavior.
    """

    frames = list(frames)

    if not frames:
        raise ValueError("There are no image frames to save.")

    if len(frames) != plan.output_frames:
        raise ValueError(
            f"The plan expects {plan.output_frames} frame(s), "
            f"but received {len(frames)}."
        )

    destination = plan.output_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    combined_metadata = dict(metadata or {})

    for key, value in frames[0].info.items():
        combined_metadata.setdefault(key, value)

    original_mode = source_mode or plan.source_mode

    temporary_suffix = _FORMAT_EXTENSIONS.get(
        plan.output_format.upper(),
        destination.suffix,
    )

    with tempfile.NamedTemporaryFile(
        prefix=".easyquadrat_",
        suffix=temporary_suffix,
        dir=str(destination.parent),
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

    # These are only sanitized copies made for MPO-derived raster encoding.
    # The caller's processed images and their metadata remain untouched.
    owned_sanitized_frames: list[Image.Image] = []

    try:
        save_frames: list[Image.Image] = frames
        omitted_mpo_metadata_keys: set[str] = set()

        # -------------------------------------------------------------
        # Minimal MPO-only metadata safeguard.
        #
        # This branch is entered only when the planned source container
        # is MPO. Non-MPO sources retain the previous metadata path.
        # -------------------------------------------------------------

        if plan.source_format.upper() == "MPO":
            filtered_metadata = {}

            for key, value in combined_metadata.items():
                if _is_mpo_container_sensitive_metadata_key(key):
                    if value:
                        omitted_mpo_metadata_keys.add(str(key))
                    continue

                filtered_metadata[key] = value

            combined_metadata = filtered_metadata

            sanitized_frames: list[Image.Image] = []

            for frame in frames:
                sensitive_keys = [
                    key
                    for key in frame.info
                    if _is_mpo_container_sensitive_metadata_key(key)
                ]

                if not sensitive_keys:
                    sanitized_frames.append(frame)
                    continue

                # Pillow can inherit encoder settings from Image.info.
                # Make a pixel-identical copy before removing those fields.
                safe_frame = frame.copy()

                for key in sensitive_keys:
                    if safe_frame.info.get(key):
                        omitted_mpo_metadata_keys.add(str(key))

                    safe_frame.info.pop(key, None)

                owned_sanitized_frames.append(safe_frame)
                sanitized_frames.append(safe_frame)

            if owned_sanitized_frames:
                save_frames = sanitized_frames

        options, warnings = _metadata_options(
            combined_metadata,
            plan.output_format,
            original_mode,
            frames[0].mode,
            orientation_normalized,
        )

        if omitted_mpo_metadata_keys:
            fields = ", ".join(sorted(omitted_mpo_metadata_keys))

            warnings.append(
                "Container-sensitive EXIF/XMP metadata was omitted from "
                "this standalone MPO-derived image because its references "
                "may depend on the original MPO container. "
                f"Metadata fields omitted: {fields}. "
                "The original MPO file and raster pixels were not modified."
            )

        if plan.strategy == "jpegtran":
            executable = jpegtran or find_jpegtran()
            jpeg_source_path = Path(source_path or plan.source_path)

            if (
                executable is None
                or plan.crop_box is None
                or len(frames) != 1
            ):
                raise RuntimeError(
                    "Verified lossless JPEG cropping is unavailable."
                )

            left, top, right, bottom = plan.crop_box

            result = subprocess.run(
                [
                    executable,
                    "-copy", "all",
                    "-perfect",
                    "-crop",
                    f"{right - left}x{bottom - top}+{left}+{top}",
                    "-outfile", str(temporary),
                    str(jpeg_source_path),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            if (
                result.returncode != 0
                or not temporary.is_file()
                or temporary.stat().st_size == 0
            ):
                reason = (
                    result.stderr
                    or result.stdout
                    or "jpegtran failed."
                ).strip()

                raise RuntimeError(reason)

            with Image.open(temporary) as actual:
                actual.load()

                if (actual.format or "").upper() != "JPEG":
                    raise RuntimeError(
                        "Lossless JPEG crop did not produce a JPEG file."
                    )

                if (
                    actual.size != frames[0].size
                    or actual.mode != frames[0].mode
                    or actual.tobytes() != frames[0].tobytes()
                ):
                    raise RuntimeError(
                        "Verified lossless JPEG crop differs from "
                        "the expected raster."
                    )

            pixels_verified = True

        elif plan.strategy == "jpegtran_rotate":
            executable = jpegtran or find_jpegtran()
            jpeg_source_path = Path(source_path or plan.source_path)

            if executable is None or len(frames) != 1:
                raise RuntimeError(
                    "Verified JPEG coefficient rotation is unavailable."
                )

            _run_jpegtran_rotation(
                executable=executable,
                source_path=jpeg_source_path,
                temporary=temporary,
                expected=frames[0],
            )

            # The tool's perfect coefficient-domain transform is lossless,
            # but this is not a byte-for-byte comparison of decoded raster
            # pixels against the Pillow-transposed image. Do not report that
            # such a raster comparison was performed.
            pixels_verified = False

        elif plan.strategy == "jpeg_quality":
            if len(save_frames) != 1:
                raise ValueError(
                    "JPEG output supports one frame only."
                )

            image = save_frames[0]

            if image.mode not in JPEG_MODES:
                raise ValueError(
                    f"Cannot save mode {image.mode!r} as JPEG without "
                    "an explicit pixel conversion."
                )

            jpeg_source_path = Path(source_path or plan.source_path)

            # A processed/cropped Pillow image is normally no longer the
            # original JpegImageFile. Recover the encoding settings from the
            # source file rather than using quality="keep" on the derivative.
            encoding_options, encoding_warnings = (
                _jpeg_encoding_options_from_source(
                    jpeg_source_path,
                    image.mode,
                )
            )

            jpeg_options = dict(options)
            jpeg_options.update(encoding_options)

            image.save(
                temporary,
                format="JPEG",
                **jpeg_options,
            )

            expected_tables = encoding_options.get("qtables")

            expected_subsampling = encoding_options.get("subsampling")

            if not isinstance(expected_subsampling, int):
                expected_subsampling = None

            _verify_jpeg_output(
                temporary,
                image,
                expected_quantization_tables=expected_tables,
                expected_subsampling=expected_subsampling,
            )

            warnings.extend(encoding_warnings)

            pixels_verified = False

            warnings.append(
                "JPEG re-encoding completed and the encoded file's format, "
                "dimensions, pixel mode, and selected coding settings were "
                "verified. This does not mean pixel values were preserved "
                "exactly; JPEG re-encoding is lossy."
            )

        elif plan.strategy == "lossless":
            fallback_warnings = _save_lossless_temp(
                save_frames,
                temporary,
                plan.output_format,
                options,
            )

            warnings.extend(fallback_warnings)

            # Verify against the caller's original processed frames rather
            # than trusting a sanitized metadata copy as the expected raster.
            verify_lossless_output(
                temporary,
                frames,
            )

            pixels_verified = True

        else:
            raise ValueError(
                f"Unknown output strategy: {plan.strategy!r}."
            )

        # The temporary file is in the destination directory, so replacement
        # happens on the same filesystem. Failed encoding or verification
        # leaves the previous destination intact.
        os.replace(temporary, destination)

        return SaveResult(
            path=destination,
            image_format=plan.output_format,
            strategy=plan.strategy,
            pixels_verified=pixels_verified,
            warnings=tuple(warnings),
        )

    finally:
        temporary.unlink(missing_ok=True)

        # Close only copies created internally for MPO metadata sanitization.
        # Never close or mutate an image supplied by the calling method.
        for frame in owned_sanitized_frames:
            frame.close()


# ---------------------------------------------------------------------
# Exact-byte output for a true no-op
# ---------------------------------------------------------------------

def copy_original_exact(
    source_path: str | Path,
    output_path: str | Path,
) -> SaveResult:
    """
    Atomically copy the original file bytes when the requested operation
    does not change the image.

    This preserves the original compression stream, metadata, and all
    contained frames/pages. The destination name can retain the source
    suffix even when Pillow identifies the internal container as MPO.
    """

    source = Path(source_path)
    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        prefix=".easyquadrat_copy_",
        suffix=source.suffix or destination.suffix,
        dir=str(destination.parent),
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

    try:
        shutil.copy2(
            source,
            temporary,
        )

        if source.stat().st_size != temporary.stat().st_size:
            raise RuntimeError(
                "Exact-copy verification failed: file size changed."
            )

        with source.open("rb") as original, temporary.open("rb") as copied:
            while True:
                original_chunk = original.read(1024 * 1024)
                copied_chunk = copied.read(1024 * 1024)

                if original_chunk != copied_chunk:
                    raise RuntimeError(
                        "Exact-copy verification failed: file bytes differ."
                    )

                if not original_chunk:
                    break

        os.replace(
            temporary,
            destination,
        )

        meta = inspect_image(destination)

        return SaveResult(
            path=destination,
            image_format=meta["format"],
            strategy="byte_copy",
            pixels_verified=True,
        )

    finally:
        temporary.unlink(missing_ok=True)