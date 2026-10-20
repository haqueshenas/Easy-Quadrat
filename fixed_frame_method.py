"""
Easy Quadrat v2.0.0
Fixed Frame image-processing method.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License
AI coding assistance: OpenAI GPT-6
"""

from __future__ import annotations

import io
import tempfile
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from threading import Event, RLock

from PIL import Image, ImageOps

from PySide6.QtCore import (
    Qt,
    QPointF,
    QRectF,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
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

from core.image_output import (
    OutputPlan,
    OutputPolicy,
    SaveResult,
    copy_original_exact,
    find_jpegtran,
    normalize_policy,
    plan_exact_copy,
    plan_output,
    save_planned_output,
)

from output_manager import ProcessingMethod, prepare_output_layout


# ---------------------------------------------------------------------
# Supported formats and common image helpers
# ---------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
    ".bmp", ".webp", ".mpo", ".gif",
}

JPEG_MODES = {"L", "RGB", "CMYK"}

FULL_FRAME = {
    "left": 0,
    "right": 100,
    "top": 0,
    "bottom": 100,
}


def is_full_frame(bounds: dict[str, int]) -> bool:
    """Return True only when the complete image has been selected."""
    return all(
        int(bounds[key]) == value
        for key, value in FULL_FRAME.items()
    )


def _all_frames_match_orientation_without_exif(
    source: Image.Image,
    orientation: str,
) -> bool:
    """
    Return True only when every frame already matches the requested
    orientation and no EXIF orientation normalization is necessary.

    The original frame index is restored before returning.

    Orientation convention:
    - Horizontal: width >= height
    - Vertical: height > width

    Square images therefore count as Horizontal, not Vertical.
    """
    target = str(orientation).strip().casefold()

    if target not in {"horizontal", "vertical"}:
        return False

    try:
        original_index = source.tell()
    except Exception:
        original_index = 0

    try:
        total_frames = frame_count(source)

        for index in range(total_frames):
            source.seek(index)
            source.load()

            width, height = source.size

            try:
                exif_orientation = int(
                    source.getexif().get(274, 1)
                )
            except Exception:
                # If EXIF orientation cannot be checked reliably,
                # do not risk an incorrect exact-copy decision.
                return False

            if exif_orientation != 1:
                return False

            if target == "horizontal":
                matches = width >= height
            else:
                matches = height > width

            if not matches:
                return False

        return True

    except Exception:
        # Inspection failure must not authorize an exact copy.
        return False

    finally:
        try:
            source.seek(original_index)
        except Exception:
            pass


def list_supported_images(folder: Path) -> list[Path]:
    """List supported input files in deterministic order."""
    folder = Path(folder).expanduser()

    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}")

    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def frame_count(image: Image.Image) -> int:
    """Return the number of frames/pages reported by Pillow."""
    try:
        return max(1, int(getattr(image, "n_frames", 1)))
    except (AttributeError, TypeError, ValueError, EOFError):
        return 1


MPO_THUMBNAIL_TYPES = frozenset({
    "Large Thumbnail (VGA Equivalent)",
    "Large Thumbnail (Full HD Equivalent)",
})


def _mpo_contains_primary_and_thumbnails(
    image: Image.Image,
) -> bool:
    """
    Identify MPO files containing one primary image and only thumbnails.

    Classification is conservative. If the MP Index IFD is missing,
    inconsistent, or describes another picture type, the file is treated
    as a true multi-picture source.
    """
    if (image.format or "").upper() != "MPO":
        return False

    total_frames = frame_count(image)

    if total_frames < 2:
        return False

    try:
        entries = image.mpinfo.get(0xB002, [])
    except (AttributeError, TypeError):
        return False

    if (
        not isinstance(entries, (list, tuple))
        or len(entries) != total_frames
    ):
        return False

    if not all(isinstance(entry, dict) for entry in entries):
        return False

    attributes = [entry.get("Attribute", {}) for entry in entries]

    if not all(isinstance(attribute, dict) for attribute in attributes):
        return False

    # Never infer primary status solely from frame order.
    if attributes[0].get("MPType") != "Baseline MP Primary Image":
        return False

    return all(
        attribute.get("MPType") in MPO_THUMBNAIL_TYPES
        for attribute in attributes[1:]
    )


# ---------------------------------------------------------------------
# JPEG/MPO parsing and extraction
# ---------------------------------------------------------------------

def _remove_container_sensitive_jpeg_markers(
    jpeg_bytes: bytes,
) -> tuple[bytes, dict[str, int]]:
    """
    Remove the MPO MPF index and container-sensitive EXIF/XMP markers.

    JPEG image data is retained byte-for-byte. ICC APP2 markers and
    unrelated markers are preserved.

    This function inspects marker segments before the first Start Of Scan
    and fails closed if it cannot find exactly one MPF APP2 marker.
    """
    if len(jpeg_bytes) < 4 or jpeg_bytes[:2] != b"\xff\xd8":
        raise ValueError(
            "The extracted MPO primary frame is not a JPEG stream."
        )

    output = bytearray(jpeg_bytes[:2])
    position = 2
    found_scan = False

    removed = {
        "MPF": 0,
        "EXIF": 0,
        "XMP": 0,
    }

    xmp_standard = b"http://ns.adobe.com/xap/1.0/\x00"
    xmp_extension = b"http://ns.adobe.com/xmp/extension/\x00"

    standalone_markers = {
        0x01,                 # TEM
        *range(0xD0, 0xD8),  # Restart markers
    }

    while position < len(jpeg_bytes):
        marker_start = position

        if jpeg_bytes[position] != 0xFF:
            raise ValueError(
                "Malformed JPEG header: expected a marker before "
                "Start Of Scan."
            )

        while (
            position < len(jpeg_bytes)
            and jpeg_bytes[position] == 0xFF
        ):
            position += 1

        if position >= len(jpeg_bytes):
            raise ValueError(
                "Truncated JPEG marker in primary MPO frame."
            )

        marker = jpeg_bytes[position]
        position += 1

        if marker == 0xDA:
            # Start Of Scan. Preserve the remaining JPEG stream exactly.
            if position + 2 > len(jpeg_bytes):
                raise ValueError(
                    "Truncated JPEG Start Of Scan segment."
                )

            segment_length = int.from_bytes(
                jpeg_bytes[position:position + 2],
                "big",
            )

            if (
                segment_length < 2
                or position + segment_length > len(jpeg_bytes)
            ):
                raise ValueError(
                    "Invalid JPEG Start Of Scan segment length."
                )

            output.extend(jpeg_bytes[marker_start:])
            found_scan = True
            break

        if marker == 0xD9:
            raise ValueError(
                "Unexpected JPEG EOI before Start Of Scan."
            )

        if marker in standalone_markers:
            output.extend(jpeg_bytes[marker_start:position])
            continue

        if position + 2 > len(jpeg_bytes):
            raise ValueError(
                "Truncated JPEG marker length."
            )

        segment_length = int.from_bytes(
            jpeg_bytes[position:position + 2],
            "big",
        )

        if segment_length < 2:
            raise ValueError(
                "Invalid JPEG marker-segment length."
            )

        segment_end = position + segment_length

        if segment_end > len(jpeg_bytes):
            raise ValueError(
                "JPEG marker extends beyond the primary frame data."
            )

        payload = jpeg_bytes[position + 2:segment_end]
        remove_segment = False

        if marker == 0xE2 and payload.startswith(b"MPF\x00"):
            removed["MPF"] += 1
            remove_segment = True

        elif marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            removed["EXIF"] += 1
            remove_segment = True

        elif marker == 0xE1 and (
            payload.startswith(xmp_standard)
            or payload.startswith(xmp_extension)
        ):
            removed["XMP"] += 1
            remove_segment = True

        if not remove_segment:
            output.extend(jpeg_bytes[marker_start:segment_end])

        position = segment_end

    if not found_scan:
        raise ValueError(
            "JPEG primary frame has no Start Of Scan segment."
        )

    if removed["MPF"] != 1:
        raise ValueError(
            "Expected exactly one MPF APP2 index marker in the MPO "
            f"primary JPEG, found {removed['MPF']}. Refusing to create "
            "an ambiguous standalone JPEG."
        )

    return bytes(output), removed


def _find_jpeg_stream_end(jpeg_bytes: bytes) -> int:
    """
    Return the offset immediately after the first structurally recognized
    JPEG EOI marker.

    Marker segments and entropy-coded scans are parsed rather than simply
    searching the byte stream for FF D9.
    """
    if len(jpeg_bytes) < 4 or jpeg_bytes[:2] != b"\xff\xd8":
        raise ValueError(
            "The MPO primary image does not begin with JPEG SOI."
        )

    position = 2
    in_scan = False

    standalone_markers = {
        0x01,
        *range(0xD0, 0xD8),
    }

    while position < len(jpeg_bytes):
        if in_scan:
            # Entropy-coded scan. FF00 is stuffed data; restart markers
            # do not terminate the scan.
            while position < len(jpeg_bytes):
                if jpeg_bytes[position] != 0xFF:
                    position += 1
                    continue

                position += 1

                while (
                    position < len(jpeg_bytes)
                    and jpeg_bytes[position] == 0xFF
                ):
                    position += 1

                if position >= len(jpeg_bytes):
                    raise ValueError(
                        "Truncated marker at the end of a JPEG scan."
                    )

                marker = jpeg_bytes[position]
                position += 1

                if marker == 0x00 or 0xD0 <= marker <= 0xD7:
                    continue

                if marker == 0xD9:
                    return position

                # A non-restart marker terminates this scan. Continue
                # parsing its segment using the existing marker value.
                in_scan = False
                break

        else:
            if jpeg_bytes[position] != 0xFF:
                raise ValueError(
                    "Malformed JPEG stream: expected a marker "
                    "outside a scan."
                )

            position += 1

            while (
                position < len(jpeg_bytes)
                and jpeg_bytes[position] == 0xFF
            ):
                position += 1

            if position >= len(jpeg_bytes):
                raise ValueError(
                    "Truncated JPEG marker outside a scan."
                )

            marker = jpeg_bytes[position]
            position += 1

            if marker == 0xD9:
                return position

            if marker == 0x00:
                raise ValueError(
                    "Unexpected stuffed byte outside JPEG scan data."
                )

        if marker in standalone_markers:
            continue

        # All other markers have a two-byte segment length, including SOS.
        if position + 2 > len(jpeg_bytes):
            raise ValueError(
                "Truncated JPEG marker-segment length."
            )

        segment_length = int.from_bytes(
            jpeg_bytes[position:position + 2],
            "big",
        )

        if segment_length < 2:
            raise ValueError(
                "Invalid JPEG marker-segment length."
            )

        segment_end = position + segment_length

        if segment_end > len(jpeg_bytes):
            raise ValueError(
                "JPEG marker segment extends beyond available bytes."
            )

        position = segment_end

        if marker == 0xDA:
            in_scan = True

    raise ValueError(
        "No complete JPEG EOI marker was found in the MPO primary stream."
    )


def _find_mpf_tiff_header_offset(source_path: Path) -> int:
    """
    Find the absolute offset of the TIFF header inside the MPF APP2 marker.

    MPF secondary-image DataOffset fields are relative to this location.
    Exactly one MPF APP2 marker must occur before the first JPEG scan.
    """
    source_path = Path(source_path)
    file_size = source_path.stat().st_size

    with source_path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError(
                "MPO primary JPEG does not begin with JPEG SOI."
            )

        found_offset = None

        while True:
            marker_start = handle.tell()

            if handle.read(1) != b"\xff":
                raise ValueError(
                    "Malformed JPEG marker at byte "
                    f"{marker_start}."
                )

            marker_byte = handle.read(1)

            while marker_byte == b"\xff":
                marker_byte = handle.read(1)

            if not marker_byte:
                raise ValueError(
                    "Truncated marker in JPEG header."
                )

            marker = marker_byte[0]

            if marker == 0xDA:
                break

            if marker == 0xD9:
                raise ValueError(
                    "JPEG ended before its MPF APP2 marker."
                )

            if marker == 0x00:
                raise ValueError(
                    "Unexpected stuffed byte in JPEG header."
                )

            if marker == 0x01 or 0xD0 <= marker <= 0xD7:
                continue

            raw_length = handle.read(2)

            if len(raw_length) != 2:
                raise ValueError(
                    "Truncated JPEG marker-segment length."
                )

            segment_length = int.from_bytes(raw_length, "big")

            if segment_length < 2:
                raise ValueError(
                    "Invalid JPEG marker-segment length."
                )

            payload_start = handle.tell()
            payload_length = segment_length - 2
            segment_end = payload_start + payload_length

            if segment_end > file_size:
                raise ValueError(
                    "JPEG marker segment extends beyond the source file."
                )

            if marker == 0xE2:
                identifier = handle.read(min(4, payload_length))

                if identifier == b"MPF\x00":
                    if found_offset is not None:
                        raise ValueError(
                            "Multiple MPF APP2 headers were found."
                        )

                    # TIFF header starts immediately after "MPF\0".
                    found_offset = payload_start + 4

            # Continue even after finding MPF so duplicate markers
            # in the pre-scan header are detected.
            handle.seek(segment_end)

        if found_offset is None:
            raise ValueError(
                "No MPF APP2 TIFF header was found before JPEG SOS."
            )

        return found_offset


def _extract_mpo_primary_jpeg_for_derivative(
    source_path: Path,
    destination_path: Path,
) -> dict[str, int]:
    """
    Extract and validate the primary JPEG from an MPO file.

    The MPF primary Size field is not treated as the sole authority for
    locating the end of the JPEG. The next image's absolute location is
    calculated from the MPF TIFF-header offset and DataOffset.

    The actual JPEG EOI must be found before or at the next image boundary.
    Any gap after EOI and before the next image may contain only 0x00 or
    0xFF padding.

    The standalone JPEG must decode and match the primary MPO raster
    pixel-for-pixel before the derivative is written.
    """
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    with Image.open(source_path) as source:
        if (source.format or "").upper() != "MPO":
            raise ValueError(
                "The source is not recognized as an MPO file."
            )

        if not _mpo_contains_primary_and_thumbnails(source):
            raise ValueError(
                "The MPO does not have a verified primary-plus-thumbnail layout."
            )

        entries = source.mpinfo.get(0xB002, [])

        if (
            not isinstance(entries, (list, tuple))
            or len(entries) < 2
            or not isinstance(entries[0], dict)
        ):
            raise ValueError(
                "The MPO primary or secondary image index entry is missing."
            )

        primary_entry = entries[0]

        try:
            primary_offset = int(
                primary_entry.get("DataOffset", 0)
            )
            primary_size = int(
                primary_entry["Size"]
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "The MPO primary-image size or offset is invalid."
            ) from exc

        if primary_offset != 0:
            raise ValueError(
                "The MPO primary-image offset is not zero; extraction was refused."
            )

        file_size = source_path.stat().st_size

        if primary_size <= 0 or primary_size > file_size:
            raise ValueError(
                "The MPO primary-image Size is outside the source file."
            )

        secondary_entries = list(entries[1:])

        source.seek(0)
        source.load()

        expected_size = source.size
        expected_mode = source.mode

    # Find the MPF TIFF-header location independently of MPEntry Size.
    mpf_base = _find_mpf_tiff_header_offset(source_path)

    secondary_offsets: list[int] = []

    for index, entry in enumerate(secondary_entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Secondary MP entry {index} is malformed."
            )

        try:
            data_offset = int(entry["DataOffset"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Secondary MP entry {index} has an invalid DataOffset."
            ) from exc

        if data_offset <= 0:
            raise ValueError(
                f"Secondary MP entry {index} has a non-positive DataOffset."
            )

        absolute_offset = mpf_base + data_offset

        if not 0 < absolute_offset < file_size:
            raise ValueError(
                f"Secondary MP entry {index} points outside the source file."
            )

        secondary_offsets.append(absolute_offset)

    if not secondary_offsets:
        raise ValueError(
            "No valid secondary-image offsets were found in the MPO index."
        )

    # The earliest secondary image sets the primary region's upper boundary.
    next_image_offset = min(secondary_offsets)

    # The primary image starts at byte zero, so its declared Size cannot
    # extend beyond the next image's absolute starting position.
    if next_image_offset < primary_size:
        raise ValueError(
            "The MPF entries conflict: a secondary image begins before "
            "the declared primary Size ends. Extraction was refused."
        )

    with source_path.open("rb") as handle:
        primary_region = handle.read(next_image_offset)

    if len(primary_region) != next_image_offset:
        raise OSError(
            "Could not read the complete primary-image region."
        )

    actual_end = _find_jpeg_stream_end(primary_region)

    if actual_end > next_image_offset:
        raise ValueError(
            "The primary JPEG extends beyond the next MP image boundary."
        )

    trailing_bytes = primary_region[actual_end:next_image_offset]

    # Unknown bytes between EOI and the next image are not silently discarded.
    if any(value not in (0x00, 0xFF) for value in trailing_bytes):
        raise ValueError(
            "Unrecognized non-padding bytes occur between the primary JPEG "
            "EOI and the next MP image. Extraction was refused."
        )

    primary_jpeg_bytes = primary_region[:actual_end]

    cleaned_bytes, removed_markers = (
        _remove_container_sensitive_jpeg_markers(primary_jpeg_bytes)
    )

    # Decode and compare the output candidate before writing it.
    with Image.open(source_path) as original:
        original.seek(0)
        original.load()

        if (
            original.size != expected_size
            or original.mode != expected_mode
        ):
            raise RuntimeError(
                "The MPO primary image changed during validation."
            )

        with Image.open(io.BytesIO(cleaned_bytes)) as extracted:
            extracted.load()

            if (extracted.format or "").upper() != "JPEG":
                raise ValueError(
                    "The extracted candidate is not a standalone JPEG."
                )

            if frame_count(extracted) != 1:
                raise ValueError(
                    "The extracted candidate unexpectedly contains multiple frames."
                )

            if (
                extracted.size != expected_size
                or extracted.mode != expected_mode
            ):
                raise ValueError(
                    "The extracted JPEG dimensions or mode do not match "
                    "the MPO primary image."
                )

            width, height = original.size
            strip_height = 64

            for top in range(0, height, strip_height):
                bottom = min(height, top + strip_height)
                box = (0, top, width, bottom)

                original_strip = original.crop(box)
                extracted_strip = extracted.crop(box)

                try:
                    if (
                        original_strip.tobytes()
                        != extracted_strip.tobytes()
                    ):
                        raise ValueError(
                            "The extracted JPEG does not preserve the "
                            "MPO primary raster pixel-for-pixel."
                        )
                finally:
                    original_strip.close()
                    extracted_strip.close()

    # Only write a JPEG that has passed the complete raster comparison.
    destination_path.write_bytes(cleaned_bytes)

    return {
        **removed_markers,
        "MPF_SIZE_DELTA": actual_end - primary_size,
        "RASTER_PIXELS_VERIFIED": 1,
        "EOI_RECOVERY_USED": 0,
        "JPEG_STREAM_END": actual_end,
        "NEXT_MP_IMAGE_OFFSET": next_image_offset,
        "MPF_BOUNDARY_VERIFIED": int(
            actual_end == next_image_offset
        ),
        "PADDING_AFTER_EOI": len(trailing_bytes),
    }


def _mpo_primary_rotation_is_eligible(
    job: "FixedFrameJob",
    bounds: dict[str, int],
    orientation: str,
    plan: OutputPlan,
    jpegtran: str | None,
) -> bool:
    """
    Check whether a primary MPO JPEG can use full-frame coefficient rotation.

    JPEG MCU-edge compatibility is left to jpegtran's -perfect switch.
    If that switch rejects the transform, the worker uses the existing
    Keep JPEG fallback rather than silently cropping image edges.
    """
    if (
        not jpegtran
        or not job.skip_mpo_thumbnails
        or job.source_format != "MPO"
        or not is_full_frame(bounds)
        or normalize_policy(plan.policy) is not OutputPolicy.KEEP_JPEG
        or plan.output_format.upper() != "JPEG"
        or plan.strategy != "jpeg_quality"
        or plan.output_frames != 1
    ):
        return False

    target = str(orientation).strip().casefold()

    if target not in {"horizontal", "vertical"}:
        return False

    try:
        with Image.open(job.source_path) as source:
            if (source.format or "").upper() != "MPO":
                return False

            if not _mpo_contains_primary_and_thumbnails(source):
                return False

            source.seek(0)
            source.load()

            width, height = source.size

            try:
                exif_orientation = int(
                    source.getexif().get(274, 1)
                )
            except (TypeError, ValueError, OverflowError):
                return False

            # Do not physically rotate a JPEG with a non-default EXIF tag.
            if exif_orientation != 1:
                return False

    except Exception:
        return False

    rotation_required = (
        target == "horizontal" and height > width
    ) or (
        target == "vertical" and width >= height
    )

    if not rotation_required:
        return False

    expected_full_rotation_box = (0, 0, height, width)

    return tuple(job.crop_box or ()) == expected_full_rotation_box


def decodeable_image(path: Path) -> bool:
    """Check whether the first frame can actually be decoded."""
    try:
        with Image.open(path) as image:
            image.seek(0)
            image.load()
            return image.width > 0 and image.height > 0
    except Exception:
        return False


def make_oriented_copy(
    source: Image.Image,
    orientation: str,
) -> Image.Image:
    """
    Make an independent copy in the requested orientation.

    Horizontal and Vertical use an exact 90-degree transpose when required,
    without interpolation or resizing.
    """
    result = source.copy()
    result.info.update(source.info)

    target = orientation.casefold()

    if target not in {"horizontal", "vertical"}:
        return result

    current = (
        "vertical"
        if result.height > result.width
        else "horizontal"
    )

    if current != target:
        rotated = result.transpose(Image.Transpose.ROTATE_90)
        rotated.info.update(result.info)
        result.close()
        result = rotated

    return result


def normalized_oriented_copy(
    source_frame: Image.Image,
    orientation: str,
) -> Image.Image:
    """Apply EXIF orientation, then the user's physical orientation choice."""
    normalized = ImageOps.exif_transpose(source_frame)

    try:
        return make_oriented_copy(normalized, orientation)
    finally:
        if normalized is not source_frame:
            normalized.close()


def crop_percent(
    image: Image.Image,
    bounds: dict[str, int],
) -> Image.Image:
    """Crop using pixel boundaries derived from percentage values."""
    width, height = image.size

    box = (
        round(bounds["left"] * width / 100),
        round(bounds["top"] * height / 100),
        round(bounds["right"] * width / 100),
        round(bounds["bottom"] * height / 100),
    )

    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(
            "The selected frame is empty at this image's pixel resolution."
        )

    cropped = image.crop(box)
    cropped.info.update(image.info)
    return cropped


def load_preview_image(path: Path) -> Image.Image:
    """Load the first frame with EXIF orientation normalized for display."""
    with Image.open(path) as source:
        source.seek(0)
        source.load()

        normalized = ImageOps.exif_transpose(source)

        try:
            result = normalized.copy()
            result.info.update(normalized.info)
            return result
        finally:
            if normalized is not source:
                normalized.close()


# ---------------------------------------------------------------------
# Persistent logging
# ---------------------------------------------------------------------

class FixedFrameLogger:
    """Append timestamped run, output, warning and error information."""

    def __init__(self, log_path: Path) -> None:
        self.path = Path(log_path)
        self._lock = RLock()
        self._write_error: str | None = None

    @property
    def write_error(self) -> str | None:
        with self._lock:
            return self._write_error

    def write(self, message: object = "") -> None:
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        lines = str(message).splitlines() or [""]
        payload = "".join(
            f"[{timestamp}] {line}\n"
            for line in lines
        )

        with self._lock:
            try:
                self.path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with self.path.open(
                    "a",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    handle.write(payload)
                    handle.flush()

            except Exception as exc:
                if self._write_error is None:
                    self._write_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

    def start_run(
        self,
        *,
        input_directory: Path,
        output_directory: Path,
        bounds: dict[str, int],
        orientation: str,
        jpegtran: str | None,
    ) -> None:
        self.write("=" * 78)
        self.write("Easy Quadrat v2.0.0 — Fixed Frame")
        self.write("Run status: STARTED")
        self.write(f"Input directory: {input_directory}")
        self.write(f"Output directory: {output_directory}")
        self.write(f"Frame boundaries (%): {bounds}")
        self.write(f"Orientation selection: {orientation}")
        self.write(
            "jpegtran: "
            + (str(jpegtran) if jpegtran else "not available")
        )
        self.write("Output plans will be recorded below.")
        self.write("=" * 78)

    def log_job_plan(self, job: "FixedFrameJob") -> None:
        self.write(f"INPUT: {job.source_path}")

        if job.planning_error:
            self.write("Planning status: FAILED")
            self.write(job.planning_error)

            if job.planning_traceback:
                self.write(job.planning_traceback)
            return

        self.write(
            "Source inspection: "
            f"format={job.source_format}; "
            f"mode={job.source_mode}; "
            f"frames={job.pages}"
        )

        if job.exact_copy:
            self.write("Operation: exact-byte copy (no-op)")
            self.write(f"Output format: {job.exact_copy_format}")
            self.write(f"Output path: {job.exact_copy_path}")
            self.write("Verification: exact file-byte comparison.")
            return

        if job.skip_mpo_thumbnails:
            self.write(
                "MPO classification: one primary image plus thumbnail(s)."
            )
            self.write(
                "Processing decision: only the primary image will be "
                "included in the derived output; thumbnail frame(s) "
                "remain untouched in the original file."
            )
            self.write(
                "Source frame indexes selected for processing: "
                f"{job.processing_frame_indices}"
            )

        if job.lossless_plan is not None:
            plan = job.lossless_plan
            self.write(
                "Central lossless plan: "
                f"format={plan.output_format}; "
                f"mode={plan.output_mode}; "
                f"frames={plan.output_frames}; "
                f"strategy={plan.strategy}"
            )
            self.write(f"Lossless destination: {plan.output_path}")
            self.write(f"Lossless planning note: {plan.reason}")

        if job.keep_jpeg_plan is not None:
            plan = job.keep_jpeg_plan
            self.write(
                "Central Keep JPEG plan: "
                f"format={plan.output_format}; "
                f"strategy={plan.strategy}; "
                f"path={plan.output_path}"
            )

        for number, plan in enumerate(
            job.keep_jpeg_frame_plans,
            start=1,
        ):
            self.write(
                f"Central Keep JPEG picture {number} plan: "
                f"format={plan.output_format}; "
                f"strategy={plan.strategy}; "
                f"path={plan.output_path}"
            )

    def log_selected_job(self, job: "FixedFrameJob") -> None:
        if job.planning_error:
            return

        if job.exact_copy:
            self.write(
                f"Selected plan: exact-byte copy -> {job.exact_copy_path}"
            )
            return

        if job.selected_frame_plans:
            self.write(
                f"Selected multi-output policy for {job.source_path.name}: "
                f"{job.selected_frame_plans[0].policy.value}"
            )

            for number, plan in enumerate(
                job.selected_frame_plans,
                start=1,
            ):
                self.write(
                    f"Selected output {number}: "
                    f"format={plan.output_format}; "
                    f"strategy={plan.strategy}; "
                    f"path={plan.output_path}"
                )
            return

        plan = job.selected_plan

        if plan is None:
            self.write(
                f"No selected plan for {job.source_path.name}."
            )
            return

        self.write(
            f"Selected job plan: {job.source_path.name}; "
            f"policy={plan.policy.value}; "
            f"format={plan.output_format}; "
            f"mode={plan.output_mode}; "
            f"frames={plan.output_frames}; "
            f"strategy={plan.strategy}"
        )
        self.write(f"Selected output path: {plan.output_path}")


# ---------------------------------------------------------------------
# Output job
# ---------------------------------------------------------------------

@dataclass
class FixedFrameJob:
    """The input and centrally planned output(s) for one source file."""

    source_path: Path

    source_format: str = ""
    source_mode: str = ""
    output_mode: str = ""
    pages: int = 1

    frame_indices_to_process: tuple[int, ...] = ()
    skip_mpo_thumbnails: bool = False

    crop_box: tuple[int, int, int, int] | None = None
    orientation_normalized: bool = False

    exact_copy: bool = False
    exact_copy_path: Path | None = None
    exact_copy_format: str = ""

    lossless_plan: OutputPlan | None = None
    keep_jpeg_plan: OutputPlan | None = None
    keep_jpeg_frame_plans: list[OutputPlan] = field(
        default_factory=list
    )

    selected_plan: OutputPlan | None = None
    selected_frame_plans: list[OutputPlan] = field(
        default_factory=list
    )

    planning_error: str | None = None
    planning_traceback: str | None = None

    @property
    def processing_frame_indices(self) -> tuple[int, ...]:
        """Return the original source frame indexes selected for processing."""
        if self.frame_indices_to_process:
            return self.frame_indices_to_process

        return tuple(range(self.pages))

    @property
    def output_page_count(self) -> int:
        """Number of raster pages expected in the derived output."""
        return len(self.processing_frame_indices)

    @property
    def output_paths(self) -> list[Path]:
        """Return every selected destination for this input."""
        if self.planning_error:
            return []

        if self.exact_copy:
            return (
                [self.exact_copy_path]
                if self.exact_copy_path is not None
                else []
            )

        if self.selected_frame_plans:
            return [
                Path(plan.output_path)
                for plan in self.selected_frame_plans
            ]

        if self.selected_plan is not None:
            return [Path(self.selected_plan.output_path)]

        return []


def _make_central_output_plan(
    job: FixedFrameJob,
    output_directory: Path,
    policy: OutputPolicy,
    jpegtran: str | None,
    prefix: str = "C",
) -> OutputPlan:
    """Delegate format, filename and strategy decisions to image_output."""
    if job.crop_box is None:
        raise ValueError(
            "Cannot build an output plan without a valid crop rectangle."
        )

    return plan_output(
        job.source_path,
        output_directory,
        prefix,
        normalize_policy(policy),
        "crop",
        output_mode=job.output_mode,
        output_frames=job.output_page_count,
        crop_box=job.crop_box,
        orientation_normalized=job.orientation_normalized,
        jpegtran=jpegtran,
    )


def plan_fixed_frame_job(
    source_path: Path,
    output_directory: Path,
    bounds: dict[str, int],
    orientation: str,
    jpegtran: str | None,
) -> FixedFrameJob:
    """
    Create the output plans for one image.

    Recognized MPO files containing one primary image and thumbnail entries
    produce a derivative from the primary image only. Other MPO files retain
    the separate-picture Keep JPEG route and multi-page Lossless route.
    """
    source_path = Path(source_path).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()

    with Image.open(source_path) as source:
        source_format = (source.format or "").upper()

        if not source_format:
            raise ValueError("Could not identify the image format.")

        pages = frame_count(source)
        source_mode = source.mode

        job = FixedFrameJob(
            source_path=source_path,
            source_format=source_format,
            source_mode=source_mode,
            pages=pages,
        )

        full_frame = is_full_frame(bounds)
        original_orientation = (
            orientation.strip().casefold() == "original"
        )

        already_matches_orientation = (
            full_frame
            and not original_orientation
            and _all_frames_match_orientation_without_exif(
                source,
                orientation,
            )
        )

        if full_frame and (
            original_orientation or already_matches_orientation
        ):
            exact_path, exact_format = plan_exact_copy(
                source_path,
                output_directory,
                "C",
            )

            job.exact_copy = True
            job.exact_copy_path = exact_path
            job.exact_copy_format = exact_format
            return job

        if _mpo_contains_primary_and_thumbnails(source):
            job.frame_indices_to_process = (0,)
            job.skip_mpo_thumbnails = True
        else:
            job.frame_indices_to_process = tuple(range(pages))

        try:
            exif_orientation = int(
                source.getexif().get(274, 1)
            )
        except Exception:
            exif_orientation = 1

        source.seek(0)
        first_frame = source.copy()

        try:
            normalized = ImageOps.exif_transpose(first_frame)

            try:
                normalized_width, normalized_height = normalized.size

                rotation_required = (
                    orientation.casefold() == "horizontal"
                    and normalized_height > normalized_width
                ) or (
                    orientation.casefold() == "vertical"
                    and normalized_width >= normalized_height
                )

                orientation_normalized = (
                    exif_orientation != 1
                    or rotation_required
                )

                oriented = make_oriented_copy(
                    normalized,
                    orientation,
                )

                try:
                    width, height = oriented.size

                    crop_box = (
                        round(bounds["left"] * width / 100),
                        round(bounds["top"] * height / 100),
                        round(bounds["right"] * width / 100),
                        round(bounds["bottom"] * height / 100),
                    )

                    output_mode = oriented.mode

                    if (
                        crop_box[0] >= crop_box[2]
                        or crop_box[1] >= crop_box[3]
                    ):
                        raise ValueError(
                            "The selected frame is empty at this image's "
                            "pixel resolution."
                        )

                finally:
                    oriented.close()

            finally:
                if normalized is not first_frame:
                    normalized.close()

        finally:
            first_frame.close()

        job.crop_box = crop_box
        job.output_mode = output_mode
        job.orientation_normalized = orientation_normalized

    job.lossless_plan = _make_central_output_plan(
        job,
        output_directory,
        OutputPolicy.LOSSLESS,
        jpegtran,
    )
    job.selected_plan = job.lossless_plan

    if (
        job.source_format == "JPEG"
        and job.pages == 1
        and job.output_mode in JPEG_MODES
    ):
        try:
            alternate = _make_central_output_plan(
                job,
                output_directory,
                OutputPolicy.KEEP_JPEG,
                jpegtran,
            )

            if str(alternate.output_format).upper() == "JPEG":
                job.keep_jpeg_plan = alternate

        except Exception:
            job.keep_jpeg_plan = None

    elif (
        job.source_format == "MPO"
        and job.pages > 1
        and job.output_mode in JPEG_MODES
    ):
        if job.skip_mpo_thumbnails:
            try:
                alternate = _make_central_output_plan(
                    job,
                    output_directory,
                    OutputPolicy.KEEP_JPEG,
                    jpegtran,
                    prefix="C_Primary",
                )

                if (
                    str(alternate.output_format).upper() == "JPEG"
                    and alternate.output_frames == 1
                ):
                    job.keep_jpeg_plan = alternate

            except Exception:
                job.keep_jpeg_plan = None

        else:
            frame_plans: list[OutputPlan] = []

            try:
                for picture_number in range(1, job.pages + 1):
                    alternate = plan_output(
                        job.source_path,
                        output_directory,
                        f"C_Picture_{picture_number}",
                        OutputPolicy.KEEP_JPEG,
                        "crop",
                        output_mode=job.output_mode,
                        output_frames=1,
                        crop_box=job.crop_box,
                        orientation_normalized=job.orientation_normalized,
                        jpegtran=jpegtran,
                    )

                    if str(alternate.output_format).upper() != "JPEG":
                        raise ValueError(
                            "The central planner did not create a JPEG plan "
                            f"for MPO picture {picture_number}."
                        )

                    frame_plans.append(alternate)

                if len(frame_plans) == job.pages:
                    job.keep_jpeg_frame_plans = frame_plans

            except Exception:
                job.keep_jpeg_frame_plans = []

    return job


# ---------------------------------------------------------------------
# Preview widget
# ---------------------------------------------------------------------

class FramePreview(QWidget):
    """Display the selected image and a translucent crop frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._pixmap: QPixmap | None = None
        self._message = "Select an input folder containing readable images."

        self._left = 0.0
        self._top = 0.0
        self._right = 1.0
        self._bottom = 1.0

        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )

    @property
    def aspect_ratio(self) -> float:
        if self._pixmap is None or self._pixmap.isNull():
            return 4 / 3

        return self._pixmap.width() / max(1, self._pixmap.height())

    def set_message(self, message: str) -> None:
        self._pixmap = None
        self._message = message
        self.update()

    def set_image(self, image: Image.Image) -> None:
        display_image = image.convert("RGB")

        try:
            qimage = QImage(
                display_image.tobytes(),
                display_image.width,
                display_image.height,
                display_image.width * 3,
                QImage.Format.Format_RGB888,
            ).copy()

            self._pixmap = QPixmap.fromImage(qimage)
            self._message = ""

        finally:
            display_image.close()

        self.update()

    def set_frame(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> None:
        self._left = left / 100
        self._top = top / 100
        self._right = right / 100
        self._bottom = bottom / 100
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            True,
        )
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            True,
        )

        painter.fillRect(self.rect(), QColor("#f5f6f4"))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#5f665f"))
            painter.drawText(
                self.rect().adjusted(12, 12, -12, -12),
                Qt.AlignmentFlag.AlignCenter,
                self._message,
            )
            painter.end()
            return

        rect = QRectF(self.rect())

        painter.drawPixmap(
            rect,
            self._pixmap,
            QRectF(self._pixmap.rect()),
        )

        left = rect.left() + rect.width() * self._left
        right = rect.left() + rect.width() * self._right
        top = rect.top() + rect.height() * self._top
        bottom = rect.top() + rect.height() * self._bottom

        shade = QColor(25, 30, 25, 120)

        for shaded_rect in (
            QRectF(
                rect.left(), rect.top(),
                rect.width(), max(0.0, top - rect.top()),
            ),
            QRectF(
                rect.left(), bottom,
                rect.width(), max(0.0, rect.bottom() - bottom),
            ),
            QRectF(
                rect.left(), top,
                max(0.0, left - rect.left()),
                max(0.0, bottom - top),
            ),
            QRectF(
                right, top,
                max(0.0, rect.right() - right),
                max(0.0, bottom - top),
            ),
        ):
            painter.fillRect(shaded_rect, shade)

        painter.setPen(QPen(QColor("#D000D0"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(
            QRectF(left, top, right - left, bottom - top)
        )

        painter.end()


class DualRangeSlider(QWidget):
    """Two-handle percentage range slider."""

    valuesChanged = Signal(int, int)

    def __init__(
        self,
        orientation: Qt.Orientation,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.orientation = orientation
        self.lower = 0
        self.upper = 100
        self.active: str | None = None
        self.radius = 7.0
        self.margin = 8.0

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        if orientation == Qt.Orientation.Horizontal:
            self.setMinimumHeight(22)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
        else:
            self.setMinimumWidth(22)
            self.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Expanding,
            )

    def set_values(
        self,
        lower: int,
        upper: int,
        emit_signal: bool = False,
    ) -> None:
        lower = max(0, min(100, int(lower)))
        upper = max(lower, min(100, int(upper)))

        changed = (lower, upper) != (self.lower, self.upper)
        self.lower, self.upper = lower, upper
        self.update()

        if changed and emit_signal:
            self.valuesChanged.emit(lower, upper)

    def _centres(self) -> tuple[QPointF, QPointF]:
        if self.orientation == Qt.Orientation.Horizontal:
            usable = max(1.0, self.width() - 2 * self.margin)
            y = self.height() / 2

            return (
                QPointF(
                    self.margin + usable * self.lower / 100,
                    y,
                ),
                QPointF(
                    self.margin + usable * self.upper / 100,
                    y,
                ),
            )

        usable = max(1.0, self.height() - 2 * self.margin)
        x = self.width() / 2

        return (
            QPointF(
                x,
                self.margin + usable * self.lower / 100,
            ),
            QPointF(
                x,
                self.margin + usable * self.upper / 100,
            ),
        )

    def _value_at(self, position: float) -> int:
        extent = (
            self.width()
            if self.orientation == Qt.Orientation.Horizontal
            else self.height()
        )
        usable = max(1.0, extent - 2 * self.margin)

        return max(
            0,
            min(
                100,
                round((position - self.margin) / usable * 100),
            ),
        )

    def _move_handle(self, position: float) -> None:
        if self.active is None:
            return

        value = self._value_at(position)

        if self.active == "lower":
            self.set_values(
                min(value, self.upper),
                self.upper,
                True,
            )
        else:
            self.set_values(
                self.lower,
                max(value, self.lower),
                True,
            )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        lower, upper = self._centres()

        if self.orientation == Qt.Orientation.Horizontal:
            start = QPointF(self.margin, self.height() / 2)
            end = QPointF(
                self.width() - self.margin,
                self.height() / 2,
            )
        else:
            start = QPointF(self.width() / 2, self.margin)
            end = QPointF(
                self.width() / 2,
                self.height() - self.margin,
            )

        base_pen = QPen(QColor("#d9ded6"), 4)
        base_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        active_pen = QPen(QColor("#57863c"), 4)
        active_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        painter.setPen(base_pen)
        painter.drawLine(start, lower)
        painter.drawLine(upper, end)

        painter.setPen(active_pen)
        painter.drawLine(lower, upper)

        for point in (lower, upper):
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(QPen(QColor("#57863c"), 2))
            painter.drawEllipse(
                QRectF(
                    point.x() - self.radius,
                    point.y() - self.radius,
                    2 * self.radius,
                    2 * self.radius,
                )
            )

        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        position = (
            event.position().x()
            if self.orientation == Qt.Orientation.Horizontal
            else event.position().y()
        )

        lower, upper = self._centres()

        lower_pos = (
            lower.x()
            if self.orientation == Qt.Orientation.Horizontal
            else lower.y()
        )
        upper_pos = (
            upper.x()
            if self.orientation == Qt.Orientation.Horizontal
            else upper.y()
        )

        self.active = (
            "lower"
            if abs(position - lower_pos) <= abs(position - upper_pos)
            else "upper"
        )

        self._move_handle(position)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self.active is None:
            super().mouseMoveEvent(event)
            return

        position = (
            event.position().x()
            if self.orientation == Qt.Orientation.Horizontal
            else event.position().y()
        )

        self._move_handle(position)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.active = None
            event.accept()
            return

        super().mouseReleaseEvent(event)


class PreviewWithRangeControls(QWidget):
    """Keep range sliders aligned with the displayed image edges."""

    horizontalValuesChanged = Signal(int, int)
    verticalValuesChanged = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 290)

        self.preview = FramePreview(self)
        self.vertical_range = DualRangeSlider(
            Qt.Orientation.Vertical,
            self,
        )
        self.horizontal_range = DualRangeSlider(
            Qt.Orientation.Horizontal,
            self,
        )

        self.horizontal_range.valuesChanged.connect(
            self.horizontalValuesChanged.emit
        )
        self.vertical_range.valuesChanged.connect(
            self.verticalValuesChanged.emit
        )

        self.slider_width = 22
        self.slider_height = 22

    @property
    def aspect_ratio(self) -> float:
        return self.preview.aspect_ratio

    def set_image(self, image: Image.Image) -> None:
        self.preview.set_image(image)
        self._layout_children()

    def set_message(self, message: str) -> None:
        self.preview.set_message(message)
        self._layout_children()

    def set_frame(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> None:
        self.horizontal_range.set_values(left, right)
        self.vertical_range.set_values(top, bottom)
        self.preview.set_frame(left, top, right, bottom)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_children()

    def _layout_children(self) -> None:
        available_width = max(1, self.width() - self.slider_width)
        available_height = max(1, self.height() - self.slider_height)
        ratio = max(0.1, self.aspect_ratio)

        image_width = max(
            1,
            min(available_width, int(available_height * ratio)),
        )
        image_height = max(1, round(image_width / ratio))

        if image_height > available_height:
            image_height = available_height
            image_width = max(1, round(image_height * ratio))

        total_width = self.slider_width + image_width
        total_height = image_height + self.slider_height

        x = max(0, (self.width() - total_width) // 2)
        y = max(0, (self.height() - total_height) // 2)

        self.vertical_range.setGeometry(
            x, y, self.slider_width, image_height
        )
        self.preview.setGeometry(
            x + self.slider_width, y, image_width, image_height
        )
        self.horizontal_range.setGeometry(
            x + self.slider_width,
            y + image_height,
            image_width,
            self.slider_height,
        )


# ---------------------------------------------------------------------
# Background batch worker
# ---------------------------------------------------------------------

class FixedFrameWorker(QThread):
    progress_changed = Signal(int, int, str)
    completed = Signal(object)

    def __init__(
        self,
        jobs: list[FixedFrameJob],
        bounds: dict[str, int],
        orientation: str,
        jpegtran: str | None,
        logger: FixedFrameLogger,
    ) -> None:
        super().__init__()

        self.jobs = list(jobs)
        self.bounds = dict(bounds)
        self.orientation = orientation
        self.jpegtran = jpegtran
        self.logger = logger
        self.cancel_event = Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def _crop_all_frames(
        self,
        source: Image.Image,
        frame_indices: tuple[int, ...] | None = None,
    ) -> tuple[list[Image.Image], list[dict]]:
        """Crop selected source frames and snapshot their metadata."""
        cropped_frames: list[Image.Image] = []
        frame_metadata: list[dict] = []

        total_frames = frame_count(source)

        indexes = (
            tuple(range(total_frames))
            if frame_indices is None
            else tuple(frame_indices)
        )

        if not indexes:
            raise ValueError(
                "No source frames were selected for processing."
            )

        if len(set(indexes)) != len(indexes):
            raise ValueError(
                "Source frame selection contains duplicate indexes."
            )

        invalid = [
            index for index in indexes
            if not 0 <= index < total_frames
        ]

        if invalid:
            raise ValueError(
                f"Invalid source frame index/indices: {invalid}; "
                f"the source contains {total_frames} frame(s)."
            )

        try:
            for index in indexes:
                source.seek(index)
                source.load()
                raw_frame = source.copy()

                try:
                    oriented = normalized_oriented_copy(
                        raw_frame,
                        self.orientation,
                    )

                    try:
                        frame_metadata.append(dict(oriented.info))
                        cropped_frames.append(
                            crop_percent(oriented, self.bounds)
                        )
                    finally:
                        oriented.close()

                finally:
                    raw_frame.close()

            return cropped_frames, frame_metadata

        except Exception:
            for frame in cropped_frames:
                frame.close()
            raise

    def _save_mpo_primary_with_jpegtran_rotation(
        self,
        job: FixedFrameJob,
        crop: Image.Image,
        metadata: dict,
        plan: OutputPlan,
    ) -> tuple[SaveResult, dict[str, int]]:
        """
        Extract the MPO primary JPEG, report extraction validation, and then
        request a full-frame coefficient-domain rotation through jpegtran.

        Extraction and rotation are logged separately so a failure in one
        stage cannot be mistaken for a failure in the other.
        """
        if not self.jpegtran:
            raise RuntimeError(
                "jpegtran is not available for this operation."
            )

        with tempfile.TemporaryDirectory(
            prefix="easyquadrat_mpo_primary_",
        ) as temp_directory:
            extracted_path = (
                Path(temp_directory) / "primary_without_mpf.jpg"
            )

            # ---------------------------------------------------------
            # Stage 1: extract and validate the primary JPEG.
            # ---------------------------------------------------------
            try:
                removed_markers = (
                    _extract_mpo_primary_jpeg_for_derivative(
                        job.source_path,
                        extracted_path,
                    )
                )
            except Exception as exc:
                self.logger.write(
                    "MPO primary JPEG extraction: FAILED; "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

            # IMPORTANT: log extraction success BEFORE jpegtran is called.
            # This record remains available even if coefficient rotation
            # subsequently fails.
            self.logger.write(
                "MPO primary JPEG extraction: SUCCESS; "
                f"standalone JPEG validated; "
                f"MPF markers removed={removed_markers['MPF']}; "
                f"EXIF markers removed={removed_markers['EXIF']}; "
                f"XMP markers removed={removed_markers['XMP']}; "
                f"MPF Size adjustment="
                f"{removed_markers['MPF_SIZE_DELTA']:+d} bytes; "
                f"JPEG EOI offset="
                f"{removed_markers['JPEG_STREAM_END']}; "
                f"next MP image offset="
                f"{removed_markers['NEXT_MP_IMAGE_OFFSET']}; "
                f"exact MPF boundary match="
                f"{'yes' if removed_markers['MPF_BOUNDARY_VERIFIED'] else 'no'}; "
                f"padding after EOI="
                f"{removed_markers['PADDING_AFTER_EOI']} bytes; "
                "primary raster pixel check=passed; "
                "the original MPO file was not modified."
            )

            # ---------------------------------------------------------
            # Stage 2: request JPEG coefficient-domain rotation.
            # ---------------------------------------------------------
            rotation_plan = replace(
                plan,
                source_path=extracted_path,
                source_format="JPEG",
                source_mode=job.source_mode,
                strategy="jpegtran_rotate",
                reason=(
                    "MPO primary JPEG extracted and pixel-verified as a "
                    "standalone JPEG; MPF index removed; full-frame "
                    "coefficient rotation requested with jpegtran -perfect."
                ),
            )

            try:
                result = save_planned_output(
                    [crop],
                    rotation_plan,
                    metadata=metadata,
                    source_path=extracted_path,
                    source_mode=job.source_mode,
                    orientation_normalized=False,
                    jpegtran=self.jpegtran,
                )
            except Exception as exc:
                self.logger.write(
                    "MPO primary coefficient rotation/save: FAILED; "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

            self.logger.write(
                "MPO primary coefficient rotation/save: SUCCESS; "
                "the central output saver returned a successful result."
            )

        return result, removed_markers

    def _save_mpo_as_separate_jpegs(
        self,
        job: FixedFrameJob,
        crops: list[Image.Image],
        frame_metadata: list[dict],
    ) -> tuple[list[SaveResult], list[str]]:
        """
        Save each processed MPO picture into its own centrally planned JPEG.

        Container-level EXIF/XMP records are not blindly copied into
        standalone JPEG files.
        """
        plans = job.selected_frame_plans

        if len(plans) != len(crops):
            raise RuntimeError(
                "The number of planned JPEG outputs does not match the "
                "number of decoded MPO pictures."
            )

        if len(frame_metadata) != len(crops):
            raise RuntimeError(
                "The number of metadata records does not match the "
                "number of decoded MPO pictures."
            )

        results: list[SaveResult] = []
        warnings: list[str] = []
        removed_mpo_metadata = False

        for index, (crop, plan, metadata) in enumerate(
            zip(crops, plans, frame_metadata),
            start=1,
        ):
            output_metadata = dict(metadata)

            if output_metadata.pop("exif", None):
                removed_mpo_metadata = True

            if output_metadata.pop("xmp", None):
                removed_mpo_metadata = True

            crop.info.pop("exif", None)
            crop.info.pop("xmp", None)

            self.logger.write(
                f"MPO picture {index}/{len(crops)} -> {plan.output_path}"
            )

            result = save_planned_output(
                [crop],
                plan,
                metadata=output_metadata,
                source_path=job.source_path,
                source_mode=job.source_mode,
                orientation_normalized=job.orientation_normalized,
                jpegtran=self.jpegtran,
            )

            results.append(result)

            self.logger.write(
                f"MPO picture {index} result: SUCCESS; "
                f"format={result.image_format}; "
                f"strategy={result.strategy}; "
                f"pixel values verified={result.pixels_verified}; "
                f"path={result.path}"
            )

            for warning in result.warnings:
                warnings.append(
                    f"{job.source_path.name}: {warning}"
                )
                self.logger.write(f"WARNING: {warning}")

        if removed_mpo_metadata:
            warning = (
                f"{job.source_path.name}: Raw MPO-derived EXIF/XMP metadata "
                "was not copied to the separate JPEG files. MPO-specific "
                "offsets and container metadata may not remain valid in "
                "standalone JPEGs. The JPEGs were re-encoded and their "
                "pixel values may differ from the lossless raster."
            )
            warnings.append(warning)
            self.logger.write("WARNING: " + warning)

        return results, warnings

    def run(self) -> None:
        total = len(self.jobs)
        attempted = 0
        succeeded = 0
        errors: list[str] = []
        warnings: list[str] = []
        cancelled = False
        output_files_succeeded = 0

        self.logger.write("Worker started.")
        self.logger.write(f"Input job count: {total}")

        for index, job in enumerate(self.jobs, start=1):
            if self.cancel_event.is_set():
                cancelled = True
                self.logger.write(
                    "Cancellation requested before the next image."
                )
                break

            source_path = Path(job.source_path)
            crops: list[Image.Image] = []

            self.logger.write("")
            self.logger.write(
                f"IMAGE {index}/{total}: {source_path.name}"
            )

            try:
                if job.planning_error:
                    raise ValueError(job.planning_error)

                if job.exact_copy:
                    if job.exact_copy_path is None:
                        raise RuntimeError(
                            "No exact-copy destination was planned."
                        )

                    self.logger.write("Operation: exact-byte copy.")
                    self.logger.write(
                        f"Destination: {job.exact_copy_path}"
                    )

                    result = copy_original_exact(
                        source_path,
                        job.exact_copy_path,
                    )

                    output_files_succeeded += 1
                    succeeded += 1

                    self.logger.write("Result: SUCCESS")
                    self.logger.write(
                        f"Output format: {result.image_format}"
                    )
                    self.logger.write(f"Output path: {result.path}")
                    self.logger.write(
                        "Verification: source and destination file bytes "
                        "were compared."
                    )

                else:
                    with Image.open(source_path) as source:
                        if (
                            (source.format or "").upper()
                            != job.source_format
                        ):
                            raise RuntimeError(
                                "Detected image format changed after preflight."
                            )

                        crops, frame_metadata = self._crop_all_frames(
                            source,
                            job.processing_frame_indices,
                        )

                    try:
                        if job.selected_frame_plans:
                            self.logger.write(
                                "Selected policy: keep_jpeg"
                            )
                            self.logger.write(
                                "Operation: save each MPO picture as a "
                                "separate JPEG."
                            )

                            results, job_warnings = (
                                self._save_mpo_as_separate_jpegs(
                                    job,
                                    crops,
                                    frame_metadata,
                                )
                            )

                            warnings.extend(job_warnings)
                            output_files_succeeded += len(results)
                            succeeded += 1

                            self.logger.write(
                                f"Input result: SUCCESS; "
                                f"JPEG files created={len(results)}"
                            )

                        else:
                            plan = job.selected_plan

                            if plan is None:
                                raise RuntimeError(
                                    "No central output plan was selected."
                                )

                            if not frame_metadata:
                                raise RuntimeError(
                                    "No frame metadata was decoded."
                                )

                            self.logger.write(
                                f"Selected policy: {plan.policy.value}"
                            )
                            self.logger.write(
                                f"Selected strategy: {plan.strategy}"
                            )
                            self.logger.write(
                                f"Selected output format: {plan.output_format}"
                            )
                            self.logger.write(
                                f"Selected output mode: {plan.output_mode}"
                            )
                            self.logger.write(
                                f"Expected output frames: {plan.output_frames}"
                            )
                            self.logger.write(
                                f"Destination: {plan.output_path}"
                            )

                            output_metadata = dict(frame_metadata[0])
                            mpo_derivative_warnings: list[str] = []

                            if job.skip_mpo_thumbnails:
                                # Do not attach container-sensitive metadata
                                # to a standalone derivative.
                                had_exif = bool(
                                    output_metadata.pop("exif", None)
                                )
                                had_xmp = bool(
                                    output_metadata.pop("xmp", None)
                                )

                                crops[0].info.pop("exif", None)
                                crops[0].info.pop("xmp", None)

                                mpo_derivative_warnings.append(
                                    f"{source_path.name}: MPO thumbnail frame(s) "
                                    "were excluded from the derived output; "
                                    "the original MPO file was not modified."
                                )

                                if had_exif or had_xmp:
                                    mpo_derivative_warnings.append(
                                        f"{source_path.name}: Raw MPO-derived "
                                        "EXIF/XMP metadata was not copied to the "
                                        "standalone derivative because its MPF "
                                        "offsets or references may be invalid "
                                        "outside the original MPO container."
                                    )

                            result = None

                            use_mpo_coefficient_rotation = (
                                len(crops) == 1
                                and _mpo_primary_rotation_is_eligible(
                                    job,
                                    self.bounds,
                                    self.orientation,
                                    plan,
                                    self.jpegtran,
                                )
                            )

                            if use_mpo_coefficient_rotation:
                                try:
                                    result, removed_markers = (
                                        self._save_mpo_primary_with_jpegtran_rotation(
                                            job,
                                            crops[0],
                                            output_metadata,
                                            plan,
                                        )
                                    )

                                    if (
                                        removed_markers["EXIF"]
                                        or removed_markers["XMP"]
                                    ) and not any(
                                        "Raw MPO-derived EXIF/XMP metadata"
                                        in item
                                        for item in mpo_derivative_warnings
                                    ):
                                        mpo_derivative_warnings.append(
                                            f"{source_path.name}: Container-sensitive "
                                            "EXIF/XMP metadata was removed from the "
                                            "standalone JPEG. The original MPO file "
                                            "remains unchanged."
                                        )

                                    self.logger.write(
                                        "MPO optimized saving strategy: "
                                        "jpegtran_rotate (coefficient-domain; "
                                        "no JPEG re-encoding)."
                                    )

                                except Exception as exc:
                                    warning = (
                                        f"{source_path.name}: MPO Keep JPEG "
                                        "coefficient-domain path failed; "
                                        f"{type(exc).__name__}: {exc}. "
                                        "See the preceding extraction/rotation "
                                        "diagnostic entries. Falling back to "
                                        "Keep JPEG re-encoding; pixel values "
                                        "may change."
                                    )

                                    warnings.append(warning)
                                    self.logger.write("WARNING: " + warning)

                            if result is None:
                                result = save_planned_output(
                                    crops,
                                    plan,
                                    metadata=output_metadata,
                                    source_path=source_path,
                                    source_mode=job.source_mode,
                                    orientation_normalized=(
                                        job.orientation_normalized
                                    ),
                                    jpegtran=self.jpegtran,
                                )

                            succeeded += 1
                            output_files_succeeded += 1

                            self.logger.write("Result: SUCCESS")
                            self.logger.write(
                                f"Saved path: {result.path}"
                            )
                            self.logger.write(
                                f"Actual output format: {result.image_format}"
                            )
                            self.logger.write(
                                f"Saving strategy: {result.strategy}"
                            )

                            if result.strategy == "jpegtran_rotate":
                                self.logger.write(
                                    "JPEG processing note: coefficient-domain "
                                    "rotation was used; JPEG coefficients were "
                                    "not re-encoded. Raster bytes were not "
                                    "compared against a Pillow-transposed image."
                                )

                            self.logger.write(
                                f"Pixel values verified: "
                                f"{result.pixels_verified}"
                            )

                            for warning in result.warnings:
                                entry = (
                                    f"{source_path.name}: {warning}"
                                )
                                warnings.append(entry)
                                self.logger.write(
                                    f"WARNING: {warning}"
                                )

                            for warning in mpo_derivative_warnings:
                                warnings.append(warning)
                                self.logger.write(
                                    f"WARNING: {warning}"
                                )

                    finally:
                        for crop in crops:
                            crop.close()

            except Exception as exc:
                error_text = (
                    f"{source_path.name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error_text)

                self.logger.write("Result: FAILED")
                self.logger.write(error_text)
                self.logger.write("Traceback:")
                self.logger.write(traceback.format_exc())

            attempted = index

            self.progress_changed.emit(
                attempted,
                total,
                source_path.name,
            )

        if cancelled:
            run_status = "CANCELLED"
        elif errors:
            run_status = "COMPLETED WITH ERRORS"
        else:
            run_status = "COMPLETED"

        self.logger.write("")
        self.logger.write("=" * 78)
        self.logger.write(f"Run status: {run_status}")
        self.logger.write(f"Input jobs: {total}")
        self.logger.write(f"Attempted jobs: {attempted}")
        self.logger.write(f"Successful input jobs: {succeeded}")
        self.logger.write(f"Failed input jobs: {len(errors)}")
        self.logger.write(
            f"Output files successfully saved: {output_files_succeeded}"
        )
        self.logger.write(f"Warnings: {len(warnings)}")
        self.logger.write(f"Cancelled: {cancelled}")
        self.logger.write(f"Log file: {self.logger.path}")
        self.logger.write("=" * 78)

        self.completed.emit({
            "total": total,
            "attempted": attempted,
            "succeeded": succeeded,
            "output_files_succeeded": output_files_succeeded,
            "errors": errors,
            "warnings": warnings,
            "cancelled": cancelled,
            "log_path": str(self.logger.path),
        })


# ---------------------------------------------------------------------
# Progress dialog
# ---------------------------------------------------------------------

class FixedFrameProgressDialog(QDialog):
    cancel_requested = Signal()

    def __init__(
        self,
        total: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Cropping images")
        self.setFixedSize(440, 155)

        self.finished_processing = False
        self.cancel_sent = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        self.status_label = QLabel("Preparing images...")
        self.status_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, total)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._request_cancel)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(buttons)

    def update_progress(
        self,
        current: int,
        total: int,
        filename: str,
    ) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.status_label.setText(
            f"Processing image {current} of {total}:\n{filename}"
        )

    def _request_cancel(self) -> None:
        if self.finished_processing or self.cancel_sent:
            return

        self.cancel_sent = True
        self.cancel_button.setEnabled(False)
        self.status_label.setText(
            "Cancellation requested. Finishing the current image..."
        )
        self.cancel_requested.emit()

    def mark_finished(self) -> None:
        self.finished_processing = True

    def closeEvent(self, event) -> None:
        if self.finished_processing:
            event.accept()
        else:
            event.ignore()
            self._request_cancel()


# ---------------------------------------------------------------------
# Fixed Frame window
# ---------------------------------------------------------------------

class FixedFrameDialog(QDialog):
    """Fixed Frame settings, preview and batch image processing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Easy Quadrat — Fixed Frame")

        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)

        self.parent_content: QWidget | None = None
        self.parent_content_enabled = True

        self.image_files: list[Path] = []
        self.preview_source: Image.Image | None = None

        self.worker: FixedFrameWorker | None = None
        self.progress_dialog: FixedFrameProgressDialog | None = None
        self.worker_result: dict | None = None
        self.run_logger: FixedFrameLogger | None = None

        self.input_valid = False
        self.syncing = False
        self.bounds = dict(FULL_FRAME)
        self.spin_boxes: dict[str, QSpinBox] = {}

        self._build_ui()
        self._apply_bounds(self.bounds)
        self._set_responsive_geometry()
        self._lock_main_content()
        self._set_frame_controls_enabled(False)
        self._update_run_enabled()

    def _set_responsive_geometry(self) -> None:
        screen = (
            self.parentWidget().screen()
            if self.parentWidget()
            else None
        )

        if screen is None:
            screen = QApplication.primaryScreen()

        if screen is None:
            self.setMinimumSize(700, 470)
            self.resize(1100, 720)
            return

        area = screen.availableGeometry()

        minimum_width = min(760, max(600, area.width() - 40))
        minimum_height = min(560, max(430, area.height() - 40))

        width = min(1180, max(minimum_width, area.width() - 40))
        height = min(780, max(minimum_height, area.height() - 40))

        self.setMinimumSize(minimum_width, minimum_height)
        self.resize(width, height)

        self.move(
            area.x() + (area.width() - width) // 2,
            area.y() + (area.height() - height) // 2,
        )

    def _lock_main_content(self) -> None:
        parent = self.parentWidget()

        if parent is None:
            return

        central_method = getattr(parent, "centralWidget", None)

        if callable(central_method):
            content = central_method()

            if isinstance(content, QWidget):
                self.parent_content = content
                self.parent_content_enabled = content.isEnabled()
                content.setEnabled(False)

    def _unlock_main_content(self) -> None:
        if self.parent_content is not None:
            self.parent_content.setEnabled(
                self.parent_content_enabled
            )
            self.parent_content = None

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 18)
        outer.setSpacing(12)

        root = QWidget()
        self.root_content = root

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Fixed Frame")
        title.setObjectName("dialogTitle")
        title.setStyleSheet(
            "font-size: 22px; "
            "font-weight: 700; "
            "color: #57863c;"
        )

        subtitle = QLabel(
            "Set four percentage boundaries and apply the same frame "
            "to an image collection."
        )
        subtitle.setObjectName("sectionDescription")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        main_row = QHBoxLayout()
        main_row.setSpacing(16)

        left_panel = QWidget()
        left_panel.setMinimumWidth(285)
        left_panel.setMaximumWidth(385)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        folders_group = QGroupBox("Image folders")
        folders_layout = QVBoxLayout(folders_group)

        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("Select input folder")

        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Select output folder")

        folders_layout.addWidget(QLabel("Input folder"))
        folders_layout.addLayout(
            self._path_row(self.input_edit, self._choose_input)
        )
        folders_layout.addWidget(QLabel("Output folder"))
        folders_layout.addLayout(
            self._path_row(self.output_edit, self._choose_output)
        )

        left_layout.addWidget(folders_group)

        self.bounds_group = QGroupBox("Exact frame values (%)")
        bounds_grid = QGridLayout(self.bounds_group)

        for key, label, row, column in (
            ("left", "Left", 0, 0),
            ("right", "Right", 0, 2),
            ("top", "Top", 1, 0),
            ("bottom", "Bottom", 1, 2),
        ):
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix("%")
            spin.setFixedWidth(82)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin.setKeyboardTracking(False)

            spin.valueChanged.connect(
                lambda value, name=key: self._value_changed(name, value)
            )

            self.spin_boxes[key] = spin
            bounds_grid.addWidget(QLabel(label), row, column)
            bounds_grid.addWidget(spin, row, column + 1)

        self.reset_button = QPushButton("Reset frame")
        self.reset_button.clicked.connect(self._reset_frame)
        bounds_grid.addWidget(self.reset_button, 2, 0, 1, 4)

        left_layout.addWidget(self.bounds_group)

        self.orientation_group = QGroupBox("Initial orientation")
        orientation_layout = QVBoxLayout(self.orientation_group)

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(
            ["Original", "Horizontal", "Vertical"]
        )
        self.orientation_combo.setCurrentText("Original")
        self.orientation_combo.currentTextChanged.connect(
            self._update_preview
        )

        orientation_layout.addWidget(self.orientation_combo)

        orientation_note = QLabel(
            "Original preserves the source orientation. Horizontal and "
            "Vertical rotate the image by 90° only when required."
        )
        orientation_note.setObjectName("sectionDescription")
        orientation_note.setWordWrap(True)
        orientation_layout.addWidget(orientation_note)

        left_layout.addWidget(self.orientation_group)
        left_layout.addStretch(1)

        preview_group = QGroupBox("Preview and frame controls")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 12, 10, 10)
        preview_layout.setSpacing(7)

        preview_layout.addWidget(QLabel("Preview image"))

        self.preview_combo = QComboBox()
        self.preview_combo.currentIndexChanged.connect(self._load_preview)
        preview_layout.addWidget(self.preview_combo)

        self.preview_panel = PreviewWithRangeControls()
        self.preview_panel.horizontalValuesChanged.connect(
            self._horizontal_changed
        )
        self.preview_panel.verticalValuesChanged.connect(
            self._vertical_changed
        )
        preview_layout.addWidget(self.preview_panel, 1)

        self.preview_status = QLabel(
            "Select an input folder with at least one readable image."
        )
        self.preview_status.setObjectName("sectionDescription")
        self.preview_status.setWordWrap(True)
        preview_layout.addWidget(self.preview_status)

        main_row.addWidget(left_panel)
        main_row.addWidget(preview_group, 1)
        layout.addLayout(main_row, 1)

        buttons = QHBoxLayout()

        self.run_button = QPushButton("Crop images")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumWidth(125)
        self.run_button.setFixedHeight(34)
        self.run_button.clicked.connect(self._run_cropping)

        buttons.addStretch(1)
        buttons.addWidget(self.run_button)

        layout.addLayout(buttons)
        outer.addWidget(root)

    @staticmethod
    def _path_row(
        edit: QLineEdit,
        callback,
    ) -> QHBoxLayout:
        row = QHBoxLayout()

        browse = QPushButton("Browse")
        browse.setFixedWidth(76)
        browse.clicked.connect(callback)

        row.addWidget(edit, 1)
        row.addWidget(browse)

        return row

    def _set_frame_controls_enabled(self, enabled: bool) -> None:
        self.bounds_group.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.orientation_group.setEnabled(enabled)
        self.preview_combo.setEnabled(enabled)
        self.preview_panel.setEnabled(enabled)

        if not enabled:
            self.preview_panel.set_message(
                "Select an input folder with at least one readable image."
            )

    def _update_run_enabled(self) -> None:
        has_output = bool(self.output_edit.text().strip())

        self.run_button.setEnabled(
            self.input_valid
            and has_output
            and self.worker is None
        )

    def _apply_bounds(self, bounds: dict[str, int]) -> None:
        values = {
            key: max(0, min(100, int(value)))
            for key, value in bounds.items()
        }

        if values["left"] > values["right"]:
            values["right"] = values["left"]

        if values["top"] > values["bottom"]:
            values["bottom"] = values["top"]

        self.syncing = True

        try:
            self.bounds = values

            for key, value in values.items():
                spin = self.spin_boxes[key]
                blocked = spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(blocked)

            self.preview_panel.set_frame(
                values["left"],
                values["top"],
                values["right"],
                values["bottom"],
            )

        finally:
            self.syncing = False

    def _value_changed(self, key: str, value: int) -> None:
        if self.syncing:
            return

        values = dict(self.bounds)
        values[key] = int(value)

        if key == "left" and values["left"] > values["right"]:
            values["right"] = values["left"]
        elif key == "right" and values["right"] < values["left"]:
            values["left"] = values["right"]
        elif key == "top" and values["top"] > values["bottom"]:
            values["bottom"] = values["top"]
        elif key == "bottom" and values["bottom"] < values["top"]:
            values["top"] = values["bottom"]

        self._apply_bounds(values)

    def _horizontal_changed(self, lower: int, upper: int) -> None:
        if self.syncing:
            return

        values = dict(self.bounds)
        values.update(left=lower, right=upper)
        self._apply_bounds(values)

    def _vertical_changed(self, lower: int, upper: int) -> None:
        if self.syncing:
            return

        values = dict(self.bounds)
        values.update(top=lower, bottom=upper)
        self._apply_bounds(values)

    def _reset_frame(self) -> None:
        self._apply_bounds(FULL_FRAME)

    def _choose_input(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select input image folder",
            self.input_edit.text() or str(Path.home()),
        )

        if folder:
            self.input_edit.setText(folder)
            self._refresh_images()

    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_edit.text() or str(Path.home()),
        )

        if folder:
            self.output_edit.setText(folder)
            self._update_run_enabled()

    def _refresh_images(self) -> None:
        folder = Path(self.input_edit.text().strip()).expanduser()

        self.image_files = []
        self.input_valid = False
        self._set_frame_controls_enabled(False)

        self.preview_combo.blockSignals(True)
        self.preview_combo.clear()
        self.preview_combo.blockSignals(False)

        self._close_preview()

        try:
            candidates = list_supported_images(folder)
        except (OSError, ValueError) as exc:
            self.preview_status.setText(str(exc))
            self._update_run_enabled()

            self._show_message(
                "Input folder error",
                str(exc),
                QMessageBox.Icon.Warning,
            )
            return

        unreadable = []

        for path in candidates:
            if decodeable_image(path):
                self.image_files.append(path)
            else:
                unreadable.append(path.name)

        if not self.image_files:
            self.preview_panel.set_message(
                "No readable supported images were found."
            )
            self.preview_status.setText(
                "Supported: JPG/JPEG, PNG, TIF/TIFF, BMP, WEBP, MPO and GIF."
            )

            if unreadable:
                self._show_message(
                    "No readable images",
                    f"{len(unreadable)} supported file(s) were found, "
                    "but none could be decoded.",
                    QMessageBox.Icon.Warning,
                    details="\n".join(unreadable),
                )

            self._update_run_enabled()
            return

        self.preview_combo.blockSignals(True)

        for path in self.image_files:
            self.preview_combo.addItem(path.name, str(path))

        self.preview_combo.setCurrentIndex(0)
        self.preview_combo.blockSignals(False)

        self.input_valid = True
        self._set_frame_controls_enabled(True)
        self._load_preview()

        status = f"{len(self.image_files)} readable image(s) found."

        if unreadable:
            status += f" {len(unreadable)} unreadable file(s) skipped."

        self.preview_status.setText(status)
        self._update_run_enabled()

    def _close_preview(self) -> None:
        if self.preview_source is not None:
            self.preview_source.close()
            self.preview_source = None

    def _load_preview(self, *args) -> None:
        selected = self.preview_combo.currentData()

        if not selected:
            return

        path = Path(selected)
        self._close_preview()

        try:
            self.preview_source = load_preview_image(path)

            self.preview_status.setText(
                f"{path.name} · {self.preview_source.width} × "
                f"{self.preview_source.height} · {self.preview_source.mode}"
            )

            self._update_preview()

        except Exception as exc:
            self.input_valid = False
            self._set_frame_controls_enabled(False)
            self.preview_status.setText(
                f"{path.name}: {type(exc).__name__}: {exc}"
            )
            self._update_run_enabled()

    def _update_preview(self, *args) -> None:
        if self.preview_source is None:
            return

        oriented = None

        try:
            oriented = make_oriented_copy(
                self.preview_source,
                self.orientation_combo.currentText(),
            )

            self.preview_panel.set_image(oriented)
            self.preview_panel.set_frame(
                self.bounds["left"],
                self.bounds["top"],
                self.bounds["right"],
                self.bounds["bottom"],
            )

        except Exception as exc:
            self.preview_panel.set_message(
                "The image preview could not be updated."
            )
            self.preview_status.setText(str(exc))

        finally:
            if oriented is not None:
                oriented.close()

    def _show_message(
        self,
        title: str,
        message: str,
        icon: QMessageBox.Icon,
        buttons: QMessageBox.StandardButton = (
            QMessageBox.StandardButton.Ok
        ),
        details: str | None = None,
        default: QMessageBox.StandardButton | None = None,
    ) -> QMessageBox.StandardButton:
        box = QMessageBox(icon, title, message, buttons, self)

        if details:
            box.setDetailedText(details)

        if default is not None:
            box.setDefaultButton(default)

        box.setWindowModality(Qt.WindowModality.WindowModal)

        return QMessageBox.StandardButton(box.exec())

    def _ask_overwrite(self, count: int) -> bool:
        answer = self._show_message(
            "Overwrite existing outputs?",
            f"{count} output file(s) already exist. "
            "Files with matching names will be replaced. Continue?",
            QMessageBox.Icon.Warning,
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
            ),
            default=QMessageBox.StandardButton.No,
        )

        return answer == QMessageBox.StandardButton.Yes

    def _choose_output_policy(
        self,
        jobs: list[FixedFrameJob],
    ) -> OutputPolicy | None:
        """Choose between centrally planned Lossless and Keep JPEG routes."""
        flagged = [
            job
            for job in jobs
            if (
                not job.planning_error
                and not job.exact_copy
                and job.lossless_plan is not None
                and job.lossless_plan.needs_format_change
            )
        ]

        jpeg_candidates = [
            job
            for job in jobs
            if (
                not job.planning_error
                and not job.exact_copy
                and (
                    job.keep_jpeg_plan is not None
                    or (
                        job.pages > 1
                        and len(job.keep_jpeg_frame_plans) == job.pages
                    )
                )
            )
        ]

        if not flagged and not jpeg_candidates:
            return normalize_policy(OutputPolicy.LOSSLESS)

        detail_lines = []

        for job in flagged:
            plan = job.lossless_plan

            if plan is not None:
                detail_lines.append(
                    f"• {job.source_path.name}: {plan.reason}"
                )

        message = (
            "Some images require an output-format decision.\n\n"
            "Lossless output (recommended) preserves processed raster "
            "pixel values when supported by the selected operation. "
            "Some JPEG/MPO-derived images use another format in this mode.\n\n"
        )

        if jpeg_candidates:
            message += (
                "Keep JPEG: eligible outputs remain JPEG. Depending on the "
                "operation and JPEG block alignment, output may be re-encoded. "
                "Re-encoding is lossy, even at quality 100, so pixel values "
                "may change.\n\n"
                "For an MPO explicitly identified as one primary image plus "
                "thumbnail(s), only the primary image is included in the "
                "derived output. Other multi-picture MPO files retain their "
                "separate-picture output route.\n\n"
            )
        else:
            message += (
                "No eligible Keep JPEG route is available for this "
                "collection.\n\n"
            )

        if detail_lines:
            message += (
                "Lossless output planning details:\n"
                + "\n".join(detail_lines)
            )

        box = QMessageBox(
            QMessageBox.Icon.Warning,
            "Choose output format and quality",
            message,
            QMessageBox.StandardButton.NoButton,
            self,
        )

        lossless_button = box.addButton(
            (
                "Lossless output (recommended)"
                if jpeg_candidates
                else "Continue with lossless output"
            ),
            QMessageBox.ButtonRole.AcceptRole,
        )

        jpeg_button = None

        if jpeg_candidates:
            jpeg_button = box.addButton(
                "Keep JPEG",
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
            return normalize_policy(OutputPolicy.LOSSLESS)

        if jpeg_button is not None and clicked is jpeg_button:
            return normalize_policy(OutputPolicy.KEEP_JPEG)

        return None

    @staticmethod
    def _apply_selected_policy(
        jobs: list[FixedFrameJob],
        policy: OutputPolicy,
    ) -> None:
        """Choose precomputed central plans without modifying their contents."""
        normalized_policy = normalize_policy(policy)

        for job in jobs:
            job.selected_frame_plans = []

            if job.planning_error or job.exact_copy:
                continue

            if normalized_policy == OutputPolicy.KEEP_JPEG:
                if job.keep_jpeg_frame_plans:
                    job.selected_plan = None
                    job.selected_frame_plans = list(
                        job.keep_jpeg_frame_plans
                    )
                    continue

                if job.keep_jpeg_plan is not None:
                    job.selected_plan = job.keep_jpeg_plan
                    continue

            job.selected_plan = job.lossless_plan

    def _run_cropping(self) -> None:
        if not self.input_valid or not self.image_files:
            self._show_message(
                "Input images required",
                "Select a valid input folder containing readable images first.",
                QMessageBox.Icon.Warning,
            )
            return

        if not self.output_edit.text().strip():
            self._show_message(
                "Output folder required",
                "Please select an output folder.",
                QMessageBox.Icon.Warning,
            )
            return

        if (
            self.bounds["left"] == self.bounds["right"]
            or self.bounds["top"] == self.bounds["bottom"]
        ):
            self._show_message(
                "Empty cropping frame",
                "The cropping frame is empty. Adjust the boundaries.",
                QMessageBox.Icon.Warning,
            )
            return

        input_directory = (
            Path(self.input_edit.text().strip())
            .expanduser()
            .resolve()
        )

        output_directory = (
            Path(self.output_edit.text().strip())
            .expanduser()
            .resolve()
        )

        if input_directory == output_directory:
            self._show_message(
                "Separate folders required",
                "Please choose a different output folder to prevent "
                "generated files from being processed as new inputs.",
                QMessageBox.Icon.Warning,
            )
            return

        try:
            layout = prepare_output_layout(
                output_directory,
                ProcessingMethod.FIXED_FRAME,
            )
            output_root = Path(layout.root).expanduser().resolve()

        except (OSError, ValueError) as exc:
            self._show_message(
                "Output folder error",
                str(exc),
                QMessageBox.Icon.Warning,
            )
            return

        if input_directory == output_root:
            self._show_message(
                "Separate folders required",
                "Please choose a different output folder. This prevents "
                "generated outputs from being read as new inputs.",
                QMessageBox.Icon.Warning,
            )
            return

        jpegtran = find_jpegtran()

        logger = FixedFrameLogger(output_root / "Log.txt")
        self.run_logger = logger

        logger.start_run(
            input_directory=input_directory,
            output_directory=output_root,
            bounds=self.bounds,
            orientation=self.orientation_combo.currentText(),
            jpegtran=jpegtran,
        )

        jobs: list[FixedFrameJob] = []

        for path in self.image_files:
            try:
                job = plan_fixed_frame_job(
                    source_path=path,
                    output_directory=output_root,
                    bounds=self.bounds,
                    orientation=self.orientation_combo.currentText(),
                    jpegtran=jpegtran,
                )
            except Exception as exc:
                job = FixedFrameJob(
                    source_path=path,
                    planning_error=f"{type(exc).__name__}: {exc}",
                    planning_traceback=traceback.format_exc(),
                )

            jobs.append(job)
            logger.log_job_plan(job)

        policy = self._choose_output_policy(jobs)

        if policy is None:
            logger.write(
                "Run cancelled at output-policy selection. "
                "No image processing was started."
            )
            return

        self._apply_selected_policy(jobs, policy)

        logger.write("")
        logger.write(f"User-selected output policy: {policy.value}")

        for job in jobs:
            logger.log_selected_job(job)

        # Detect collisions across every output, including MPO derivatives.
        output_map: dict[str, list[tuple[Path, Path]]] = {}

        for job in jobs:
            for output in job.output_paths:
                output = Path(output)

                output_map.setdefault(
                    str(output.resolve()).casefold(),
                    [],
                ).append((Path(job.source_path), output))

        collisions = [
            items
            for items in output_map.values()
            if len(items) > 1
        ]

        if collisions:
            details = "\n".join(
                ", ".join(source.name for source, _ in items)
                for items in collisions
            )

            logger.write("Run aborted: output filename collision.")
            logger.write(details)

            self._show_message(
                "Output filename collision",
                "Multiple inputs would produce the same output name. "
                "Rename the affected files before processing.",
                QMessageBox.Icon.Warning,
                details=details,
            )
            return

        existing_outputs = [
            output
            for job in jobs
            for output in job.output_paths
            if output.exists()
        ]

        if (
            existing_outputs
            and not self._ask_overwrite(len(existing_outputs))
        ):
            logger.write(
                "Run cancelled at overwrite confirmation. "
                "No image processing was started."
            )
            return

        logger.write(
            f"Existing output files to be replaced: {len(existing_outputs)}"
        )
        logger.write("Starting batch worker.")

        self.worker_result = None
        self.root_content.setEnabled(False)
        self.back_button.setEnabled(False)
        self.run_button.setEnabled(False)

        self.progress_dialog = FixedFrameProgressDialog(
            len(jobs),
            self,
        )

        self.worker = FixedFrameWorker(
            jobs=jobs,
            bounds=self.bounds,
            orientation=self.orientation_combo.currentText(),
            jpegtran=jpegtran,
            logger=logger,
        )

        self.worker.progress_changed.connect(
            self.progress_dialog.update_progress
        )
        self.progress_dialog.cancel_requested.connect(
            self.worker.request_cancel
        )

        self.worker.completed.connect(self._worker_completed)
        self.worker.finished.connect(self._worker_finished)

        self.progress_dialog.show()
        self.worker.start()

    @Slot(object)
    def _worker_completed(self, result: dict) -> None:
        self.worker_result = result

        if self.progress_dialog is not None:
            self.progress_dialog.mark_finished()
            self.progress_dialog.accept()

    @Slot()
    def _worker_finished(self) -> None:
        self.root_content.setEnabled(True)
        self.back_button.setEnabled(True)

        self.worker = None

        if self.progress_dialog is not None:
            self.progress_dialog.deleteLater()
            self.progress_dialog = None

        self._update_run_enabled()

        if not self.worker_result:
            if self.run_logger is not None:
                self.run_logger.write(
                    "Worker finished without returning a result object."
                )
            return

        result = self.worker_result

        total = result["total"]
        attempted = result["attempted"]
        succeeded = result["succeeded"]

        output_files_succeeded = result.get(
            "output_files_succeeded",
            succeeded,
        )

        errors = result["errors"]
        warnings = result.get("warnings", [])
        cancelled = result["cancelled"]
        log_path = result.get("log_path", "")

        if cancelled:
            title = "Cropping cancelled"
            icon = QMessageBox.Icon.Warning
            summary = (
                "Processing cancelled.\n\n"
                f"Attempted input jobs: {attempted} of {total}\n"
                f"Successful input jobs: {succeeded}\n"
                f"Failed input jobs: {len(errors)}\n"
                f"Output files saved: {output_files_succeeded}"
            )

        elif errors:
            title = "Completed with errors"
            icon = QMessageBox.Icon.Warning
            summary = (
                "Processing finished with errors.\n\n"
                f"Input jobs: {total}\n"
                f"Successful input jobs: {succeeded}\n"
                f"Failed input jobs: {len(errors)}\n"
                f"Output files saved: {output_files_succeeded}"
            )

        else:
            title = "Cropping completed"
            icon = QMessageBox.Icon.Information
            summary = (
                f"All {succeeded} input image(s) were processed "
                "successfully.\n"
                f"Output files saved: {output_files_succeeded}"
            )

        detail_parts: list[str] = []

        if log_path:
            detail_parts.append(f"Log file:\n{log_path}")

        if errors:
            detail_parts.append(
                "Errors:\n" + "\n".join(errors)
            )

        if warnings:
            detail_parts.append(
                "Warnings from image processing:\n"
                + "\n".join(warnings)
            )

        if self.run_logger is not None:
            self.run_logger.write(
                f"UI summary: {title}; "
                f"attempted={attempted}/{total}; "
                f"succeeded_input_jobs={succeeded}; "
                f"successful_output_files={output_files_succeeded}; "
                f"errors={len(errors)}; warnings={len(warnings)}."
            )

            if self.run_logger.write_error:
                detail_parts.append(
                    "WARNING: The log file could not be written completely.\n"
                    + self.run_logger.write_error
                )

        self._show_message(
            title,
            summary,
            icon,
            details="\n\n".join(detail_parts) if detail_parts else None,
        )

    def reject(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        self._close_preview()
        self._unlock_main_content()
        super().reject()

    def done(self, result: int) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        self._close_preview()
        self._unlock_main_content()
        super().done(result)

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            return

        self._close_preview()
        self._unlock_main_content()
        event.accept()