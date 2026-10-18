
"""
Easy Quadrat v2.0.0
Image loading and validation.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from dataclasses import dataclass
from pathlib import Path
import warnings

from PIL import Image, UnidentifiedImageError

from core.image_catalog import is_supported_image_path


def _build_extension_format_map() -> dict[str, frozenset[str]]:
    """Build a map from supported extensions to Pillow format names."""

    Image.init()

    readable_formats = set(Image.OPEN)
    writable_formats = set(Image.SAVE)
    mapping: dict[str, set[str]] = {}

    for extension, image_format in Image.registered_extensions().items():
        if (
            image_format in readable_formats
            and image_format in writable_formats
        ):
            mapping.setdefault(extension.casefold(), set()).add(
                image_format
            )

    return {
        extension: frozenset(formats)
        for extension, formats in mapping.items()
    }


EXPECTED_FORMATS_BY_EXTENSION = _build_extension_format_map()


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or validated."""


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """Decoded image and metadata."""

    source_path: Path
    image: Image.Image
    image_format: str
    width: int
    height: int
    mode: str


def load_image(path: str | Path) -> LoadedImage:
    """
    Load one image and validate its extension against its actual format.

    The decoded image mode and pixel dimensions are preserved.
    No resize, crop, rotation, or color conversion is applied.
    """

    raw_path = str(path).strip()

    if not raw_path:
        raise ValueError("The image path cannot be empty.")

    image_path = Path(raw_path).expanduser()

    if not image_path.exists():
        raise FileNotFoundError(
            f"The image file does not exist: {image_path}"
        )

    if not image_path.is_file():
        raise IsADirectoryError(
            f"The supplied path is not a file: {image_path}"
        )

    if not is_supported_image_path(image_path):
        raise ImageLoadError(
            f"Unsupported image extension: {image_path.suffix}"
        )

    expected_formats = EXPECTED_FORMATS_BY_EXTENSION.get(
        image_path.suffix.casefold()
    )

    if not expected_formats:
        raise ImageLoadError(
            f"No supported image decoder and encoder are registered for "
            f"the extension: {image_path.suffix}"
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(image_path) as source_image:
                actual_format = (source_image.format or "").upper()

                source_image.load()

                width, height = source_image.size
                mode = source_image.mode

                pixels = source_image.copy()
                pixels.info = dict(source_image.info)

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageLoadError(
            f"Could not read image '{image_path}': {exc}"
        ) from exc

    if actual_format not in expected_formats:
        pixels.close()

        expected_text = ", ".join(sorted(expected_formats))

        raise ImageLoadError(
            f"The file extension does not match the image content. "
            f"Extension: '{image_path.suffix}'. "
            f"Detected format: '{actual_format or 'unknown'}'. "
            f"Expected: {expected_text}."
        )

    return LoadedImage(
        source_path=image_path.resolve(),
        image=pixels,
        image_format=actual_format,
        width=width,
        height=height,
        mode=mode,
    )