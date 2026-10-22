
"""
Easy Quadrat v2.0.0
Image input catalog.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from pathlib import Path

from PIL import Image


def _get_supported_extensions() -> frozenset[str]:
    """
    Return extensions for formats supported for both reading and saving
    by the active Pillow installation.
    """

    Image.init()

    readable_formats = set(Image.OPEN)
    writable_formats = set(Image.SAVE)

    return frozenset(
        extension.casefold()
        for extension, image_format in Image.registered_extensions().items()
        if image_format in readable_formats
        and image_format in writable_formats
    )


SUPPORTED_IMAGE_EXTENSIONS = _get_supported_extensions()


def is_supported_image_path(path: str | Path) -> bool:
    """Check whether a path has a supported image extension."""

    return Path(path).suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS


def list_image_files(directory: str | Path) -> list[Path]:
    """
    Return supported image candidates from one directory.

    Subdirectories are not searched recursively.
    File contents are validated separately by the image loader.
    """

    raw_directory = str(directory).strip()

    if not raw_directory:
        raise ValueError("The input directory cannot be empty.")

    folder = Path(raw_directory).expanduser()

    if not folder.exists():
        raise FileNotFoundError(
            f"The input directory does not exist: {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"The input path is not a directory: {folder}"
        )

    image_files = [
        path
        for path in folder.iterdir()
        if path.is_file() and is_supported_image_path(path)
    ]

    return sorted(
        image_files,
        key=lambda path: (path.name.casefold(), path.name),
    )