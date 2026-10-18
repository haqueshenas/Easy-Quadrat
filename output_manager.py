"""
Easy Quadrat v2.0.0
Output directory management and processing logs.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from app_config import APP_VERSION


class ProcessingMethod(str, Enum):
    """The five original processing methods."""

    MANUAL = "Manual"
    FIXED_FRAME = "Fixed Frame"
    CLASSIC_I = "Auto (Classic I)"
    CLASSIC_II = "Auto (Classic II)"
    AI_ASSISTED = "Auto (AI assisted)"


_LOG_LABELS = {
    ProcessingMethod.CLASSIC_I: "Automated (classic I)",
    ProcessingMethod.CLASSIC_II: "Automated (classic II)",
    ProcessingMethod.AI_ASSISTED: "Auto (AI assisted)",
}


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Resolved paths for one selected processing method."""

    root: Path
    method: ProcessingMethod
    cropped_images: Path | None = None
    segmented_images: Path | None = None
    labels: Path | None = None
    log_file: Path | None = None


def prepare_output_layout(
    output_directory: str | Path,
    method: ProcessingMethod | str,
) -> OutputLayout:
    """
    Validate an existing output directory and prepare required folders.

    Existing folders are reused; their contents are not deleted.

    Manual:
        Cropped images/
        Segmented images/
        Labels/

    Fixed Frame:
        Output images are written directly into root.

    Classic I, Classic II, and AI assisted:
        Cropped images/
        Segmented images/
        Log.txt
    """

    raw_directory = str(output_directory).strip()

    if not raw_directory:
        raise ValueError("Please select the output directory.")

    try:
        selected_method = ProcessingMethod(method)
    except (ValueError, TypeError) as exc:
        allowed = ", ".join(item.value for item in ProcessingMethod)

        raise ValueError(
            f"Unknown processing method: {method!r}. "
            f"Supported methods: {allowed}."
        ) from exc

    root = Path(raw_directory).expanduser()

    if not root.exists():
        raise FileNotFoundError(
            f"The output directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"The output path is not a directory: {root}"
        )

    root = root.resolve()

    cropped_images = None
    segmented_images = None
    labels = None
    log_file = None

    if selected_method is ProcessingMethod.MANUAL:
        cropped_images = root / "Cropped images"
        segmented_images = root / "Segmented images"
        labels = root / "Labels"

    elif selected_method is ProcessingMethod.FIXED_FRAME:
        # Preserve the original output arrangement.
        # Fixed Frame writes C_<original filename> into root.
        pass

    else:
        cropped_images = root / "Cropped images"
        segmented_images = root / "Segmented images"
        log_file = root / "Log.txt"

    for directory in (
        cropped_images,
        segmented_images,
        labels,
    ):
        if directory is not None:
            directory.mkdir(exist_ok=True)

    return OutputLayout(
        root=root,
        method=selected_method,
        cropped_images=cropped_images,
        segmented_images=segmented_images,
        labels=labels,
        log_file=log_file,
    )


class ProcessingRunLog:
    """
    Write the processing log used by the three automated methods.

    The legacy Log.txt convention is retained. Per-image errors and warnings
    identify the affected filename. A failed run is not reported as normally
    completed.
    """

    def __init__(
        self,
        log_path: str | Path,
        method_name: str,
        settings_text: str,
        total_images: int,
    ) -> None:
        if total_images < 0:
            raise ValueError("The image count cannot be negative.")

        raw_path = str(log_path).strip()

        if not raw_path:
            raise ValueError("The log file path cannot be empty.")

        self.path = Path(raw_path).expanduser()
        self._closed = False

        if not self.path.parent.is_dir():
            raise NotADirectoryError(
                f"The log directory does not exist: {self.path.parent}"
            )

        # Match the legacy application's behavior: each run replaces
        # the previous Log.txt in the same output directory.
        self._stream = self.path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        )

        try:
            self._write(
                f"\nEasy Quadrat Version {APP_VERSION}\n"
                "..........................\n\n"
                f"Cropping method: {method_name}\n\n"
                f"Settings: {settings_text}\n\n"
                f"Total images: {total_images}\n\n"
                f"Starting time: {self._timestamp()}\n\n"
                "List of processed images:\n"
                ".........................\n"
            )
        except Exception:
            self._stream.close()
            self._closed = True
            raise

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("The processing log is closed.")

        self._stream.write(text)
        self._stream.flush()

    def record_success(self, filename: str) -> None:
        """Record an image processed successfully."""

        self._write(f"{filename}\n")

    def record_warning(
        self,
        filename: str,
        warning: BaseException | str,
    ) -> None:
        """Record a non-fatal warning and identify its affected output."""

        if isinstance(warning, BaseException):
            details = f"{type(warning).__name__}: {warning}"
        else:
            details = str(warning)

        details = details.strip()

        if not details:
            return

        self._write(
            f"\nWarning for '{filename}': {details}\n"
        )

    def record_error(
        self,
        filename: str,
        error: BaseException | str,
    ) -> None:
        """Record an image-processing error."""

        if isinstance(error, BaseException):
            details = f"{type(error).__name__}: {error}"
        else:
            details = str(error)

        self._write(
            f"\nError processing '{filename}': {details}\n"
        )

    def finish(self) -> None:
        """Record normal completion and close the log."""

        if self._closed:
            return

        try:
            self._write(
                ".........................\n\n"
                f"Completion time: {self._timestamp()}\n"
            )
        finally:
            self._close_stream()

    def close(self) -> None:
        """Close a run that did not finish normally."""

        if self._closed:
            return

        try:
            self._write(
                "\nRun ended before normal completion.\n"
                f"End time: {self._timestamp()}\n"
            )
        finally:
            self._close_stream()

    def _close_stream(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True

    def __enter__(self) -> "ProcessingRunLog":
        if self._closed:
            raise RuntimeError("The processing log is closed.")

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        if self._closed:
            return False

        if exc_type is None:
            self.finish()
        else:
            try:
                self._write(
                    "\nRun aborted because of an unhandled exception:\n"
                    f"{exc_type.__name__}: {exc_value}\n"
                    f"End time: {self._timestamp()}\n"
                )
            finally:
                self._close_stream()

        # Do not suppress the original exception.
        return False