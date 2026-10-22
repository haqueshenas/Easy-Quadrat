# -*- mode: python ; coding: utf-8 -*-
"""
Easy Quadrat v2.0.0
PyInstaller Windows x64 one-directory build.

Build using the shared Python environment:
    C:/envs/shared/Scripts/python.exe -m PyInstaller EasyQuadrat.spec

This file configures the build; it is not the release executable.
"""

import hashlib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(SPECPATH).resolve()
APP_NAME = "EasyQuadrat"

ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"
VENDOR_DIR = ROOT / "vendor" / "libjpeg-turbo"

JPEGTRAN_EXE = VENDOR_DIR / "win-x64" / "jpegtran.exe"
JPEGTRAN_DLL = VENDOR_DIR / "win-x64" / "jpeg62.dll"

ICON_FILE = ASSETS_DIR / "EQ.ico"

LICENSE_FILE = ROOT / "LICENSE"
AGPL_LICENSE_FILE = ROOT / "LICENSE-AGPL-3.0.txt"
NOTICES_FILE = ROOT / "THIRD_PARTY_NOTICES.txt"

MODEL_FILES = (
    "easyquadrat.pt",
    "easyquadrat2.pt",
    "EQ_Frameless_1.pt",
    "EQ_Frameless_2.pt",
)

REQUIRED_PATHS = (
    ROOT / "main.py",
    ICON_FILE,
    ASSETS_DIR,
    MODELS_DIR,
    VENDOR_DIR / "LICENSE.md",
    VENDOR_DIR / "README.ijg",
    VENDOR_DIR / "README.md",
    JPEGTRAN_EXE,
    JPEGTRAN_DLL,
    LICENSE_FILE,
    AGPL_LICENSE_FILE,
    NOTICES_FILE,
)

for required_path in REQUIRED_PATHS:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required Easy Quadrat build file was not found: "
            f"{required_path}"
        )


EXPECTED_MODELS = {
    "easyquadrat.pt": (
        6744355,
        "ef318ef36e1fc1df8b31df46903b3aa26200366912194415f4b774ef4e91cc2d",
    ),
    "easyquadrat2.pt": (
        54786955,
        "ab0a9e8c6dfa34aa8f72a4b2cade9e1b5037204899da73fd23305ed464a08fee",
    ),
    "EQ_Frameless_1.pt": (
        6742691,
        "dd4f8fe4eeae7533cccc9947d57a718176aeec920f705303d61b77f87d7293ac",
    ),
    "EQ_Frameless_2.pt": (
        6742883,
        "3f557ee46bfe8cd45360913647ef20cb55f84cc4ee1ea85698c923344e679324",
    ),
}

for model_filename in MODEL_FILES:
    model_path = MODELS_DIR / model_filename

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Required built-in model was not found: {model_path}"
        )

    expected_size, expected_sha256 = EXPECTED_MODELS[model_filename]
    actual_size = model_path.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            f"Incorrect model size for {model_filename}: "
            f"expected {expected_size} bytes, got {actual_size}"
        )

    digest = hashlib.sha256()

    with model_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)

    actual_sha256 = digest.hexdigest()

    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {model_filename}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    print(
        f"MODEL_OK: {model_filename} | "
        f"{actual_size} bytes | SHA-256 verified"
    )


# ---------------------------------------------------------------------
# Data files and native binaries
# ---------------------------------------------------------------------

datas = [
    (str(ASSETS_DIR), "assets"),
    (str(MODELS_DIR), "models"),

    # Keep original upstream documentation.
    (str(VENDOR_DIR / "LICENSE.md"), "vendor/libjpeg-turbo"),
    (str(VENDOR_DIR / "README.ijg"), "vendor/libjpeg-turbo"),
    (str(VENDOR_DIR / "README.md"), "vendor/libjpeg-turbo"),

    # jpegtran is launched as a separate executable.
    (
        str(JPEGTRAN_EXE),
        "vendor/libjpeg-turbo/win-x64",
    ),

    # These license and notice files are included in the bundle.
    # Final release assembly must also place accessible copies beside
    # EasyQuadrat.exe, outside the _internal directory.
    (str(LICENSE_FILE), "."),
    (str(AGPL_LICENSE_FILE), "."),
    (str(NOTICES_FILE), "."),
]

# jpeg62.dll must remain beside jpegtran.exe.
binaries = [
    (
        str(JPEGTRAN_DLL),
        "vendor/libjpeg-turbo/win-x64",
    ),
]

hiddenimports = []

# Ultralytics uses package resources and dynamic imports.
# PyInstaller's installed hooks handle the associated dependencies
# where applicable.
ultralytics_datas, ultralytics_binaries, ultralytics_hiddenimports = (
    collect_all("ultralytics")
)

datas += ultralytics_datas
binaries += ultralytics_binaries
hiddenimports += ultralytics_hiddenimports


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)


# ---------------------------------------------------------------------
# One-directory executable
# ---------------------------------------------------------------------

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON_FILE),
    contents_directory="_internal",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)