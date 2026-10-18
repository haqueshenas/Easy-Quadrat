"""
Easy Quadrat v2.0.0
Application configuration and product metadata.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from pathlib import Path

APP_NAME = "Easy Quadrat"
APP_VERSION = "2.0.0"
APP_BUILD_STATUS = "Development build"

APP_SHORT_TAGLINE = (
    "The traditional quadrat, brought into "
    "image-based crop phenotyping."
)

APP_DESCRIPTION = (
    "Easy Quadrat is nothing but the same quadrat "
    "traditionally used in agronomy and crop science. "
    "Now, it has been brought into the field of modern "
    "phenotyping, for cropping and sampling; not from "
    "the plants themselves but rather from the images "
    "of plants."
)

APP_LIMITATION = (
    "Easy Quadrat supports locating and extracting quadrat "
    "regions from plant images. It does not, by itself, "
    "measure or validate crop phenotypic traits."
)

APP_AUTHOR = "Abbas Haghshenas"
APP_ORGANIZATION = "Easy-Phenotyping Lab (EPL)"
APP_WEBSITE = "https://haqueshenas.github.io/EPL/"
APP_REPOSITORY = "https://github.com/haqueshenas/Easy-Quadrat"
APP_EMAIL = "haqueshenas@gmail.com"

APP_LICENSE = "MIT License"
AI_CODING_ASSISTANCE = "OpenAI GPT-6"

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

APP_ICON_PATH = ASSETS_DIR / "EQ.ico"
APP_LOGO_PATH = ASSETS_DIR / "EQ.png"

# EPL visual identity
COLOR_GREEN = "#57863c"
COLOR_BACKGROUND = "#f5f6f4"
COLOR_TEXT = "#202520"
COLOR_MUTED = "#5f665f"
COLOR_BORDER = "#e2e6df"
COLOR_WHITE = "#ffffff"
COLOR_LIGHT_GREEN = "#e8eee5"