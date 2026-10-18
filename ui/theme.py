
"""
Easy Quadrat v2.0.0
Centralized application stylesheet.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6
"""

from app_config import (
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_GREEN,
    COLOR_LIGHT_GREEN,
    COLOR_MUTED,
    COLOR_TEXT,
    COLOR_WHITE,
)


STYLESHEET = f"""
QMainWindow, QDialog {{
    background-color: {COLOR_BACKGROUND};
}}

QWidget#root {{
    background-color: {COLOR_BACKGROUND};
}}

/* -------------------------------------------------------------
   Logo
   No frame, no border, and no rounded corners.
   ------------------------------------------------------------- */

QLabel#appLogo {{
    background-color: transparent;
    border: none;
    border-radius: 0px;
    padding: 0px;
}}

QLabel#logoFallback {{
    color: {COLOR_GREEN};
    background-color: transparent;
    border: none;
    border-radius: 0px;
    font-size: 22px;
    font-weight: 700;
}}

/* -------------------------------------------------------------
   Main header
   ------------------------------------------------------------- */

QLabel#eyebrow {{
    color: {COLOR_GREEN};
    font-size: 10px;
    font-weight: 700;
}}

QLabel#appTitle {{
    color: {COLOR_TEXT};
    font-size: 27px;
    font-weight: 700;
}}

QLabel#subtitle {{
    color: {COLOR_MUTED};
    font-size: 12px;
}}

QLabel#versionPill {{
    color: #3f652d;
    background-color: {COLOR_LIGHT_GREEN};
    border: none;
    border-radius: 8px;
    padding: 7px 10px;
    font-size: 10px;
    font-weight: 700;
}}

/* -------------------------------------------------------------
   Separators and section headings
   ------------------------------------------------------------- */

QFrame#headerRule,
QFrame#footerLine {{
    background-color: {COLOR_BORDER};
    border: none;
    max-height: 1px;
}}

QLabel#sectionTitle {{
    color: {COLOR_TEXT};
    font-size: 16px;
    font-weight: 700;
}}

QLabel#sectionDescription {{
    color: {COLOR_MUTED};
    font-size: 11px;
}}

/* -------------------------------------------------------------
   Processing method cards
   ------------------------------------------------------------- */

QFrame#workflowCard {{
    background-color: {COLOR_WHITE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 10px;
}}

QLabel#cardNumber {{
    color: {COLOR_GREEN};
    font-size: 9px;
    font-weight: 700;
}}

QLabel#cardTitle {{
    color: {COLOR_TEXT};
    font-size: 13px;
    font-weight: 700;
}}

QLabel#cardText {{
    color: {COLOR_MUTED};
    font-size: 10px;
}}

/* -------------------------------------------------------------
   Primary action button
   Used for Open Fixed Frame and other primary actions.
   ------------------------------------------------------------- */

QPushButton#primaryButton {{
    color: {COLOR_WHITE};
    background-color: {COLOR_GREEN};
    border: none;
    border-radius: 6px;
    padding: 7px 18px;
    font-size: 11px;
    font-weight: 600;
}}

QPushButton#primaryButton:hover {{
    background-color: #496f32;
}}

QPushButton#primaryButton:pressed {{
    background-color: #3f652d;
}}

QPushButton#primaryButton:disabled {{
    color: {COLOR_BACKGROUND};
    background-color: #a5b69b;
}}

/* -------------------------------------------------------------
   Footer and About button
   ------------------------------------------------------------- */

QLabel#footerText {{
    color: {COLOR_MUTED};
    font-size: 10px;
}}

QPushButton#aboutButton {{
    color: {COLOR_TEXT};
    background-color: transparent;
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 600;
}}

QPushButton#aboutButton:hover {{
    color: {COLOR_GREEN};
    background-color: {COLOR_LIGHT_GREEN};
    border: 1px solid {COLOR_GREEN};
}}

QPushButton#aboutButton:pressed {{
    background-color: #dce7d6;
}}

/* -------------------------------------------------------------
   About dialog
   ------------------------------------------------------------- */

QLabel#aboutTitle {{
    color: {COLOR_TEXT};
    font-size: 21px;
    font-weight: 700;
}}

QLabel#aboutVersion {{
    color: {COLOR_GREEN};
    font-size: 11px;
    font-weight: 700;
}}

QLabel#aboutBody {{
    color: {COLOR_MUTED};
    font-size: 11px;
}}

QLabel#aboutIntro {{
    color: {COLOR_TEXT};
    font-size: 12px;
}}

QLabel#aboutCredits {{
    color: {COLOR_MUTED};
    font-size: 11px;
}}

QLabel#aboutLinks {{
    font-size: 11px;
}}

QLabel#aboutLinks a {{
    color: {COLOR_GREEN};
    text-decoration: none;
}}

QLabel#aboutLinks a:hover {{
    text-decoration: underline;
}}

QPushButton#closeButton {{
    color: {COLOR_WHITE};
    background-color: {COLOR_GREEN};
    border: none;
    border-radius: 6px;
    padding: 7px 18px;
    font-size: 11px;
    font-weight: 600;
}}

QPushButton#closeButton:hover {{
    background-color: #496f32;
}}

QPushButton#closeButton:pressed {{
    background-color: #3f652d;
}}
"""