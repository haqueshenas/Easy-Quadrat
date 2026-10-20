"""
Easy Quadrat v2.0.0
About dialog.

Author: Abbas Haghshenas
Initiative: Easy-Phenotyping Lab (EPL)
Website: https://haqueshenas.github.io/EPL/
Repository: https://github.com/haqueshenas/Easy-Quadrat
Copyright (c) 2026 Abbas Haghshenas
License: MIT License; see LICENSE
AI coding assistance: OpenAI GPT-6

Third-party software and trained model weights are separately attributed.
See THIRD_PARTY_NOTICES.txt and the applicable upstream terms.
"""

from __future__ import annotations

import html
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app_config import (
    AI_CODING_ASSISTANCE,
    APP_AUTHOR,
    APP_DESCRIPTION,
    APP_EMAIL,
    APP_LICENSE,
    APP_LIMITATION,
    APP_LOGO_PATH,
    APP_NAME,
    APP_ORGANIZATION,
    APP_REPOSITORY,
    APP_VERSION,
    APP_WEBSITE,
)
from ui.window_utils import configure_window


# ---------------------------------------------------------------------
# Official upstream links
# ---------------------------------------------------------------------

ULTRALYTICS_SEGMENTATION_URL = (
    "https://docs.ultralytics.com/tasks/segment"
)

ULTRALYTICS_LICENSE_URL = (
    "https://www.ultralytics.com/license"
)

AGPL_LICENSE_URL = (
    "https://www.gnu.org/licenses/agpl-3.0.html"
)


# ---------------------------------------------------------------------
# Link helpers
# ---------------------------------------------------------------------

def _extract_url(value: str) -> str:
    """
    Extract a usable HTTP(S) URL from either a raw URL or a Markdown link.

    app_config.py may store APP_WEBSITE and APP_REPOSITORY in Markdown
    link form. Extract their destinations before constructing HTML links.
    """
    text = str(value or "").strip()

    markdown_match = re.search(
        r"\]\((https?://[^)\s]+)\)",
        text,
    )

    if markdown_match:
        return markdown_match.group(1)

    plain_match = re.search(
        r"https?://[^\s)\]]+",
        text,
    )

    if plain_match:
        return plain_match.group(0)

    return ""


def _external_link(
    url: str,
    label: str,
) -> str:
    """Build an escaped HTML link, or plain text if no URL is available."""
    if not url:
        return html.escape(label)

    safe_url = html.escape(url, quote=True)
    safe_label = html.escape(label)

    return f'<a href="{safe_url}">{safe_label}</a>'


# ---------------------------------------------------------------------
# About dialog
# ---------------------------------------------------------------------

class AboutDialog(QDialog):
    """Application information, model notices, and acknowledgements."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(f"About {APP_NAME}")
        self.setMinimumSize(560, 520)
        self.resize(650, 640)
        self.setModal(True)

        configure_window(
            self,
            parent=parent,
            keep_on_top=True,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 23, 26, 22)
        root.setSpacing(14)

        # -------------------------------------------------------------
        # Header: logo, application name, version, organization
        # -------------------------------------------------------------

        heading = QHBoxLayout()
        heading.setContentsMargins(0, 0, 0, 0)
        heading.setSpacing(15)

        logo_label = QLabel()
        logo_label.setObjectName("appLogo")
        logo_label.setFixedSize(66, 66)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if APP_LOGO_PATH.is_file():
            pixmap = QPixmap(str(APP_LOGO_PATH))

            if not pixmap.isNull():
                logo_label.setPixmap(
                    pixmap.scaled(
                        66,
                        66,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                logo_label.setText("EQ")
                logo_label.setObjectName("logoFallback")
        else:
            logo_label.setText("EQ")
            logo_label.setObjectName("logoFallback")

        heading.addWidget(
            logo_label,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(5)

        title = QLabel(APP_NAME)
        title.setObjectName("aboutTitle")

        # Use the application version, not a temporary build status.
        version = QLabel(f"Version {APP_VERSION}")
        version.setObjectName("aboutVersion")
        version.setWordWrap(True)

        organization = QLabel(APP_ORGANIZATION)
        organization.setObjectName("aboutBody")
        organization.setWordWrap(True)

        title_column.addWidget(title)
        title_column.addWidget(version)
        title_column.addWidget(organization)
        title_column.addStretch(1)

        heading.addLayout(title_column, 1)
        root.addLayout(heading)

        header_rule = QFrame()
        header_rule.setObjectName("headerRule")
        header_rule.setFixedHeight(1)
        root.addWidget(header_rule)

        # -------------------------------------------------------------
        # Scrollable informational content
        # -------------------------------------------------------------

        scroll_area = QScrollArea()
        scroll_area.setObjectName("aboutScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content.setObjectName("aboutContent")

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(14)

        # -------------------------------------------------------------
        # Main product description
        # -------------------------------------------------------------

        introduction = QLabel(APP_DESCRIPTION)
        introduction.setObjectName("aboutIntro")
        introduction.setWordWrap(True)
        introduction.setAlignment(Qt.AlignmentFlag.AlignTop)

        content_layout.addWidget(introduction)

        # -------------------------------------------------------------
        # Scope and scientific limitations
        # -------------------------------------------------------------

        limitation = QLabel(APP_LIMITATION)
        limitation.setObjectName("aboutBody")
        limitation.setWordWrap(True)
        limitation.setAlignment(Qt.AlignmentFlag.AlignTop)

        content_layout.addWidget(limitation)

        # -------------------------------------------------------------
        # Developer, copyright, and application source license
        # -------------------------------------------------------------

        credits = QLabel(
            f"<b>Developer:</b> {html.escape(APP_AUTHOR)}<br>"
            "<b>Copyright:</b> © 2026 Abbas Haghshenas<br>"
            f"<b>Initiative:</b> {html.escape(APP_ORGANIZATION)}<br>"
            f"<b>Application source license:</b> "
            f"{html.escape(APP_LICENSE)}; see LICENSE<br>"
            f"<b>AI coding assistance:</b> "
            f"{html.escape(AI_CODING_ASSISTANCE)}"
        )
        credits.setObjectName("aboutCredits")
        credits.setTextFormat(Qt.TextFormat.RichText)
        credits.setWordWrap(True)

        content_layout.addWidget(credits)

        # -------------------------------------------------------------
        # Built-in models and third-party licensing
        # -------------------------------------------------------------

        third_party_title = QLabel(
            "Built-in models and third-party licensing"
        )
        third_party_title.setObjectName("sectionTitle")
        third_party_title.setWordWrap(True)

        content_layout.addWidget(third_party_title)

        third_party_notice = QLabel(
            "<b>Separate licensing:</b> The MIT License identified "
            "above applies to the Easy Quadrat application source as "
            "specified in the root LICENSE file. It does not, by itself, "
            "replace the applicable terms for third-party software or "
            "model weights.<br><br>"

            "<b>Built-in models:</b><br>"
            "&bull; <b>EQ1 Standard</b> — "
            "easyquadrat.pt (YOLOv8n-seg).<br>"
            "&bull; <b>EQ2 Standard</b> — "
            "easyquadrat2.pt (YOLOv8m-seg).<br>"
            "&bull; <b>EQ Frameless 1</b> — "
            "EQ_Frameless_1.pt.<br>"
            "&bull; <b>EQ Frameless 2</b> — "
            "EQ_Frameless_2.pt.<br><br>"

            "The Frameless variants are intended to reduce the "
            "influence of the quadrat frame on detection. Model "
            "architecture, provenance, and licensing information "
            "should be read together with the bundled "
            "THIRD_PARTY_NOTICES.txt file.<br><br>"

            "<b>Ultralytics and YOLO licensing:</b> The Ultralytics "
            "YOLOv8 framework and related trained model weights are "
            "subject to third-party licensing separate from the Easy Quadrat "
            "application source. Ultralytics identifies the <b>GNU Affero "
            "General Public License v3.0 (AGPL-3.0)</b> as its open-source "
            "license and also offers an <b>Ultralytics Enterprise License</b> "
            "for use cases covered by a commercial agreement. Ultralytics "
            "states that its trained YOLO models are covered by AGPL-3.0 by "
            "default; a different commercial license applies only where a "
            "valid agreement covers the relevant components and use. The "
            "applicable terms should be confirmed for the specific bundled "
            "model files and intended use. Consult the official Ultralytics "
            "licensing information and the license text linked below before "
            "redistributing or adapting these components.<br><br>"

            "<b>Custom models:</b> Model files selected by users may "
            "have separate authorship, licensing, and redistribution "
            "conditions. Users are responsible for ensuring that their "
            "use of custom models is consistent with the applicable "
            "terms."
        )

        third_party_notice.setObjectName("aboutBody")
        third_party_notice.setTextFormat(Qt.TextFormat.RichText)
        third_party_notice.setWordWrap(True)
        third_party_notice.setAlignment(Qt.AlignmentFlag.AlignTop)

        content_layout.addWidget(third_party_notice)

        # -------------------------------------------------------------
        # Official upstream references
        # -------------------------------------------------------------

        upstream_links = QLabel(
            _external_link(
                ULTRALYTICS_SEGMENTATION_URL,
                "Ultralytics YOLO segmentation documentation",
            )
            + "<br>"
            + _external_link(
                ULTRALYTICS_LICENSE_URL,
                "Ultralytics licensing information",
            )
            + "<br>"
            + _external_link(
                AGPL_LICENSE_URL,
                "GNU AGPL-3.0 license text",
            )
        )

        upstream_links.setObjectName("aboutLinks")
        upstream_links.setTextFormat(Qt.TextFormat.RichText)
        upstream_links.setOpenExternalLinks(True)
        upstream_links.setWordWrap(True)

        content_layout.addWidget(upstream_links)

        # -------------------------------------------------------------
        # Project links
        # -------------------------------------------------------------

        website_url = _extract_url(APP_WEBSITE)
        repository_url = _extract_url(APP_REPOSITORY)

        contact_url = (
            f"mailto:{APP_EMAIL.strip()}"
            if APP_EMAIL and APP_EMAIL.strip()
            else ""
        )

        project_links = [
            _external_link(
                website_url,
                "EPL website",
            ),
            _external_link(
                repository_url,
                "Easy Quadrat GitHub repository",
            ),
            _external_link(
                contact_url,
                "Contact",
            ),
        ]

        links = QLabel(
            "&nbsp;&nbsp;·&nbsp;&nbsp;".join(project_links)
        )
        links.setObjectName("aboutLinks")
        links.setTextFormat(Qt.TextFormat.RichText)
        links.setOpenExternalLinks(True)
        links.setWordWrap(True)

        content_layout.addWidget(links)
        content_layout.addStretch(1)

        scroll_area.setWidget(content)
        root.addWidget(scroll_area, 1)

        # -------------------------------------------------------------
        # Close button
        # -------------------------------------------------------------

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        close_button = QPushButton("Close")
        close_button.setObjectName("closeButton")
        close_button.setMinimumWidth(88)
        close_button.setFixedHeight(32)
        close_button.clicked.connect(self.accept)

        button_row.addWidget(close_button)
        root.addLayout(button_row)
