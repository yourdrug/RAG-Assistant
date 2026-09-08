"""Preview unit kind -- classification of preview display units."""

from __future__ import annotations

from enum import StrEnum


class PreviewUnitKind(StrEnum):
    PAGE = "page"
    SECTION = "section"
    DOCUMENT = "document"
