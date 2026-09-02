"""Page content type classification for dry-run preview."""

from __future__ import annotations

from enum import StrEnum


class PageContentType(StrEnum):
    TEXT = "text"
    SCAN = "scan"
    GARBLED = "garbled"
    EMPTY = "empty"
    TABLE = "table"
    OCR = "ocr"
    IMAGE_ONLY = "image_only"


class PreviewUnitKind(StrEnum):
    PAGE = "page"
    SECTION = "section"
    DOCUMENT = "document"
