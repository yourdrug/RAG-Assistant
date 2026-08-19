"""PDF page content type classification."""

from __future__ import annotations

from enum import StrEnum


class PageContentType(StrEnum):
    TEXT = "text"
    SCAN = "scan"
    GARBLED = "garbled"
    EMPTY = "empty"
    TABLE = "table"
    OCR = "ocr"
