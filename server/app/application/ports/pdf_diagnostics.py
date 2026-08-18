"""PDF diagnostics ports — abstract interfaces for PDF analysis infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class PDFPageClassifierPort(Protocol):
    def classify_page(self, text: str, chars: int, page=None) -> tuple[str, str]: ...


@runtime_checkable
class PDFTextCleanerPort(Protocol):
    def clean(self, text: str) -> str: ...


@runtime_checkable
class PDFOcrPort(Protocol):
    def ocr_pages(self, pdf, pages: list[int], filename: str) -> dict[int, str]: ...


@runtime_checkable
class PDFDocumentPort(Protocol):
    def open(self, path: str): ...
    def get_page_count(self, doc) -> int: ...
    def get_page_text(self, doc, index: int) -> str: ...
    def find_tables(self, page) -> bool: ...
    def close(self, doc) -> None: ...


@runtime_checkable
class PDFStoragePort(Protocol):
    def download_to_temp(self, key: str) -> Path: ...
