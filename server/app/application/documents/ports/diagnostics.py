"""PDF diagnostics ports — abstract interfaces for PDF analysis infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from pathlib import Path


class PDFDocumentHandle(Protocol):
    def load_page(self, index: int) -> object: ...


@runtime_checkable
class PDFPageClassifierPort(Protocol):
    def classify_page(self, text: str, chars: int, page: object | None = None) -> tuple[str, str]: ...


@runtime_checkable
class PDFTextCleanerPort(Protocol):
    def clean(self, text: str) -> str: ...


@runtime_checkable
class PDFOcrPort(Protocol):
    def ocr_pages(self, pdf: PDFDocumentHandle, pages: list[int], filename: str) -> dict[int, str]: ...


@runtime_checkable
class PDFDocumentPort(Protocol):
    def open(self, path: str) -> PDFDocumentHandle: ...
    def get_page_count(self, doc: PDFDocumentHandle) -> int: ...
    def get_page_text(self, doc: PDFDocumentHandle, index: int) -> str: ...
    def find_tables(self, page: object) -> bool: ...
    def render_page_image(self, doc: PDFDocumentHandle, index: int, dpi: int = 100) -> bytes: ...
    def close(self, doc: PDFDocumentHandle) -> None: ...


@runtime_checkable
class PDFStoragePort(Protocol):
    async def download_to_temp(self, key: str) -> Path: ...
