"""Application service for PDF quality diagnostics and dry-run preview."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from domain.value_objects.page_content_type import PageContentType

logger = logging.getLogger("default")


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
class FileStoragePort(Protocol):
    def download_to_temp(self, key: str) -> Path: ...


@dataclass(frozen=True)
class PageDiagnostic:
    page: int
    type: str
    chars: int = 0
    description: str = ""


@dataclass(frozen=True)
class DocumentDiagnoseResult:
    document_id: int
    filename: str
    total_pages: int
    pages: list[PageDiagnostic] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DryRunPageResult:
    page: int
    type: str
    content_type: str = PageContentType.TEXT
    chars: int = 0
    preview: str = ""


@dataclass(frozen=True)
class DryRunResult:
    filename: str
    total_pages: int = 0
    pages: list[DryRunPageResult] = field(default_factory=list)
    total_chars: int = 0
    quality_score: float = 0.0
    warning: str | None = None
    full_text_preview: str = ""
    summary: dict[str, int] = field(default_factory=dict)


class PDFDiagnosticService:
    def __init__(
        self,
        classifier: PDFPageClassifierPort,
        text_cleaner: PDFTextCleanerPort,
        ocr: PDFOcrPort,
        pdf_doc: PDFDocumentPort,
        storage: FileStoragePort,
        *,
        max_dry_run_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self._classifier = classifier
        self._cleaner = text_cleaner
        self._ocr = ocr
        self._pdf = pdf_doc
        self._storage = storage
        self._max_bytes = max_dry_run_bytes

    async def diagnose_document(self, document_id: int, source_path: str) -> DocumentDiagnoseResult | None:
        temp_path = self._storage.download_to_temp(source_path)
        try:
            doc = self._pdf.open(str(temp_path))
            total_pages = self._pdf.get_page_count(doc)
            page_diagnostics = []

            for i in range(total_pages):
                text = self._pdf.get_page_text(doc, i)
                chars = len(text.strip())
                ptype, desc = self._classifier.classify_page(text, chars)
                page_diagnostics.append(PageDiagnostic(page=i + 1, type=ptype, chars=chars, description=desc))

            self._pdf.close(doc)

            types = [p.type for p in page_diagnostics]
            summary = {
                PageContentType.TEXT: types.count(PageContentType.TEXT),
                PageContentType.SCAN: types.count(PageContentType.SCAN),
                PageContentType.GARBLED: types.count(PageContentType.GARBLED),
                PageContentType.EMPTY: types.count(PageContentType.EMPTY),
                PageContentType.TABLE: types.count(PageContentType.TABLE),
            }

            return DocumentDiagnoseResult(
                document_id=document_id,
                filename=source_path,
                total_pages=total_pages,
                pages=page_diagnostics,
                summary=summary,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def analyze_text_layer(self, pdf_path: Path) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        doc = self._pdf.open(str(pdf_path))
        page_results: list[DryRunPageResult] = []
        types_count: dict[str, int] = {PageContentType.TEXT: 0, PageContentType.SCAN: 0, PageContentType.GARBLED: 0, PageContentType.EMPTY: 0, PageContentType.TABLE: 0}
        total_chars = 0

        for i in range(self._pdf.get_page_count(doc)):
            page_num = i + 1
            page = doc.load_page(i) if hasattr(doc, "load_page") else None
            raw_text = self._pdf.get_page_text(doc, i)
            chars = len(raw_text.strip())

            has_table = False
            if page is not None:
                has_table = self._pdf.find_tables(page)

            if has_table:
                ptype = PageContentType.TABLE
            else:
                ptype, _desc = self._classifier.classify_page(raw_text, chars, page=page)

            types_count[ptype] = types_count.get(ptype, 0) + 1

            cleaned = self._cleaner.clean(raw_text) if raw_text else ""
            total_chars += len(cleaned)

            page_results.append(
                DryRunPageResult(
                    page=page_num,
                    type=ptype,
                    content_type=PageContentType.TABLE if has_table else PageContentType.TEXT,
                    chars=chars,
                    preview=cleaned[:200] if cleaned else raw_text[:200],
                )
            )

        self._pdf.close(doc)
        return page_results, types_count, total_chars

    def ocr_problem_pages(
        self, pdf_path: Path, page_results: list[DryRunPageResult]
    ) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        problem_pages = [p.page for p in page_results if p.type in (PageContentType.SCAN, PageContentType.EMPTY)]
        if not problem_pages:
            return page_results, {}, 0

        doc = self._pdf.open(str(pdf_path))
        ocr_results = self._ocr.ocr_pages(doc, problem_pages, pdf_path.name)
        self._pdf.close(doc)

        new_types: dict[str, int] = {PageContentType.TEXT: 0, PageContentType.SCAN: 0, PageContentType.GARBLED: 0, PageContentType.EMPTY: 0, PageContentType.TABLE: 0}
        total_chars = 0

        for pr in page_results:
            if pr.page in ocr_results:
                ocr_text = ocr_results[pr.page]
                if ocr_text:
                    ocr_text = self._cleaner.clean(ocr_text)
                    if ocr_text:
                        pr = DryRunPageResult(
                            page=pr.page,
                            type=PageContentType.TEXT,
                            content_type=PageContentType.OCR,
                            chars=len(ocr_text),
                            preview=ocr_text[:200],
                        )
                    else:
                        pr = DryRunPageResult(
                            page=pr.page, type=PageContentType.SCAN, content_type=PageContentType.OCR, chars=pr.chars, preview=pr.preview
                        )
                else:
                    pr = DryRunPageResult(
                        page=pr.page, type=PageContentType.SCAN, content_type=PageContentType.OCR, chars=pr.chars, preview=pr.preview
                    )

            new_types[pr.type] = new_types.get(pr.type, 0) + 1
            total_chars += pr.chars

        return page_results, new_types, total_chars
