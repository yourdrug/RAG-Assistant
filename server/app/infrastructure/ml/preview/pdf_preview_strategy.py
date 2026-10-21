"""PDF preview strategy — wraps existing PDFDiagnosticService methods."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from domain.knowledge.enums import PageContentType
from domain.ingestion.preview import PreviewUnitKind

from domain.ingestion.preview import DryRunPageResult
from config import settings
from infrastructure.ml.ingestion.pdf import parse_pdf

if TYPE_CHECKING:
    from langchain.schema import Document
    from application.documents.diagnostics import PDFDiagnosticService

logger = logging.getLogger("default")


class PdfPreviewStrategy:
    def __init__(self, diag_service: PDFDiagnosticService) -> None:
        self._svc = diag_service

    def supports(self, extension: str) -> bool:
        return extension == ".pdf"

    def analyze(self, path: Path) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        if settings.ocr_enabled and settings.ocr_engine == "paddleocr_vl":
            return self.analyze_vl(path)
        page_results, types_count, total_chars = self._svc.analyze_text_layer(path)
        enriched = [
            DryRunPageResult(
                page=p.page,
                type=p.type,
                content_type=p.content_type,
                chars=p.chars,
                preview=p.preview,
                full_text=p.full_text,
                problem_spans=p.problem_spans,
                previous_type=p.previous_type,
                unit_kind=PreviewUnitKind.PAGE,
                label=f"Стр. {p.page}",
            )
            for p in page_results
        ]
        return enriched, types_count, total_chars

    def analyze_vl(self, path: Path) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        documents = parse_pdf(path)
        grouped: dict[int, list[Document]] = {}
        for document in documents:
            grouped.setdefault(document.metadata["page"], []).append(document)
        # Preserve blank native pages as well as OCR pages in the preview.
        original, _, _ = self._svc.analyze_text_layer(path)
        results = []
        counts = PageContentType.empty_counts()
        for unit in original:
            blocks = grouped.get(unit.page, [])
            text = "\n\n".join(block.page_content for block in blocks)
            has_table = any(b.metadata.get("content_type") == PageContentType.TABLE for b in blocks)
            is_ocr = any(b.metadata.get("ocr_engine") == "paddleocr_vl" for b in blocks)
            kind = PageContentType.TABLE if has_table else PageContentType.TEXT if text else unit.type
            results.append(
                DryRunPageResult(
                    page=unit.page,
                    type=kind,
                    content_type=PageContentType.TABLE
                    if has_table
                    else PageContentType.OCR
                    if is_ocr
                    else unit.content_type,
                    chars=len(text),
                    preview=text[:200],
                    full_text=text,
                    previous_type=unit.type if is_ocr else None,
                    unit_kind=PreviewUnitKind.PAGE,
                    label=f"Стр. {unit.page}",
                )
            )
            counts[kind] = counts.get(kind, 0) + 1
        return results, counts, sum(unit.chars for unit in results)

    def ocr_problem_units(
        self,
        path: Path,
        units: list[DryRunPageResult],
        unit_ids: list[int],
    ) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        filtered = [u for u in units if u.page in unit_ids]
        result, ocr_types_count, _total_chars = self._svc.ocr_problem_pages(path, filtered)
        merged = []
        by_page = {u.page: u for u in result}
        for u in units:
            if u.page in by_page:
                updated = by_page[u.page]
                merged.append(
                    DryRunPageResult(
                        page=updated.page,
                        type=updated.type,
                        content_type=updated.content_type,
                        chars=updated.chars,
                        preview=updated.preview,
                        full_text=updated.full_text,
                        problem_spans=updated.problem_spans,
                        previous_type=updated.previous_type,
                        unit_kind=PreviewUnitKind.PAGE,
                        label=f"Стр. {updated.page}",
                    )
                )
            else:
                merged.append(u)

        # Recompute types_count from ALL merged units (not just OCR'd ones)
        types_count = PageContentType.empty_counts()
        total_chars = 0
        for u in merged:
            types_count[u.type] = types_count.get(u.type, 0) + 1
            total_chars += u.chars

        return merged, types_count, total_chars
