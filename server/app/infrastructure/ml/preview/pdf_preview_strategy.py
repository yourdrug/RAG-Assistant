"""PDF preview strategy — wraps existing PDFDiagnosticService methods."""

from __future__ import annotations

import logging
from pathlib import Path

from domain.value_objects.page_content_type import PreviewUnitKind

from application.services.pdf_diagnostic_service import DryRunPageResult, PDFDiagnosticService

logger = logging.getLogger("default")


class PdfPreviewStrategy:
    def __init__(self, diag_service: PDFDiagnosticService) -> None:
        self._svc = diag_service

    def supports(self, extension: str) -> bool:
        return extension == ".pdf"

    def analyze(self, path: Path) -> tuple[list[DryRunPageResult], dict[str, int], int]:
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
        from domain.value_objects.page_content_type import PageContentType

        types_count: dict[str, int] = {
            PageContentType.TEXT: 0,
            PageContentType.SCAN: 0,
            PageContentType.GARBLED: 0,
            PageContentType.EMPTY: 0,
            PageContentType.TABLE: 0,
        }
        total_chars = 0
        for u in merged:
            types_count[u.type] = types_count.get(u.type, 0) + 1
            total_chars += u.chars

        return merged, types_count, total_chars
