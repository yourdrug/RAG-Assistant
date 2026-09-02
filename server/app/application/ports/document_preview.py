"""Protocol for document preview strategies (PDF, DOCX, RTF)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from application.services.pdf_diagnostic_service import DryRunPageResult


@runtime_checkable
class DocumentPreviewStrategy(Protocol):
    def supports(self, extension: str) -> bool: ...

    def analyze(self, path: Path) -> tuple[list[DryRunPageResult], dict[str, int], int]: ...

    def ocr_problem_units(
        self,
        path: Path,
        units: list[DryRunPageResult],
        unit_ids: list[int],
    ) -> tuple[list[DryRunPageResult], dict[str, int], int]: ...
