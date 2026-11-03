"""Factory for selecting the appropriate preview strategy by file extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

from infrastructure.ml.ingestion.registry import FORMAT_HANDLERS, FormatContext


if TYPE_CHECKING:
    from application.documents import DocumentPreviewStrategy
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry


class PreviewStrategyFactory:
    @staticmethod
    def for_extension(
        extension: str,
        *,
        diag_service: object | None = None,
        domain_registry: DomainProfileRegistry | None = None,
        domain_settings: DomainSettingsPort | None = None,
    ) -> DocumentPreviewStrategy:
        ext = extension.lower()

        if ext == ".doc":
            message = (
                "Legacy .doc format (Word 97-2003) is not supported for preview. "
                "Convert the file to .docx and try again."
            )
            raise ValueError(message)

        handler = FORMAT_HANDLERS.get(ext)

        if handler is None or handler.preview is None:
            message = f"Unsupported extension for dry-run preview: {ext}"
            raise ValueError(message)

        return handler.preview(FormatContext(domain_registry, domain_settings, diag_service))

    @staticmethod
    def supported_extensions() -> set[str]:
        return {ext for ext, handler in FORMAT_HANDLERS.items() if handler.preview is not None}
