"""Factory for selecting the appropriate preview strategy by file extension."""

from __future__ import annotations


from application.documents import DocumentPreviewStrategy
from infrastructure.ml.ingestion.registry import FORMAT_HANDLERS, FormatContext


class PreviewStrategyFactory:
    @staticmethod
    def for_extension(
        extension: str,
        *,
        diag_service: object | None = None,
        domain_registry=None,
        domain_settings=None,
    ) -> DocumentPreviewStrategy:
        ext = extension.lower()
        if ext == ".doc":
            raise ValueError(
                "Legacy .doc format (Word 97-2003) is not supported for preview. "
                "Convert the file to .docx and try again."
            )
        handler = FORMAT_HANDLERS.get(ext)
        if handler is None or handler.preview is None:
            raise ValueError(f"Unsupported extension for dry-run preview: {ext}")
        return handler.preview(FormatContext(domain_registry, domain_settings, diag_service))

    @staticmethod
    def supported_extensions() -> set[str]:
        return {ext for ext, handler in FORMAT_HANDLERS.items() if handler.preview is not None}
