"""Factory for selecting the appropriate preview strategy by file extension."""

from __future__ import annotations


from application.ports.document_preview import DocumentPreviewStrategy
from infrastructure.ml.preview.docx_preview_strategy import DocxPreviewStrategy
from infrastructure.ml.preview.pdf_preview_strategy import PdfPreviewStrategy
from infrastructure.ml.preview.rtf_preview_strategy import RtfPreviewStrategy

_EXTENSION_MAP: dict[str, type[DocumentPreviewStrategy]] = {
    ".pdf": PdfPreviewStrategy,
    ".docx": DocxPreviewStrategy,
    ".rtf": RtfPreviewStrategy,
}


class PreviewStrategyFactory:
    @staticmethod
    def for_extension(
        extension: str,
        *,
        diag_service: object | None = None,
    ) -> DocumentPreviewStrategy:
        ext = extension.lower()
        if ext == ".doc":
            raise ValueError(
                "Файлы .doc (старый формат Word 97-2003) не поддерживаются в preview. "
                "Конвертируйте файл в .docx и повторите попытку."
            )
        cls = _EXTENSION_MAP.get(ext)
        if cls is None:
            raise ValueError(f"Unsupported extension for dry-run preview: {ext}")
        if cls is PdfPreviewStrategy:
            return PdfPreviewStrategy(diag_service)  # type: ignore[arg-type]
        if cls is DocxPreviewStrategy:
            return DocxPreviewStrategy()
        if cls is RtfPreviewStrategy:
            return RtfPreviewStrategy()
        raise ValueError(f"No strategy for {ext}")

    @staticmethod
    def supported_extensions() -> set[str]:
        return set(_EXTENSION_MAP.keys())
