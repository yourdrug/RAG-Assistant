"""Re-exports for backward compatibility: ``from infrastructure.ml.ingestion import X``."""

from config import settings  # noqa: F401 — re-exported for test patching

from infrastructure.ml.ingestion.docx import (
    docx_table_to_markdown,
    parse_docx,
    parse_docx_sections,
)
from infrastructure.ml.ingestion.markdown import (
    extract_date_from_filename,
    parse_markdown,
    parse_markdown_sections,
    split_markdown_tables,
)
from infrastructure.ml.ingestion.ocr import (
    get_paddle_ocr,
    get_surya_predictors,
    ocr_image_paddle,
    ocr_image_surya,
    ocr_pdf_pages,
)
from infrastructure.ml.ingestion.pdf import (
    parse_pdf,
    pymupdf_table_to_markdown,
)
from infrastructure.ml.ingestion.registry import PARSERS
from infrastructure.ml.ingestion.rtf import parse_rtf
from infrastructure.ml.ingestion.splitting import (
    GENERAL_SEPARATORS,
    LEGAL_SEPARATORS,
    TABLE_BATCH_ROWS,
    extract_article_number,
    merge_pdf_pages,
    split_documents,
    split_documents_legal,
)
from infrastructure.ml.ingestion.txt import parse_txt
from infrastructure.ml.ingestion.utils import clean_pdf_text

__all__ = [
    # OCR
    "get_paddle_ocr",
    "get_surya_predictors",
    "ocr_image_paddle",
    "ocr_image_surya",
    "ocr_pdf_pages",
    # PDF
    "pymupdf_table_to_markdown",
    "parse_pdf",
    # DOCX
    "docx_table_to_markdown",
    "parse_docx",
    "parse_docx_sections",
    # RTF
    "parse_rtf",
    # TXT
    "parse_txt",
    # Markdown
    "parse_markdown",
    "split_markdown_tables",
    "parse_markdown_sections",
    "extract_date_from_filename",
    # Splitting
    "merge_pdf_pages",
    "split_documents",
    "split_documents_legal",
    "extract_article_number",
    "GENERAL_SEPARATORS",
    "LEGAL_SEPARATORS",
    "TABLE_BATCH_ROWS",
    # Utils
    "clean_pdf_text",
    # Registry
    "PARSERS",
]
