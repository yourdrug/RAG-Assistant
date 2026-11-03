"""Ingestion parsing, OCR and chunking exports."""

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
from infrastructure.ml.ingestion.rtf import parse_rtf
from infrastructure.ml.ingestion.splitting import (
    LEGAL_SEPARATORS,
    TABLE_BATCH_ROWS,
    extract_article_number,
    merge_pdf_pages,
    split_documents,
)
from infrastructure.ml.ingestion.txt import parse_txt
from infrastructure.ml.ingestion.utils import clean_pdf_text


__all__ = [
    "LEGAL_SEPARATORS",
    "TABLE_BATCH_ROWS",
    # Utils
    "clean_pdf_text",
    # DOCX
    "docx_table_to_markdown",
    "extract_article_number",
    "extract_date_from_filename",
    # OCR
    "get_paddle_ocr",
    "get_surya_predictors",
    # Splitting
    "merge_pdf_pages",
    "ocr_image_paddle",
    "ocr_image_surya",
    "ocr_pdf_pages",
    "parse_docx",
    "parse_docx_sections",
    # Markdown
    "parse_markdown",
    "parse_markdown_sections",
    "parse_pdf",
    # RTF
    "parse_rtf",
    # TXT
    "parse_txt",
    # PDF
    "pymupdf_table_to_markdown",
    "split_documents",
    "split_markdown_tables",
]
