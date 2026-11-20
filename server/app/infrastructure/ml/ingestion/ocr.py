"""Render PDF pages for the PaddleOCR-VL HTTP service."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import fitz
from langchain.schema import Document
from PIL import Image

from config import OCRSection, process_section
from infrastructure.ml.ingestion.ocr_vl import recognize_page


if TYPE_CHECKING:
    from application.documents.ports.diagnostics import PDFDocumentHandle


def ocr_vl_pdf_pages(doc: PDFDocumentHandle, page_nums: list[int], filename: str) -> list[Document]:
    results: list[Document] = []

    dpi = process_section(OCRSection).ocr_dpi
    colorspace = fitz.Colorspace(fitz.CS_RGB)

    for page_num in page_nums:
        page = cast("fitz.Page", doc.load_page(page_num - 1))
        pix = page.get_pixmap(dpi=dpi, colorspace=colorspace, alpha=False)
        with Image.frombytes("RGB", (pix.width, pix.height), pix.samples) as image:
            result = recognize_page(image)

        image_count = len(page.get_images(full=True))
        results.extend(
            Document(
                page_content=block.text,
                metadata={
                    "page": page_num,
                    "source": filename,
                    "content_type": block.kind,
                    "ocr_engine": "paddleocr_vl",
                    "has_images": True,
                    "image_count": image_count,
                    "text_quality": 0.0,
                },
            )
            for block in result.blocks
        )

    return results


def ocr_pdf_pages(doc: PDFDocumentHandle, page_nums: list[int], filename: str) -> dict[int, str]:
    texts: dict[int, list[str]] = {page_num: [] for page_num in page_nums}

    for document in ocr_vl_pdf_pages(doc, page_nums, filename):
        texts[document.metadata["page"]].append(document.page_content)

    return {page_num: "\n\n".join(blocks) for page_num, blocks in texts.items()}
