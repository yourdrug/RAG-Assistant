"""PaddleOCR adapter — infrastructure."""

from __future__ import annotations

import functools
import logging

import fitz
import numpy as np
from config import settings
from PIL import Image

log = logging.getLogger("default")


@functools.lru_cache(maxsize=1)
def _get_paddle_ocr():
    from paddleocr import PaddleOCR

    log.info("Loading PaddleOCR (lang=%s) ...", settings.ocr_lang_paddle)
    return PaddleOCR(
        use_angle_cls=True,
        lang=settings.ocr_lang_paddle,
        show_log=False,
    )


def ocr_image_paddle(image) -> str:
    ocr = _get_paddle_ocr()
    result = ocr.ocr(np.array(image), cls=True)
    lines = []
    for block in result or []:
        for entry in block or []:
            text = entry[1][0]
            if text and text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def ocr_pdf_pages(doc, page_nums: list[int], filename: str) -> dict:
    results = {}
    zoom = settings.ocr_dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page_num in page_nums:
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        text = ""
        if settings.ocr_engine in ("paddleocr", "auto"):
            text = ocr_image_paddle(img)

        if not text and settings.ocr_engine in ("surya", "auto"):
            try:
                from infrastructure.ocr.surya_ocr import ocr_image_surya

                text = ocr_image_surya(img)
            except ImportError:
                pass

        results[page_num] = text

    return results
