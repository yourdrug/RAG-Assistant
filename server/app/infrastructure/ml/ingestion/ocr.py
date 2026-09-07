"""OCR functions — lazy-loaded PaddleOCR and Surya runners."""

from __future__ import annotations

import functools
import logging

import numpy as np
from PIL import Image

from config import settings

log = logging.getLogger("detailed")


@functools.lru_cache(maxsize=1)
def get_paddle_ocr():
    from paddleocr import PaddleOCR

    log.info("Loading PaddleOCR (lang=%s) ...", settings.ocr_lang_paddle)
    return PaddleOCR(
        use_angle_cls=True,
        lang=settings.ocr_lang_paddle,
        show_log=False,
    )


@functools.lru_cache(maxsize=1)
def get_surya_predictors():
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor

    log.info("Loading Surya OCR ...")
    return (RecognitionPredictor(), DetectionPredictor())


def ocr_image_paddle(image) -> str:
    ocr = get_paddle_ocr()
    result = ocr.ocr(np.array(image), cls=True)
    lines = []
    for block in result or []:
        for entry in block or []:
            text = entry[1][0]
            if text and text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def ocr_image_surya(image) -> str:
    rec_predictor, det_predictor = get_surya_predictors()
    predictions = rec_predictor([image], [settings.ocr_lang_surya], det_predictor)
    lines = [line.text.strip() for line in predictions[0].text_lines if line.text.strip()]
    return "\n".join(lines)


def ocr_pdf_pages(doc, page_nums: list[int], filename: str) -> dict:
    import fitz

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
            text = ocr_image_surya(img)

        results[page_num] = text

    return results
