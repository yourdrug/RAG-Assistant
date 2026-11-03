"""Construct PaddleOCR after the ingestion runner loads this module lazily."""

from paddleocr import PaddleOCR


def create_paddle_ocr(language: str) -> PaddleOCR:
    return PaddleOCR(use_angle_cls=True, lang=language, show_log=False)
