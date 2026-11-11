"""OCR engine settings."""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.ownership import Static


class OCRSection(BaseModel):
    ocr_engine: str = "paddleocr"
    # Deployment-only endpoint: never supplied by an uploaded document or dynamic settings.
    ocr_vl_url: Static[str] = "http://ocr-vl:8080"
    ocr_vl_timeout: Static[float] = Field(default=180, gt=0)
    ocr_enabled: bool = True
    ocr_lang_paddle: str = "ru"
    ocr_lang_surya: list[str] = ["ru", "en"]
    ocr_dpi: int = 300
    ocr_min_chars: int = 40
    # Recognitions below this confidence are dropped as noise before indexing.
    ocr_min_confidence: float = Field(default=0.5, ge=0, le=1)
