"""OCR engine settings."""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.ownership import Static


class OCRSection(BaseModel):
    # Deployment-only endpoint: never supplied by an uploaded document or dynamic settings.
    ocr_vl_url: Static[str] = "http://ocr-vl:8080"
    ocr_vl_timeout: Static[float] = Field(default=180, gt=0)
    ocr_enabled: bool = True
    ocr_dpi: int = 300
    ocr_min_chars: int = 40
