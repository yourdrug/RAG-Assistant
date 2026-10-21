"""Local document OCR client; no cloud fallback or document-controlled URLs."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from config import settings
from domain.shared.exceptions import AppServerError


if TYPE_CHECKING:
    from PIL import Image


class OCRServiceError(AppServerError):
    """The selected OCR engine could not produce a usable page."""

    def __init__(self) -> None:
        super().__init__("Локальный OCR недоступен или вернул некорректный результат")


class OCRBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(pattern="^(text|table)$")
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class OCRPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks: list[OCRBlock] = Field(min_length=1)


def recognize_page(image: Image.Image) -> OCRPage:
    """Send only rendered pixels to the deployment's local service."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    try:
        with httpx.Client(timeout=settings.ocr_vl_timeout, trust_env=False) as client:
            response = client.post(
                f"{settings.ocr_vl_url.rstrip('/')}/recognize",
                content=buffer.getvalue(),
                headers={"Content-Type": "image/png"},
            )
            response.raise_for_status()
            return OCRPage.model_validate(response.json())
    except (httpx.HTTPError, ValidationError, ValueError):
        # Do not include response bodies, URLs or document text in processing errors.
        raise OCRServiceError from None
