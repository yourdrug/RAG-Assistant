"""HTTP client for the deployment's PaddleOCR-VL service."""

from __future__ import annotations

import io
import threading
from typing import TYPE_CHECKING, Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from config import JobsSection, OCRSection, process_section
from domain.shared.exceptions import AppServerError


if TYPE_CHECKING:
    from PIL import Image


class OCRClientPool:
    """Share a bounded connection pool between ingestion threads in this process."""

    def __init__(self) -> None:
        self.client: httpx.Client | None = None
        self.lock = threading.Lock()

    def get_client(self) -> httpx.Client:
        with self.lock:
            if self.client is None:
                # Pages of one job are recognized sequentially, so job slots bound concurrent requests.
                jobs = process_section(JobsSection)
                connections = max(jobs.document_max_concurrent, jobs.worker_max_concurrent)
                self.client = httpx.Client(
                    timeout=process_section(OCRSection).ocr_vl_timeout,
                    limits=httpx.Limits(max_connections=connections, max_keepalive_connections=connections),
                    trust_env=False,
                )

            return self.client

    def close(self) -> None:
        """Release pooled connections after processing tasks have stopped."""
        with self.lock:
            if self.client is not None:
                self.client.close()
                self.client = None


ocr_client_pool = OCRClientPool()


class OCRServiceError(AppServerError):
    """PaddleOCR-VL could not produce a usable page."""

    def __init__(self) -> None:
        super().__init__("Сервис OCR недоступен или вернул некорректный результат")


class OCRBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(pattern="^(text|table|title|section-header|section_header)$")
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class OCRPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks: list[OCRBlock] = Field(min_length=1)


def recognize_page(image: Image.Image) -> OCRPage:
    """Send only rendered pixels to the deployment's local service."""
    ocr_settings = process_section(OCRSection)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    try:
        response = ocr_client_pool.get_client().post(
            f"{ocr_settings.ocr_vl_url.rstrip('/')}/recognize",
            content=buffer.getvalue(),
            headers={"Content-Type": "image/png"},
        )
        response.raise_for_status()
        return OCRPage.model_validate(response.json())
    except (httpx.HTTPError, ValidationError, ValueError):
        # Do not include response bodies, URLs or document text in processing errors.
        raise OCRServiceError from None
