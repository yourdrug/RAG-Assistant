"""Isolated local PaddleOCR-VL-1.6 service. Receives pixels, never URLs."""
from __future__ import annotations
from numpy.typing import NDArray
from typing import Any, Protocol
from collections.abc import AsyncIterator, Iterable

from contextlib import asynccontextmanager
import io
import logging
import threading

from fastapi import FastAPI, HTTPException, Request
import numpy as np
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from tables import UNREADABLE, table_markdown


class OCRPrediction(Protocol):
    @property
    def json(self) -> dict[str, Any]: ...
    def get(self, key: str, default: dict[str, Any]) -> dict[str, Any]: ...


class OCRPipeline(Protocol):
    def predict(self, image: NDArray[np.uint8], **options: bool) -> Iterable[OCRPrediction]: ...


log = logging.getLogger("ocr_vl")
MAX_BYTES = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000
inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Heavy ML dependency belongs only to this service, not the RAG server.
    from paddleocr import PaddleOCRVL

    app.state.pipeline = await run_in_threadpool(
        PaddleOCRVL,
        pipeline_version="v1.6",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        layout_shape_mode="rect",
    )
    yield
    del app.state.pipeline


app = FastAPI(lifespan=lifespan)


def predict_blocks(pipeline: OCRPipeline, image: NDArray[np.uint8], *, preprocess: bool=True) -> tuple[list[dict[str, Any]], OCRPrediction]:
    results = list(
        pipeline.predict(
            image,
            use_doc_orientation_classify=preprocess,
            use_doc_unwarping=preprocess,
            format_block_content=True,
        )
    )
    if len(results) != 1:
        raise ValueError("Expected one page")
    result = results[0]
    payload = result.json["res"]
    blocks: list[dict[str, Any]] = payload["parsing_res_list"]
    return blocks, result


def recognize(pipeline: OCRPipeline, image: NDArray[np.uint8]) -> dict[str, list[dict[str, str]]]:
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(503, "OCR занят; повторите обработку позже")
    try:
        blocks, result = predict_blocks(pipeline, image)
        output = []
        for block in blocks:
            content = block["block_content"].strip()
            kind = {
                "table": "table", "doc_title": "title", "title": "title",
                "paragraph_title": "section-header", "section-header": "section-header",
                "section_header": "section-header",
            }.get(block["block_label"], "text")
            if kind == "table":
                content = table_markdown(content)
                # The model has no calibrated handwriting confidence. Retry only
                # visibly incomplete tables; never infer values from totals.
                if not content or UNREADABLE in content:
                    corrected = result.get("doc_preprocessor_res", {}).get("output_img", image)
                    x0, y0, x1, y1 = [int(value) for value in block["block_bbox"]]
                    height, width = corrected.shape[:2]
                    crop = corrected[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
                    if crop.size:
                        retry, _ = predict_blocks(pipeline, crop, preprocess=False)
                        tables = [item for item in retry if item["block_label"] == "table"]
                        if len(tables) == 1:
                            candidate = table_markdown(tables[0]["block_content"])
                            if candidate and (
                                not content or candidate.count(UNREADABLE) < content.count(UNREADABLE)
                            ):
                                content = candidate
                if not content:
                    # Keep the failure visible in preview and searchable output.
                    content = f"| Значение |\n| --- |\n| {UNREADABLE} |"
            if content:
                output.append({"kind": kind, "text": content.replace("�", UNREADABLE)})
        if not output:
            raise ValueError("Empty OCR result")
        return {"blocks": output}
    finally:
        inference_lock.release()


def recognize_bytes(pipeline: OCRPipeline, body: bytes) -> dict[str, list[dict[str, str]]]:
    try:
        with Image.open(io.BytesIO(body)) as source:
            if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Слишком большое изображение")
            # Paddle pipelines accept BGR; preserve all original color channels.
            image = np.array(source.convert("RGB"))[:, :, ::-1].copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Некорректное изображение") from exc
    return recognize(pipeline, image)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recognize")
async def recognize_request(request: Request) -> dict[str, list[dict[str, str]]]:
    if request.headers.get("content-type") != "image/png":
        raise HTTPException(415, "Ожидается PNG")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BYTES:
            raise HTTPException(413, "Слишком большое изображение")
    try:
        return await run_in_threadpool(recognize_bytes, request.app.state.pipeline, bytes(body))
    except HTTPException:
        raise
    except Exception as exc:
        # Model exceptions can include private document content: log only type.
        log.error("Local OCR failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Ошибка локального OCR") from exc
