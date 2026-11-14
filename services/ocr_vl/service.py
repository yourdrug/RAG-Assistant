"""Isolated local PaddleOCR-VL-1.6 service. Receives pixels, never URLs."""
from __future__ import annotations
from numpy.typing import NDArray
from typing import Any, Protocol
from collections.abc import AsyncIterator, Iterable

from contextlib import asynccontextmanager
import io
import logging
import os
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
    def predict(self, image: NDArray[np.uint8], **options: object) -> Iterable[OCRPrediction]: ...


log = logging.getLogger("ocr_vl")
MAX_BYTES = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000
inference_lock = threading.Lock()
# Per-block pixel cap for the VL model; lower it on small GPUs (e.g. 501760 for 6 GB). Empty: library default.
VL_MAX_PIXELS = int(os.environ["OCR_VL_MAX_PIXELS"]) if os.environ.get("OCR_VL_MAX_PIXELS") else None


def enable_fp16_fallback() -> None:
    """Load the VL model in fp16 on GPUs without bf16 (Turing/Volta) instead of fp32.

    PaddleX picks bf16 when supported and fp32 otherwise, and always keeps the vision encoder in
    fp32; together with loading-time copies this does not fit a 6 GB card. The whole model is
    built in fp16 and the checkpoint is read and cast on CPU, so the GPU only ever holds fp16
    weights. Opt-in, CUDA only: fp32 stays more accurate on large non-bf16 cards.
    """
    # Heavy ML dependencies, imported only inside the isolated service at startup.
    import paddle  # noqa: PLC0415
    from paddlex.inference.models.common.transformers.transformers import model_utils  # noqa: PLC0415
    from paddlex.inference.models.doc_vlm import predictor  # noqa: PLC0415
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl import _paddleocr_vl as modeling  # noqa: PLC0415

    # No public dtype or load-device hook in PaddleX 3.6: wrap the private builder and loader.
    original_build = predictor.DocVLMLocalPredictor._build  # noqa: SLF001
    original_load = model_utils._load_part_state_dict_from_safetensors  # noqa: SLF001

    def build_fp16(self: Any, **kwargs: Any) -> Any:
        if self.dtype == "float32":
            self.dtype = "float16"
            log.info("Loading PaddleOCR-VL in float16 (bf16 unavailable)")
        return original_build(self, **kwargs)

    def load_on_cpu(*args: Any, **kwargs: Any) -> Any:
        # "expected" place follows the current device, so the checkpoint lands in host memory.
        previous = paddle.device.get_device()
        paddle.device.set_device("cpu")
        try:
            return original_load(*args, **kwargs)
        finally:
            paddle.device.set_device(previous)

    # Upstream keeps the vision encoder in fp32 only to dodge a ROCm/MIOpen bf16 conv bug; on CUDA
    # an fp16 encoder halves its memory (~1.9 GB -> ~0.95 GB), which a 6 GB card needs.
    modeling.PaddleOCRVLForConditionalGeneration._keep_in_fp32_modules = None  # noqa: SLF001
    predictor.DocVLMLocalPredictor._build = build_fp16  # noqa: SLF001
    model_utils._load_part_state_dict_from_safetensors = load_on_cpu  # noqa: SLF001


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Heavy ML dependency belongs only to this service, not the RAG server.
    from paddleocr import PaddleOCRVL

    if os.environ.get("OCR_VL_FP16_FALLBACK", "").lower() in {"1", "true", "yes"}:
        enable_fp16_fallback()

    app.state.pipeline = await run_in_threadpool(
        PaddleOCRVL,
        pipeline_version="v1.6",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
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
            # paddleocr 3.6 accepts the layout shape mode per prediction, not in the constructor.
            layout_shape_mode="rect",
            max_pixels=VL_MAX_PIXELS,
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
