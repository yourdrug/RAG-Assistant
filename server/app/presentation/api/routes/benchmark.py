"""API endpoint for running the RAG quality benchmark as a background job."""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from fastapi import APIRouter, Depends
from infrastructure.worker.queue import enqueue_benchmark

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import get_uow_factory
from presentation.api.routes.common import create_background_job
from presentation.api.schemas import BenchmarkRequest, BenchmarkResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["benchmark"])


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    req: BenchmarkRequest,
    admin: dict = Depends(require_admin),
):
    job_id = await create_background_job(get_uow_factory(), "benchmark")

    q_path = req.questions_path or str(Path(settings.data_dir) / "test_questions.json")
    o_dir = req.out_dir or str(Path(settings.data_dir) / "benchmark_results")
    k = req.top_k or settings.retriever_top_k
    judge = req.judge_model or settings.llm_model

    await enqueue_benchmark(
        questions_path=q_path,
        out_dir=o_dir,
        top_k=k,
        judge_model=judge,
        job_id=job_id,
    )
    return BenchmarkResponse(status="started")
