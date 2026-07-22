"""Benchmark endpoint — run RAG benchmark via API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from infrastructure.auth.fastapi_dependencies import require_admin

from presentation.api.dependencies import create_benchmark_service
from presentation.api.schemas import BenchmarkRequest, BenchmarkResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["benchmark"])


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    req: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    admin: dict = Depends(require_admin),
):
    service = create_benchmark_service()

    def _run():
        try:
            service.execute(
                questions_path=req.questions_path,
                out_dir=req.out_dir,
                top_k=req.top_k,
                judge_model=req.judge_model,
            )
        except Exception as e:
            logger.error("Benchmark failed: %s", e)

    background_tasks.add_task(_run)
    return BenchmarkResponse(status="started")
