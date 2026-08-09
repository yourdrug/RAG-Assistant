"""API endpoint for running the RAG quality benchmark as a background job."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from config import settings
from fastapi import APIRouter, BackgroundTasks, Depends

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_benchmark_service, get_uow_factory
from presentation.api.routes.common import create_background_job
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

    job_id = await create_background_job(get_uow_factory(), "benchmark")

    q_path = req.questions_path or str(Path(settings.data_dir) / "test_questions.json")
    o_dir = req.out_dir or str(Path(settings.data_dir) / "benchmark_results")
    k = req.top_k or settings.retriever_top_k
    judge = req.judge_model or settings.llm_model

    def _run():
        uow_factory = get_uow_factory()

        async def _update(status: str, error: str | None = None):
            async with uow_factory.create(master=True) as uow:
                if status == "running":
                    await uow.background_jobs.mark_running(job_id)
                elif status == "done":
                    await uow.background_jobs.mark_done(job_id)
                elif status == "failed":
                    await uow.background_jobs.mark_failed(job_id, error or "unknown")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_update("running"))
            service.run(
                questions_path=q_path,
                out_dir=o_dir,
                top_k=k,
                judge_model=judge,
            )
            loop.run_until_complete(_update("done"))
        except Exception as e:
            logger.error("Benchmark failed: %s", e)
            loop.run_until_complete(_update("failed", str(e)[:500]))
        finally:
            loop.close()

    background_tasks.add_task(_run)
    return BenchmarkResponse(status="started")
