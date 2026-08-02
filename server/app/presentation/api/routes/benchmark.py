"""Benchmark endpoint — run RAG benchmark via API."""

from __future__ import annotations

import logging

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

    def _run():
        uow_factory = get_uow_factory()
        import asyncio

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
            service.execute(
                questions_path=req.questions_path,
                out_dir=req.out_dir,
                top_k=req.top_k,
                judge_model=req.judge_model,
            )
            loop.run_until_complete(_update("done"))
        except Exception as e:
            logger.error("Benchmark failed: %s", e)
            loop.run_until_complete(_update("failed", str(e)[:500]))
        finally:
            loop.close()

    background_tasks.add_task(_run)
    return BenchmarkResponse(status="started")
