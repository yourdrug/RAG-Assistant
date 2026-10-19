"""API endpoints for running and viewing RAG quality benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from application.benchmark import BenchmarkExportService
from fastapi.responses import StreamingResponse

from application.ports.rate_limit import RateLimitPolicyName
from application.benchmark import BenchmarkResultService
from application.jobs.service import JobService
from fastapi import APIRouter, Depends, HTTPException, Request

from presentation.api.auth_dependencies import require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.constants import JobType
from presentation.api.dependencies import (
    create_benchmark_config,
    create_benchmark_export_service,
    create_benchmark_result_service,
    create_idempotency_store,
    create_job_enqueuer,
    create_job_service,
    get_idempotency_key,
)
from presentation.api.idempotency import complete_idempotency, reserve_idempotency
from presentation.api.helpers import summary_to_response
from presentation.api.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResultDetail,
    BenchmarkResultsListResponse,
    CurrentUser,
)

router = APIRouter(tags=["benchmark"])


@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.BENCHMARK))],
)
async def run_benchmark(
    request: Request,
    req: BenchmarkRequest,
    admin: CurrentUser = Depends(require_admin),
    job_service: JobService = Depends(create_job_service),
    job_enqueuer=Depends(create_job_enqueuer),
    bench_cfg=Depends(create_benchmark_config),
    idempotency_key: str | None = Depends(get_idempotency_key),
    idempotency_store=Depends(create_idempotency_store),
):
    if req.out_dir is not None and req.out_dir != str(Path(bench_cfg.data_dir) / "benchmark_results"):
        raise HTTPException(
            status_code=422, detail="out_dir is no longer supported; use /benchmark/results/{run_id}/export"
        )
    k = req.top_k or bench_cfg.retriever_top_k
    judge = req.judge_model or bench_cfg.llm_model

    cached, reservation = await reserve_idempotency(
        idempotency_store,
        idempotency_key,
        f"{admin.kind}:{admin.role}:{admin.id}",
        f"{request.method}:{request.url.path}",
        req.model_dump(mode="json"),
    )
    if cached is not None:
        return BenchmarkResponse(**cached)

    job_id = await job_service.create_job(JobType.BENCHMARK)

    await job_enqueuer.enqueue_benchmark(
        dataset=req.dataset,
        top_k=k,
        judge_model=judge,
        job_id=job_id,
    )

    response = BenchmarkResponse(status="started")

    await complete_idempotency(idempotency_store, reservation, response.model_dump(mode="json"))

    return response


@router.get("/benchmark/results", response_model=BenchmarkResultsListResponse)
async def list_benchmark_results(
    admin: CurrentUser = Depends(require_admin),
    service: BenchmarkResultService = Depends(create_benchmark_result_service),
):
    result = await service.list_results()
    return BenchmarkResultsListResponse(
        results=[summary_to_response(s) for s in result.results],
        total=result.total,
    )


@router.get("/benchmark/results/{run_id}", response_model=BenchmarkResultDetail)
async def get_benchmark_result(
    run_id: int,
    admin: CurrentUser = Depends(require_admin),
    service: BenchmarkResultService = Depends(create_benchmark_result_service),
):
    detail = await service.get_result(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Benchmark result not found")

    return BenchmarkResultDetail(
        id=detail.id,
        summary=summary_to_response(detail.summary),
        per_question_results=detail.per_question_results,
    )


@router.get("/benchmark/results/{run_id}/export")
async def export_benchmark_result(
    run_id: int,
    format: Literal["csv", "json"] = "json",
    admin: CurrentUser = Depends(require_admin),
    service: BenchmarkExportService = Depends(create_benchmark_export_service),
):
    try:
        await service.header(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        service.export(run_id, format),
        media_type="text/csv" if format == "csv" else "application/json",
        headers={"Content-Disposition": f'attachment; filename="benchmark-{run_id}.{format}"'},
    )


@router.post("/benchmark/results/{run_id}/artifacts/{format}")
async def retry_benchmark_report(
    run_id: int,
    format: Literal["csv", "json"],
    admin: CurrentUser = Depends(require_admin),
    service: BenchmarkExportService = Depends(create_benchmark_export_service),
):
    try:
        ready = await service.save_report(run_id, format)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ready" if ready else "failed"}
