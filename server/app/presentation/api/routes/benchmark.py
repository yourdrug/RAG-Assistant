"""API endpoints for running and viewing RAG quality benchmarks."""

from __future__ import annotations

from pathlib import Path

from application.ports.rate_limit import RateLimitPolicyName
from application.services.benchmark_result_service import BenchmarkResultService
from application.services.job_service import JobService
from fastapi import APIRouter, Depends, HTTPException

from presentation.api.auth_dependencies import require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.constants import JobType
from presentation.api.dependencies import (
    create_benchmark_config,
    create_benchmark_result_service,
    create_idempotency_store,
    create_job_enqueuer,
    create_job_service,
    get_idempotency_key,
)
from presentation.api.helpers import summary_to_response, validate_data_path_within_dir
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
    req: BenchmarkRequest,
    admin: CurrentUser = Depends(require_admin),
    job_service: JobService = Depends(create_job_service),
    job_enqueuer=Depends(create_job_enqueuer),
    bench_cfg=Depends(create_benchmark_config),
    idempotency_key: str | None = Depends(get_idempotency_key),
    idempotency_store=Depends(create_idempotency_store),
):
    # Idempotency: return cached result if key already executed
    if idempotency_key:
        cached = await idempotency_store.get(idempotency_key, admin.id)
        if cached:
            return BenchmarkResponse(**cached)

    job_id = await job_service.create_job(JobType.BENCHMARK)

    q_path = req.questions_path or str(Path(bench_cfg.data_dir) / "test_questions.json")
    o_dir = req.out_dir or str(Path(bench_cfg.data_dir) / "benchmark_results")
    q_path = validate_data_path_within_dir(q_path, bench_cfg.data_dir)
    o_dir = validate_data_path_within_dir(o_dir, bench_cfg.data_dir)
    k = req.top_k or bench_cfg.retriever_top_k
    judge = req.judge_model or bench_cfg.llm_model

    await job_enqueuer.enqueue_benchmark(
        questions_path=q_path,
        out_dir=o_dir,
        top_k=k,
        judge_model=judge,
        job_id=job_id,
    )

    response = BenchmarkResponse(status="started")

    # Idempotency: store result for future duplicate requests
    if idempotency_key:
        await idempotency_store.store(idempotency_key, admin.id, response.model_dump())

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
