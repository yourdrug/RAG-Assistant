"""API endpoints for running and viewing RAG quality benchmarks."""

from __future__ import annotations

from pathlib import Path

from application.services.benchmark_result_service import (
    BenchmarkResultService,
    BenchmarkResultSummary as ResultSummaryDTO,
)
from application.services.job_service import JobService
from config import settings
from fastapi import APIRouter, Depends, HTTPException
from infrastructure.worker.queue import enqueue_benchmark

from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import JobType
from presentation.api.dependencies import create_benchmark_result_service, create_job_service
from presentation.api.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResultDetail,
    BenchmarkResultsListResponse,
    BenchmarkResultSummary,
)

router = APIRouter(tags=["benchmark"])


def _summary_to_response(s: ResultSummaryDTO) -> BenchmarkResultSummary:
    return BenchmarkResultSummary(
        id=s.id,
        config_json=s.config_json,
        summary_metrics=s.summary_metrics,
        duration_sec=s.duration_sec,
        llm_evaluated=s.llm_evaluated,
        dataset=s.dataset,
        sweep_id=s.sweep_id,
        creation_date=s.creation_date,
    )


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    req: BenchmarkRequest,
    admin: dict = Depends(require_admin),
    job_service: JobService = Depends(create_job_service),
):
    job_id = await job_service.create_job(JobType.BENCHMARK)

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


@router.get("/benchmark/results", response_model=BenchmarkResultsListResponse)
async def list_benchmark_results(
    admin: dict = Depends(require_admin),
    service: BenchmarkResultService = Depends(create_benchmark_result_service),
):
    result = await service.list_results()
    return BenchmarkResultsListResponse(
        results=[_summary_to_response(s) for s in result.results],
        total=result.total,
    )


@router.get("/benchmark/results/{run_id}", response_model=BenchmarkResultDetail)
async def get_benchmark_result(
    run_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkResultService = Depends(create_benchmark_result_service),
):
    detail = await service.get_result(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Benchmark result not found")

    return BenchmarkResultDetail(
        id=detail.id,
        summary=_summary_to_response(detail.summary),
        per_question_results=detail.per_question_results,
    )
