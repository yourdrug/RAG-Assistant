"""API endpoints for running and viewing RAG quality benchmarks."""

from __future__ import annotations

from pathlib import Path

from application.services.benchmark_result_service import BenchmarkResultService
from config import settings
from fastapi import APIRouter, Depends, HTTPException
from infrastructure.uow_factory import UnitOfWorkFactory
from infrastructure.worker.queue import enqueue_benchmark

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_benchmark_result_service, get_uow_factory
from presentation.api.routes.common import create_background_job
from presentation.api.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResultDetail,
    BenchmarkResultsListResponse,
    BenchmarkResultSummary,
)

router = APIRouter(tags=["benchmark"])


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    req: BenchmarkRequest,
    admin: dict = Depends(require_admin),
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
):
    job_id = await create_background_job(uow_factory, "benchmark")

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
    result = service.list_results()
    return BenchmarkResultsListResponse(
        results=[
            BenchmarkResultSummary(
                filename=s.filename,
                model=s.model,
                total_questions=s.total_questions,
                total_time_sec=s.total_time_sec,
                hit_rate=s.hit_rate,
                avg_mrr=s.avg_mrr,
                avg_faithfulness=s.avg_faithfulness,
                avg_relevancy=s.avg_relevancy,
                avg_correctness=s.avg_correctness,
                avg_similarity=s.avg_similarity,
            )
            for s in result.results
        ],
        total=result.total,
    )


@router.get("/benchmark/results/{filename}", response_model=BenchmarkResultDetail)
async def get_benchmark_result(
    filename: str,
    admin: dict = Depends(require_admin),
    service: BenchmarkResultService = Depends(create_benchmark_result_service),
):
    detail = service.get_result(filename)
    if detail is None:
        raise HTTPException(status_code=404, detail="Benchmark result not found")

    return BenchmarkResultDetail(
        filename=filename,
        summary=BenchmarkResultSummary(
            filename=detail.summary.filename,
            model=detail.summary.model,
            total_questions=detail.summary.total_questions,
            total_time_sec=detail.summary.total_time_sec,
            hit_rate=detail.summary.hit_rate,
            avg_mrr=detail.summary.avg_mrr,
            avg_faithfulness=detail.summary.avg_faithfulness,
            avg_relevancy=detail.summary.avg_relevancy,
            avg_correctness=detail.summary.avg_correctness,
            avg_similarity=detail.summary.avg_similarity,
        ),
        results=detail.results,
    )
