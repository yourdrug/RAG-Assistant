"""API endpoints for running and viewing RAG quality benchmarks."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from config import settings
from fastapi import APIRouter, Depends, HTTPException
from infrastructure.worker.queue import enqueue_benchmark

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import get_uow_factory
from presentation.api.routes.common import create_background_job
from presentation.api.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResultDetail,
    BenchmarkResultsListResponse,
    BenchmarkResultSummary,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["benchmark"])

_RESULTS_GLOB = re.compile(r"^benchmark_\d{8}_\d{6}.*\.json$")


def _parse_result_filename(name: str) -> tuple[str, str]:
    """Extract timestamp and model from filename like benchmark_20260815_150138_qwen2.5_7b.json."""
    stem = Path(name).stem
    # benchmark_YYYYMMDD_HHMMSS_modeltag
    m = re.match(r"benchmark_(\d{8}_\d{6})_(.+)", stem)
    if m:
        return m.group(1), m.group(2)
    return "", stem


def _load_summary(filepath: Path) -> BenchmarkResultSummary | None:
    """Load a benchmark JSON and extract summary metrics."""
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return None

        ts, model = _parse_result_filename(filepath.name)

        faiths = [r["generator_metrics"]["faithfulness"] for r in data]
        rels = [r["generator_metrics"]["relevancy"] for r in data]
        corrs = [
            r["generator_metrics"]["correctness"]
            for r in data
            if r["generator_metrics"].get("correctness") is not None
        ]
        hit_rates = [
            r["retriever_metrics"]["hit_rate"]
            for r in data
            if r["retriever_metrics"].get("hit_rate") is not None
        ]
        mrrs = [
            r["retriever_metrics"]["mrr"]
            for r in data
            if r["retriever_metrics"].get("mrr") is not None
        ]
        sims = [r["retriever_metrics"]["avg_similarity"] for r in data]

        return BenchmarkResultSummary(
            filename=filepath.name,
            model=model or None,
            total_questions=len(data),
            total_time_sec=round(sum(r.get("latency_sec", 0) for r in data), 1),
            hit_rate=round(sum(hit_rates) / len(hit_rates), 3) if hit_rates else None,
            avg_mrr=round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
            avg_faithfulness=round(sum(faiths) / len(faiths), 1) if faiths else None,
            avg_relevancy=round(sum(rels) / len(rels), 1) if rels else None,
            avg_correctness=round(sum(corrs) / len(corrs), 1) if corrs else None,
            avg_similarity=round(sum(sims) / len(sims), 3) if sims else None,
        )
    except Exception as exc:
        logger.warning("Failed to parse benchmark result %s: %s", filepath.name, exc)
        return None


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


@router.get("/benchmark/results", response_model=BenchmarkResultsListResponse)
async def list_benchmark_results(admin: dict = Depends(require_admin)):
    results_dir = Path(settings.data_dir) / "benchmark_results"
    if not results_dir.exists():
        return BenchmarkResultsListResponse(results=[], total=0)

    files = sorted(
        [f for f in results_dir.iterdir() if f.suffix == ".json" and _RESULTS_GLOB.match(f.name)],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    summaries = []
    for f in files:
        s = _load_summary(f)
        if s is not None:
            summaries.append(s)

    return BenchmarkResultsListResponse(results=summaries, total=len(summaries))


@router.get("/benchmark/results/{filename}", response_model=BenchmarkResultDetail)
async def get_benchmark_result(filename: str, admin: dict = Depends(require_admin)):
    results_dir = Path(settings.data_dir) / "benchmark_results"
    filepath = results_dir / filename

    if not filepath.exists() or not filepath.suffix == ".json":
        raise HTTPException(status_code=404, detail="Benchmark result not found")

    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {exc}")

    summary = _load_summary(filepath)
    if summary is None:
        raise HTTPException(status_code=500, detail="Failed to parse benchmark result")

    return BenchmarkResultDetail(
        filename=filename,
        summary=summary,
        results=data,
    )
