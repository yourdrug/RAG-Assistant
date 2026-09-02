"""Admin Benchmark Lab API — questions CRUD, sweeps, runs, history/comparison."""

from __future__ import annotations

import asyncio
import json
import logging

from application.services.benchmark_services import (
    BenchmarkQuestionService,
    BenchmarkRunService,
    BenchmarkSweepService,
)
from application.services.config_service import ConfigService
from application.services.document_service import DocumentService
from application.services.job_service import JobService
from config import settings
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from infrastructure.worker.queue import enqueue_sweep

from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import (
    HISTORY_CONFIG_KEYS,
    JobType,
    RUN_CONFIG_KEY_MAP,
    SSE_HEADERS,
    SSE_MEDIA_TYPE,
)
from presentation.api.dependencies import (
    create_benchmark_question_service,
    create_benchmark_run_service,
    create_benchmark_sweep_service,
    create_config_service,
    create_document_service,
    create_job_service,
)
from presentation.api.schemas import (
    BenchmarkHistoryPoint,
    BenchmarkHistoryResponse,
    BenchmarkQuestionCreate,
    BenchmarkQuestionResponse,
    BenchmarkQuestionsImportRequest,
    BenchmarkQuestionsImportResponse,
    BenchmarkQuestionsListResponse,
    BenchmarkQuestionUpdate,
    BenchmarkRunResponse,
    BenchmarkRunsListResponse,
    RegressionCheckResponse,
    RegressionCheckResult,
    RunApplyFailed,
    RunApplyResponse,
    RunCompareResponse,
    SweepCreateRequest,
    SweepResponse,
    SweepsListResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["benchmark-admin"])


# ---------------------------------------------------------------------------
# Helpers — response mapping
# ---------------------------------------------------------------------------


def _question_to_response(q: object) -> BenchmarkQuestionResponse:
    return BenchmarkQuestionResponse(
        id=q.id,
        question=q.question,
        expected_answer=q.expected_answer,
        source_hint=q.source_hint,
        tags=q.tags,
        dataset=q.dataset,
        is_active=q.is_active,
        created_by=q.created_by,
        notes=q.notes,
        creation_date=q.creation_date,
    )


def _run_to_response(r: object) -> BenchmarkRunResponse:
    return BenchmarkRunResponse(
        id=r.id,
        sweep_id=r.sweep_id,
        config_json=r.config_json,
        summary_metrics=r.summary_metrics,
        duration_sec=r.duration_sec,
        llm_evaluated=r.llm_evaluated,
        dataset=r.dataset,
        filename=r.filename,
        creation_date=r.creation_date,
    )


def _sweep_to_response(s: object, *, job_id: int | None = None) -> SweepResponse:
    return SweepResponse(
        id=s.id,
        status=s.status,
        strategy=s.strategy,
        search_space=s.search_space,
        objective_weights=s.objective_weights,
        dataset=s.dataset,
        top_n_llm=s.top_n_llm,
        total_configs=s.total_configs,
        evaluated_configs=s.evaluated_configs,
        best_run_id=s.best_run_id,
        job_id=job_id if job_id is not None else s.job_id,
        creation_date=s.creation_date,
    )


# ---------------------------------------------------------------------------
# Questions CRUD
# ---------------------------------------------------------------------------


@router.get("/admin/benchmark/questions", response_model=BenchmarkQuestionsListResponse)
async def list_questions(
    dataset: str | None = None,
    tag: str | None = None,
    search: str | None = None,
    is_active: bool | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(create_benchmark_question_service),
):
    questions, total = await service.list(
        dataset=dataset, tag=tag, search=search, is_active=is_active, limit=limit, offset=offset
    )
    return BenchmarkQuestionsListResponse(
        questions=[_question_to_response(q) for q in questions],
        total=total,
    )


@router.post("/admin/benchmark/questions", response_model=BenchmarkQuestionResponse)
async def create_question(
    body: BenchmarkQuestionCreate,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(create_benchmark_question_service),
):
    created = await service.create(body, created_by=admin["id"])
    return _question_to_response(created)


@router.put("/admin/benchmark/questions/{question_id}", response_model=BenchmarkQuestionResponse)
async def update_question(
    question_id: int,
    body: BenchmarkQuestionUpdate,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(create_benchmark_question_service),
):
    fields = body.model_dump(exclude_unset=True)
    updated = await service.update(question_id, fields)
    return _question_to_response(updated)


@router.delete("/admin/benchmark/questions/{question_id}")
async def delete_question(
    question_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(create_benchmark_question_service),
):
    await service.delete(question_id)
    return {"deleted": True}


@router.post("/admin/benchmark/questions/import", response_model=BenchmarkQuestionsImportResponse)
async def import_questions(
    body: BenchmarkQuestionsImportRequest,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(create_benchmark_question_service),
):
    count = await service.bulk_create(body.questions, created_by=admin["id"])
    return BenchmarkQuestionsImportResponse(imported=count)


@router.get("/admin/benchmark/questions/export")
async def export_questions(
    dataset: str | None = None,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(create_benchmark_question_service),
):
    questions = await service.export(dataset=dataset)
    data = [
        {
            "question": q.question,
            "expected_answer": q.expected_answer,
            "source_hint": q.source_hint,
            "tags": q.tags,
            "dataset": q.dataset,
        }
        for q in questions
    ]
    return data


@router.get("/admin/benchmark/source-files")
async def list_source_files(
    search: str | None = None,
    admin: dict = Depends(require_admin),
    document_service: DocumentService = Depends(create_document_service),
):
    """Return distinct indexed document filenames for source_hint picker."""
    filenames = await document_service.list_source_files(search=search)
    return {"files": filenames}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


@router.post("/admin/benchmark/sweep", response_model=SweepResponse)
async def create_sweep(
    body: SweepCreateRequest,
    admin: dict = Depends(require_admin),
    service: BenchmarkSweepService = Depends(create_benchmark_sweep_service),
    job_service: JobService = Depends(create_job_service),
):
    sweep = await service.create(body)

    job_id = await job_service.create_job(JobType.SWEEP, related_id=sweep.id)

    await service.update_status(sweep.id, "pending")

    await enqueue_sweep(sweep_id=sweep.id, job_id=job_id)

    return _sweep_to_response(sweep, job_id=job_id)


@router.get("/admin/benchmark/sweep/{sweep_id}", response_model=SweepResponse)
async def get_sweep(
    sweep_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkSweepService = Depends(create_benchmark_sweep_service),
):
    sweep = await service.get(sweep_id)
    return _sweep_to_response(sweep)


@router.get("/admin/benchmark/sweeps", response_model=SweepsListResponse)
async def list_sweeps(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    admin: dict = Depends(require_admin),
    service: BenchmarkSweepService = Depends(create_benchmark_sweep_service),
):
    sweeps, total = await service.list(limit=limit, offset=offset)
    return SweepsListResponse(
        sweeps=[_sweep_to_response(s) for s in sweeps],
        total=total,
    )


@router.get("/admin/benchmark/sweep/{sweep_id}/stream")
async def sweep_progress_stream(
    sweep_id: int,
    admin: dict = Depends(require_admin),
):
    """SSE stream: live progress + new results as they complete."""

    async def event_generator():
        try:
            from arq import create_pool
            from arq.connections import RedisSettings

            redis_settings = RedisSettings.from_dsn(settings.redis_url)
            pool = await create_pool(redis_settings)
            try:
                pubsub = pool.pubsub()
                await pubsub.subscribe(f"sweep:{sweep_id}")

                while True:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=30,
                    )
                    if message and message["type"] == "message":
                        data = message["data"].decode("utf-8")
                        yield f"data: {data}\n\n"
                        parsed = json.loads(data)
                        if parsed.get("done"):
                            break
            finally:
                await pubsub.unsubscribe(f"sweep:{sweep_id}")
                await pool.close()
        except TimeoutError:
            yield ": heartbeat\n\n"
        except Exception as e:
            logger.warning("SSE stream error for sweep %d: %s", sweep_id, e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


@router.post("/admin/benchmark/sweep/{sweep_id}/cancel")
async def cancel_sweep(
    sweep_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkSweepService = Depends(create_benchmark_sweep_service),
):
    await service.cancel(sweep_id)
    return {"cancelled": True}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@router.get("/admin/benchmark/runs", response_model=BenchmarkRunsListResponse)
async def list_runs(
    sweep_id: int | None = None,
    dataset: str | None = None,
    sort_by: str = Query("creation_date", pattern="^(creation_date|duration_sec|id)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    admin: dict = Depends(require_admin),
    service: BenchmarkRunService = Depends(create_benchmark_run_service),
):
    runs, total = await service.list(
        sweep_id=sweep_id,
        dataset=dataset,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
    )
    return BenchmarkRunsListResponse(
        runs=[_run_to_response(r) for r in runs],
        total=total,
    )


@router.get("/admin/benchmark/runs/{run_id}", response_model=BenchmarkRunResponse)
async def get_run(
    run_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkRunService = Depends(create_benchmark_run_service),
):
    run = await service.get(run_id)
    return _run_to_response(run)


@router.post("/admin/benchmark/runs/{run_id}/apply", response_model=RunApplyResponse)
async def apply_run_config(
    run_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkRunService = Depends(create_benchmark_run_service),
    config_service: ConfigService = Depends(create_config_service),
):
    """Apply a run's config_json to the live system via ConfigService."""
    run = await service.get(run_id)

    config = run.config_json
    applied_keys = []
    failed_keys: list[RunApplyFailed] = []

    for config_key, param_key in RUN_CONFIG_KEY_MAP.items():
        if config_key in config:
            try:
                await config_service.update_parameter(
                    param_key, str(config[config_key]), changed_by=admin["id"]
                )
                applied_keys.append(param_key)
            except Exception as e:
                logger.warning("Failed to apply %s=%s: %s", param_key, config[config_key], e)
                failed_keys.append(RunApplyFailed(key=param_key, error=str(e)))

    return RunApplyResponse(applied=len(applied_keys), keys=applied_keys, failed=failed_keys)


@router.get("/admin/benchmark/runs/compare", response_model=RunCompareResponse)
async def compare_runs(
    ids: str = Query(..., description="Comma-separated run IDs"),
    admin: dict = Depends(require_admin),
    service: BenchmarkRunService = Depends(create_benchmark_run_service),
):
    """Compare multiple benchmark runs side by side."""
    id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]

    runs, diff = await service.compare(id_list)

    return RunCompareResponse(
        runs=[_run_to_response(r) for r in runs],
        diff=diff,
    )


# ---------------------------------------------------------------------------
# History / Trends
# ---------------------------------------------------------------------------


@router.get("/admin/benchmark/history", response_model=BenchmarkHistoryResponse)
async def benchmark_history(
    metric: str | None = None,
    dataset: str | None = None,
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(100, ge=1, le=1000),
    admin: dict = Depends(require_admin),
    service: BenchmarkRunService = Depends(create_benchmark_run_service),
):
    """Get benchmark history as time-series data for trend charts."""
    runs, _total = await service.list(dataset=dataset, sort_by="creation_date", sort_order="asc", limit=limit)

    points = []
    for r in runs:
        metrics = r.summary_metrics or {}
        config_summary = {k: r.config_json.get(k) for k in HISTORY_CONFIG_KEYS if k in r.config_json}
        points.append(
            BenchmarkHistoryPoint(
                run_id=r.id,
                creation_date=r.creation_date,
                metrics=metrics,
                config_summary=config_summary,
                dataset=r.dataset,
                llm_evaluated=r.llm_evaluated,
            )
        )

    return BenchmarkHistoryResponse(points=points, total=len(points))


@router.get("/admin/benchmark/regression-check", response_model=RegressionCheckResponse)
async def regression_check(
    run_id: int | None = Query(
        None,
        description="DB run ID to check; if omitted, compares last two history entries",
    ),
    admin: dict = Depends(require_admin),
    service: BenchmarkRunService = Depends(create_benchmark_run_service),
):
    """Check for regression: compare a run (or latest) against the last baseline."""
    from config import settings
    from infrastructure.ml.benchmark_history import compare_runs, get_last_baseline, load_history

    data_dir = str(settings.data_dir)
    baseline = get_last_baseline(data_dir)
    if baseline is None:
        return RegressionCheckResponse(passed=True, results=[])

    if run_id is not None:
        run = await service.get(run_id)
        current = {
            "metrics": run.summary_metrics or {},
            "config": run.config_json,
        }
    else:
        history = load_history(data_dir)
        if len(history) < 2:
            return RegressionCheckResponse(passed=True, results=[])
        current = history[-1]

    result = compare_runs(current, baseline)
    return RegressionCheckResponse(
        passed=result["passed"],
        results=[
            RegressionCheckResult(
                metric=r["metric"],
                baseline=r.get("baseline"),
                current=r.get("current"),
                delta=r.get("delta"),
                threshold=r["threshold"],
                failed=r["failed"],
                note=r.get("note"),
            )
            for r in result["results"]
        ],
    )
