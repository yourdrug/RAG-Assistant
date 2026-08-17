"""Admin Benchmark Lab API — questions CRUD, sweeps, runs, history/comparison."""

from __future__ import annotations

import json
import logging

from application.services.benchmark_services import (
    BenchmarkQuestionService,
    BenchmarkRunService,
    BenchmarkSweepService,
)
from application.services.config_service import ConfigService
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from infrastructure.uow_factory import UnitOfWorkFactory
from infrastructure.worker.queue import enqueue_sweep

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import (
    create_config_service,
    get_benchmark_question_service,
    get_benchmark_run_service,
    get_benchmark_sweep_service,
    get_uow_factory,
)
from presentation.api.routes.common import create_background_job
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
    RunApplyResponse,
    RunCompareResponse,
    SweepCreateRequest,
    SweepResponse,
    SweepsListResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["benchmark-admin"])


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
    service: BenchmarkQuestionService = Depends(get_benchmark_question_service),
):
    questions, total = await service.list(
        dataset=dataset, tag=tag, search=search, is_active=is_active, limit=limit, offset=offset
    )
    return BenchmarkQuestionsListResponse(
        questions=[
            BenchmarkQuestionResponse(
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
            for q in questions
        ],
        total=total,
    )


@router.post("/admin/benchmark/questions", response_model=BenchmarkQuestionResponse)
async def create_question(
    body: BenchmarkQuestionCreate,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(get_benchmark_question_service),
):
    created = await service.create(body, created_by=admin["id"])
    return BenchmarkQuestionResponse(
        id=created.id,
        question=created.question,
        expected_answer=created.expected_answer,
        source_hint=created.source_hint,
        tags=created.tags,
        dataset=created.dataset,
        is_active=created.is_active,
        created_by=created.created_by,
        notes=created.notes,
        creation_date=created.creation_date,
    )


@router.put("/admin/benchmark/questions/{question_id}", response_model=BenchmarkQuestionResponse)
async def update_question(
    question_id: int,
    body: BenchmarkQuestionUpdate,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(get_benchmark_question_service),
):
    fields = body.model_dump(exclude_unset=True)
    try:
        updated = await service.update(question_id, fields)
    except Exception as e:
        from domain.exceptions import EntityNotFound

        if isinstance(e, EntityNotFound):
            raise HTTPException(status_code=404, detail="Question not found")
        raise
    return BenchmarkQuestionResponse(
        id=updated.id,
        question=updated.question,
        expected_answer=updated.expected_answer,
        source_hint=updated.source_hint,
        tags=updated.tags,
        dataset=updated.dataset,
        is_active=updated.is_active,
        created_by=updated.created_by,
        notes=updated.notes,
        creation_date=updated.creation_date,
    )


@router.delete("/admin/benchmark/questions/{question_id}")
async def delete_question(
    question_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(get_benchmark_question_service),
):
    try:
        await service.delete(question_id)
    except Exception as e:
        from domain.exceptions import EntityNotFound

        if isinstance(e, EntityNotFound):
            raise HTTPException(status_code=404, detail="Question not found")
        raise
    return {"deleted": True}


@router.post("/admin/benchmark/questions/import", response_model=BenchmarkQuestionsImportResponse)
async def import_questions(
    body: BenchmarkQuestionsImportRequest,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(get_benchmark_question_service),
):
    count = await service.bulk_create(body.questions, created_by=admin["id"])
    return BenchmarkQuestionsImportResponse(imported=count)


@router.get("/admin/benchmark/questions/export")
async def export_questions(
    dataset: str | None = None,
    admin: dict = Depends(require_admin),
    service: BenchmarkQuestionService = Depends(get_benchmark_question_service),
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
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
):
    """Return distinct indexed document filenames for source_hint picker."""
    async with uow_factory.create() as uow:
        filenames = await uow.documents.list_distinct_filenames(search=search, limit=100)
    return {"files": filenames}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


@router.post("/admin/benchmark/sweep", response_model=SweepResponse)
async def create_sweep(
    body: SweepCreateRequest,
    admin: dict = Depends(require_admin),
    service: BenchmarkSweepService = Depends(get_benchmark_sweep_service),
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
):
    sweep = await service.create(body)

    job_id = await create_background_job(uow_factory, "sweep", related_id=sweep.id)

    await service.update_status(sweep.id, "pending")

    await enqueue_sweep(sweep_id=sweep.id, job_id=job_id)

    return SweepResponse(
        id=sweep.id,
        status="pending",
        strategy=sweep.strategy,
        search_space=sweep.search_space,
        objective_weights=sweep.objective_weights,
        dataset=sweep.dataset,
        top_n_llm=sweep.top_n_llm,
        total_configs=sweep.total_configs,
        evaluated_configs=sweep.evaluated_configs,
        best_run_id=sweep.best_run_id,
        job_id=job_id,
        creation_date=sweep.creation_date,
    )


@router.get("/admin/benchmark/sweep/{sweep_id}", response_model=SweepResponse)
async def get_sweep(
    sweep_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkSweepService = Depends(get_benchmark_sweep_service),
):
    try:
        sweep = await service.get(sweep_id)
    except Exception as e:
        from domain.exceptions import EntityNotFound

        if isinstance(e, EntityNotFound):
            raise HTTPException(status_code=404, detail="Sweep not found")
        raise
    return SweepResponse(
        id=sweep.id,
        status=sweep.status,
        strategy=sweep.strategy,
        search_space=sweep.search_space,
        objective_weights=sweep.objective_weights,
        dataset=sweep.dataset,
        top_n_llm=sweep.top_n_llm,
        total_configs=sweep.total_configs,
        evaluated_configs=sweep.evaluated_configs,
        best_run_id=sweep.best_run_id,
        job_id=sweep.job_id,
        creation_date=sweep.creation_date,
    )


@router.get("/admin/benchmark/sweeps", response_model=SweepsListResponse)
async def list_sweeps(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    admin: dict = Depends(require_admin),
    service: BenchmarkSweepService = Depends(get_benchmark_sweep_service),
):
    sweeps, total = await service.list(limit=limit, offset=offset)
    return SweepsListResponse(
        sweeps=[
            SweepResponse(
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
                job_id=s.job_id,
                creation_date=s.creation_date,
            )
            for s in sweeps
        ],
        total=total,
    )


@router.get("/admin/benchmark/sweep/{sweep_id}/stream")
async def sweep_progress_stream(
    sweep_id: int,
    admin: dict = Depends(require_admin),
):
    """SSE stream: live progress + new results as they complete."""
    import asyncio

    from config import settings

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
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/admin/benchmark/sweep/{sweep_id}/cancel")
async def cancel_sweep(
    sweep_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkSweepService = Depends(get_benchmark_sweep_service),
):
    try:
        await service.cancel(sweep_id)
    except Exception as e:
        from domain.exceptions import EntityNotFound, ValidationError

        if isinstance(e, EntityNotFound):
            raise HTTPException(status_code=404, detail="Sweep not found")
        if isinstance(e, ValidationError):
            raise HTTPException(status_code=400, detail=str(e.detail))
        raise
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
    service: BenchmarkRunService = Depends(get_benchmark_run_service),
):
    runs, total = await service.list(
        sweep_id=sweep_id, dataset=dataset, sort_by=sort_by, sort_order=sort_order,
        limit=limit, offset=offset,
    )
    return BenchmarkRunsListResponse(
        runs=[
            BenchmarkRunResponse(
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
            for r in runs
        ],
        total=total,
    )


@router.get("/admin/benchmark/runs/{run_id}", response_model=BenchmarkRunResponse)
async def get_run(
    run_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkRunService = Depends(get_benchmark_run_service),
):
    try:
        run = await service.get(run_id)
    except Exception as e:
        from domain.exceptions import EntityNotFound

        if isinstance(e, EntityNotFound):
            raise HTTPException(status_code=404, detail="Run not found")
        raise
    return BenchmarkRunResponse(
        id=run.id,
        sweep_id=run.sweep_id,
        config_json=run.config_json,
        summary_metrics=run.summary_metrics,
        duration_sec=run.duration_sec,
        llm_evaluated=run.llm_evaluated,
        dataset=run.dataset,
        filename=run.filename,
        creation_date=run.creation_date,
    )


@router.post("/admin/benchmark/runs/{run_id}/apply", response_model=RunApplyResponse)
async def apply_run_config(
    run_id: int,
    admin: dict = Depends(require_admin),
    service: BenchmarkRunService = Depends(get_benchmark_run_service),
    config_service: ConfigService = Depends(create_config_service),
):
    """Apply a run's config_json to the live system via ConfigService."""
    try:
        run = await service.get(run_id)
    except Exception as e:
        from domain.exceptions import EntityNotFound

        if isinstance(e, EntityNotFound):
            raise HTTPException(status_code=404, detail="Run not found")
        raise

    config = run.config_json
    applied_keys = []

    key_mapping = {
        "top_k": "retriever_top_k",
        "fetch_k": "retriever_fetch_k",
        "dense_weight": "dense_weight",
        "sparse_weight": "sparse_weight",
        "rrf_k": "rrf_k",
        "rerank_min_score": "rerank_min_score",
        "rerank_score_gap_ratio": "rerank_score_gap_ratio",
    }

    for config_key, param_key in key_mapping.items():
        if config_key in config:
            try:
                await config_service.update_parameter(
                    param_key, str(config[config_key]), changed_by=admin["id"]
                )
                applied_keys.append(param_key)
            except Exception as e:
                logger.warning("Failed to apply %s=%s: %s", param_key, config[config_key], e)

    return RunApplyResponse(applied=len(applied_keys), keys=applied_keys)


@router.get("/admin/benchmark/runs/compare", response_model=RunCompareResponse)
async def compare_runs(
    ids: str = Query(..., description="Comma-separated run IDs"),
    admin: dict = Depends(require_admin),
    service: BenchmarkRunService = Depends(get_benchmark_run_service),
):
    """Compare multiple benchmark runs side by side."""
    id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]

    try:
        runs, diff = await service.compare(id_list)
    except Exception as e:
        from domain.exceptions import EntityNotFound, ValidationError

        if isinstance(e, EntityNotFound):
            raise HTTPException(status_code=404, detail=str(e.detail))
        if isinstance(e, ValidationError):
            raise HTTPException(status_code=400, detail=str(e.detail))
        raise

    return RunCompareResponse(
        runs=[
            BenchmarkRunResponse(
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
            for r in runs
        ],
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
    service: BenchmarkRunService = Depends(get_benchmark_run_service),
):
    """Get benchmark history as time-series data for trend charts."""
    runs, _total = await service.list(
        dataset=dataset, sort_by="creation_date", sort_order="asc", limit=limit
    )

    points = []
    for r in runs:
        metrics = r.summary_metrics or {}
        config_summary = {
            k: r.config_json.get(k)
            for k in ("top_k", "fetch_k", "dense_weight", "sparse_weight", "rrf_k")
            if k in r.config_json
        }
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
