"""Admin Benchmark Lab API — questions CRUD, sweeps, runs, history/comparison."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
import logging
from typing import Annotated, Any

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from application.admin import ConfigService
from application.benchmark import (
    HISTORY_CONFIG_KEYS,
    BenchmarkHistoryPort,
    BenchmarkQuestionService,
    BenchmarkResultService,
    BenchmarkRunService,
    BenchmarkSweepLauncher,
    BenchmarkSweepService,
)
from application.documents import DocumentQueryService
from application.ports.rate_limit import RateLimitPolicyName
from application.ports.route_configs import StorageConfig
from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import SSE_HEADERS, SSE_MEDIA_TYPE
from presentation.api.dependencies import (
    create_benchmark_history_port,
    create_benchmark_question_service,
    create_benchmark_result_service,
    create_benchmark_run_service,
    create_benchmark_sweep_launcher,
    create_benchmark_sweep_service,
    create_config_service,
    create_document_query_service,
    create_storage_config,
)
from presentation.api.helpers import (
    question_create_to_dto,
    question_to_response,
    run_to_response,
    sweep_create_to_dto,
    sweep_to_response,
)
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import (
    BenchmarkCancelResponse,
    BenchmarkDeleteResponse,
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
    CurrentUser,
    RegressionCheckResponse,
    RegressionCheckResult,
    RunApplyFailed,
    RunApplyResponse,
    RunCompareResponse,
    SourceFilesResponse,
    SweepCreateRequest,
    SweepResponse,
    SweepsListResponse,
)


logger = logging.getLogger("default")


@dataclass(kw_only=True)
class QuestionQuery:
    dataset: str | None = None
    tag: str | None = None
    search: str | None = None
    is_active: bool | None = None
    limit: Annotated[int, Query(ge=1, le=500)] = 50
    offset: Annotated[int, Query(ge=0)] = 0


@dataclass(kw_only=True)
class RunQuery:
    sweep_id: int | None = None
    dataset: str | None = None
    sort_by: Annotated[str, Query(pattern="^(creation_date|duration_sec|id)$")] = "creation_date"
    sort_order: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc"
    limit: Annotated[int, Query(ge=1, le=500)] = 50
    offset: Annotated[int, Query(ge=0)] = 0


@dataclass(kw_only=True)
class HistoryQuery:
    metric: str | None = None
    dataset: str | None = None
    days: Annotated[int, Query(ge=1, le=365)] = 30
    limit: Annotated[int, Query(ge=1, le=1000)] = 100


router = APIRouter(tags=["benchmark-admin"])


# ---------------------------------------------------------------------------
# Questions CRUD
# ---------------------------------------------------------------------------


@router.get("/admin/benchmark/datasets")
async def list_datasets(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkQuestionService, Depends(create_benchmark_question_service)],
) -> list[str]:
    return await service.list_datasets()


@router.get("/admin/benchmark/questions")
async def list_questions(
    *,
    options: Annotated[QuestionQuery, Depends()],
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkQuestionService, Depends(create_benchmark_question_service)],
) -> BenchmarkQuestionsListResponse:
    questions, total = await service.list(
        dataset=options.dataset,
        tag=options.tag,
        search=options.search,
        is_active=options.is_active,
        limit=options.limit,
        offset=options.offset,
    )
    return BenchmarkQuestionsListResponse(
        questions=[question_to_response(q) for q in questions],
        total=total,
    )


@router.post(
    "/admin/benchmark/questions",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def create_question(
    *,
    body: BenchmarkQuestionCreate,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkQuestionService, Depends(create_benchmark_question_service)],
) -> BenchmarkQuestionResponse:
    created = await service.create(question_create_to_dto(body), created_by=admin.id)
    return question_to_response(created)


@router.put(
    "/admin/benchmark/questions/{question_id}",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def update_question(
    *,
    question_id: int,
    body: BenchmarkQuestionUpdate,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkQuestionService, Depends(create_benchmark_question_service)],
) -> BenchmarkQuestionResponse:
    fields = body.model_dump(exclude_unset=True)
    updated = await service.update(question_id, fields)
    return question_to_response(updated)


@router.delete(
    "/admin/benchmark/questions/{question_id}",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def delete_question(
    *,
    question_id: int,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkQuestionService, Depends(create_benchmark_question_service)],
) -> BenchmarkDeleteResponse:
    await service.delete(question_id)
    return BenchmarkDeleteResponse(deleted=True)


@router.post(
    "/admin/benchmark/questions/import",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def import_questions(
    *,
    body: BenchmarkQuestionsImportRequest,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkQuestionService, Depends(create_benchmark_question_service)],
) -> BenchmarkQuestionsImportResponse:
    count = await service.bulk_create(
        [question_create_to_dto(q) for q in body.questions], created_by=admin.id
    )
    return BenchmarkQuestionsImportResponse(imported=count)


@router.get("/admin/benchmark/questions/export", response_model=None)
async def export_questions(
    *,
    dataset: str | None = None,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkQuestionService, Depends(create_benchmark_question_service)],
) -> list[dict[str, object]]:
    questions = await service.export(dataset=dataset)
    return [
        {
            "question": q.question,
            "expected_answer": q.expected_answer,
            "source_hint": q.source_hint,
            "annotations": q.annotations,
            "case_id": q.case_id,
            "variant_of": q.variant_of,
            "question_style": q.question_style,
            "tags": q.tags,
            "dataset": q.dataset,
            "is_active": q.is_active,
        }
        for q in questions
    ]


@router.get("/admin/benchmark/source-files")
async def list_source_files(
    *,
    search: str | None = None,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    query: Annotated[DocumentQueryService, Depends(create_document_query_service)],
) -> SourceFilesResponse:
    """Return distinct indexed document filenames for source_hint picker."""
    filenames = await query.list_source_files(search=search)
    return SourceFilesResponse(files=filenames)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


@router.post(
    "/admin/benchmark/sweep",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.BENCHMARK))],
)
async def create_sweep(
    *,
    body: SweepCreateRequest,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    launcher: Annotated[BenchmarkSweepLauncher, Depends(create_benchmark_sweep_launcher)],
) -> SweepResponse:
    sweep, job_id = await launcher.start(sweep_create_to_dto(body))
    return sweep_to_response(sweep, job_id=job_id)


@router.post(
    "/admin/benchmark/sweep/{sweep_id}/resume",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.BENCHMARK))],
)
async def resume_sweep(
    *,
    sweep_id: int,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    launcher: Annotated[BenchmarkSweepLauncher, Depends(create_benchmark_sweep_launcher)],
) -> SweepResponse:
    sweep, job_id = await launcher.resume(sweep_id)
    return sweep_to_response(sweep, job_id=job_id)


@router.get("/admin/benchmark/sweep/{sweep_id}")
async def get_sweep(
    *,
    sweep_id: int,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkSweepService, Depends(create_benchmark_sweep_service)],
) -> SweepResponse:
    sweep = await service.get(sweep_id)
    return sweep_to_response(sweep)


@router.get("/admin/benchmark/sweeps")
async def list_sweeps(
    *,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkSweepService, Depends(create_benchmark_sweep_service)],
) -> SweepsListResponse:
    sweeps, total = await service.list(limit=limit, offset=offset)
    return SweepsListResponse(
        sweeps=[sweep_to_response(s) for s in sweeps],
        total=total,
    )


@router.get("/admin/benchmark/sweep/{sweep_id}/progress")
async def get_sweep_progress(
    *,
    sweep_id: int,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkSweepService, Depends(create_benchmark_sweep_service)],
) -> dict[str, Any] | None:
    return await service.progress(sweep_id)


@router.get("/admin/benchmark/sweep/{sweep_id}/stream")
async def sweep_progress_stream(
    *,
    sweep_id: int,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    storage_cfg: Annotated[StorageConfig, Depends(create_storage_config)],
) -> StreamingResponse:
    """SSE stream: live progress + new results as they complete."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            redis_settings = RedisSettings.from_dsn(storage_cfg.redis_url)
            pool = await create_pool(redis_settings)
            try:
                pubsub = pool.pubsub()
                await pubsub.subscribe(f"sweep:{sweep_id}")

                while True:
                    try:
                        message = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True, timeout=30),
                            timeout=30,
                        )
                    except TimeoutError:
                        # idle timeout must NOT end the stream — keep it
                        # alive with a heartbeat and keep waiting for progress
                        # / the final done event.
                        yield ": heartbeat\n\n"
                        continue

                    if message and message["type"] == "message":
                        data = message["data"].decode("utf-8")
                        yield f"data: {data}\n\n"
                        parsed = json.loads(data)

                        if parsed.get("done"):
                            break
            finally:
                await pubsub.unsubscribe(f"sweep:{sweep_id}")
                await pool.close()
        except Exception as e:
            logger.warning("SSE stream error for sweep %d: %s", sweep_id, e, exc_info=True)
            yield f"data: {json.dumps({'error': 'SSE stream error'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


@router.post(
    "/admin/benchmark/sweep/{sweep_id}/cancel",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def cancel_sweep(
    *,
    sweep_id: int,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkSweepService, Depends(create_benchmark_sweep_service)],
) -> BenchmarkCancelResponse:
    await service.cancel(sweep_id)
    return BenchmarkCancelResponse(cancelled=True)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@router.get("/admin/benchmark/runs")
async def list_runs(
    *,
    options: Annotated[RunQuery, Depends()],
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkRunService, Depends(create_benchmark_run_service)],
) -> BenchmarkRunsListResponse:
    runs, total = await service.list(
        sweep_id=options.sweep_id,
        dataset=options.dataset,
        sort_by=options.sort_by,
        sort_order=options.sort_order,
        limit=options.limit,
        offset=options.offset,
    )
    return BenchmarkRunsListResponse(
        runs=[run_to_response(r) for r in runs],
        total=total,
    )


@router.get("/admin/benchmark/runs/{run_id:int}")
async def get_run(
    *,
    run_id: int,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkRunService, Depends(create_benchmark_run_service)],
) -> BenchmarkRunResponse:
    run = await service.get(run_id)
    return run_to_response(run)


@router.post(
    "/admin/benchmark/runs/{run_id}/apply",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.BENCHMARK))],
)
async def apply_run_config(
    *,
    run_id: int,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkRunService, Depends(create_benchmark_run_service)],
    config_service: Annotated[ConfigService, Depends(create_config_service)],
) -> RunApplyResponse:
    """Apply a run's config_json to the live system via ConfigService."""
    result = await service.apply_config(run_id, admin.id, config_service)
    return RunApplyResponse(
        applied=result.applied,
        keys=result.keys,
        failed=[RunApplyFailed(**f) for f in result.failed],
    )


@router.get("/admin/benchmark/runs/compare")
async def compare_runs(
    *,
    ids: Annotated[str, Query(description="Comma-separated run IDs")],
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkRunService, Depends(create_benchmark_run_service)],
) -> RunCompareResponse:
    """Compare multiple benchmark runs side by side."""
    try:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
    except ValueError as err:
        raise HTTPException(
            status_code=422, detail="Invalid run IDs: must be comma-separated integers"
        ) from err

    if not id_list:
        raise HTTPException(status_code=422, detail="At least one run ID required")

    runs, diff = await service.compare(id_list)

    return RunCompareResponse(
        runs=[run_to_response(r) for r in runs],
        diff=diff,
    )


# ---------------------------------------------------------------------------
# History / Trends
# ---------------------------------------------------------------------------


@router.get("/admin/benchmark/history")
async def benchmark_history(
    *,
    options: Annotated[HistoryQuery, Depends()],
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BenchmarkRunService, Depends(create_benchmark_run_service)],
) -> BenchmarkHistoryResponse:
    """Get benchmark history as time-series data for trend charts."""
    runs, _total = await service.list(
        dataset=options.dataset, sort_by="creation_date", sort_order="asc", limit=options.limit
    )

    points = []

    for r in runs:
        if r.id is None:
            message = "BenchmarkRun saved with None id"
            raise RuntimeError(message)

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


@router.get("/admin/benchmark/regression-check")
async def regression_check(
    *,
    run_id: Annotated[
        int | None, Query(description="DB run ID to check; if omitted, compares last two history entries")
    ] = None,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    result_service: Annotated[BenchmarkResultService, Depends(create_benchmark_result_service)],
    history_port: Annotated[BenchmarkHistoryPort, Depends(create_benchmark_history_port)],
) -> RegressionCheckResponse:
    """Check for regression: compare a run (or latest) against the last baseline."""
    output = await result_service.check_regression(run_id, history_port)
    return RegressionCheckResponse(
        passed=output.passed,
        run_id=output.run_id,
        baseline_run_id=output.baseline_run_id,
        results=[
            RegressionCheckResult(
                metric=r.metric,
                baseline=r.baseline,
                current=r.current,
                delta=r.delta,
                threshold=r.threshold,
                failed=r.failed,
                note=r.note,
            )
            for r in output.results
        ],
    )
