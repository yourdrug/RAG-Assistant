"""Admin jobs endpoints — background job listing."""

from __future__ import annotations

from application.ports.unit_of_work_factory import UnitOfWorkFactory
from fastapi import APIRouter, Depends, Query

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import get_uow_factory
from presentation.api.schemas import JobResponse, JobsListResponse, JobsStatsResponse

router = APIRouter(tags=["admin-jobs"])


@router.get("/admin/jobs", response_model=JobsListResponse)
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    admin: dict = Depends(require_admin),
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
):
    async with uow_factory.create() as uow:
        jobs = await uow.background_jobs.list_recent(limit=limit, offset=offset)
        stats = await uow.background_jobs.count_by_status()
    total = sum(stats.values())
    return JobsListResponse(
        total=total,
        jobs=[
            JobResponse(
                id=j.id,
                job_type=j.job_type,
                status=j.status,
                related_id=j.related_id,
                request_id=j.request_id,
                started_at=j.started_at,
                finished_at=j.finished_at,
                error_message=j.error_message,
                creation_date=j.creation_date,
            )
            for j in jobs
        ],
    )


@router.get("/admin/jobs/stats", response_model=JobsStatsResponse)
async def jobs_stats(
    admin: dict = Depends(require_admin),
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
):
    async with uow_factory.create() as uow:
        stats = await uow.background_jobs.count_by_status()
    total = sum(stats.values())
    return JobsStatsResponse(total=total, by_status=stats)
