"""Admin jobs endpoints — background job listing."""

from __future__ import annotations

from application.services.job_service import JobService
from fastapi import APIRouter, Depends, Query

from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import DEFAULT_PAGE_LIMIT, DEFAULT_PAGE_OFFSET, MAX_PAGE_LIMIT
from presentation.api.dependencies import create_job_service
from presentation.api.schemas import JobResponse, JobsListResponse, JobsStatsResponse

router = APIRouter(tags=["admin-jobs"])


@router.get("/admin/jobs", response_model=JobsListResponse)
async def list_jobs(
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(DEFAULT_PAGE_OFFSET, ge=0),
    admin: dict = Depends(require_admin),
    service: JobService = Depends(create_job_service),
):
    jobs = await service.list_recent(limit=limit, offset=offset)
    stats = await service.count_by_status()
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
    service: JobService = Depends(create_job_service),
):
    stats = await service.count_by_status()
    total = sum(stats.values())
    return JobsStatsResponse(total=total, by_status=stats)
