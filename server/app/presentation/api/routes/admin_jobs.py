"""Admin jobs endpoints — background job listing."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from application.jobs.service import JobService
from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import DEFAULT_PAGE_LIMIT, DEFAULT_PAGE_OFFSET, MAX_PAGE_LIMIT
from presentation.api.dependencies import create_job_service
from presentation.api.helpers import persisted_id
from presentation.api.schemas import CurrentUser, JobResponse, JobsListResponse, JobsStatsResponse


router = APIRouter(tags=["admin-jobs"])


@router.get("/admin/jobs", response_model=JobsListResponse)
async def list_jobs(
    *,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
    offset: Annotated[int, Query(ge=0)] = DEFAULT_PAGE_OFFSET,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[JobService, Depends(create_job_service)],
) -> JobsListResponse:
    jobs = await service.list_recent(limit=limit, offset=offset)
    stats = await service.count_by_status()
    total = sum(stats.values())
    return JobsListResponse(
        total=total,
        jobs=[
            JobResponse(
                id=persisted_id(j.id),
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
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[JobService, Depends(create_job_service)],
) -> JobsStatsResponse:
    stats = await service.count_by_status()
    total = sum(stats.values())
    return JobsStatsResponse(total=total, by_status=stats)


@router.get("/admin/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    *,
    job_id: int,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[JobService, Depends(create_job_service)],
) -> JobResponse:
    job = await service.get(job_id)
    return JobResponse(
        id=job_id,
        job_type=job.job_type,
        status=job.status,
        related_id=job.related_id,
        request_id=job.request_id,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error_message=job.error_message,
        creation_date=job.creation_date,
    )
