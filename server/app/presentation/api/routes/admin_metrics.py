"""Admin metrics endpoint — Prometheus metrics as JSON."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from application.admin.metrics import MetricsService
from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_metrics_service
from presentation.api.schemas import CurrentUser, MetricsResponse


router = APIRouter(tags=["admin-metrics"])


@router.get("/admin/metrics", response_model=MetricsResponse)
async def get_metrics(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    metrics_service: Annotated[MetricsService, Depends(create_metrics_service)],
) -> MetricsResponse:
    result = metrics_service.collect()
    return MetricsResponse(
        db_pool=result.db_pool,
        qdrant=result.qdrant,
        bm25=result.bm25,
        rag=result.rag,
        ingestion=result.ingestion,
        http_requests=result.http_requests,
    )
