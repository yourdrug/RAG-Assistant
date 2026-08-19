"""Admin metrics endpoint — Prometheus metrics as JSON."""

from __future__ import annotations

from application.services.metrics_service import MetricsService
from fastapi import APIRouter, Depends

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_metrics_service
from presentation.api.schemas import MetricsResponse

router = APIRouter(tags=["admin-metrics"])


@router.get("/admin/metrics", response_model=MetricsResponse)
async def get_metrics(
    admin: dict = Depends(require_admin),
    metrics_service: MetricsService = Depends(create_metrics_service),
):
    result = metrics_service.collect()
    return MetricsResponse(
        db_pool=result.db_pool,
        qdrant=result.qdrant,
        bm25=result.bm25,
        ollama=result.ollama,
        rag=result.rag,
        ingestion=result.ingestion,
        http_requests=result.http_requests,
    )
