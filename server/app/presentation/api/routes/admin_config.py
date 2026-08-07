"""Admin config endpoints — dynamic config management + system info."""

from __future__ import annotations

import logging

from application.services.config_service import ConfigService
from config import settings
from fastapi import APIRouter, Depends
from infrastructure.logging.actions import log_action
from qdrant_client import QdrantClient

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_config_service
from presentation.api.routes.health import get_ollama_models, get_qdrant_status
from presentation.api.schemas import (
    ConfigParamResponse,
    ConfigParamUpdateRequest,
    ModelsInfoResponse,
    VectorDBCollectionInfo,
    VectorDBInfoResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["admin-config"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/admin/config", response_model=list[ConfigParamResponse])
async def list_config(
    admin: dict = Depends(require_admin),
    config_service: ConfigService = Depends(create_config_service),
):
    rows = await config_service.list_parameters()
    return [
        ConfigParamResponse(
            key=r.key,
            value=r.value,
            value_type=r.value_type,
            category=r.category,
            description=r.description,
            min_value=r.min_value,
            max_value=r.max_value,
        )
        for r in rows
    ]


@router.put("/admin/config/{key}", response_model=ConfigParamResponse)
async def update_config(
    key: str,
    body: ConfigParamUpdateRequest,
    admin: dict = Depends(require_admin),
    config_service: ConfigService = Depends(create_config_service),
):
    param = await config_service.update_parameter(key, body.value, changed_by=admin["id"])
    log_action("config.update", user_id=admin["id"], details={"key": key, "value": body.value[:100]})
    return ConfigParamResponse(
        key=param.key,
        value=param.value,
        value_type=param.value_type,
        category=param.category,
        description=param.description,
        min_value=param.min_value,
        max_value=param.max_value,
    )


@router.get("/admin/models/info", response_model=ModelsInfoResponse)
async def models_info(admin: dict = Depends(require_admin)):
    ollama_models = await get_ollama_models()

    return ModelsInfoResponse(
        llm_model=settings.llm_model,
        embed_model=settings.embed_model,
        rerank_model=settings.rerank_model,
        rerank_device=settings.rerank_device,
        ocr_engine=settings.ocr_engine,
        ocr_enabled=settings.ocr_enabled,
        ollama_models=ollama_models,
    )


@router.get("/admin/vectordb/info", response_model=VectorDBInfoResponse)
async def vectordb_info(admin: dict = Depends(require_admin)):
    collections: list[VectorDBCollectionInfo] = []

    qdrant_status = get_qdrant_status(timeout=5)

    if qdrant_status == "ok":
        try:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
            col_list = client.get_collections()

            for col in col_list.collections:
                info = client.get_collection(col.name)

                vectors_cfg = info.config.params.vectors if info.config.params.vectors else None
                hnsw_cfg = info.config.hnsw_config

                collections.append(
                    VectorDBCollectionInfo(
                        name=col.name,
                        points_count=info.points_count or 0,
                        vectors_count=info.vectors_count or 0,
                        indexed_vectors_count=info.indexed_vectors_count or 0,
                        segments_count=info.segments_count or 0,
                        status=str(info.status.value) if hasattr(info.status, "value") else str(info.status),
                        optimizer_status=(
                            str(info.optimizer_status.value)
                            if hasattr(info.optimizer_status, "value")
                            else str(info.optimizer_status)
                        ),
                        hnsw_m=hnsw_cfg.m if hnsw_cfg else None,
                        hnsw_ef_construct=hnsw_cfg.ef_construct if hnsw_cfg else None,
                        on_disk_payload=info.config.params.on_disk_payload if info.config.params else None,
                        vector_size=vectors_cfg.size if vectors_cfg else None,
                        distance=(
                            str(vectors_cfg.distance.value)
                            if vectors_cfg and hasattr(vectors_cfg.distance, "value")
                            else str(vectors_cfg.distance)
                            if vectors_cfg
                            else None
                        ),
                    )
                )
        except Exception as e:
            qdrant_status = f"error: {e}"

    return VectorDBInfoResponse(
        collections=collections,
        active_collection=settings.collection_name,
        qdrant_status=qdrant_status,
    )
