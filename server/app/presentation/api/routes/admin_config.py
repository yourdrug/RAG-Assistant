"""Admin config endpoints — dynamic config management + system info."""

from __future__ import annotations

import logging

from application.services.config_admin_service import ConfigAdminService
from application.services.config_service import ConfigService
from fastapi import APIRouter, Depends
from infrastructure.logging.actions import log_action

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_config_admin_service, create_config_service
from presentation.api.schemas import (
    ConfigParamResponse,
    ConfigParamUpdateRequest,
    ModelsInfoResponse,
    OpenRouterModelInfo,
    OpenRouterModelsResponse,
    VectorDBCollectionInfo,
    VectorDBInfoResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["admin-config"])


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
async def models_info(
    admin: dict = Depends(require_admin),
    admin_service: ConfigAdminService = Depends(create_config_admin_service),
):
    info = await admin_service.get_models_info()
    return ModelsInfoResponse(
        llm_provider=info.llm_provider,
        llm_model=info.llm_model,
        embed_model=info.embed_model,
        rerank_model=info.rerank_model,
        device=info.device,
        embed_device=info.embed_device,
        rerank_device=info.rerank_device,
        ocr_engine=info.ocr_engine,
        ocr_enabled=info.ocr_enabled,
        ollama_models=info.ollama_models,
        openrouter_model=info.openrouter_model,
    )


@router.get("/admin/models/openrouter", response_model=OpenRouterModelsResponse)
async def openrouter_models(
    admin: dict = Depends(require_admin),
    admin_service: ConfigAdminService = Depends(create_config_admin_service),
):
    info = await admin_service.get_openrouter_models()
    return OpenRouterModelsResponse(
        models=[OpenRouterModelInfo(**m) for m in info.models],
        active_model=info.active_model,
    )


@router.get("/admin/vectordb/info", response_model=VectorDBInfoResponse)
async def vectordb_info(
    admin: dict = Depends(require_admin),
    admin_service: ConfigAdminService = Depends(create_config_admin_service),
):
    info = admin_service.get_vectordb_info()
    return VectorDBInfoResponse(
        collections=[
            VectorDBCollectionInfo(
                name=c.name,
                points_count=c.points_count,
                vectors_count=c.vectors_count,
                indexed_vectors_count=c.indexed_vectors_count,
                segments_count=c.segments_count,
                status=c.status,
                optimizer_status=c.optimizer_status,
                hnsw_m=c.hnsw_m,
                hnsw_ef_construct=c.hnsw_ef_construct,
                on_disk_payload=c.on_disk_payload,
                vector_size=c.vector_size,
                distance=c.distance,
            )
            for c in info.collections
        ],
        active_collection=info.active_collection,
        qdrant_status=info.qdrant_status,
    )
