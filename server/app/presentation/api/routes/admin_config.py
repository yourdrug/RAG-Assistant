"""Admin config endpoints — dynamic config management + system info."""

from __future__ import annotations

import logging

from application.ports.rate_limit import RateLimitPolicyName
from application.services.config_admin_service import ConfigAdminService
from application.services.config_service import ConfigService
from fastapi import APIRouter, Depends, HTTPException

from presentation.api.auth_dependencies import require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.constants import QUESTION_LOG_MAX_CHARS, STATIC_CONFIG_KEYS
from presentation.api.dependencies import (
    create_action_logger,
    create_config_admin_service,
    create_config_masker,
    create_config_service,
)
from presentation.api.schemas import (
    ConfigParamResponse,
    ConfigParamUpdateRequest,
    CurrentUser,
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
    admin: CurrentUser = Depends(require_admin),
    config_service: ConfigService = Depends(create_config_service),
    masker=Depends(create_config_masker),
):
    rows = await config_service.list_parameters()
    return [
        ConfigParamResponse(
            key=r.key,
            value=masker.mask_value(r.value) if r.key in masker.sensitive_keys else r.normalize(r.value),
            value_type=r.value_type,
            category=r.category,
            description=r.description,
            min_value=r.min_value,
            max_value=r.max_value,
            domain_key=r.domain_key,
        )
        for r in rows
        if r.key not in STATIC_CONFIG_KEYS
    ]


@router.put(
    "/admin/config/{key}",
    response_model=ConfigParamResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def update_config(
    key: str,
    body: ConfigParamUpdateRequest,
    admin: CurrentUser = Depends(require_admin),
    config_service: ConfigService = Depends(create_config_service),
    log=Depends(create_action_logger),
    masker=Depends(create_config_masker),
    domain: str | None = None,
):
    if key in STATIC_CONFIG_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"'{key}' is a static parameter — set it in server/.env and restart",
        )
    param = await config_service.update_parameter(key, body.value, changed_by=admin.id, domain_key=domain)
    is_sensitive = key in masker.sensitive_keys
    masked_value = masker.mask_value(body.value) if is_sensitive else body.value[:QUESTION_LOG_MAX_CHARS]
    log(
        "config.update",
        user_id=admin.id,
        details={"key": key, "value": masked_value, "domain": domain},
    )
    return ConfigParamResponse(
        key=param.key,
        value=masker.mask_value(param.value) if param.key in masker.sensitive_keys else param.value,
        value_type=param.value_type,
        category=param.category,
        description=param.description,
        min_value=param.min_value,
        max_value=param.max_value,
        domain_key=param.domain_key,
    )


@router.get("/admin/models/info", response_model=ModelsInfoResponse)
async def models_info(
    admin: CurrentUser = Depends(require_admin),
    admin_service: ConfigAdminService = Depends(create_config_admin_service),
):
    info = await admin_service.get_models_info()
    return ModelsInfoResponse(
        llm_provider=info.llm_provider,
        llm_model=info.llm_model,
        tei_embed_url=info.tei_embed_url,
        tei_rerank_url=info.tei_rerank_url,
        embed_model=info.embed_model,
        rerank_model=info.rerank_model,
        ocr_engine=info.ocr_engine,
        ocr_enabled=info.ocr_enabled,
        ollama_models=info.ollama_models,
        openrouter_model=info.openrouter_model,
        ml_provider=info.ml_provider,
    )


@router.get("/admin/models/openrouter", response_model=OpenRouterModelsResponse)
async def openrouter_models(
    admin: CurrentUser = Depends(require_admin),
    admin_service: ConfigAdminService = Depends(create_config_admin_service),
):
    info = await admin_service.get_openrouter_models()
    return OpenRouterModelsResponse(
        models=[OpenRouterModelInfo(**m) for m in info.models],
        active_model=info.active_model or "",
    )


@router.get("/admin/vectordb/info", response_model=VectorDBInfoResponse)
async def vectordb_info(
    admin: CurrentUser = Depends(require_admin),
    admin_service: ConfigAdminService = Depends(create_config_admin_service),
):
    info = await admin_service.get_vectordb_info()
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
