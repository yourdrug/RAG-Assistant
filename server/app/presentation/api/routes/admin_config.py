"""Admin config endpoints — dynamic config management + system info."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from application.admin import (
    ConfigAdminService,
    ConfigMaskerPort,
    ConfigService,
    ConfigurationStatus,
    CorpusControlPort,
    EffectiveParameter,
)
from application.ports.action_logger import ActionLoggerPort
from application.ports.rate_limit import RateLimitPolicyName
from application.runtime.dynamic_params import DYNAMIC_FIELDS_MAP
from application.runtime.setting_keys import SECRET_SETTING_KEYS
from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import QUESTION_LOG_MAX_CHARS, STATIC_CONFIG_KEYS
from presentation.api.dependencies import (
    create_action_logger,
    create_config_admin_service,
    create_config_masker,
    create_config_service,
    create_corpus_control,
)
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import (
    ConfigParamResponse,
    ConfigParamUpdateRequest,
    CurrentUser,
    LLMModelInfo,
    LLMModelsResponse,
    ModelsInfoResponse,
    VectorDBCollectionInfo,
    VectorDBInfoResponse,
)


logger = logging.getLogger("default")

router = APIRouter(tags=["admin-config"])


@router.get("/admin/config/effective", response_model=list[EffectiveParameter])
async def effective_configuration(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    config_service: Annotated[ConfigService, Depends(create_config_service)],
) -> list[EffectiveParameter]:
    """Admin-only applied values; secrets expose presence only."""
    return await config_service.effective_parameters()


@router.get("/admin/config/status", response_model=ConfigurationStatus)
async def configuration_status(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    config_service: Annotated[ConfigService, Depends(create_config_service)],
) -> ConfigurationStatus:
    """Compare the saved DB snapshot with the snapshot applied by this API process."""
    return await config_service.configuration_status()


@router.get("/admin/config", response_model=list[ConfigParamResponse])
async def list_config(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    config_service: Annotated[ConfigService, Depends(create_config_service)],
    masker: Annotated[ConfigMaskerPort, Depends(create_config_masker)],
) -> list[ConfigParamResponse]:
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
        if r.key not in SECRET_SETTING_KEYS and (r.domain_key is not None or r.key in DYNAMIC_FIELDS_MAP)
    ]


@router.put(
    "/admin/config/{key}",
    response_model=ConfigParamResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def update_config(
    *,
    key: str,
    body: ConfigParamUpdateRequest,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    config_service: Annotated[ConfigService, Depends(create_config_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
    masker: Annotated[ConfigMaskerPort, Depends(create_config_masker)],
    domain: str | None = None,
) -> ConfigParamResponse:
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
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    admin_service: Annotated[ConfigAdminService, Depends(create_config_admin_service)],
) -> ModelsInfoResponse:
    info = await admin_service.get_models_info()
    return ModelsInfoResponse(
        llm_base_url=info.llm_base_url,
        llm_model=info.llm_model,
        llm_auxiliary_model=info.llm_auxiliary_model,
        llm_api_key_configured=info.llm_api_key_configured,
        embedding_base_url=info.embedding_base_url,
        embed_model=info.embed_model,
        embedding_api_key_configured=info.embedding_api_key_configured,
        rerank_base_url=info.rerank_base_url,
        rerank_model=info.rerank_model,
        rerank_api_format=info.rerank_api_format,
        rerank_api_key_configured=info.rerank_api_key_configured,
        ocr_enabled=info.ocr_enabled,
    )


@router.get("/admin/models/llm", response_model=LLMModelsResponse)
async def llm_models(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    admin_service: Annotated[ConfigAdminService, Depends(create_config_admin_service)],
) -> LLMModelsResponse:
    info = await admin_service.get_llm_models()
    return LLMModelsResponse(
        models=[LLMModelInfo(**m) for m in info.models],
        active_model=info.active_model or "",
    )


@router.get("/admin/vectordb/info", response_model=VectorDBInfoResponse)
async def vectordb_info(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    admin_service: Annotated[ConfigAdminService, Depends(create_config_admin_service)],
) -> VectorDBInfoResponse:
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


class CorpusSwitchRequest(BaseModel):
    endpoint: str = Field(min_length=1, max_length=2048)
    collection: str = Field(min_length=1, max_length=255)
    expected_generation: str = Field(min_length=1, max_length=100)


@router.get("/admin/corpus/status", response_model=None)
async def corpus_status(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    control: Annotated[CorpusControlPort, Depends(create_corpus_control)],
) -> dict[str, Any]:
    return await control.status()


@router.post(
    "/admin/corpus/switch", dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))], response_model=None
)
async def switch_corpus(
    *,
    body: CorpusSwitchRequest,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    control: Annotated[CorpusControlPort, Depends(create_corpus_control)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> dict[str, Any]:
    result = await control.switch(body.endpoint, body.collection, body.expected_generation)
    log("corpus.switch", user_id=admin.id, details={"generation": result["active"]["generation"]})
    return result
