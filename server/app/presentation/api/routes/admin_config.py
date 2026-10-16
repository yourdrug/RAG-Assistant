"""Admin config endpoints — dynamic config management + system info."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field
from config import settings
from domain.value_objects.cost_budget import CostRate
from application.services.budget_admin_service import BudgetAdminService
from application.ports.corpus_control import CorpusControlPort

from application.ports.provider_configuration import ConfigurationStatus, EffectiveParameter
from application.ports.rate_limit import RateLimitPolicyName
from application.services.config_admin_service import ConfigAdminService
from application.services.config_service import ConfigService
from fastapi import APIRouter, Depends, HTTPException

from presentation.api.auth_dependencies import require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.constants import QUESTION_LOG_MAX_CHARS, STATIC_CONFIG_KEYS
from domain.value_objects.dynamic_config import DYNAMIC_FIELDS_MAP, SECRET_SETTING_KEYS
from presentation.api.dependencies import (
    create_action_logger,
    create_budget_admin_service,
    create_corpus_control,
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


@router.get("/admin/config/effective", response_model=list[EffectiveParameter])
async def effective_configuration(
    admin: CurrentUser = Depends(require_admin),
    config_service: ConfigService = Depends(create_config_service),
):
    """Admin-only applied values; secrets expose presence only."""
    return await config_service.effective_parameters()


@router.get("/admin/config/status", response_model=ConfigurationStatus)
async def configuration_status(
    admin: CurrentUser = Depends(require_admin),
    config_service: ConfigService = Depends(create_config_service),
):
    """Compare the saved DB snapshot with the snapshot applied by this API process."""
    return await config_service.configuration_status()


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
        if r.key not in SECRET_SETTING_KEYS and (r.domain_key is not None or r.key in DYNAMIC_FIELDS_MAP)
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


class CostRateResponse(BaseModel):
    provider: Literal['openrouter', 'deepinfra']
    model: str = Field(min_length=1, max_length=200)
    version: int = Field(ge=1)
    unit: Literal['tokens'] = 'tokens'
    input_usd_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    output_usd_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    context_tokens: int = Field(ge=1, le=10_000_000)


class CostRateRequest(CostRateResponse):
    input_usd_per_million: Decimal = Field(gt=0, allow_inf_nan=False)


@router.get('/admin/cost/rates', response_model=list[CostRateResponse])
async def cost_rates(
    admin: CurrentUser = Depends(require_admin),
    service: BudgetAdminService = Depends(create_budget_admin_service),
):
    return await service.rates()


@router.post(
    '/admin/cost/rates',
    status_code=201,
    response_model=CostRateRequest,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def add_cost_rate(
    body: CostRateRequest,
    admin: CurrentUser = Depends(require_admin),
    service: BudgetAdminService = Depends(create_budget_admin_service),
):
    return await service.add_rate(CostRate(**body.model_dump()))


@router.get('/admin/cost/usage')
async def recent_cost_usage(
    admin: CurrentUser = Depends(require_admin),
    service: BudgetAdminService = Depends(create_budget_admin_service),
):
    return await service.usage()


@router.get('/admin/cost/models')
async def active_cost_models(
    admin: CurrentUser = Depends(require_admin),
    service: BudgetAdminService = Depends(create_budget_admin_service),
):
    models = []
    if settings.ml_provider == 'deepinfra':
        for model in (settings.deepinfra_embed_model, settings.deepinfra_rerank_model):
            models.append(await service.model_status('deepinfra', model))
    return {'enabled': settings.cost_rate_limit_enabled, 'models': models}


@router.get('/admin/cost/pending')
async def pending_cost_usage(
    admin: CurrentUser = Depends(require_admin),
    service: BudgetAdminService = Depends(create_budget_admin_service),
):
    return await service.pending()


class CorpusSwitchRequest(BaseModel):
    endpoint: str = Field(min_length=1, max_length=2048)
    collection: str = Field(min_length=1, max_length=255)
    expected_generation: str = Field(min_length=1, max_length=100)


@router.get('/admin/corpus/status')
async def corpus_status(
    admin: CurrentUser = Depends(require_admin), control: CorpusControlPort = Depends(create_corpus_control)
):
    return await control.status()


@router.post('/admin/corpus/switch', dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))])
async def switch_corpus(
    body: CorpusSwitchRequest,
    admin: CurrentUser = Depends(require_admin),
    control: CorpusControlPort = Depends(create_corpus_control),
    log=Depends(create_action_logger),
):
    result = await control.switch(body.endpoint, body.collection, body.expected_generation)
    log('corpus.switch', user_id=admin.id, details={'generation': result['active']['generation']})
    return result


class CostReconciliationRequest(BaseModel):
    actual_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    evidence: str = Field(min_length=1, max_length=512)


@router.post(
    '/admin/cost/pending/{operation_id}/reconcile',
    status_code=204,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def reconcile_cost(
    operation_id: str,
    body: CostReconciliationRequest,
    admin: CurrentUser = Depends(require_admin),
    service: BudgetAdminService = Depends(create_budget_admin_service),
    log=Depends(create_action_logger),
):
    await service.reconcile(operation_id, str(body.actual_usd), body.evidence)
    log('cost.reconcile', user_id=admin.id, details={'operation_id': operation_id})
