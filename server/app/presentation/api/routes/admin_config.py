"""Admin config endpoints — dynamic config management + system info."""

from __future__ import annotations

from decimal import Decimal
import logging
from typing import Annotated, Any, Literal

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
from application.budget import BudgetAdminService
from application.budget.dto import BudgetUsageRows, CostRateRow
from application.ports.action_logger import ActionLoggerPort
from application.ports.rate_limit import RateLimitPolicyName
from application.runtime.dynamic_params import DYNAMIC_FIELDS_MAP
from application.runtime.enums import MLProvider
from application.runtime.setting_keys import SECRET_SETTING_KEYS
from config import CostLimitSection, MLProviderSection, process_section
from domain.runtime.cost import CostRate
from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import QUESTION_LOG_MAX_CHARS, STATIC_CONFIG_KEYS
from presentation.api.dependencies import (
    create_action_logger,
    create_budget_admin_service,
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
        llm_provider=info.llm_provider,
        llm_model=info.llm_model,
        tei_embed_url=info.tei_embed_url,
        tei_rerank_url=info.tei_rerank_url,
        embed_model=info.embed_model,
        rerank_model=info.rerank_model,
        ocr_engine=info.ocr_engine,
        ocr_enabled=info.ocr_enabled,
        openrouter_model=info.openrouter_model,
        ml_provider=info.ml_provider,
    )


@router.get("/admin/models/openrouter", response_model=OpenRouterModelsResponse)
async def openrouter_models(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    admin_service: Annotated[ConfigAdminService, Depends(create_config_admin_service)],
) -> OpenRouterModelsResponse:
    info = await admin_service.get_openrouter_models()
    return OpenRouterModelsResponse(
        models=[OpenRouterModelInfo(**m) for m in info.models],
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


class CostRateResponse(BaseModel):
    provider: Literal["openrouter", "deepinfra"]
    model: str = Field(min_length=1, max_length=200)
    version: int = Field(ge=1)
    unit: Literal["tokens"] = "tokens"
    input_usd_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    output_usd_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    context_tokens: int = Field(ge=1, le=10_000_000)


class CostRateRequest(CostRateResponse):
    input_usd_per_million: Decimal = Field(gt=0, allow_inf_nan=False)


@router.get("/admin/cost/rates", response_model=list[CostRateResponse])
async def cost_rates(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BudgetAdminService, Depends(create_budget_admin_service)],
) -> list[CostRateRow]:
    return await service.rates()


@router.post(
    "/admin/cost/rates",
    status_code=201,
    response_model=CostRateRequest,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def add_cost_rate(
    *,
    body: CostRateRequest,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BudgetAdminService, Depends(create_budget_admin_service)],
) -> CostRate:
    return await service.add_rate(
        CostRate(**body.model_dump(), input_only=body.provider == MLProvider.DEEPINFRA.value)
    )


@router.get("/admin/cost/usage", response_model=None)
async def recent_cost_usage(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BudgetAdminService, Depends(create_budget_admin_service)],
) -> BudgetUsageRows:
    return await service.usage()


@router.get("/admin/cost/models", response_model=None)
async def active_cost_models(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BudgetAdminService, Depends(create_budget_admin_service)],
) -> dict[str, object]:
    ml_provider_settings = process_section(MLProviderSection)

    models = []

    if ml_provider_settings.ml_provider == MLProvider.DEEPINFRA.value:
        models.extend(
            [
                await service.model_status(MLProvider.DEEPINFRA.value, model)
                for model in (
                    ml_provider_settings.deepinfra_embed_model,
                    ml_provider_settings.deepinfra_rerank_model,
                )
            ]
        )

    return {"enabled": process_section(CostLimitSection).cost_rate_limit_enabled, "models": models}


@router.get("/admin/cost/pending", response_model=None)
async def pending_cost_usage(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BudgetAdminService, Depends(create_budget_admin_service)],
) -> BudgetUsageRows:
    return await service.pending()


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


class CostReconciliationRequest(BaseModel):
    actual_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    evidence: str = Field(min_length=1, max_length=512)


@router.post(
    "/admin/cost/pending/{operation_id}/reconcile",
    status_code=204,
    response_model=None,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def reconcile_cost(
    *,
    operation_id: str,
    body: CostReconciliationRequest,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[BudgetAdminService, Depends(create_budget_admin_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> None:
    await service.reconcile(operation_id, str(body.actual_usd), body.evidence)
    log("cost.reconcile", user_id=admin.id, details={"operation_id": operation_id})
