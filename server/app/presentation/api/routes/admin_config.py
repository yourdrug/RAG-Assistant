"""Admin config endpoints — dynamic config management + system info."""

from __future__ import annotations

import logging

import httpx
from application.uow import UnitOfWork
from config import settings
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import get_uow

logger = logging.getLogger("default")

router = APIRouter(tags=["admin-config"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ConfigParamResponse(BaseModel):
    key: str
    value: str
    value_type: str
    category: str
    description: str | None = None
    min_value: float | None = None
    max_value: float | None = None


class ConfigParamUpdateRequest(BaseModel):
    value: str


class ModelsInfoResponse(BaseModel):
    llm_model: str
    embed_model: str
    rerank_model: str
    rerank_device: str
    ocr_engine: str
    ocr_enabled: bool
    ollama_models: list[str] | None = None


class VectorDBCollectionInfo(BaseModel):
    name: str
    points_count: int
    vectors_count: int
    indexed_vectors_count: int
    segments_count: int
    status: str
    optimizer_status: str
    hnsw_m: int | None = None
    hnsw_ef_construct: int | None = None
    on_disk_payload: bool | None = None
    vector_size: int | None = None
    distance: str | None = None


class VectorDBInfoResponse(BaseModel):
    collections: list[VectorDBCollectionInfo]
    active_collection: str
    qdrant_status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DYNAMIC_FIELDS = {
    "retriever_fetch_k": ("retriever_fetch_k", int),
    "retriever_top_k": ("retriever_top_k", int),
    "retriever_fetch_k_broad": ("retriever_fetch_k_broad", int),
    "retriever_top_k_broad": ("retriever_top_k_broad", int),
    "history_window": ("history_window", int),
    "chunk_size": ("chunk_size", int),
    "chunk_overlap": ("chunk_overlap", int),
    "hybrid_enabled": ("hybrid_enabled", bool),
    "bm25_fetch_k": ("bm25_fetch_k", int),
    "rrf_k": ("rrf_k", int),
    "dense_weight": ("dense_weight", float),
    "sparse_weight": ("sparse_weight", float),
    "embed_batch_size": ("embed_batch_size", int),
}


def _apply_config_to_settings(key: str, raw_value: str, value_type: str) -> None:
    """Apply a config value from DB to the in-memory settings object."""
    attr, expected_type = _DYNAMIC_FIELDS.get(key, (key, None))
    if not hasattr(settings, attr):
        return

    try:
        if expected_type is bool:
            setattr(settings, attr, raw_value.lower() in ("true", "1", "yes"))
        elif expected_type is int:
            setattr(settings, attr, int(raw_value))
        elif expected_type is float:
            setattr(settings, attr, float(raw_value))
        else:
            setattr(settings, attr, raw_value)
    except (ValueError, TypeError) as e:
        logger.warning("Failed to apply config %s=%r: %s", key, raw_value, e)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/admin/config", response_model=list[ConfigParamResponse])
async def list_config(
    admin: dict = Depends(require_admin),
    uow: UnitOfWork = Depends(get_uow),
):
    rows = await uow.config_parameters.get_all()
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
    uow: UnitOfWork = Depends(get_uow),
):
    param = await uow.config_parameters.get_by_key(key)
    if param is None:
        raise HTTPException(status_code=404, detail=f"Parameter '{key}' not found")

    # Validate type
    try:
        if param.value_type == "int":
            val = int(body.value)
            if param.min_value is not None and val < param.min_value:
                raise HTTPException(status_code=400, detail=f"Value must be >= {param.min_value}")
            if param.max_value is not None and val > param.max_value:
                raise HTTPException(status_code=400, detail=f"Value must be <= {param.max_value}")
        elif param.value_type == "float":
            val = float(body.value)
            if param.min_value is not None and val < param.min_value:
                raise HTTPException(status_code=400, detail=f"Value must be >= {param.min_value}")
            if param.max_value is not None and val > param.max_value:
                raise HTTPException(status_code=400, detail=f"Value must be <= {param.max_value}")
        elif param.value_type == "bool":
            if body.value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise HTTPException(status_code=400, detail="Value must be a boolean")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid value: {e}")

    await uow.config_parameters.update_value(key, body.value)

    # Apply to in-memory settings immediately
    _apply_config_to_settings(key, body.value, param.value_type)
    logger.info("Config updated: %s = %s", key, body.value)

    param.value = body.value
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
    ollama_models = None
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

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
    qdrant_status = "unknown"

    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
        col_list = client.get_collections()
        qdrant_status = "ok"

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
                        str(vectors_cfg.distance.value) if vectors_cfg and hasattr(vectors_cfg.distance, "value")
                        else str(vectors_cfg.distance) if vectors_cfg
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
