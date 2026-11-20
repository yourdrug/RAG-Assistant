"""Compatibility and system information behavior across the admin move."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.admin import (
    ConfigAdminService,
    ConfigService,
    ConfigurationStatus,
    LLMModelsInfo,
    VectorDBCollectionInfo,
)
from domain.shared.exceptions import ValidationError


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.admin", "application.admin.config_admin_service", "ConfigAdminService"),
        ("application.admin", "application.admin.dto", "ModelsInfo"),
        ("application.admin", "application.admin.dto", "VectorDBInfo"),
        ("application.admin", "application.admin.dto", "VectorDBCollectionInfo"),
        ("application.admin", "application.admin.dto", "LLMModelsInfo"),
        ("application.admin", "application.admin.config_service", "ConfigService"),
        ("application.admin", "application.admin.config_service", "validate_corpus_update_key"),
        ("application.admin", "application.admin.ports.log_buffer", "LogBufferPort"),
        ("application.admin", "application.admin.ports", "VectorDBInfoPort"),
        ("application.admin", "application.admin.ports.settings", "ConfigAdminSettingsPort"),
        ("application.admin", "application.admin.ports", "ConfigMaskerPort"),
        ("application.admin", "application.admin.ports", "CorpusControlPort"),
        ("application.admin", "application.admin.dto", "ConfigurationStatus"),
        ("application.admin", "application.admin.dto", "EffectiveParameter"),
        ("application.admin", "application.admin.ports", "ProviderConfigurationPort"),
    ],
)
def test_public_exports_share_canonical_objects(public: str, implementation: str, name: str) -> None:
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )


@pytest.mark.asyncio
async def test_models_info_reports_configured_endpoints() -> None:
    settings = SimpleNamespace(
        llm_base_url="https://openrouter.ai/api/v1",
        llm_model="router-model",
        llm_auxiliary_model="router-model",
        llm_api_key_configured=True,
        embedding_base_url="http://embed/v1",
        embedding_model="embed-model",
        embedding_api_key_configured=False,
        rerank_base_url="http://rerank",
        rerank_model="rerank-model",
        rerank_api_format="tei",
        rerank_api_key_configured=True,
        ocr_enabled=True,
    )
    info = await ConfigAdminService(AsyncMock(), settings).get_models_info()
    assert info.llm_base_url == "https://openrouter.ai/api/v1"
    assert info.llm_model == "router-model"
    assert info.llm_api_key_configured
    assert info.embedding_base_url == "http://embed/v1"
    assert info.embed_model == "embed-model"
    assert not info.embedding_api_key_configured
    assert info.rerank_model == "rerank-model"
    assert info.rerank_api_format == "tei"
    assert info.rerank_api_key_configured
    assert info.ocr_enabled


@pytest.mark.asyncio
async def test_vectordb_defaults_and_collection_metadata() -> None:
    probe = AsyncMock()
    probe.get_status.return_value = "ok"
    probe.get_collections.return_value = [
        {"name": "empty"},
        {"name": "active", "points_count": 12, "hnsw_m": 16, "on_disk_payload": False},
    ]
    info = await ConfigAdminService(probe, SimpleNamespace(collection_name="active")).get_vectordb_info()
    assert info.qdrant_status == "ok"
    assert info.active_collection == "active"
    assert info.collections == [
        VectorDBCollectionInfo(name="empty"),
        VectorDBCollectionInfo(name="active", points_count=12, hnsw_m=16, on_disk_payload=False),
    ]


@pytest.mark.asyncio
async def test_unhealthy_vectordb_skips_collection_fetch() -> None:
    probe = AsyncMock()
    probe.get_status.return_value = "error: unavailable"
    info = await ConfigAdminService(probe, SimpleNamespace(collection_name="active")).get_vectordb_info()
    assert info.qdrant_status == "error: unavailable"
    assert info.collections == []
    probe.get_collections.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_models_fetcher_is_optional_and_preserves_payload() -> None:
    settings = SimpleNamespace(llm_model="active")
    service = ConfigAdminService(AsyncMock(), settings)
    assert await service.get_llm_models() == LLMModelsInfo()
    fetcher = AsyncMock(return_value=[{"id": "m", "pricing": {"prompt": 0.5}}])
    service = ConfigAdminService(AsyncMock(), settings, fetcher)
    assert await service.get_llm_models() == LLMModelsInfo(fetcher.return_value, "active")
    fetcher.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_llm_models_fetch_failure_propagates() -> None:
    fetcher = AsyncMock(side_effect=RuntimeError("unavailable"))
    service = ConfigAdminService(AsyncMock(), SimpleNamespace(), fetcher)
    with pytest.raises(RuntimeError, match="unavailable"):
        await service.get_llm_models()


@pytest.mark.asyncio
async def test_uninitialized_configuration_status_and_effective_parameters() -> None:
    service = ConfigService(MagicMock(), MagicMock())
    assert await service.configuration_status() == ConfigurationStatus(
        saved_version=None, applied_version=None, ready=False, error="Configuration not initialized"
    )
    with pytest.raises(ValidationError, match="Configuration not initialized"):
        await service.effective_parameters()
