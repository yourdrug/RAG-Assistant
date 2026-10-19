"""Compatibility and system information behavior across the admin move."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.admin import (
    ConfigAdminService,
    ConfigService,
    ConfigurationStatus,
    OpenRouterModelsInfo,
    VectorDBCollectionInfo,
)
from domain.exceptions import ValidationError


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.admin", "application.admin.config_admin_service", "ConfigAdminService"),
        ("application.admin", "application.admin.dto", "ModelsInfo"),
        ("application.admin", "application.admin.dto", "VectorDBInfo"),
        ("application.admin", "application.admin.dto", "VectorDBCollectionInfo"),
        ("application.admin", "application.admin.dto", "OpenRouterModelsInfo"),
        ("application.admin", "application.admin.config_service", "ConfigService"),
        ("application.admin", "application.admin.config_service", "validate_corpus_update_key"),
        ("application.admin", "application.admin.log_buffer", "LogBufferPort"),
        ("application.admin", "application.admin.ports", "OllamaProbePort"),
        ("application.admin", "application.admin.ports", "VectorDBInfoPort"),
        ("application.admin", "application.admin.settings", "ConfigAdminSettingsPort"),
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
@pytest.mark.parametrize("ml_provider", ["tei", "deepinfra"])
@pytest.mark.parametrize("llm_provider", ["ollama", "openrouter"])
async def test_models_info_preserves_provider_selection(ml_provider: str, llm_provider: str) -> None:
    settings = SimpleNamespace(
        ml_provider=ml_provider,
        llm_provider=llm_provider,
        llm_model="llm",
        tei_embed_url="http://embed",
        tei_rerank_url="http://rerank",
        deepinfra_embed_model="embed-model",
        deepinfra_rerank_model="rerank-model",
        ocr_engine="surya",
        ocr_enabled=True,
        openrouter_model="router-model",
    )
    ollama = AsyncMock()
    ollama.get_models.return_value = ["local-model"]
    info = await ConfigAdminService(ollama, AsyncMock(), settings).get_models_info()
    assert info.ollama_models == ["local-model"]
    assert info.llm_provider == llm_provider
    assert info.llm_model == "llm"
    assert info.ml_provider == ml_provider
    assert info.embed_model == ("embed-model" if ml_provider == "deepinfra" else "http://embed")
    assert info.rerank_model == ("rerank-model" if ml_provider == "deepinfra" else "http://rerank")
    assert info.openrouter_model == ("router-model" if llm_provider == "openrouter" else None)
    assert info.ocr_enabled
    assert info.ocr_engine == "surya"


@pytest.mark.asyncio
async def test_vectordb_defaults_and_collection_metadata() -> None:
    probe = AsyncMock()
    probe.get_status.return_value = "ok"
    probe.get_collections.return_value = [
        {"name": "empty"},
        {"name": "active", "points_count": 12, "hnsw_m": 16, "on_disk_payload": False},
    ]
    info = await ConfigAdminService(
        AsyncMock(), probe, SimpleNamespace(collection_name="active")
    ).get_vectordb_info()
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
    info = await ConfigAdminService(
        AsyncMock(), probe, SimpleNamespace(collection_name="active")
    ).get_vectordb_info()
    assert info.qdrant_status == "error: unavailable"
    assert info.collections == []
    probe.get_collections.assert_not_awaited()


@pytest.mark.asyncio
async def test_openrouter_fetcher_is_optional_and_preserves_payload() -> None:
    settings = SimpleNamespace(openrouter_model="active")
    service = ConfigAdminService(AsyncMock(), AsyncMock(), settings)
    assert await service.get_openrouter_models() == OpenRouterModelsInfo()
    fetcher = AsyncMock(return_value=[{"id": "m", "pricing": {"prompt": 0.5}}])
    service = ConfigAdminService(AsyncMock(), AsyncMock(), settings, fetcher)
    assert await service.get_openrouter_models() == OpenRouterModelsInfo(fetcher.return_value, "active")
    fetcher.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_openrouter_fetch_failure_propagates() -> None:
    fetcher = AsyncMock(side_effect=RuntimeError("unavailable"))
    service = ConfigAdminService(AsyncMock(), AsyncMock(), SimpleNamespace(), fetcher)
    with pytest.raises(RuntimeError, match="unavailable"):
        await service.get_openrouter_models()


@pytest.mark.asyncio
async def test_uninitialized_configuration_status_and_effective_parameters() -> None:
    service = ConfigService(MagicMock(), MagicMock())
    assert await service.configuration_status() == ConfigurationStatus(
        saved_version=None, applied_version=None, ready=False, error="Configuration not initialized"
    )
    with pytest.raises(ValidationError, match="Configuration not initialized"):
        await service.effective_parameters()
