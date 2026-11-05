"""API lifespan must remain available while corpus operations are fenced."""

from __future__ import annotations

import logging.config
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
import pytest

from config import settings
from domain.shared.exceptions import ValidationError
import main


@pytest.fixture
def startup_components(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    listener = SimpleNamespace(start=AsyncMock())
    vector_store = SimpleNamespace(ensure_collection=AsyncMock())
    infrastructure = SimpleNamespace(
        db=SimpleNamespace(uow_factory=Mock()),
        bootstrap_configuration=AsyncMock(),
        corpus_control=SimpleNamespace(is_ready=AsyncMock(return_value=True)),
        events=SimpleNamespace(
            config_listener=listener,
            outbox_listener=SimpleNamespace(start=AsyncMock()),
            outbox_dispatcher=Mock(),
        ),
        services=SimpleNamespace(api_key_provider=Mock()),
        ml=SimpleNamespace(vector_store_repo=vector_store, ml_clients=Mock()),
    )
    container = SimpleNamespace(init=Mock(), infrastructure=infrastructure)
    monkeypatch.setattr(main, "Container", Mock(return_value=container))
    monkeypatch.setattr(main, "redis_client", SimpleNamespace(init=AsyncMock()))
    monkeypatch.setattr(main, "database", SimpleNamespace(connect=AsyncMock()))
    monkeypatch.setattr(main, "scheduler", SimpleNamespace(startup=AsyncMock()))
    monkeypatch.setattr(main, "bootstrap_admin", AsyncMock())
    monkeypatch.setattr(main, "attach_log_buffer", Mock())
    monkeypatch.setattr(logging.config, "dictConfig", Mock())
    monkeypatch.setattr(main, "init_breakers", Mock())
    monkeypatch.setattr(main, "collect_infra_metrics", AsyncMock())
    shutdown = AsyncMock()
    monkeypatch.setattr(main, "shutdown_resources", shutdown)
    return SimpleNamespace(container=container, infrastructure=infrastructure, shutdown=shutdown)


@pytest.mark.asyncio
@pytest.mark.parametrize("corpus_ready", [False, True])
async def test_api_lifespan_continues_when_corpus_is_not_ready(
    startup_components: SimpleNamespace, caplog: pytest.LogCaptureFixture, corpus_ready: bool
) -> None:
    infrastructure = startup_components.infrastructure
    infrastructure.corpus_control.is_ready.return_value = corpus_ready
    ensure = infrastructure.ml.vector_store_repo.ensure_collection

    if not corpus_ready:
        ensure.side_effect = ValidationError("Corpus generation is not ready; retry after synchronization")

    app = FastAPI()
    async with main.lifespan(app):
        assert app.state.container is startup_components.container
        infrastructure.bootstrap_configuration.assert_awaited_once_with(role="api")
        infrastructure.events.config_listener.start.assert_awaited_once()
        infrastructure.events.outbox_listener.start.assert_awaited_once()

        if corpus_ready:
            ensure.assert_awaited_once_with(vector_size=settings.embed_dim, reset=False)
        else:
            ensure.assert_not_awaited()
            assert "Skipping Qdrant collection initialization" in caplog.text

    startup_components.shutdown.assert_awaited_once()
    assert app.state.container is None


@pytest.mark.asyncio
async def test_api_lifespan_does_not_hide_collection_initialization_failure(
    startup_components: SimpleNamespace,
) -> None:
    startup_components.infrastructure.ml.vector_store_repo.ensure_collection.side_effect = RuntimeError(
        "collection initialization failed"
    )
    app = FastAPI()
    with pytest.raises(RuntimeError, match="collection initialization failed"):
        async with main.lifespan(app):
            pytest.fail("failed initialization admitted")

    startup_components.shutdown.assert_awaited_once()
    assert app.state.container is None
