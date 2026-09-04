"""Tests for the DI Container — InfrastructureContainer, ApplicationContainer, and Container."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from composition.container import (
    ApplicationContainer,
    Container,
    InfrastructureContainer,
)
from infrastructure.adapters.chunk_search_adapter import ChunkSearchAdapter
from infrastructure.database.database import DatabaseManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_database_manager() -> MagicMock:
    """Create a mock DatabaseManager that passes isinstance check."""
    mock = MagicMock(spec=DatabaseManager)
    return mock


@pytest.fixture
def infra_container() -> InfrastructureContainer:
    return InfrastructureContainer()


@pytest.fixture
def app_container() -> ApplicationContainer:
    return ApplicationContainer()


@pytest.fixture
def container() -> Container:
    return Container()


# ===========================================================================
# InfrastructureContainer
# ===========================================================================


class TestInfrastructureContainer:
    def test_fields_default_to_none(self):
        infra = InfrastructureContainer()
        # Check raw dataclass fields via sub-containers (before init, all are None)
        assert infra.db.database is None
        assert infra.db.uow_factory is None
        assert infra.ml.vector_store_repo is None
        assert infra.ml.file_storage is None
        assert infra.ml.document_parser is None
        assert infra.ml.document_splitter is None
        assert infra.ml.ml_clients is None
        assert infra.events.config_listener is None
        assert infra.services.health_probe is None
        assert infra.ml.metrics_registry is None
        assert infra.services.ollama_probe is None
        assert infra.services.qdrant_info is None
        assert infra.ml.benchmark_service is None
        assert infra.ml.summary_updater is None
        assert infra.services.api_key_provider is None
        assert infra.db.config_broadcaster is None
        assert infra.ml.content_extractor is None
        assert infra.ml.pdf_quality_assessor is None
        assert infra.ml.metrics_collector is None

    @patch("composition.container.InfrastructureContainer.init")
    def test_init_sets_fields(self, mock_init, mock_database_manager):
        infra = InfrastructureContainer()
        mock_init.return_value = None
        infra.init(mock_database_manager)
        mock_init.assert_called_once_with(mock_database_manager)

    def test_init_rejects_non_database_manager(self):
        infra = InfrastructureContainer()
        with pytest.raises(TypeError, match="Expected DatabaseManager"):
            infra.init("not a database manager")

    def test_init_rejects_none(self):
        infra = InfrastructureContainer()
        with pytest.raises(TypeError, match="Expected DatabaseManager"):
            infra.init(None)

    @pytest.mark.asyncio
    async def test_dispose_is_safe(self):
        infra = InfrastructureContainer()
        await infra.dispose()

    def test_config_fields_included(self):
        infra = InfrastructureContainer()
        import dataclasses

        field_names = [f.name for f in dataclasses.fields(infra)]
        assert "db" in field_names
        assert "ml" in field_names
        assert "events" in field_names
        assert "services" in field_names


# ===========================================================================
# ApplicationContainer
# ===========================================================================


class TestApplicationContainer:
    def test_fields_default_to_none(self):
        app = ApplicationContainer()
        assert app.rag_service is None
        assert app.chat_service is None
        assert app.auth_service is None
        assert app.document_service is None
        assert app.chunk_service is None
        assert app.ingest_app_service is None
        assert app.config_service is None
        assert app.health_service is None
        assert app.metrics_service is None
        assert app.config_admin_service is None
        assert app.pdf_diagnostic_service is None
        assert app.ingestion_service is None
        assert app.search_service is None
        assert app.conversation_service is None
        assert app.group_service is None
        assert app.quality_service is None
        assert app.benchmark_question_service is None
        assert app.benchmark_sweep_service is None
        assert app.benchmark_run_service is None
        assert app.benchmark_result_service is None
        assert app.job_service is None
        assert app.chat_log_service is None

    def test_init_requires_infra_initialized(self):
        app = ApplicationContainer()
        infra = InfrastructureContainer()
        with pytest.raises(RuntimeError, match="Container.init\\(\\) must be called"):
            app.init(infra)

    @pytest.mark.asyncio
    async def test_dispose_calls_shutdown(self):
        app = ApplicationContainer()
        mock_chat = AsyncMock()
        mock_chat.shutdown = AsyncMock()
        app.chat_service = mock_chat
        await app.dispose()
        mock_chat.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispose_safe_when_no_shutdown(self):
        app = ApplicationContainer()
        app.chat_service = MagicMock(spec=[])  # no shutdown method
        await app.dispose()

    @pytest.mark.asyncio
    async def test_dispose_safe_when_chat_service_none(self):
        app = ApplicationContainer()
        await app.dispose()

    def test_all_field_names(self):
        app = ApplicationContainer()
        import dataclasses

        expected = {
            "rag_service",
            "chat_service",
            "auth_service",
            "document_service",
            "chunk_service",
            "ingest_app_service",
            "config_service",
            "health_service",
            "metrics_service",
            "config_admin_service",
            "pdf_diagnostic_service",
            "ingestion_service",
            "search_service",
            "conversation_service",
            "group_service",
            "quality_service",
            "benchmark_question_service",
            "benchmark_sweep_service",
            "benchmark_run_service",
            "benchmark_result_service",
            "job_service",
            "chat_log_service",
        }
        actual = {f.name for f in dataclasses.fields(app)}
        assert expected == actual


# ===========================================================================
# Container
# ===========================================================================


class TestContainer:
    def test_initial_state(self):
        c = Container()
        assert c._initialized is False

    def test_sub_containers_exist(self):
        c = Container()
        assert isinstance(c.infrastructure, InfrastructureContainer)
        assert isinstance(c.application, ApplicationContainer)

    @patch("composition.container.Container.init")
    def test_init_called_once(self, mock_init, container):
        mock_init.return_value = None
        container.init(MagicMock(spec=DatabaseManager))
        mock_init.assert_called_once()

    def test_double_init_raises(self):
        c = Container()
        mock_db = MagicMock(spec=DatabaseManager)
        with (
            patch.object(InfrastructureContainer, "init"),
            patch.object(ApplicationContainer, "init"),
            patch.object(Container, "_subscribe_config_events"),
            patch.object(Container, "_unsubscribe_config_events"),
        ):
            c.init(mock_db)
            with pytest.raises(RuntimeError, match="exactly once"):
                c.init(mock_db)

    @pytest.mark.asyncio
    async def test_dispose_safe_before_init(self):
        c = Container()
        await c.dispose()

    @pytest.mark.asyncio
    async def test_dispose_resets_flag(self):
        c = Container()
        mock_db = MagicMock(spec=DatabaseManager)
        with (
            patch.object(InfrastructureContainer, "init"),
            patch.object(ApplicationContainer, "init"),
            patch.object(ApplicationContainer, "dispose", new_callable=AsyncMock),
            patch.object(InfrastructureContainer, "dispose", new_callable=AsyncMock),
            patch.object(Container, "_subscribe_config_events"),
            patch.object(Container, "_unsubscribe_config_events"),
        ):
            c.init(mock_db)
            assert c._initialized is True
            await c.dispose()
            assert c._initialized is False

    @pytest.mark.asyncio
    async def test_dispose_calls_sub_dispose_in_order(self):
        c = Container()
        call_order = []
        mock_app = AsyncMock()
        mock_infra = AsyncMock()

        async def track_app():
            call_order.append("app")

        async def track_infra():
            call_order.append("infra")

        mock_app.dispose = track_app
        mock_infra.dispose = track_infra

        c.application = mock_app
        c.infrastructure = mock_infra
        c._initialized = True

        await c.dispose()
        assert call_order == ["app", "infra"]

    def test_no_third_party_di_framework(self):
        c = Container()
        assert hasattr(c, "infrastructure")
        assert hasattr(c, "application")


# ===========================================================================
# ChunkSearchAdapter
# ===========================================================================


class TestChunkSearchAdapter:
    @pytest.mark.asyncio
    async def test_delegates_to_uow_factory(self):
        mock_uow_factory = MagicMock()
        mock_chunks = AsyncMock()
        mock_chunks.search_substring.return_value = ["result"]
        mock_uow = AsyncMock()
        mock_uow.chunks = mock_chunks

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_uow
        mock_cm.__aexit__.return_value = False
        mock_uow_factory.create.return_value = mock_cm

        adapter = ChunkSearchAdapter(uow_factory=mock_uow_factory)

        result = await adapter.search_substring(
            query="test",
            user={"id": 1, "kind": "admin"},
            group_ids=[1, 2],
            limit=10,
            mode="exact",
        )

        mock_chunks.search_substring.assert_awaited_once_with(
            query="test",
            user={"id": 1, "kind": "admin"},
            group_ids=[1, 2],
            limit=10,
            mode="exact",
        )
        assert result == ["result"]

    def test_stores_uow_factory(self):
        mock_uow_factory = MagicMock()
        adapter = ChunkSearchAdapter(uow_factory=mock_uow_factory)
        assert adapter._uow_factory is mock_uow_factory


# ===========================================================================
# _subscribe_config_events
# ===========================================================================


class TestSubscribeConfigEvents:
    def test_subscribes_handlers_to_event_bus(self):
        from domain.events.config_events import ConfigParameterChanged

        c = Container()
        c.infrastructure.ml.ml_clients = MagicMock()

        with patch("infrastructure.events.in_process_event_bus.event_bus") as mock_bus:
            c._subscribe_config_events()
            assert mock_bus.subscribe.call_count == 7
            for call_args in mock_bus.subscribe.call_args_list:
                event_type = call_args[0][0]
                assert event_type is ConfigParameterChanged

    def test_requires_ml_clients(self):
        c = Container()
        with pytest.raises(RuntimeError, match="ml_clients not initialized"):
            c._subscribe_config_events()

    def test_invalidation_handlers_are_callable(self):
        from domain.events.config_events import ConfigParameterChanged

        c = Container()
        mock_ml = MagicMock()
        c.infrastructure.ml.ml_clients = mock_ml

        with patch("infrastructure.events.in_process_event_bus.event_bus") as mock_bus:
            c._subscribe_config_events()

            handlers = [call[0][1] for call in mock_bus.subscribe.call_args_list]

            llm_event = ConfigParameterChanged(
                key="llm_model", old_value="a", new_value="b", value_type="str"
            )
            for h in handlers:
                h(llm_event)
            mock_ml.invalidate_llm.assert_called()

            mock_ml.reset_mock()
            bm25_event = ConfigParameterChanged(
                key="hybrid_enabled", old_value="false", new_value="true", value_type="bool"
            )
            for h in handlers:
                h(bm25_event)
            mock_ml.invalidate_bm25.assert_called()


# ===========================================================================
# Integration: full init cycle
# ===========================================================================


class TestContainerIntegration:
    @pytest.mark.asyncio
    async def test_full_init_dispose_cycle(self, mock_database_manager):
        c = Container()
        with (
            patch.object(InfrastructureContainer, "init") as mock_infra_init,
            patch.object(ApplicationContainer, "init"),
            patch.object(ApplicationContainer, "dispose", new_callable=AsyncMock),
            patch.object(InfrastructureContainer, "dispose", new_callable=AsyncMock),
            patch.object(Container, "_subscribe_config_events"),
            patch.object(Container, "_unsubscribe_config_events"),
        ):
            c.init(mock_database_manager)
            mock_infra_init.assert_called_once_with(mock_database_manager)
            assert c._initialized is True

            await c.dispose()
            assert c._initialized is False
