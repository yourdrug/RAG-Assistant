"""Tests for the DI Container — lifecycle, wiring, and config event subscription."""

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


@pytest.fixture
def mock_database_manager() -> MagicMock:
    return MagicMock(spec=DatabaseManager)


# ===========================================================================
# InfrastructureContainer — type safety and lifecycle
# ===========================================================================


class TestInfrastructureContainer:
    def test_init_rejects_non_database_manager(self):
        infra = InfrastructureContainer()
        with pytest.raises(TypeError, match="Expected DatabaseManager"):
            infra.init("not a database manager")

    def test_init_rejects_none(self):
        infra = InfrastructureContainer()
        with pytest.raises(TypeError, match="Expected DatabaseManager"):
            infra.init(None)

    @pytest.mark.asyncio
    async def test_dispose_is_safe_before_init(self):
        infra = InfrastructureContainer()
        await infra.dispose()

    def test_has_expected_sub_containers(self):
        infra = InfrastructureContainer()
        assert hasattr(infra, "db")
        assert hasattr(infra, "ml")
        assert hasattr(infra, "events")
        assert hasattr(infra, "services")


# ===========================================================================
# ApplicationContainer — lifecycle
# ===========================================================================


class TestApplicationContainer:
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


# ===========================================================================
# Container — lifecycle and wiring
# ===========================================================================


class TestContainer:
    def test_initial_state(self):
        c = Container()
        assert c._initialized is False

    def test_sub_containers_exist(self):
        c = Container()
        assert isinstance(c.infrastructure, InfrastructureContainer)
        assert isinstance(c.application, ApplicationContainer)

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


# ===========================================================================
# ChunkSearchAdapter — wiring
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

        user = MagicMock(user_id=1, user_kind="admin", user_role="user", group_ids=[1, 2])
        result = await adapter.search_substring(
            query="test",
            user=user,
            limit=10,
            mode="exact",
        )

        mock_chunks.search_substring.assert_awaited_once_with(
            query="test",
            user=user,
            limit=10,
            mode="exact",
        )
        assert result == ["result"]


# ===========================================================================
# _subscribe_config_events — wiring
# ===========================================================================


class TestSubscribeConfigEvents:
    def test_subscribes_handlers_to_event_bus(self):
        from domain.events.config_events import ConfigParameterChanged

        c = Container()
        c.infrastructure.ml.ml_clients = MagicMock()

        with patch("infrastructure.events.in_process_event_bus.event_bus") as mock_bus:
            c._subscribe_config_events()
            assert mock_bus.subscribe.call_count == 9
            for call_args in mock_bus.subscribe.call_args_list:
                event_type = call_args[0][0]
                assert event_type is ConfigParameterChanged

    def test_ml_clients_none_before_init(self):
        c = Container()
        # Before init(), ml_clients is None
        assert c.infrastructure.ml.ml_clients is None

    def test_invalidation_handlers_invalidate_on_event(self):
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
