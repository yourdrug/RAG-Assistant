"""Tests for the DI Container — lifecycle, wiring, and config event subscription."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest

from application.runtime.config import ConfigParameter
from config import Settings
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.pack_metadata import Missing
from tests.fakes import FakeUnitOfWorkFactory, require_present


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


from application.runtime.config import ConfigParameterChanged, ConfigValueType
from application.spaces import DomainPackNotLoadedError
from composition.application import ApplicationContainer
from composition.container import Container
from composition.domain_packs import NEUTRAL_WIRING, LoadedPack, PackWiring
from composition.infrastructure import InfrastructureContainer
from composition.pack_catalog import LoadedPackCatalog
from composition.service_providers import create_pack_hooks
from composition.utils import ContainerNotInitializedError
from config import settings
from domain.identity.roles import UserKind
from domain.identity.user_context import UserContext
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.knowledge.space import SpaceRef
from domain.rag.context import RagContext
from infrastructure.adapters.chunk_search_adapter import ChunkSearchAdapter
from infrastructure.database.database import DatabaseManager
from infrastructure.domain_profile.packs import NEUTRAL_PACK, NEUTRAL_PACK_KEY
from infrastructure.events.in_process_event_bus import InProcessEventBus
from infrastructure.events.postgres_config_listener import PostgresConfigListener
from infrastructure.ml.config.config_subscribers import audit_log_config_change
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration


if TYPE_CHECKING:
    from infrastructure.ml.rag.rag_steps import RagOperations


@pytest.fixture
def mock_database_manager() -> MagicMock:
    return MagicMock(spec=DatabaseManager)


# ===========================================================================
# InfrastructureContainer — type safety and lifecycle
# ===========================================================================


class TestInfrastructureContainer:
    def test_init_rejects_non_database_manager(self) -> None:
        infra = InfrastructureContainer()
        with pytest.raises(TypeError, match="Expected DatabaseManager"):
            infra.init(cast("DatabaseManager", "not a database manager"))

    def test_init_rejects_none(self) -> None:
        infra = InfrastructureContainer()
        with pytest.raises(TypeError, match="Expected DatabaseManager"):
            infra.init(cast("DatabaseManager", None))

    @pytest.mark.asyncio
    async def test_dispose_is_safe_before_init(self) -> None:
        infra = InfrastructureContainer()
        await infra.dispose()

    def test_has_expected_sub_containers(self) -> None:
        infra = InfrastructureContainer()
        assert hasattr(infra, "db")
        assert hasattr(infra, "ml")
        assert hasattr(infra, "events")
        assert hasattr(infra, "services")


# ===========================================================================
# ApplicationContainer — lifecycle
# ===========================================================================


class TestApplicationContainer:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("update_source", ["hot_reload", "db_resync"])
    async def test_shared_pii_redactor_uses_config_loaded_after_init(  # noqa: PLR0915 - One scenario verifies the complete transaction/lifecycle.
        self, monkeypatch: pytest.MonkeyPatch, update_source: str
    ) -> None:
        monkeypatch.setattr(settings, "pii_redaction_enabled", False)
        bus = InProcessEventBus()
        monkeypatch.setattr("infrastructure.events.in_process_event_bus.event_bus", bus)
        container = Container()
        container.infrastructure = MagicMock()
        container.infrastructure.pack_catalog = neutral_catalog()
        with patch("composition.application.create_ingestion_service", return_value=MagicMock()):
            container.application.init(container.infrastructure)

        configuration = SettingsProviderConfiguration(
            Settings(
                _env_file=None,
                embedding_base_url="http://tei-embed:8080",
                rerank_base_url="http://tei-rerank:8080",
            ),
            bus=bus,
        )
        # LivePIIRedactor consumes the process singleton, so apply to that instance.
        configuration.settings = settings
        monkeypatch.setattr(settings, "embedding_base_url", "http://tei-embed:8080")
        monkeypatch.setattr(settings, "rerank_base_url", "http://tei-rerank:8080")
        container.infrastructure.configuration = configuration

        chat_redactor = require_present(container.application.chat_service)._pii_redactor
        neutral_space = RagContext(UserContext(1, UserKind.INTERNAL), space=SpaceRef(1, NEUTRAL_PACK_KEY))
        preparation = require_present(container.application.rag_service).service(neutral_space).preparation
        rag_redactor = require_present(cast("RagOperations", preparation).pii_redactor)
        assert chat_redactor is rag_redactor
        text = "Контакт: private@example.com"
        assert chat_redactor.redact(text) == text
        assert rag_redactor.scan_and_redact(text) == (text, [])

        row = ConfigParameter("pii_redaction_enabled", "true", ConfigValueType.BOOL.value, "toggles")

        if update_source == "hot_reload":
            configuration.apply([row])
        else:
            uow = AsyncMock()
            uow.config_parameters.get_all.return_value = [row]
            factory = container.infrastructure.db.uow_factory
            factory.create.return_value.__aenter__ = AsyncMock(return_value=uow)
            factory.create.return_value.__aexit__ = AsyncMock(return_value=False)
            await PostgresConfigListener(factory, configuration=configuration, database=MagicMock()).resync(
                trigger="reconnect"
            )

        assert settings.pii_redaction_enabled is True
        assert chat_redactor.redact(text) == "Контакт: ***"
        assert rag_redactor.scan_and_redact(text) == ("Контакт: ***", ["email"])

    def test_init_requires_infra_initialized(self) -> None:
        app = ApplicationContainer()
        infra = InfrastructureContainer()
        with pytest.raises(RuntimeError, match=r"Container.init\(\) must be called"):
            app.init(infra)

    @pytest.mark.asyncio
    async def test_dispose_calls_shutdown(self) -> None:
        app = ApplicationContainer()
        mock_chat = AsyncMock()
        mock_chat.shutdown = AsyncMock()
        app.chat_service = mock_chat
        await app.dispose()
        mock_chat.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispose_safe_when_no_shutdown(self) -> None:
        app = ApplicationContainer()
        app.chat_service = MagicMock(spec=[])  # no shutdown method
        await app.dispose()

    @pytest.mark.asyncio
    async def test_dispose_safe_when_chat_service_none(self) -> None:
        app = ApplicationContainer()
        await app.dispose()


# ===========================================================================
# Container — lifecycle and wiring
# ===========================================================================


class TestContainer:
    def test_initial_state(self) -> None:
        c = Container()
        assert c._initialized is False

    def test_sub_containers_exist(self) -> None:
        c = Container()
        assert isinstance(c.infrastructure, InfrastructureContainer)
        assert isinstance(c.application, ApplicationContainer)

    def test_double_init_raises(self) -> None:
        c = Container()
        mock_db = MagicMock(spec=DatabaseManager)
        with (
            patch.object(InfrastructureContainer, "init"),
            patch.object(ApplicationContainer, "init"),
            patch.object(Container, "subscribe_config_events"),
            patch.object(Container, "unsubscribe_config_events"),
        ):
            c.init(mock_db)
            with pytest.raises(RuntimeError, match="exactly once"):
                c.init(mock_db)

    @pytest.mark.asyncio
    async def test_dispose_safe_before_init(self) -> None:
        c = Container()
        await c.dispose()

    @pytest.mark.asyncio
    async def test_dispose_resets_flag(self) -> None:
        c = Container()
        mock_db = MagicMock(spec=DatabaseManager)
        with (
            patch.object(InfrastructureContainer, "init"),
            patch.object(ApplicationContainer, "init"),
            patch.object(ApplicationContainer, "dispose", new_callable=AsyncMock),
            patch.object(InfrastructureContainer, "dispose", new_callable=AsyncMock),
            patch.object(Container, "subscribe_config_events"),
            patch.object(Container, "unsubscribe_config_events"),
        ):
            c.init(mock_db)
            assert c._initialized is True
            await c.dispose()
            assert c._initialized is False

    @pytest.mark.asyncio
    async def test_dispose_calls_sub_dispose_in_order(self) -> None:
        c = Container()
        call_order = []
        mock_app = AsyncMock()
        mock_infra = AsyncMock()
        mock_infra.unsubscribe_config_events = MagicMock()

        async def track_app() -> None:
            call_order.append("app")

        async def track_infra() -> None:
            call_order.append("infra")

        mock_app.dispose = track_app
        mock_infra.dispose = track_infra

        c.application = mock_app
        c.infrastructure = mock_infra
        c._initialized = True

        await c.dispose()
        assert call_order == ["app", "infra"]

    @pytest.mark.asyncio
    async def test_full_init_dispose_cycle(self, mock_database_manager: MagicMock) -> None:
        c = Container()
        with (
            patch.object(InfrastructureContainer, "init") as mock_infra_init,
            patch.object(ApplicationContainer, "init"),
            patch.object(ApplicationContainer, "dispose", new_callable=AsyncMock),
            patch.object(InfrastructureContainer, "dispose", new_callable=AsyncMock),
            patch.object(Container, "subscribe_config_events"),
            patch.object(Container, "unsubscribe_config_events"),
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
    @pytest.mark.parametrize("selection", [None, Missing("superseded")])
    async def test_delegates_to_uow_factory(self, selection: Any) -> None:
        mock_uow_factory = MagicMock()
        mock_chunks = AsyncMock()
        matches: list[ChunkRecord] = [MagicMock(spec=ChunkRecord)]
        mock_chunks.search_substring.return_value = matches
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
            selection=selection,
        )

        mock_chunks.search_substring.assert_awaited_once_with(
            query="test",
            user=user,
            limit=10,
            mode="exact",
            selection=selection,
            space_id=None,
            document_id=None,
            document_ids=None,
        )
        assert result == matches


# ===========================================================================
# subscribe_config_events — wiring
# ===========================================================================


class TestSubscribeConfigEvents:
    def test_subscribes_handlers_to_event_bus(self) -> None:
        c = Container()
        c.infrastructure.ml.ml_clients = MagicMock()

        with patch("composition.infrastructure.event_bus") as mock_bus:
            c.subscribe_config_events()
            assert mock_bus.subscribe.call_args_list == [
                call(ConfigParameterChanged, c.infrastructure.request_configuration_sync),
                call(ConfigParameterChanged, c.infrastructure.invalidate_config_clients),
                call(ConfigParameterChanged, audit_log_config_change),
            ]
            c.subscribe_config_events()
            assert mock_bus.subscribe.call_count == 3
            c.unsubscribe_config_events()
            assert mock_bus.unsubscribe.call_args_list == mock_bus.subscribe.call_args_list

    def test_ml_clients_none_before_init(self) -> None:
        c = Container()
        # Before init(), ml_clients is None
        assert c.infrastructure.ml.ml_clients is None

    def test_invalidation_handlers_invalidate_on_event(self) -> None:
        c = Container()
        mock_ml = MagicMock()
        c.infrastructure.ml.ml_clients = mock_ml
        c.infrastructure.configuration = MagicMock(applying=True, switching_clients=False)

        with patch("composition.infrastructure.event_bus") as mock_bus:
            c.subscribe_config_events()

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

            mock_ml.invalidate_bm25.assert_not_called()


def neutral_catalog(**packs: LoadedPack) -> LoadedPackCatalog:
    loaded = {NEUTRAL_PACK_KEY: LoadedPack(NEUTRAL_PACK, NEUTRAL_WIRING), **packs}
    return LoadedPackCatalog(
        loaded,
        {key: DomainProfileRegistry() for key in loaded},
        uow_factory=FakeUnitOfWorkFactory(),
        domain_settings=MagicMock(),
        cache_invalidator=None,
    )


def test_pack_document_hooks_are_built_with_their_pack_profiles() -> None:
    hooks = MagicMock()
    law = replace(NEUTRAL_PACK, key="law")
    catalog = neutral_catalog(law=LoadedPack(law, PackWiring(document_hooks=hooks)))
    uow = FakeUnitOfWorkFactory()

    assert catalog.build_hooks("law", uow) is hooks.return_value
    hooks.assert_called_once_with(uow, catalog.domain_registry("law"), ANY, None)


def test_neutral_pack_has_no_document_hooks() -> None:
    assert neutral_catalog().build_hooks(NEUTRAL_PACK_KEY, FakeUnitOfWorkFactory()) is None


def test_space_cannot_use_a_pack_that_is_not_loaded() -> None:
    with pytest.raises(DomainPackNotLoadedError):
        neutral_catalog().document_hooks("belarus_law")


def test_create_pack_hooks_requires_an_initialized_catalog() -> None:
    with pytest.raises(ContainerNotInitializedError, match="pack_catalog not initialized"):
        create_pack_hooks(InfrastructureContainer(), FakeUnitOfWorkFactory())
