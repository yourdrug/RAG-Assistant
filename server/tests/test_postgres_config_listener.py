"""Tests for PostgresConfigListener."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from domain.events.config_events import ConfigParameterChanged
from infrastructure.events.postgres_config_listener import PostgresConfigListener


def test_on_notify_publishes_event():
    bus = MagicMock()
    factory = MagicMock()
    listener = PostgresConfigListener(bus, factory)

    payload = (
        '{"key": "retriever_top_k", "old_value": "4", "new_value": "8", '
        '"value_type": "int", "changed_by": 1}'
    )
    listener._on_notify(None, None, "config_changed", payload)

    bus.publish.assert_called_once()
    event = bus.publish.call_args[0][0]
    assert isinstance(event, ConfigParameterChanged)
    assert event.key == "retriever_top_k"
    assert event.new_value == "8"
    assert event.old_value == "4"
    assert event.value_type == "int"
    assert event.changed_by == 1


def test_on_notify_malformed_payload_does_not_raise():
    bus = MagicMock()
    factory = MagicMock()
    listener = PostgresConfigListener(bus, factory)
    listener._on_notify(None, None, "config_changed", "not valid json")
    bus.publish.assert_not_called()


def test_on_notify_refetch_payload_skips_direct_publish():
    bus = MagicMock()
    factory = MagicMock()
    listener = PostgresConfigListener(bus, factory)

    payload = '{"key": "chunk_size", "refetch": True}'
    listener._on_notify(None, None, "config_changed", payload)
    bus.publish.assert_not_called()


def test_on_notify_missing_optional_fields():
    bus = MagicMock()
    factory = MagicMock()
    listener = PostgresConfigListener(bus, factory)

    payload = '{"key": "chunk_size", "new_value": "900", "value_type": "int"}'
    listener._on_notify(None, None, "config_changed", payload)

    bus.publish.assert_called_once()
    event = bus.publish.call_args[0][0]
    assert event.old_value is None
    assert event.changed_by is None


@pytest.mark.asyncio
async def test_refetch_and_publish_reads_from_db():
    bus = MagicMock()
    fake_param = MagicMock()
    fake_param.key = "chunk_size"
    fake_param.value = "900"
    fake_param.value_type = "int"

    fake_uow = AsyncMock()
    fake_uow.config_parameters.get_by_key.return_value = fake_param

    fake_factory = MagicMock()
    fake_factory.create.return_value.__aenter__ = AsyncMock(return_value=fake_uow)
    fake_factory.create.return_value.__aexit__ = AsyncMock(return_value=False)

    listener = PostgresConfigListener(bus, fake_factory)
    await listener._refetch_and_publish("chunk_size")

    bus.publish.assert_called_once()
    event = bus.publish.call_args[0][0]
    assert event.key == "chunk_size"
    assert event.new_value == "900"


@pytest.mark.asyncio
async def test_refetch_and_publish_param_not_found():
    bus = MagicMock()
    fake_uow = AsyncMock()
    fake_uow.config_parameters.get_by_key.return_value = None

    fake_factory = MagicMock()
    fake_factory.create.return_value.__aenter__ = AsyncMock(return_value=fake_uow)
    fake_factory.create.return_value.__aexit__ = AsyncMock(return_value=False)

    listener = PostgresConfigListener(bus, fake_factory)
    await listener._refetch_and_publish("nonexistent")

    bus.publish.assert_not_called()


def test_is_connected_false_initially():
    bus = MagicMock()
    factory = MagicMock()
    listener = PostgresConfigListener(bus, factory)
    assert listener.is_connected is False


@pytest.mark.asyncio
async def test_stop_sets_stopped_flag():
    bus = MagicMock()
    factory = MagicMock()
    listener = PostgresConfigListener(bus, factory)
    await listener.stop()
    assert listener._stopped is True


@pytest.mark.asyncio
async def test_resync_publishes_only_changed_params():
    bus = MagicMock()
    fake_rows = [
        MagicMock(key="chunk_size", value="900", value_type="int"),
        MagicMock(key="hybrid_enabled", value="true", value_type="bool"),
    ]

    fake_uow = AsyncMock()
    fake_uow.config_parameters.get_all.return_value = fake_rows

    fake_factory = MagicMock()
    fake_factory.create.return_value.__aenter__ = AsyncMock(return_value=fake_uow)
    fake_factory.create.return_value.__aexit__ = AsyncMock(return_value=False)

    import config

    original_chunk_size = getattr(config.settings, "chunk_size", None)
    original_hybrid = getattr(config.settings, "hybrid_enabled", None)
    try:
        config.settings.chunk_size = 500
        config.settings.hybrid_enabled = False

        listener = PostgresConfigListener(bus, fake_factory)
        await listener.resync(trigger="manual")

        assert bus.publish.call_count == 2
        first_event = bus.publish.call_args_list[0][0][0]
        assert first_event.key == "chunk_size"
        assert first_event.new_value == "900"
        second_event = bus.publish.call_args_list[1][0][0]
        assert second_event.key == "hybrid_enabled"
        assert second_event.new_value == "true"
    finally:
        if original_chunk_size is not None:
            config.settings.chunk_size = original_chunk_size
        if original_hybrid is not None:
            config.settings.hybrid_enabled = original_hybrid


@pytest.mark.asyncio
async def test_resync_skips_unchanged_params():
    bus = MagicMock()
    fake_rows = [
        MagicMock(key="chunk_size", value="500", value_type="int"),
        MagicMock(key="hybrid_enabled", value="false", value_type="bool"),
    ]

    fake_uow = AsyncMock()
    fake_uow.config_parameters.get_all.return_value = fake_rows

    fake_factory = MagicMock()
    fake_factory.create.return_value.__aenter__ = AsyncMock(return_value=fake_uow)
    fake_factory.create.return_value.__aexit__ = AsyncMock(return_value=False)

    import config

    original_chunk_size = getattr(config.settings, "chunk_size", None)
    original_hybrid = getattr(config.settings, "hybrid_enabled", None)
    try:
        config.settings.chunk_size = 500
        config.settings.hybrid_enabled = False

        listener = PostgresConfigListener(bus, fake_factory)
        await listener.resync(trigger="manual")

        bus.publish.assert_not_called()
    finally:
        if original_chunk_size is not None:
            config.settings.chunk_size = original_chunk_size
        if original_hybrid is not None:
            config.settings.hybrid_enabled = original_hybrid


@pytest.mark.asyncio
async def test_resync_logs_failure_without_raising():
    bus = MagicMock()
    fake_factory = MagicMock()
    fake_factory.create.side_effect = RuntimeError("DB down")

    listener = PostgresConfigListener(bus, fake_factory)
    await listener.resync(trigger="periodic")
    bus.publish.assert_not_called()
