"""Storage rotation and container teardown preserve in-flight requests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Never
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.runtime.config import ConfigParameter, ConfigParameterChanged, ConfigValueType
from composition.infrastructure import InfrastructureContainer
from config import settings
from infrastructure.events.in_process_event_bus import InProcessEventBus
from infrastructure.events.postgres_config_listener import PostgresConfigListener
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
from infrastructure.storage.file_storage import LazyStorage
from tests.fakes import FakeUnitOfWorkFactory
from tests.s3_fakes import Body
from tests.test_s3_storage_async import client_harness


if TYPE_CHECKING:
    from types import SimpleNamespace

    from infrastructure.storage.file_storage import S3Storage


def backend_with_credentials(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, access_key: str, secret_key: str
) -> tuple[S3Storage, MagicMock, SimpleNamespace, MagicMock]:
    monkeypatch.setattr(settings, "s3_endpoint", endpoint)
    monkeypatch.setattr(settings, "s3_access_key", access_key)
    monkeypatch.setattr(settings, "s3_secret_key", secret_key)
    return client_harness(monkeypatch)


def assert_backend_credentials(session: MagicMock, endpoint: str, access_key: str, secret_key: str) -> None:
    kwargs = session.create_client.call_args.kwargs
    assert kwargs["endpoint_url"] == endpoint
    assert kwargs["aws_access_key_id"] == access_key
    assert kwargs["aws_secret_access_key"] == secret_key


@pytest.mark.asyncio
async def test_rotation_retains_active_backend_and_closes_all(monkeypatch: pytest.MonkeyPatch) -> None:
    first, first_client, first_context, first_session = backend_with_credentials(
        monkeypatch, "http://old-endpoint:9000", "old-test-access", "old-test-secret"
    )
    body = Body([b"old"])
    body.release = asyncio.Event()
    first_client.get_object.return_value = {"Body": body}
    await first.upload_file("init", b"x")
    second, second_client, second_context, second_session = backend_with_credentials(
        monkeypatch, "http://new-endpoint:9000", "new-test-access", "new-test-secret"
    )
    second.bucket = "new-bucket"
    backends = iter([first, second])
    monkeypatch.setattr("infrastructure.storage.file_storage.create_storage", lambda: next(backends))
    storage = LazyStorage()
    pending = asyncio.create_task(storage.download_to_temp("a.txt"))
    await asyncio.wait_for(body.started.wait(), 5)
    storage.invalidate()
    await storage.upload_file("b.txt", b"new")
    second_client.put_object.assert_awaited_once_with(Bucket="new-bucket", Key="b.txt", Body=b"new")

    for session, endpoint, access_key, secret_key in (
        (first_session, "http://old-endpoint:9000", "old-test-access", "old-test-secret"),
        (second_session, "http://new-endpoint:9000", "new-test-access", "new-test-secret"),
    ):
        assert_backend_credentials(session, endpoint, access_key, secret_key)

    closing = asyncio.create_task(storage.aclose())
    await asyncio.sleep(0)
    first_context.__aexit__.assert_not_awaited()
    body.release.set()
    path = await pending
    assert path.read_bytes() == b"old"
    path.unlink()
    await closing
    await storage.aclose()
    first_context.__aexit__.assert_awaited_once()
    second_context.__aexit__.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closed"):
        await storage.upload_file("late", b"x")


@pytest.mark.asyncio
async def test_close_error_does_not_skip_other_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    backends = [AsyncMock(), AsyncMock()]
    backends[0].aclose.side_effect = OSError("close failure")
    choices = iter(backends)
    monkeypatch.setattr("infrastructure.storage.file_storage.create_storage", lambda: next(choices))
    storage = LazyStorage()
    await storage.upload_file("first", b"x")
    storage.invalidate()
    await storage.upload_file("second", b"x")
    with pytest.raises(OSError, match="close failure"):
        await storage.aclose()

    for backend in backends:
        backend.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_unused_lazy_storage_close_does_not_create_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> Never:
        pytest.fail("shutdown created a backend")

    monkeypatch.setattr("infrastructure.storage.file_storage.create_storage", forbidden)
    await LazyStorage().aclose()


@pytest.mark.asyncio
async def test_s3_config_is_snapshotted_per_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    storage, client, _, session = client_harness(monkeypatch)
    monkeypatch.setattr(settings, "s3_bucket", "changed")
    monkeypatch.setattr(settings, "s3_region", "changed-region")
    await storage.upload_file("a", b"x")
    client.put_object.assert_awaited_once_with(Bucket="bucket", Key="a", Body=b"x")
    assert session.create_client.call_args.kwargs["region_name"] != "changed-region"
    config = session.create_client.call_args.kwargs["config"]
    assert config.max_pool_connections == 2
    assert config.s3["addressing_style"] == "path"
    await storage.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["s3_bucket", "s3_endpoint", "s3_access_key", "s3_secret_key", "s3_region"])
async def test_legacy_config_event_preserves_storage_and_bm25_reader(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.setattr(settings, key, "old")
    storage = MagicMock()
    clients = MagicMock()
    event = ConfigParameterChanged(key=key, old_value="old", new_value="new", value_type=ConfigValueType.STR)
    bus = InProcessEventBus()
    monkeypatch.setattr("composition.infrastructure.event_bus", bus)
    factory = FakeUnitOfWorkFactory()
    await factory._uow.config_parameters.save(ConfigParameter(key, "new", ConfigValueType.STR, "legacy"))
    configuration = SettingsProviderConfiguration(settings, bus=bus)
    infra = InfrastructureContainer(configuration=configuration)
    infra.ml.file_storage = storage
    infra.ml.ml_clients = clients
    listener = PostgresConfigListener(factory, configuration, database=MagicMock())
    infra.events.config_listener = listener
    infra.subscribe_config_events()
    try:
        bus.publish(event)
        await asyncio.gather(*listener.transport.background_tasks)
        assert configuration.status().ready
        assert getattr(settings, key) == "old"
    finally:
        infra.unsubscribe_config_events()
        await listener.stop()
    storage.invalidate.assert_not_called()
    clients.invalidate_bm25.assert_not_called()
