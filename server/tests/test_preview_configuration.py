"""DB preview settings reach existing consumers without restarting them."""

from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

from fastapi import HTTPException, UploadFile
import pytest

from application.admin import ConfigService
from application.documents.diagnostics import PDFDiagnosticService
from application.documents.preview_cache import PreviewCache
from application.ports.file_storage import FileStorage
from config import Settings
from domain.runtime.config import ConfigParameter
from domain.shared.exceptions import ValidationError
from infrastructure.events.in_process_event_bus import InProcessEventBus
from infrastructure.ml.config import settings_adapters
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
from infrastructure.ml.config.settings_adapters import LivePreviewSettings
from presentation.api.helpers import read_upload_with_limit


type PreviewConfiguration = tuple[Settings, SettingsProviderConfiguration, LivePreviewSettings]


@pytest.fixture
def preview_configuration(monkeypatch: pytest.MonkeyPatch) -> PreviewConfiguration:
    model = Settings(preview_max_size_mb=2, preview_cache_ttl_seconds=100)
    monkeypatch.setattr(settings_adapters, "settings", model)
    configuration = SettingsProviderConfiguration(model, bus=InProcessEventBus())
    return model, configuration, LivePreviewSettings()


@pytest.mark.asyncio
async def test_admin_update_changes_existing_preview_consumers(
    preview_configuration: PreviewConfiguration,
    fake_uow_factory: Any,
) -> None:
    model, configuration, live = preview_configuration
    factory = fake_uow_factory

    for key, value in (("preview_max_size_mb", "2"), ("preview_cache_ttl_seconds", "100")):
        await factory._uow.config_parameters.save(ConfigParameter(key, value, "int", "ingestion"))

    service = ConfigService(factory, InProcessEventBus(), configuration)
    diagnostic = PDFDiagnosticService(Mock(), Mock(), Mock(), Mock(), Mock(), settings=live)
    configuration.apply(await factory._uow.config_parameters.get_all())
    listed = {row.key: row for row in await service.list_parameters()}
    assert {"preview_max_size_mb", "preview_cache_ttl_seconds"} <= listed.keys()
    await service.update_parameter("preview_max_size_mb", "1")
    await service.update_parameter("preview_cache_ttl_seconds", "5")
    effective = {row.key: row for row in await service.effective_parameters()}
    assert effective["preview_max_size_mb"].configured_value == "1"
    assert effective["preview_max_size_mb"].applied_value == "2"
    assert effective["preview_max_size_mb"].editable
    configuration.apply(await factory._uow.config_parameters.get_all())
    assert live.preview_cache_ttl_seconds == 5
    data = b"x" * (1024 * 1024 + 1)
    with pytest.raises(HTTPException) as error:
        await read_upload_with_limit(UploadFile(BytesIO(data)), diagnostic.max_preview_bytes)
    assert error.value.status_code == 413
    assert model.max_upload_size_mb == 50
    await service.update_parameter("preview_max_size_mb", "2")
    configuration.apply(await factory._uow.config_parameters.get_all())
    assert await read_upload_with_limit(UploadFile(BytesIO(data)), diagnostic.max_preview_bytes) == data


@pytest.mark.asyncio
@pytest.mark.parametrize("access", ["bytes", "path", "cleanup"])
async def test_reduced_ttl_expires_existing_entries(
    preview_configuration: PreviewConfiguration, monkeypatch: pytest.MonkeyPatch, access: str
) -> None:
    _model, configuration, live = preview_configuration
    clock = Mock(return_value=100.0)
    monkeypatch.setattr("application.documents.preview_cache.time.monotonic", clock)
    storage = AsyncMock(spec=FileStorage)
    cache = PreviewCache(storage, settings=live)
    preview_id = await cache.store(b"pdf")
    key = storage.upload_file.await_args.args[0]
    clock.return_value = 110.0
    configuration.apply([ConfigParameter("preview_cache_ttl_seconds", "5", "int", "ingestion")])

    if access == "bytes":
        assert await cache.get_bytes(preview_id) is None
    elif access == "path":
        async with cache.get_path(preview_id) as path:
            assert path is None
    else:
        await cache.cleanup_expired()

    storage.delete_file.assert_awaited_once_with(key)
    storage.download_to_temp.assert_not_awaited()


@pytest.mark.asyncio
async def test_extended_ttl_keeps_existing_preview_available(
    preview_configuration: PreviewConfiguration, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _model, configuration, live = preview_configuration
    clock = Mock(return_value=100.0)
    monkeypatch.setattr("application.documents.preview_cache.time.monotonic", clock)
    storage = AsyncMock(spec=FileStorage)
    cache = PreviewCache(storage, settings=live)
    preview_id = await cache.store(b"pdf")
    configuration.apply([ConfigParameter("preview_cache_ttl_seconds", "200", "int", "ingestion")])
    clock.return_value = 250.0
    path = tmp_path / "preview.pdf"
    path.write_bytes(b"pdf")
    storage.download_to_temp.return_value = path
    assert await cache.get_bytes(preview_id) == b"pdf"
    storage.delete_file.assert_not_awaited()


@pytest.mark.parametrize(
    "key,value",
    [("preview_max_size_mb", "0"), ("preview_max_size_mb", "64"), ("preview_cache_ttl_seconds", "0")],
)
def test_invalid_preview_snapshot_keeps_applied_values(
    preview_configuration: PreviewConfiguration, key: str, value: str
) -> None:
    model, configuration, _live = preview_configuration
    with pytest.raises(ValidationError):
        configuration.apply([ConfigParameter(key, value, "int", "ingestion")])
    assert model.preview_max_size_mb == 2
    assert model.preview_cache_ttl_seconds == 100
