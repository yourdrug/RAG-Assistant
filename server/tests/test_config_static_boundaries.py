"""Static/secret ownership is enforced below HTTP and on legacy events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

from pydantic import SecretStr
import pytest

from application.admin import ConfigService
from application.runtime.config import ConfigParameter, ConfigValueType
from application.runtime.setting_keys import SECRET_SETTING_KEYS, STATIC_SETTING_KEYS
from config import Settings
from domain.shared.exceptions import ValidationError
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
@pytest.mark.parametrize("key", sorted(SECRET_SETTING_KEYS | STATIC_SETTING_KEYS | {"collection_name"}))
async def test_direct_service_cannot_update_deployment_or_pending_keys(key: str) -> None:
    factory = FakeUnitOfWorkFactory()
    row = ConfigParameter(key, "original", ConfigValueType.STR, "legacy")
    await factory._uow.fake_config_parameters.save(row)
    service = ConfigService(factory, MagicMock())
    with pytest.raises(ValidationError):
        await service.update_parameter(key, "****")
    assert row.value == "original"


@pytest.mark.parametrize("key", ["s3_secret_key", "s3_access_key", "s3_endpoint", "s3_bucket", "s3_region"])
def test_legacy_storage_row_does_not_change_deployment_settings(key: str) -> None:
    model = Settings(_env_file=None, **cast("dict[str, Any]", {key: "deployment-value"}))
    bus = MagicMock()
    configuration = SettingsProviderConfiguration(model, bus=bus)
    row = ConfigParameter(key, "****", ConfigValueType.STR, "legacy")
    configuration.apply([row])
    value = getattr(model, key)
    assert (value.get_secret_value() if isinstance(value, SecretStr) else value) == "deployment-value"
    bus.publish_strict.assert_not_called()


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_ml_provider_comes_from_deployment_and_ignores_legacy_database_row(
    source: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ML_PROVIDER", raising=False)
    env = tmp_path / ".env"
    env.write_text("ML_PROVIDER=deepinfra\n")

    if source == "environment":
        env.write_text("ML_PROVIDER=tei\n")
        monkeypatch.setenv("ML_PROVIDER", "deepinfra")

    model = Settings(_env_file=env)
    configuration = SettingsProviderConfiguration(model, bus=MagicMock())
    configuration.apply([ConfigParameter("ml_provider", "tei", ConfigValueType.STR, "legacy")])
    assert model.ml_provider == "deepinfra"
    assert configuration.status().ready
    parameter = next(p for p in configuration.effective_parameters([]) if p.key == "ml_provider")
    assert parameter.source == "deployment"
    assert not parameter.editable


def test_dynamic_environment_and_dotenv_are_ignored_but_constructor_kwargs_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "legacy-env-model")
    monkeypatch.setenv("OCR_LANG_SURYA", "not-json")
    env = tmp_path / ".env"
    env.write_text(
        "OPENROUTER_MODEL=legacy-dotenv-model\nOCR_LANG_SURYA=not-json\nCURATOR_SCOPE_MAX_IDS=42\n"
    )
    model = Settings(_env_file=env)
    assert model.openrouter_model == "qwen/qwen-2.5-7b-instruct"
    assert model.curator_scope_max_ids == 42
    assert Settings(_env_file=env, openrouter_model="explicit-model").openrouter_model == "explicit-model"


def test_removed_classifier_threshold_in_env_and_dotenv_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DOCUMENT_DOMAIN_MARKER_THRESHOLD", "obsolete-value")
    env = tmp_path / ".env"
    env.write_text("DOCUMENT_DOMAIN_MARKER_THRESHOLD=obsolete-value\nCURATOR_SCOPE_MAX_IDS=42\n")
    model = Settings(_env_file=env)
    assert not hasattr(model, "document_domain_marker_threshold")
    assert model.curator_scope_max_ids == 42


@pytest.mark.asyncio
async def test_removed_classifier_threshold_in_database_stays_inactive() -> None:
    key = "document_domain_marker_threshold"
    factory = FakeUnitOfWorkFactory()
    row = ConfigParameter(key, "obsolete-value", ConfigValueType.FLOAT, "legacy")
    await factory._uow.fake_config_parameters.save(row)
    model = Settings(_env_file=None)
    bus = MagicMock()
    configuration = SettingsProviderConfiguration(model, bus=bus)
    configuration.apply([row])
    assert configuration.status().ready
    assert configuration.snapshot_version([row]) == configuration.snapshot_version([])
    assert not hasattr(model, key)
    assert key not in {param.key for param in configuration.effective_parameters([row])}
    bus.publish.assert_not_called()
    service = ConfigService(factory, bus, configuration)
    assert await service.list_parameters() == []
    with pytest.raises(ValidationError, match="not an active dynamic parameter"):
        await service.update_parameter(key, "3.0")
    assert row.value == "obsolete-value"


def test_preview_limits_ignore_legacy_environment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREVIEW_MAX_SIZE_MB", "7")
    monkeypatch.setenv("PREVIEW_CACHE_TTL_SECONDS", "90")
    model = Settings(_env_file=None)
    assert model.preview_max_size_mb == 50
    assert model.preview_cache_ttl_seconds == 1800
    assert not {"preview_max_size_mb", "preview_cache_ttl_seconds"} & STATIC_SETTING_KEYS


@pytest.mark.parametrize("key", ["preview_max_size_mb", "preview_cache_ttl_seconds"])
def test_preview_limits_must_be_positive(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        Settings(_env_file=None, **cast("dict[str, Any]", {key: 0}))
