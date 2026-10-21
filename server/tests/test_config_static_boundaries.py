"""Static/secret ownership is enforced below HTTP and on legacy events."""

from unittest.mock import MagicMock

import pytest

from application.admin import ConfigService
from config import Settings
from domain.runtime.config import ConfigParameter, ConfigValueType
from domain.runtime.setting_keys import SECRET_SETTING_KEYS, STATIC_SETTING_KEYS
from domain.shared.exceptions import ValidationError
from fakes import FakeUnitOfWorkFactory
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration


@pytest.mark.asyncio
@pytest.mark.parametrize("key", sorted(SECRET_SETTING_KEYS | STATIC_SETTING_KEYS | {"collection_name"}))
async def test_direct_service_cannot_update_deployment_or_pending_keys(key):
    factory = FakeUnitOfWorkFactory()
    row = ConfigParameter(key, "original", ConfigValueType.STR, "legacy")
    await factory._uow.config_parameters.save(row)
    service = ConfigService(factory, MagicMock())
    with pytest.raises(ValidationError):
        await service.update_parameter(key, "****")
    assert row.value == "original"


@pytest.mark.parametrize("key", ["s3_secret_key", "s3_access_key", "s3_endpoint", "s3_bucket", "s3_region"])
def test_legacy_storage_row_does_not_change_deployment_settings(key):
    model = Settings(_env_file=None, **{key: "deployment-value"})
    bus = MagicMock()
    configuration = SettingsProviderConfiguration(model, bus=bus)
    row = ConfigParameter(key, "****", ConfigValueType.STR, "legacy")
    configuration.apply([row])
    assert getattr(model, key) == "deployment-value"
    bus.publish_strict.assert_not_called()


def test_dynamic_environment_and_dotenv_are_ignored_but_constructor_kwargs_work(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "legacy-env-model")
    monkeypatch.setenv("OCR_LANG_SURYA", "not-json")
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=legacy-dotenv-model\nOCR_LANG_SURYA=not-json\nCURATOR_SCOPE_MAX_IDS=42\n")
    model = Settings(_env_file=env)
    assert model.llm_model == "qwen2.5:7b"
    assert model.curator_scope_max_ids == 42
    assert Settings(_env_file=env, llm_model="explicit-model").llm_model == "explicit-model"


def test_removed_classifier_threshold_in_env_and_dotenv_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCUMENT_DOMAIN_MARKER_THRESHOLD", "obsolete-value")
    env = tmp_path / ".env"
    env.write_text("DOCUMENT_DOMAIN_MARKER_THRESHOLD=obsolete-value\nCURATOR_SCOPE_MAX_IDS=42\n")
    model = Settings(_env_file=env)
    assert not hasattr(model, "document_domain_marker_threshold")
    assert model.curator_scope_max_ids == 42


@pytest.mark.asyncio
async def test_removed_classifier_threshold_in_database_stays_inactive():
    key = "document_domain_marker_threshold"
    factory = FakeUnitOfWorkFactory()
    row = ConfigParameter(key, "obsolete-value", ConfigValueType.FLOAT, "legacy")
    await factory._uow.config_parameters.save(row)
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


def test_preview_limits_ignore_legacy_environment_settings(monkeypatch):
    monkeypatch.setenv("PREVIEW_MAX_SIZE_MB", "7")
    monkeypatch.setenv("PREVIEW_CACHE_TTL_SECONDS", "90")
    model = Settings(_env_file=None)
    assert model.preview_max_size_mb == 50
    assert model.preview_cache_ttl_seconds == 1800
    assert not {"preview_max_size_mb", "preview_cache_ttl_seconds"} & STATIC_SETTING_KEYS


@pytest.mark.parametrize("key", ["preview_max_size_mb", "preview_cache_ttl_seconds"])
def test_preview_limits_must_be_positive(key):
    with pytest.raises(ValueError, match=key):
        Settings(_env_file=None, **{key: 0})
