"""Static/secret ownership is enforced below HTTP and on legacy events."""

from unittest.mock import MagicMock

import pytest

from application.admin import ConfigService
from config import Settings, settings
from domain.entities.config_parameter import ConfigParameter
from domain.events.config_events import ConfigParameterChanged
from domain.exceptions import ValidationError
from domain.value_objects.config_value_type import ConfigValueType
from domain.value_objects.dynamic_config import SECRET_SETTING_KEYS, STATIC_SETTING_KEYS
from fakes import FakeUnitOfWorkFactory
from infrastructure.ml.config.config_subscribers import apply_to_settings, invalidate_storage_cache


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
def test_legacy_storage_event_does_not_change_env_or_resources(monkeypatch, key):
    monkeypatch.setattr(settings, key, "deployment-value")
    storage, clients = MagicMock(), MagicMock()
    event = ConfigParameterChanged(key=key, old_value=None, new_value="****", value_type=ConfigValueType.STR)
    apply_to_settings(event)
    invalidate_storage_cache(event, storage, clients)
    assert getattr(settings, key) == "deployment-value"
    storage.invalidate.assert_not_called()
    clients.invalidate_bm25.assert_not_called()


def test_dynamic_environment_and_dotenv_are_ignored_but_constructor_kwargs_work(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "legacy-env-model")
    monkeypatch.setenv("OCR_LANG_SURYA", "not-json")
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=legacy-dotenv-model\nOCR_LANG_SURYA=not-json\nCURATOR_SCOPE_MAX_IDS=42\n")
    model = Settings(_env_file=env)
    assert model.llm_model == "qwen2.5:7b"
    assert model.curator_scope_max_ids == 42
    assert Settings(_env_file=env, llm_model="explicit-model").llm_model == "explicit-model"
