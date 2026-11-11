"""Settings are split into sections without changing the flat field contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_overloads, get_type_hints
from unittest.mock import MagicMock

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings, SettingsConfigDict
import pytest

from application.runtime.config import ConfigParameter, ConfigValueType
from application.runtime.enums import AppStage
from config import (
    ALL_SECTIONS,
    RetrievalSection,
    Settings,
    get_request_settings,
    process_section,
    request_section,
    request_settings,
    settings,
)
from config.ownership import SettingOwnership
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration


if TYPE_CHECKING:
    from collections.abc import Callable


CONTRACT_PATH = Path(__file__).parent / "fixtures" / "settings_contract.json"


def current_contract() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}

    for name, field in sorted(Settings.model_fields.items()):
        default = field.default
        result[name] = {
            "annotation": repr(field.annotation),
            "required": field.is_required(),
            "default": None if default is PydanticUndefined else repr(default),
            # Ownership markers are checked by test_configuration_sources, not by this contract.
            "metadata": repr([item for item in field.metadata if not isinstance(item, SettingOwnership)]),
        }

    return result


def test_flat_field_contract_is_unchanged() -> None:
    # Env names, DB parameter keys and model_fields introspection all depend on flat fields.
    expected = json.loads(CONTRACT_PATH.read_text())
    assert current_contract() == expected


def test_derived_properties_are_preserved() -> None:
    for name in (
        "db_slave_hosts_list",
        "db_slave_ports_list",
        "redis_url",
        "allowed_origins_list",
        "tz",
        "uptime_seconds",
    ):
        assert isinstance(getattr(Settings, name), property), name


class SpikeSectionA(BaseModel):
    spike_alpha: int = 1

    @property
    def spike_alpha_doubled(self) -> int:
        return self.spike_alpha * 2


class SpikeSectionB(BaseModel):
    spike_beta: str = "b"


class SpikeSettings(BaseSettings, SpikeSectionA, SpikeSectionB):
    model_config = SettingsConfigDict(extra="ignore")  # type: ignore[mutable-override]


def test_mixin_sections_load_from_env_and_share_live_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SPIKE_ALPHA", "7")
    env = tmp_path / ".env"
    env.write_text("SPIKE_BETA=from-dotenv\n")

    model = SpikeSettings(_env_file=env)

    assert model.spike_alpha == 7
    assert model.spike_beta == "from-dotenv"
    assert set(SpikeSettings.model_fields) == {"spike_alpha", "spike_beta"}
    assert SpikeSettings.model_config.get("extra") == "ignore"
    assert isinstance(model, SpikeSectionA)
    assert isinstance(model, SpikeSectionB)

    section: SpikeSectionA = model
    model.__dict__.update({"spike_alpha": 11})
    assert section.spike_alpha_doubled == 22

    copied = model.model_copy(update={"spike_beta": "copied"})
    assert isinstance(copied, SpikeSectionB)
    assert copied.spike_beta == "copied"
    assert model.spike_beta == "from-dotenv"


def test_every_field_belongs_to_exactly_one_section() -> None:
    owners: dict[str, list[str]] = {}

    for section in ALL_SECTIONS:
        for name in section.model_fields:
            owners.setdefault(name, []).append(section.__name__)

    duplicated = {name: sections for name, sections in owners.items() if len(sections) > 1}
    assert duplicated == {}
    assert set(owners) == set(Settings.model_fields)


def test_settings_is_instance_of_every_section() -> None:
    for section in ALL_SECTIONS:
        assert isinstance(settings, section), section.__name__


def test_process_section_is_the_live_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieval = process_section(RetrievalSection)
    monkeypatch.setattr(settings, "retriever_top_k", 13)
    assert retrieval is settings
    assert retrieval.retriever_top_k == 13


def test_request_section_prefers_admitted_snapshot() -> None:
    snapshot = settings.model_copy(update={"retriever_top_k": settings.retriever_top_k + 5})
    assert request_section(RetrievalSection) is settings

    token = request_settings.set(snapshot)
    try:
        assert request_section(RetrievalSection) is snapshot
        assert request_section(RetrievalSection).retriever_top_k == snapshot.retriever_top_k
    finally:
        request_settings.reset(token)

    assert get_request_settings() is settings


def test_section_sees_hot_reloaded_dynamic_value() -> None:
    model = Settings(_env_file=None, **cast("dict[str, Any]", {}))
    configuration = SettingsProviderConfiguration(model, bus=MagicMock())
    section: RetrievalSection = model
    new_value = model.retriever_top_k + 3

    configuration.apply([ConfigParameter("retriever_top_k", str(new_value), ConfigValueType.INT, "rag")])

    assert section.retriever_top_k == new_value


def strong_security_values() -> dict[str, object]:
    return {
        "jwt_secret_key": "a" * 64,
        "admin_email": "admin@example.test",
        "admin_password": "strong-admin-password",
        "redis_password": "strong-redis-password",
        "s3_access_key": "strong-access-key",
        "s3_secret_key": "strong-secret-key",
        "allowed_origins": "https://app.example.test",
    }


def test_security_check_accepts_strong_credentials() -> None:
    candidate = settings.model_copy(update=strong_security_values())
    assert candidate.check_security() == []


def test_security_check_reports_every_weak_value_in_stable_order() -> None:
    weak = strong_security_values() | {
        "jwt_secret_key": "change-me-in-production",
        "admin_password": "admin",
        "redis_password": "password",
        "s3_secret_key": "minioadmin",
        "allowed_origins": " * ",
        "stage": AppStage.PROD,
    }
    errors = settings.model_copy(update=weak).check_security()
    prefixes = ["JWT_SECRET_KEY", "ADMIN_PASSWORD", "REDIS_PASSWORD", "S3_ACCESS_KEY", "ALLOWED_ORIGINS"]
    assert [error.split(" ", 1)[0] for error in errors] == prefixes


def test_security_check_scopes_conditional_rules() -> None:
    # Admin password matters only with an admin account; wildcard origins only in production.
    relaxed = strong_security_values() | {"admin_email": "", "admin_password": "", "allowed_origins": "*"}
    candidate = settings.model_copy(update=relaxed | {"stage": AppStage.DEVELOPMENT})
    assert candidate.check_security() == []


@pytest.mark.parametrize("accessor", [process_section, request_section])
def test_section_accessors_have_one_overload_per_section(accessor: Callable[..., object]) -> None:
    # A section without an overload would make its callers fail type checking.
    covered = [get_type_hints(overload)["return"] for overload in get_overloads(accessor)]
    assert sorted(covered, key=repr) == sorted(ALL_SECTIONS, key=repr)
