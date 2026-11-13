"""Executable configuration ownership and scoped database seeding contracts."""

from __future__ import annotations

from typing import Any

import pytest

import application.runtime.dynamic_params
from application.runtime.enums import LLMProvider
import application.runtime.setting_keys
from application.runtime.setting_keys import SECRET_SETTING_KEYS as SENSITIVE_KEYS
from config import Settings, settings
from config.ownership import SettingOwnership, field_ownership, keys_owned_by
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.knowledge.enums import DocDomain
from infrastructure import initialization
from tests.fakes import FakeUnitOfWorkFactory


def test_every_setting_has_exactly_one_owner() -> None:
    groups = (
        application.runtime.setting_keys.STATIC_SETTING_KEYS,
        application.runtime.setting_keys.SECRET_SETTING_KEYS,
        application.runtime.setting_keys.GLOBAL_DYNAMIC_SETTING_KEYS,
        application.runtime.setting_keys.METADATA_SETTING_KEYS,
    )
    assert set().union(*groups) == set(Settings.model_fields)
    assert sum(map(len, groups)) == len(Settings.model_fields)


def test_ownership_markers_match_setting_key_registry() -> None:
    # Section field markers are the source of truth; Application mirrors them without importing config.
    keys = application.runtime.setting_keys
    assert keys_owned_by(Settings, SettingOwnership.SECRET) == keys.SECRET_SETTING_KEYS
    assert keys_owned_by(Settings, SettingOwnership.STATIC) == keys.STATIC_SETTING_KEYS
    assert keys_owned_by(Settings, SettingOwnership.METADATA) == keys.METADATA_SETTING_KEYS


def test_unmarked_settings_are_exactly_the_dynamic_ones() -> None:
    unmarked = {name for name in Settings.model_fields if field_ownership(Settings, name) is None}
    assert unmarked == application.runtime.setting_keys.GLOBAL_DYNAMIC_SETTING_KEYS


def test_only_database_owned_settings_are_seeded() -> None:
    seeded = {row.key for row in initialization.build_defaults()}
    assert seeded == application.runtime.setting_keys.TARGET_DYNAMIC_SETTING_KEYS
    assert not seeded & (
        application.runtime.setting_keys.SECRET_SETTING_KEYS
        | application.runtime.setting_keys.STATIC_SETTING_KEYS
        | application.runtime.setting_keys.METADATA_SETTING_KEYS
    )


def test_future_dynamic_registry_excludes_secrets_and_static_s3() -> None:
    assert (
        not application.runtime.setting_keys.TARGET_DYNAMIC_SETTING_KEYS
        & application.runtime.setting_keys.SECRET_SETTING_KEYS
    )
    assert {"s3_endpoint", "s3_bucket", "s3_region"} <= application.runtime.setting_keys.STATIC_SETTING_KEYS
    assert {"s3_access_key", "s3_secret_key"} <= application.runtime.setting_keys.SECRET_SETTING_KEYS
    assert "curator_scope_max_ids" in application.runtime.setting_keys.STATIC_SETTING_KEYS


def test_secret_masking_uses_complete_canonical_list() -> None:
    assert SENSITIVE_KEYS is application.runtime.setting_keys.SECRET_SETTING_KEYS
    assert {"admin_password", "qdrant_api_key"} <= SENSITIVE_KEYS


def test_dynamic_registry_keys_match_target_ownership() -> None:
    active = {param.key for param in application.runtime.dynamic_params.DYNAMIC_PARAMS}
    assert active == application.runtime.setting_keys.TARGET_DYNAMIC_SETTING_KEYS


def test_domain_parameters_keep_scope_and_stay_out_of_global_map() -> None:
    expected = {
        "legal_chunk_max_tokens": DocDomain.LEGAL,
        "legal_chunk_target_tokens": DocDomain.LEGAL,
        "decree_chunk_max_tokens": DocDomain.DECREE,
        "decree_chunk_target_tokens": DocDomain.DECREE,
        "contract_chunk_max_tokens": DocDomain.CONTRACT,
        "contract_chunk_target_tokens": DocDomain.CONTRACT,
    }
    scoped = {p.key: p.domain_key for p in application.runtime.dynamic_params.DYNAMIC_PARAMS if p.domain_key}
    assert scoped == expected
    assert not scoped.keys() & application.runtime.dynamic_params.DYNAMIC_FIELDS_MAP.keys()
    pairs = [(p.key, p.domain_key) for p in application.runtime.dynamic_params.DYNAMIC_PARAMS]
    assert len(pairs) == len(set(pairs))


@pytest.mark.asyncio
async def test_domain_profile_defaults_are_seeded_in_declared_scope() -> None:
    registry = DomainProfileRegistry()

    for profile in (LegalDomainProfile(), DecreeDomainProfile(), GeneralDomainProfile()):
        registry.register(profile)

    defaults: dict[tuple[str, str | None], tuple[str, str]] = {
        (param.key, profile.key): (param.value, param.value_type)
        for profile in registry.all()
        for param in profile.config_defaults()
    }
    factory = FakeUnitOfWorkFactory()
    await initialization.seed_domain_config_defaults(factory, registry)
    rows = await factory._uow.fake_config_parameters.get_all()
    assert {(row.key, row.domain_key): (row.value, row.value_type) for row in rows} == defaults
    assert all(row.category == f"domain:{row.domain_key}" for row in rows)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("llm_provider", LLMProvider.OPENROUTER.value),
        ("openrouter_model", "qwen/qwen-2.5-7b-instruct"),
        ("ml_provider", "deepinfra"),
    ],
)
def test_provider_defaults(key: str, expected: Any) -> None:
    assert Settings.model_fields[key].default == expected


def test_seed_uses_defaults_and_preserves_domain_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", LLMProvider.OPENROUTER.value)
    monkeypatch.setattr(settings, "openrouter_model", "existing-deployment-model")
    rows = {(p.key, p.domain_key): p for p in initialization.build_defaults()}
    assert rows[("llm_provider", None)].value == LLMProvider.OPENROUTER.value
    assert rows[("openrouter_model", None)].value == "qwen/qwen-2.5-7b-instruct"
    assert rows[("legal_chunk_max_tokens", DocDomain.LEGAL)].value == str(
        LegalDomainProfile().chunking_policy().max_tokens
    )
    assert ("legal_chunk_max_tokens", None) not in rows


@pytest.mark.asyncio
async def test_seed_keeps_existing_database_values_and_is_repeatable() -> None:
    factory = FakeUnitOfWorkFactory()
    existing = initialization.build_defaults()
    model = next(p for p in existing if p.key == "openrouter_model")
    model.value = "preserved-db-model"
    scoped = next(p for p in existing if p.key == "legal_chunk_max_tokens")
    scoped.value = "1777"

    for row in (model, scoped):
        await factory._uow.fake_config_parameters.save(row)

    await initialization.seed_config_defaults(factory)
    await initialization.seed_config_defaults(factory)
    rows = await factory._uow.fake_config_parameters.get_all()
    assert len(rows) == len(existing)
    values = {(p.key, p.domain_key): p.value for p in rows}
    assert values[("openrouter_model", None)] == model.value
    assert values[("legal_chunk_max_tokens", DocDomain.LEGAL)] == scoped.value


def test_profile_token_limits_are_not_global_settings_fields() -> None:
    scoped = {param.key for param in application.runtime.dynamic_params.DYNAMIC_PARAMS if param.domain_key}
    assert not scoped & Settings.model_fields.keys()
