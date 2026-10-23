"""Executable configuration ownership and scoped database seeding contracts."""

import pytest

from config import Settings, settings
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.knowledge.enums import DocDomain
import domain.runtime.dynamic_params
from domain.runtime.enums import LLMProvider
import domain.runtime.setting_keys
from fakes import FakeUnitOfWorkFactory
from infrastructure import initialization
from infrastructure.ml.config.config_subscribers import SENSITIVE_KEYS


def test_every_setting_has_exactly_one_owner():
    groups = (
        domain.runtime.setting_keys.STATIC_SETTING_KEYS,
        domain.runtime.setting_keys.SECRET_SETTING_KEYS,
        domain.runtime.setting_keys.TARGET_DYNAMIC_SETTING_KEYS,
        domain.runtime.setting_keys.METADATA_SETTING_KEYS,
    )
    assert set().union(*groups) == set(Settings.model_fields)
    assert sum(map(len, groups)) == len(Settings.model_fields)


def test_only_database_owned_settings_are_seeded():
    seeded = {row.key for row in initialization.build_defaults()}
    assert seeded == domain.runtime.setting_keys.TARGET_DYNAMIC_SETTING_KEYS
    assert not seeded & (
        domain.runtime.setting_keys.SECRET_SETTING_KEYS
        | domain.runtime.setting_keys.STATIC_SETTING_KEYS
        | domain.runtime.setting_keys.METADATA_SETTING_KEYS
    )


def test_future_dynamic_registry_excludes_secrets_and_static_s3():
    assert (
        not domain.runtime.setting_keys.TARGET_DYNAMIC_SETTING_KEYS
        & domain.runtime.setting_keys.SECRET_SETTING_KEYS
    )
    assert {"s3_endpoint", "s3_bucket", "s3_region"} <= domain.runtime.setting_keys.STATIC_SETTING_KEYS
    assert {"s3_access_key", "s3_secret_key"} <= domain.runtime.setting_keys.SECRET_SETTING_KEYS
    assert "curator_scope_max_ids" in domain.runtime.setting_keys.STATIC_SETTING_KEYS


def test_secret_masking_uses_complete_canonical_list():
    assert SENSITIVE_KEYS is domain.runtime.setting_keys.SECRET_SETTING_KEYS
    assert {"admin_password", "qdrant_api_key"} <= SENSITIVE_KEYS


def test_dynamic_registry_keys_match_target_ownership():
    active = {param.key for param in domain.runtime.dynamic_params.DYNAMIC_PARAMS}
    assert active == domain.runtime.setting_keys.TARGET_DYNAMIC_SETTING_KEYS


def test_domain_parameters_keep_scope_and_stay_out_of_global_map():
    expected = {
        "legal_chunk_size": DocDomain.LEGAL,
        "legal_chunk_overlap": DocDomain.LEGAL,
        "decree_chunk_size": DocDomain.DECREE,
        "decree_chunk_overlap": DocDomain.DECREE,
    }
    scoped = {p.key: p.domain_key for p in domain.runtime.dynamic_params.DYNAMIC_PARAMS if p.domain_key}
    assert scoped == expected
    assert not scoped.keys() & domain.runtime.dynamic_params.DYNAMIC_FIELDS_MAP.keys()
    pairs = [(p.key, p.domain_key) for p in domain.runtime.dynamic_params.DYNAMIC_PARAMS]
    assert len(pairs) == len(set(pairs))


@pytest.mark.asyncio
async def test_domain_profile_defaults_are_seeded_in_declared_scope():
    registry = DomainProfileRegistry()
    for profile in (LegalDomainProfile(), DecreeDomainProfile(), GeneralDomainProfile()):
        registry.register(profile)
    defaults = {
        (param.key, profile.key): (param.value, param.value_type)
        for profile in registry.all()
        for param in profile.config_defaults()
    }
    factory = FakeUnitOfWorkFactory()
    await initialization.seed_domain_config_defaults(factory, registry)
    rows = await factory._uow.config_parameters.get_all()
    assert {(row.key, row.domain_key): (row.value, row.value_type) for row in rows} == defaults
    assert all(row.category == f"domain:{row.domain_key}" for row in rows)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("llm_provider", LLMProvider.OLLAMA.value),
        ("llm_model", "qwen2.5:7b"),
        ("ml_provider", "tei"),
    ],
)
def test_provider_defaults(key, expected):
    assert Settings.model_fields[key].default == expected


def test_seed_uses_defaults_and_preserves_domain_scope(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", LLMProvider.OPENROUTER.value)
    monkeypatch.setattr(settings, "llm_model", "existing-deployment-model")
    rows = {(p.key, p.domain_key): p for p in initialization.build_defaults()}
    assert rows[("llm_provider", None)].value == LLMProvider.OLLAMA.value
    assert rows[("llm_model", None)].value == "qwen2.5:7b"
    assert rows[("legal_chunk_size", DocDomain.LEGAL)].value == str(settings.legal_chunk_size)
    assert ("legal_chunk_size", None) not in rows


@pytest.mark.asyncio
async def test_seed_keeps_existing_database_values_and_is_repeatable():
    factory = FakeUnitOfWorkFactory()
    existing = initialization.build_defaults()
    model = next(p for p in existing if p.key == "llm_model")
    model.value = "preserved-db-model"
    scoped = next(p for p in existing if p.key == "legal_chunk_size")
    scoped.value = "1777"
    for row in (model, scoped):
        await factory._uow.config_parameters.save(row)
    await initialization.seed_config_defaults(factory)
    await initialization.seed_config_defaults(factory)
    rows = await factory._uow.config_parameters.get_all()
    assert len(rows) == len(existing)
    values = {(p.key, p.domain_key): p.value for p in rows}
    assert values[("llm_model", None)] == model.value
    assert values[("legal_chunk_size", DocDomain.LEGAL)] == scoped.value
