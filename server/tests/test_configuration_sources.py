"""Configuration ownership contract; planned keys do not enable hot reload."""

from pathlib import Path

import pytest

from config import Settings, settings
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.knowledge.enums import DocDomain
import domain.runtime.dynamic_params
from domain.runtime.enums import LLMProvider
import domain.runtime.setting_keys
from fakes import FakeUnitOfWorkFactory
from infrastructure import initialization
from infrastructure.ml.config.config_subscribers import SENSITIVE_KEYS


def catalog_sources():
    plan = Path(__file__).resolve().parents[2] / "docs/operations/configuration-plan.md"
    table = plan.read_text().split("## Все поля Settings\n", 1)[1].split("\n¹", 1)[0]
    sources = {}
    for line in table.splitlines():
        if not line.startswith("| "):
            continue
        key, source, *_ = [cell.strip() for cell in line.split("|")[1:-1]]
        if source in {"env", "secrets", "БД", "Код / метаданные"}:
            sources[key.split()[0].lower()] = source
    return sources


def test_catalog_covers_every_current_and_planned_setting():
    sources = catalog_sources()
    planned_pools = {
        "embedding_http_max_connections",
        "embedding_http_max_keepalive_connections",
        "rerank_http_max_connections",
        "rerank_http_max_keepalive_connections",
    }
    legacy_pools = {"http_pool_max_connections", "http_pool_max_keepalive"}
    assert set(sources) == (set(Settings.model_fields) - legacy_pools) | planned_pools


def test_target_sources_match_catalog_and_are_disjoint():
    groups = {
        "env": domain.runtime.setting_keys.STATIC_SETTING_KEYS,
        "secrets": domain.runtime.setting_keys.SECRET_SETTING_KEYS,
        "БД": domain.runtime.setting_keys.TARGET_DYNAMIC_SETTING_KEYS,
        "Код / метаданные": domain.runtime.setting_keys.METADATA_SETTING_KEYS,
    }
    sources = catalog_sources()
    for source, keys in groups.items():
        assert keys == {key for key, owner in sources.items() if owner == source}
    assert sum(map(len, groups.values())) == len(set().union(*groups.values()))


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


def test_domain_profile_catalog_matches_scoped_defaults():
    plan = Path(__file__).resolve().parents[2] / "docs/operations/configuration-plan.md"
    table = plan.read_text().split("## Дополнительные настройки доменов в БД\n", 1)[1]
    table = table.split("## Docker, сборка и frontend", 1)[0]
    documented = {}
    for line in table.splitlines():
        if not line.startswith("| "):
            continue
        key, domain, value, *_ = [cell.strip() for cell in line.split("|")[1:-1]]
        if domain in {d.value for d in DocDomain}:
            documented[(key, domain)] = value
    defaults = {
        (param.key, profile.key): param.value
        for profile in (LegalDomainProfile(), DecreeDomainProfile(), GeneralDomainProfile())
        for param in profile.config_defaults()
    }
    assert defaults == documented


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
