"""Settings are split into sections without changing the flat field contract."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

from pydantic import BaseModel, SecretStr, ValidationError as PydanticValidationError
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings, SettingsConfigDict
import pytest
from redis.connection import ConnectionPool

from application.runtime.config import ConfigParameter, ConfigValueType
from application.runtime.enums import AppStage
import composition.startup_errors
from composition.startup_errors import ConfigurationErrorHook
import config as config_module
from config import (
    ALL_SECTIONS,
    ConfigurationError,
    RetrievalSection,
    Settings,
    get_request_settings,
    process_section,
    request_section,
    request_settings,
    settings,
)
from config.ownership import SettingOwnership, keys_owned_by
from infrastructure.database.database import connection_arguments
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
from infrastructure.worker.queue import arq_redis_settings


if TYPE_CHECKING:
    from collections.abc import Callable

APP_DIR = Path(__file__).resolve().parents[1] / "app"


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
    # SecretStr values: model_copy(update=...) skips validation, Settings(...) accepts both.
    return {
        "jwt_secret_key": SecretStr("a" * 64),
        "admin_email": "admin@example.test",
        "admin_password": SecretStr("strong-admin-password"),
        "redis_password": SecretStr("strong-redis-password"),
        "s3_access_key": SecretStr("strong-access-key"),
        "s3_secret_key": SecretStr("strong-secret-key"),
        "allowed_origins": "https://app.example.test",
    }


def plain_secret(values: dict[str, object], key: str) -> str:
    value = values[key]
    return value.get_secret_value() if isinstance(value, SecretStr) else str(value)


def test_security_check_accepts_strong_credentials() -> None:
    candidate = settings.model_copy(update=strong_security_values())
    assert candidate.check_security() == []


def test_security_check_reports_every_weak_value_in_stable_order() -> None:
    weak = strong_security_values() | {
        "jwt_secret_key": SecretStr("change-me-in-production"),
        "admin_password": SecretStr("admin"),
        "redis_password": SecretStr("password"),
        "s3_secret_key": SecretStr("minioadmin"),
        "allowed_origins": " * ",
        "stage": AppStage.PROD,
    }
    errors = settings.model_copy(update=weak).check_security()
    prefixes = ["JWT_SECRET_KEY", "ADMIN_PASSWORD", "REDIS_PASSWORD", "S3_ACCESS_KEY", "ALLOWED_ORIGINS"]
    assert [error.split(" ", 1)[0] for error in errors] == prefixes


def test_security_check_scopes_conditional_rules() -> None:
    # Admin password matters only with an admin account; wildcard origins only in production.
    relaxed = strong_security_values() | {
        "admin_email": "",
        "admin_password": SecretStr(""),
        "allowed_origins": "*",
    }
    candidate = settings.model_copy(update=relaxed | {"stage": AppStage.DEVELOPMENT})
    assert candidate.check_security() == []


@pytest.mark.parametrize("accessor", [process_section, request_section])
def test_section_accessors_accept_every_section(accessor: Callable[[type[BaseModel]], object]) -> None:
    for section in ALL_SECTIONS:
        assert accessor(section) is get_request_settings()


@pytest.mark.parametrize("accessor", [process_section, request_section])
@pytest.mark.parametrize("foreign", [Settings, BaseModel, SpikeSectionA])
def test_section_accessors_reject_non_section(
    accessor: Callable[[type[BaseModel]], object], foreign: type[BaseModel]
) -> None:
    with pytest.raises(TypeError, match="is not a Settings section"):
        accessor(foreign)


def test_weak_credentials_raise_configuration_error_instead_of_exiting() -> None:
    weak_key = "change-me-in-production"
    values = strong_security_values() | {"jwt_secret_key": weak_key}

    with pytest.raises(ConfigurationError, match="JWT_SECRET_KEY") as error:
        Settings(_env_file=None, **cast("dict[str, Any]", values))

    assert weak_key not in str(error.value)


def test_secret_fields_are_masked_everywhere_except_explicit_read() -> None:
    values = strong_security_values()
    model = Settings(_env_file=None, **cast("dict[str, Any]", values))
    keys = keys_owned_by(Settings, SettingOwnership.SECRET) & set(values)
    secrets = [plain_secret(values, key) for key in keys]
    outputs = (repr(model), str(model), str(model.model_dump()), model.model_dump_json())

    for text in outputs:
        assert not [secret for secret in secrets if secret in text]

    assert "jwt_secret_key=SecretStr('**********')" in repr(model)
    assert "allowed_origins='https://app.example.test'" in repr(model)
    assert model.jwt_secret_key.get_secret_value() == "a" * 64


def test_secret_marker_always_types_the_field_as_secret_str() -> None:
    for name in keys_owned_by(Settings, SettingOwnership.SECRET):
        assert Settings.model_fields[name].annotation is SecretStr, name


def test_invalid_http_pool_error_does_not_carry_loaded_secrets() -> None:
    values = strong_security_values() | {
        "embedding_http_max_connections": 1,
        "embedding_http_max_keepalive_connections": 2,
    }

    with pytest.raises(ConfigurationError, match="embedding HTTP pool") as error:
        Settings(_env_file=None, **cast("dict[str, Any]", values))

    secrets = [
        plain_secret(values, key)
        for key in ("jwt_secret_key", "redis_password", "s3_secret_key", "admin_password")
    ]
    text = f"{error.value!s} {error.value!r}"
    assert not [secret for secret in secrets if secret and secret in text]


@pytest.mark.parametrize(
    "key",
    [
        "qdrant_search_max_concurrent",
        "qdrant_write_max_concurrent",
        "llm_max_concurrent",
        "llm_generation_max_concurrent",
        "llm_auxiliary_max_concurrent",
        "embedding_max_concurrent",
        "ingestion_max_concurrent",
        "reranker_max_concurrent",
        "semaphore_acquire_timeout",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_concurrency_limits_must_be_positive(key: str, value: int) -> None:
    values = strong_security_values() | {key: value}

    with pytest.raises(PydanticValidationError) as error:
        Settings(_env_file=None, **cast("dict[str, Any]", values))

    assert [item["loc"] for item in error.value.errors()] == [(key,)]
    assert plain_secret(values, "jwt_secret_key") not in error.value.json()


def test_short_jwt_secret_is_rejected() -> None:
    short_key = "s" * 31
    values = strong_security_values() | {"jwt_secret_key": short_key}

    with pytest.raises(ConfigurationError, match="at least 32 bytes") as error:
        Settings(_env_file=None, **cast("dict[str, Any]", values))

    assert short_key not in str(error.value)
    accepted = Settings(_env_file=None, **cast("dict[str, Any]", values | {"jwt_secret_key": "s" * 32}))
    assert accepted.check_auth_security() == []


@pytest.mark.parametrize("origins", ["*", "https://app.example.test, *", "*,https://app.example.test"])
def test_production_rejects_wildcard_anywhere_in_origins(origins: str) -> None:
    candidate = settings.model_copy(
        update=strong_security_values() | {"stage": AppStage.PROD, "allowed_origins": origins}
    )
    assert any("ALLOWED_ORIGINS" in error for error in candidate.check_security())


def test_production_accepts_explicit_origins() -> None:
    explicit = "https://app.example.test,https://admin.example.test"
    candidate = settings.model_copy(
        update=strong_security_values() | {"stage": AppStage.PROD, "allowed_origins": explicit}
    )
    assert candidate.check_security() == []


@pytest.mark.parametrize("timezone", ["Mars/Olympus", "../etc/passwd", ""])
def test_unknown_timezone_is_rejected_at_startup(timezone: str) -> None:
    values = strong_security_values() | {"timezone": timezone}

    with pytest.raises(PydanticValidationError) as error:
        Settings(_env_file=None, **cast("dict[str, Any]", values))

    assert [item["loc"] for item in error.value.errors()] == [("timezone",)]


def test_known_timezone_is_accepted() -> None:
    values = strong_security_values() | {"timezone": "Europe/Minsk"}
    assert str(Settings(_env_file=None, **cast("dict[str, Any]", values)).tz) == "Europe/Minsk"


@pytest.mark.parametrize(
    "key",
    [
        "worker_max_concurrent",
        "db_pool_size",
        "redis_socket_timeout",
        "redis_max_connections",
        "llm_request_timeout",
        "llm_auxiliary_timeout",
        "llm_breaker_fail_max",
        "llm_breaker_timeout_duration",
        "qdrant_timeout",
        "rate_limit_max_buckets",
        "curator_scope_max_ids",
        "jwt_expire_minutes",
        "db_pool_recycle_sec",
        "db_statement_timeout_ms",
        "db_idle_in_transaction_timeout_ms",
        "embed_dim",
        "outbox_max_concurrent",
        "s3_max_concurrent",
        "benchmark_max_concurrent",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_static_operational_limits_must_be_positive(key: str, value: int) -> None:
    values = strong_security_values() | {key: value}

    with pytest.raises(PydanticValidationError) as error:
        Settings(_env_file=None, **cast("dict[str, Any]", values))

    assert [item["loc"] for item in error.value.errors()] == [(key,)]


@pytest.mark.parametrize(
    "key, value",
    [
        ("db_max_overflow", -1),
        ("redis_db", -1),
        ("redis_port", 0),
        ("redis_port", 65536),
        ("rag_retrieval_score_threshold", -0.1),
        ("rag_retrieval_score_threshold", 1.1),
    ],
)
def test_static_fields_reject_out_of_range_values(key: str, value: float) -> None:
    values = strong_security_values() | {key: value}

    with pytest.raises(PydanticValidationError) as error:
        Settings(_env_file=None, **cast("dict[str, Any]", values))

    assert [item["loc"] for item in error.value.errors()] == [(key,)]


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"retriever_top_k": 16, "retriever_fetch_k": 15},
            "retriever_top_k must not exceed retriever_fetch_k",
        ),
        (
            {"retriever_top_k_broad": 41, "retriever_fetch_k_broad": 40},
            "retriever_top_k_broad must not exceed retriever_fetch_k_broad",
        ),
        (
            {"llm_num_predict_narrow": 4096, "llm_num_ctx_narrow": 4096},
            "llm_num_predict_narrow must be less than llm_num_ctx_narrow",
        ),
        (
            {"llm_num_predict_broad": 9000, "llm_num_ctx_broad": 8192},
            "llm_num_predict_broad must be less than llm_num_ctx_broad",
        ),
    ],
)
def test_linked_retrieval_and_generation_limits_are_checked(overrides: dict[str, int], message: str) -> None:
    assert settings.check_retrieval() == []
    assert settings.check_generation_budget() == []

    broken = settings.model_copy(update=overrides)
    assert any(
        error.startswith(message) for error in [*broken.check_retrieval(), *broken.check_generation_budget()]
    )

    with pytest.raises(ConfigurationError, match=message):
        Settings(_env_file=None, **cast("dict[str, Any]", strong_security_values() | overrides))


def test_equal_top_k_and_fetch_k_are_accepted() -> None:
    model = settings.model_copy(update={"retriever_top_k": 15, "retriever_fetch_k": 15})
    assert model.check_retrieval() == []


@pytest.mark.parametrize("user, password", [("", "ab/cd#ef?g@h:i%j"), ("svc:user@x", "p+w/=#?")])
def test_redis_url_percent_encodes_credentials(user: str, password: str) -> None:
    model = settings.model_copy(update={"redis_user": user, "redis_password": SecretStr(password)})

    redis_kwargs = ConnectionPool.from_url(model.redis_url).connection_kwargs
    arq_settings = arq_redis_settings(model.redis_url)

    assert (redis_kwargs.get("username"), redis_kwargs["password"]) == (user or None, password)
    assert (arq_settings.username or None, arq_settings.password) == (user or None, password)
    assert (arq_settings.host, arq_settings.port) == (model.redis_host, model.redis_port)


def test_chunking_check_requires_overlap_below_size() -> None:
    assert settings.check_chunking() == []

    broken = settings.model_copy(
        update={
            "chunk_size": 150,
            "chunk_overlap": 150,
            "legal_chunk_target_tokens": 10**6,
            "chunk_overlap_tokens": settings.chunk_target_tokens,
        }
    )
    assert broken.check_chunking() == [
        "CHUNK_OVERLAP must be less than CHUNK_SIZE",
        "LEGAL_CHUNK_TARGET_TOKENS must be positive and <= max tokens",
        "CHUNK_OVERLAP_TOKENS must be non-negative and less than target tokens",
    ]


@pytest.mark.parametrize("algorithm", ["RS256", "none", "ES256"])
def test_non_hmac_jwt_algorithm_is_rejected(algorithm: str) -> None:
    # The signing key is one shared secret: only HMAC algorithms can use it.
    values = strong_security_values() | {"jwt_algorithm": algorithm}

    with pytest.raises(PydanticValidationError) as error:
        Settings(_env_file=None, **cast("dict[str, Any]", values))

    assert [item["loc"] for item in error.value.errors()] == [("jwt_algorithm",)]


def test_jwt_key_length_follows_algorithm() -> None:
    values = strong_security_values() | {"jwt_algorithm": "HS512", "jwt_secret_key": "k" * 63}

    with pytest.raises(ConfigurationError, match="at least 64 bytes for HS512"):
        Settings(_env_file=None, **cast("dict[str, Any]", values))

    accepted = Settings(_env_file=None, **cast("dict[str, Any]", values | {"jwt_secret_key": "k" * 64}))
    assert accepted.check_auth_security() == []


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"db_port": "postgres"}, "DB_PORT must be a TCP port number"),
        ({"db_port": "70000"}, "DB_PORT must be a TCP port number"),
        ({"db_slave_hosts": "r1", "db_slave_ports": "5432,x"}, "DB_SLAVE_PORTS must be a TCP port number"),
        ({"db_slave_hosts": "r1,r2", "db_slave_ports": "5432"}, "same number of entries"),
    ],
)
def test_invalid_database_endpoints_fail_at_startup(overrides: dict[str, str], message: str) -> None:
    values = strong_security_values() | overrides

    with pytest.raises(ConfigurationError, match=message):
        Settings(_env_file=None, **cast("dict[str, Any]", values))


def test_matching_database_replicas_are_accepted() -> None:
    values = strong_security_values() | {"db_slave_hosts": "r1, r2", "db_slave_ports": "5432,5433"}
    candidate = Settings(_env_file=None, **cast("dict[str, Any]", values))
    assert candidate.check_database() == []


def test_database_session_time_zone_follows_timezone_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "timezone", "Asia/Tokyo")
    options = str(connection_arguments(settings)["options"])
    assert "-c timezone=Asia/Tokyo " in options
    assert "Europe/Minsk" not in options


def test_load_settings_reports_field_errors_without_values(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "a" * 64
    monkeypatch.setenv("JWT_SECRET_KEY", secret)
    monkeypatch.setenv("JWT_EXPIRE_MINUTES", "0")
    monkeypatch.setenv("TIMEZONE", "Mars/Olympus")

    with pytest.raises(ConfigurationError) as error:
        config_module.load_settings()

    text = str(error.value)
    assert "JWT_EXPIRE_MINUTES:" in text
    assert "TIMEZONE:" in text
    assert secret not in text
    # The pydantic cause carries ``input_value``; it must not travel with the error.
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__


def test_startup_hook_reports_configuration_error_without_traceback(caplog: pytest.LogCaptureFixture) -> None:
    previous = MagicMock()
    hook = ConfigurationErrorHook(previous)
    error = ConfigurationError("Configuration errors:\n  - DB_PORT must be a TCP port number")

    with caplog.at_level(logging.CRITICAL, logger="config"):
        hook(ConfigurationError, error, None)

    previous.assert_not_called()
    assert "DB_PORT must be a TCP port number" in caplog.text

    other = RuntimeError("boom")
    hook(RuntimeError, other, None)
    previous.assert_called_once_with(RuntimeError, other, None)


def test_startup_hook_is_installed_once() -> None:
    assert isinstance(sys.excepthook, ConfigurationErrorHook)
    installed = sys.excepthook
    composition.startup_errors.install_configuration_error_hook()
    assert sys.excepthook is installed


def test_entry_point_prints_configuration_errors_not_a_traceback() -> None:
    env = os.environ | {"JWT_SECRET_KEY": "short", "DB_PORT": "5432"}
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=APP_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode != 0
    assert "Invalid configuration, the process cannot start." in result.stderr
    assert "JWT_SECRET_KEY must be a strong random value" in result.stderr
    assert "Traceback" not in result.stderr
    assert "short" not in result.stderr


@pytest.mark.parametrize("line", ["JWT_SECRET_KEY={value}", "db_password={value}"])
def test_secret_in_plain_dotenv_is_rejected_without_echoing_value(tmp_path: Path, line: str) -> None:
    value = "leaked-" + "x" * 64
    env = tmp_path / ".env"
    env.write_text("CURATOR_SCOPE_MAX_IDS=42\n" + line.format(value=value) + "\n")
    key = line.split("=", 1)[0].upper()

    with pytest.raises(ConfigurationError, match=f"{key} is a secret") as error:
        Settings(_env_file=env)

    assert value not in str(error.value)


def test_secret_in_secrets_dotenv_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The process environment outranks dotenv files, so the key must be absent there.
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    value = "from-secrets-file-" + "y" * 16
    env = tmp_path / ".env"
    env.write_text("CURATOR_SCOPE_MAX_IDS=42\n")
    secrets_file = tmp_path / ".env.secrets"
    secrets_file.write_text(f"QDRANT_API_KEY={value}\n")

    model = Settings(_env_file=(env, secrets_file))

    assert model.qdrant_api_key.get_secret_value() == value
    assert model.curator_scope_max_ids == 42
