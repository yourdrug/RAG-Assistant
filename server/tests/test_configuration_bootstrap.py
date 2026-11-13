"""Database snapshots are validated before any global/domain state is applied."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, HTTPException, Response
import httpx
from pydantic import SecretStr
import pytest

from application.admin import ConfigService
from application.health.ports import HealthCheckResult
from application.health.service import HealthService
from application.runtime.config import ConfigParameter, ConfigParameterChanged, ConfigValueType
from application.runtime.enums import AppStage, HealthStatus, LLMProvider
from composition.cli.commands import config, ingest, worker
from config import Settings, settings
from domain.ingestion.profiles.contract import ContractDomainProfile
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.runtime.cost import BudgetError
from domain.shared.exceptions import ValidationError
from infrastructure import initialization
from infrastructure.budget.runtime import BudgetRuntime
from infrastructure.budget.transport import PaidAsyncTransport
from infrastructure.database.database import database
from infrastructure.domain_profile import register_all_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.events.in_process_event_bus import InProcessEventBus
from infrastructure.events.postgres_config_listener import PostgresConfigListener
from infrastructure.ml.clients import managed_http
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
from infrastructure.redis.redis_client import redis_client
from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_config_service
from presentation.api.routes.admin_config import router
from presentation.api.routes.health import ready
from tests.fakes import FakeUnitOfWorkFactory, require_present
from tests.test_cli_domain_bootstrap import build_infrastructure


if TYPE_CHECKING:
    from application.admin.ports import ProviderConfigurationPort
    from application.budget.ports import CostBudgetPort


def row(
    key: str, value: str, value_type: ConfigValueType = ConfigValueType.STR, domain_key: str | None = None
) -> ConfigParameter:
    return ConfigParameter(key, value, value_type, "test", domain_key=domain_key)


@pytest.fixture
def configuration() -> SettingsProviderConfiguration:

    model = Settings(
        _env_file=None,
        ml_provider="tei",
        tei_embed_url="http://tei-embed:8080",
        tei_rerank_url="http://tei-rerank:8080",
    )
    adapter = DomainSettingsAdapter()
    registry = DomainProfileRegistry()
    register_all_profiles(registry, adapter)
    bus = InProcessEventBus()
    return SettingsProviderConfiguration(model, adapter, registry, bus)


def test_seed_uses_declared_defaults_not_mutated_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openrouter_model", "already-applied-model")
    rows = {p.key: p.value for p in initialization.build_defaults()}
    assert rows["openrouter_model"] == "qwen/qwen-2.5-7b-instruct"
    assert "s3_secret_key" not in rows


@pytest.mark.asyncio
async def test_seed_and_load_errors_are_fatal() -> None:
    factory = MagicMock()
    factory.create.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        await initialization.seed_config_defaults(factory)
    with pytest.raises(RuntimeError, match="database unavailable"):
        await initialization.load_config_from_db(factory)


def test_production_provider_validation_uses_deployment_provider(
    monkeypatch: pytest.MonkeyPatch, configuration: SettingsProviderConfiguration
) -> None:
    model = configuration.settings
    model.tei_embed_url = model.tei_rerank_url = ""
    model.ml_provider = "deepinfra"
    model.deepinfra_api_key = SecretStr("private-test-key")
    configuration.apply([row("ml_provider", "tei")])
    assert model.ml_provider == "deepinfra"
    assert configuration.status().ready


@pytest.mark.parametrize(
    "rows",
    [
        [row("retriever_top_k", "garbage", ConfigValueType.INT)],
        [row("retriever_top_k", "0", ConfigValueType.INT)],
        [row("ocr_lang_surya", "{}", ConfigValueType.LIST)],
        [row("retriever_top_k", "7", ConfigValueType.STR)],
    ],
)
def test_bad_snapshot_keeps_entire_previous_state_and_safe_error(
    configuration: SettingsProviderConfiguration, rows: list[ConfigParameter]
) -> None:
    configuration.settings.deepinfra_api_key = SecretStr("")
    configuration.apply([row("openrouter_model", "previous-model")])
    previous_version = configuration.status().applied_version
    with pytest.raises(ValidationError):
        configuration.apply([row("openrouter_model", "next-model"), *rows])
    assert configuration.settings.openrouter_model == "previous-model"
    status = configuration.status()
    assert status.applied_version == previous_version
    assert status.saved_version != previous_version
    assert not status.ready
    assert status.error
    assert "garbage" not in status.error


def test_domain_and_global_snapshot_apply_before_reactions(
    configuration: SettingsProviderConfiguration,
) -> None:
    observed = []
    configuration.bus.subscribe(
        ConfigParameterChanged,
        lambda _event: observed.append(
            (
                configuration.settings.openrouter_model,
                require_present(configuration.domain_settings).get("fingerprint_min_articles", "legal"),
            )
        ),
    )
    configuration.apply(
        [
            row("openrouter_model", "db-model"),
            row("fingerprint_min_articles", "7", ConfigValueType.INT, "legal"),
        ]
    )
    assert observed
    assert all(pair == ("db-model", "7") for pair in observed)


def test_subscriber_failure_is_not_reported_as_applied(configuration: SettingsProviderConfiguration) -> None:
    def broken(event: object) -> None:
        msg = "credential-like-value"
        raise RuntimeError(msg)

    configuration.bus.subscribe(ConfigParameterChanged, broken)
    with pytest.raises(RuntimeError):
        configuration.apply([row("openrouter_model", "db-model")])
    status = configuration.status()
    assert not status.ready
    assert status.applied_version is None
    assert "credential-like-value" not in require_present(status.error)


@pytest.mark.asyncio
async def test_restart_and_lost_notify_load_latest_writer_snapshot(
    configuration: SettingsProviderConfiguration,
) -> None:

    factory = FakeUnitOfWorkFactory()
    await factory._uow.fake_config_parameters.save(row("openrouter_model", "first"))
    listener = PostgresConfigListener(factory, configuration=configuration, database=MagicMock())
    await listener.resync()
    assert configuration.settings.openrouter_model == "first"
    await factory._uow.fake_config_parameters.update_value("openrouter_model", "latest")
    # Delayed/duplicate old NOTIFY must reread the writer, never replay stale values.
    listener.on_notify('{"key":"openrouter_model","new_value":"stale","value_type":"str"}')
    await asyncio.gather(*listener.transport.background_tasks)
    assert configuration.settings.openrouter_model == "latest"
    configuration.settings.openrouter_model = "restarted-process-default"
    await listener.resync(trigger="periodic")
    assert configuration.settings.openrouter_model == "latest"
    assert configuration.status().ready
    await listener.stop()


def test_provider_change_rejected_before_commit_without_required_key(
    configuration: SettingsProviderConfiguration,
) -> None:
    configuration.settings.openrouter_api_key = SecretStr("")
    with pytest.raises(ValidationError):
        configuration.validate_change("llm_provider", LLMProvider.OPENROUTER.value)


@pytest.mark.asyncio
async def test_service_validates_provider_before_any_database_write(
    configuration: SettingsProviderConfiguration,
) -> None:

    factory = FakeUnitOfWorkFactory()
    saved = row("llm_provider", LLMProvider.OPENROUTER.value)
    await factory._uow.fake_config_parameters.save(saved)
    configuration.settings.openrouter_api_key = SecretStr("")
    service = ConfigService(factory, configuration.bus, configuration)
    with pytest.raises(ValidationError):
        await service.update_parameter("llm_provider", LLMProvider.OPENROUTER.value)
    assert saved.value == LLMProvider.OPENROUTER.value


@pytest.mark.asyncio
async def test_status_reports_database_version_ahead_of_local_applied_snapshot(
    configuration: SettingsProviderConfiguration,
) -> None:

    factory = FakeUnitOfWorkFactory()
    saved = row("openrouter_model", "first")
    await factory._uow.fake_config_parameters.save(saved)
    configuration.apply([saved])
    await factory._uow.fake_config_parameters.update_value("openrouter_model", "remote-update")
    status = await ConfigService(factory, configuration.bus, configuration).configuration_status()
    assert status.saved_version != status.applied_version
    assert not status.ready
    assert configuration.settings.openrouter_model == "first"


@pytest.mark.asyncio
async def test_concurrent_resync_cannot_replay_an_older_snapshot(
    configuration: SettingsProviderConfiguration, monkeypatch: pytest.MonkeyPatch
) -> None:

    factory = FakeUnitOfWorkFactory()
    await factory._uow.fake_config_parameters.save(row("openrouter_model", "old"))
    started, release = asyncio.Event(), asyncio.Event()
    original = factory._uow.fake_config_parameters.get_all
    calls = 0

    async def get_all() -> list[ConfigParameter]:
        nonlocal calls
        calls += 1
        result = await original()

        if calls == 1:
            # Return independent rows to model READ COMMITTED snapshots.
            result = [row("openrouter_model", "old")]
            started.set()
            await release.wait()

        return result

    monkeypatch.setattr(factory._uow.fake_config_parameters, "get_all", get_all)
    listener = PostgresConfigListener(factory, configuration=configuration, database=MagicMock())
    first = asyncio.create_task(listener.synchronize())
    await started.wait()
    await factory._uow.fake_config_parameters.update_value("openrouter_model", "new")
    second = asyncio.create_task(listener.synchronize())
    release.set()
    await asyncio.gather(first, second)
    assert configuration.settings.openrouter_model == "new"


@pytest.mark.asyncio
async def test_worker_bootstrap_failure_closes_resources(monkeypatch: pytest.MonkeyPatch) -> None:

    closed = []

    async def close(label: str) -> None:
        closed.append(label)

    container = SimpleNamespace(
        init=lambda _database: None,
        infrastructure=SimpleNamespace(
            bootstrap_configuration=AsyncMock(side_effect=RuntimeError("bad config"))
        ),
        dispose=lambda: close("container"),
    )
    monkeypatch.setattr(worker, "Container", lambda: container)
    monkeypatch.setattr(database, "connect", AsyncMock())
    monkeypatch.setattr(database, "disconnect", lambda: close("database"))
    monkeypatch.setattr(redis_client, "init", AsyncMock())
    monkeypatch.setattr(redis_client, "aclose", lambda: close("redis"))
    ctx: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="bad config"):
        await worker.on_startup(ctx)
    assert closed == ["container", "redis", "database"]
    assert "container" not in ctx


@pytest.mark.asyncio
async def test_cli_bootstrap_failure_does_not_run_action_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    closed = []

    async def close(label: str) -> None:
        closed.append(label)

    infra = SimpleNamespace(
        init=lambda _database: None,
        bootstrap_configuration=AsyncMock(side_effect=RuntimeError("bad config")),
        dispose=lambda: close("container"),
    )
    monkeypatch.setattr(ingest, "InfrastructureContainer", lambda: infra)
    monkeypatch.setattr(database, "connect", AsyncMock())
    monkeypatch.setattr(database, "disconnect", lambda: close("database"))
    monkeypatch.setattr(redis_client, "init", AsyncMock())
    monkeypatch.setattr(redis_client, "aclose", lambda: close("redis"))
    action = AsyncMock()
    with pytest.raises(RuntimeError, match="bad config"):
        await ingest.run_with_infrastructure(action)
    action.assert_not_awaited()
    assert closed == ["container", "database", "redis"]


def test_deepinfra_production_can_construct_before_db_load_and_without_tei_urls(
    configuration: SettingsProviderConfiguration,
) -> None:

    model = Settings(
        _env_file=None,
        stage=AppStage.PROD,
        ml_provider="deepinfra",
        allowed_origins="https://app.example.test",
        deepinfra_api_key=SecretStr("private-test-key"),
        tei_embed_url="",
        tei_rerank_url="",
    )
    configuration.settings = model
    configuration.apply([row("ml_provider", "deepinfra")])
    assert model.ml_provider == "deepinfra"
    assert configuration.status().ready


@pytest.mark.asyncio
@pytest.mark.parametrize("corpus_ready", [None, False, True])
async def test_readiness_rejects_failed_configuration_snapshot(
    configuration: SettingsProviderConfiguration, corpus_ready: bool | None
) -> None:

    probes = SimpleNamespace(
        **{
            name: AsyncMock(return_value=HealthCheckResult(HealthStatus.OK.value))
            for name in ("check_qdrant", "check_postgres", "check_redis", "check_openrouter")
        },
        check_tei=AsyncMock(return_value={}),
    )
    service = HealthService(
        FakeUnitOfWorkFactory(),
        probes,
        SimpleNamespace(is_connected=True),
        SimpleNamespace(
            ml_provider="tei", llm_provider=LLMProvider.OPENROUTER.value, version="test", uptime_seconds=1
        ),
        configuration=configuration,
        corpus_control=SimpleNamespace(is_ready=AsyncMock(return_value=corpus_ready))
        if corpus_ready is not None
        else None,
    )

    if corpus_ready is not None:
        configuration.apply([row("openrouter_model", "ready-model")])

    result = await service.check()

    if corpus_ready is True:
        assert result.status == HealthStatus.HEALTHY.value
    else:
        assert result.status == HealthStatus.DEGRADED.value
        check = "corpus" if corpus_ready is False else "configuration"
        assert result.checks[check].status.startswith("error")

    response = Response()
    await ready(response=response, health_service=service)
    assert response.status_code == (200 if corpus_ready is True else 503)


@pytest.mark.asyncio
async def test_configuration_status_endpoint_is_admin_only_and_contains_no_secrets(
    configuration: SettingsProviderConfiguration,
) -> None:

    app = FastAPI()
    app.include_router(router)
    factory = FakeUnitOfWorkFactory()
    await factory._uow.fake_config_parameters.save(row("openrouter_model", "saved-model"))

    async def service_dependency() -> ConfigService:
        return ConfigService(factory, configuration.bus, cast("ProviderConfigurationPort", configuration))

    app.dependency_overrides[create_config_service] = service_dependency

    async def deny() -> None:
        raise HTTPException(403)

    app.dependency_overrides[require_admin] = deny
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:  # type: ignore[arg-type]  # Starlette and HTTPX disagree on the ASGI send annotation.
        assert (await client.get("/admin/config/status")).status_code == 403

        async def allow() -> object:
            return object()

        app.dependency_overrides[require_admin] = allow
        response = await client.get("/admin/config/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["saved_version"]
        assert payload["applied_version"] is None
        assert not payload["ready"]
        assert set(payload) == {"saved_version", "applied_version", "ready", "error"}


def test_domain_invalid_value_is_rejected_before_update(configuration: SettingsProviderConfiguration) -> None:
    with pytest.raises(ValidationError):
        configuration.validate_domain_change("fingerprint_min_articles", "-1", "legal")
    with pytest.raises(ValidationError):
        configuration.validate_domain_change("openrouter_model", "other", "legal")


def test_missing_snapshot_key_after_bootstrap_does_not_silently_keep_stale_value(
    configuration: SettingsProviderConfiguration,
) -> None:
    complete = initialization.build_defaults()

    pairs = {(item.key, item.domain_key) for item in complete}

    for profile in (
        LegalDomainProfile(),
        DecreeDomainProfile(),
        GeneralDomainProfile(),
        ContractDomainProfile(),
    ):
        for default in profile.config_defaults():
            if (default.key, profile.key) not in pairs:
                complete.append(
                    ConfigParameter(
                        default.key,
                        default.value,
                        default.value_type,
                        "domain:" + profile.key,
                        domain_key=profile.key,
                    )
                )

    configuration.apply(complete, require_complete=True)
    with pytest.raises(ValidationError):
        configuration.apply([item for item in complete if item.key != "openrouter_model"])
    assert not configuration.status().ready


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_config_cli_resync_uses_same_snapshot_and_reports_invalid_database(
    monkeypatch: pytest.MonkeyPatch, malformed: bool
) -> None:

    infra, bus = build_infrastructure(monkeypatch)
    infra.corpus_control = AsyncMock()
    monkeypatch.setattr(infra, "init", lambda _database: setattr(infra, "_initialized", True))
    await require_present(
        cast("FakeUnitOfWorkFactory", infra.db.uow_factory)._uow.fake_config_parameters
    ).save(row("openrouter_model", "cli-db-model"))
    await require_present(
        cast("FakeUnitOfWorkFactory", infra.db.uow_factory)._uow.fake_config_parameters
    ).save(row("retriever_top_k", "garbage" if malformed else "7", ConfigValueType.INT))
    await require_present(
        cast("FakeUnitOfWorkFactory", infra.db.uow_factory)._uow.fake_config_parameters
    ).save(row("fingerprint_min_articles", "9", ConfigValueType.INT, "legal"))
    monkeypatch.setattr(ingest, "InfrastructureContainer", lambda: infra)
    monkeypatch.setattr(database, "connect", AsyncMock())
    monkeypatch.setattr(database, "disconnect", AsyncMock())
    monkeypatch.setattr(redis_client, "init", AsyncMock())
    monkeypatch.setattr(redis_client, "aclose", AsyncMock())

    if malformed:
        with pytest.raises(ValidationError):
            await config.run_config_resync()
        assert (
            require_present(require_present(infra.configuration).settings).openrouter_model
            == "qwen/qwen-2.5-7b-instruct"
        )
    else:
        status = await config.run_config_resync()
        assert status.ready
        assert (
            require_present(require_present(infra.configuration).settings).openrouter_model == "cli-db-model"
        )
        assert require_present(infra.domain_settings).get("fingerprint_min_articles", "legal") == "9"

    assert not infra._initialized
    assert not bus._handlers[ConfigParameterChanged]


@pytest.mark.parametrize("daily", [5, 10, 14401])
def test_incompatible_chat_quotas_rejected_before_application(
    configuration: SettingsProviderConfiguration, daily: int
) -> None:
    configuration.settings.rate_limit_chat_per_minute = 10
    previous = configuration.settings.rate_limit_chat_per_day
    with pytest.raises(ValidationError, match="quota"):
        configuration.validate_change("rate_limit_chat_per_day", str(daily))
    with pytest.raises(ValidationError, match="quota"):
        configuration.apply([row("rate_limit_chat_per_day", str(daily), ConfigValueType.INT)])
    assert configuration.settings.rate_limit_chat_per_day == previous


@pytest.mark.asyncio
async def test_config_service_rejects_chat_quota_against_saved_snapshot(
    configuration: SettingsProviderConfiguration,
) -> None:

    factory = FakeUnitOfWorkFactory()
    async with factory.create() as uow:
        await uow.config_parameters.save(row("rate_limit_chat_per_minute", "10", ConfigValueType.INT))
        await uow.config_parameters.save(row("rate_limit_chat_per_day", "100", ConfigValueType.INT))
    # Validation must use durable rows, even when this replica is behind.
    configuration.settings.rate_limit_chat_per_minute = 1
    service = ConfigService(factory, configuration.bus, configuration)
    with pytest.raises(ValidationError, match="quota"):
        await service.update_parameter("rate_limit_chat_per_day", "5")
    async with factory.create() as uow:
        saved = await uow.config_parameters.get_by_key("rate_limit_chat_per_day")
        assert require_present(saved).value == "100"


@pytest.mark.asyncio
async def test_budget_policy_is_durable_before_bootstrap_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    infra, _ = build_infrastructure(monkeypatch)
    runtime = BudgetRuntime()
    session = AsyncMock()
    session.execute.return_value = MagicMock()
    session.execute.return_value.all.return_value = [
        ("cost_rate_limit_enabled", "true"),
        ("cost_hourly_limit", "1"),
        ("cost_daily_limit", "5"),
    ]
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    ledger = object()
    catalog = SimpleNamespace(contract=AsyncMock(side_effect=BudgetError("Unknown price; paid call blocked")))
    runtime.configure(cast("CostBudgetPort", ledger), factory, catalog)
    monkeypatch.setattr(infra, "budget_admin", SimpleNamespace(repository=ledger))
    infra.budget_runtime = runtime
    monkeypatch.setattr(settings, "cost_rate_limit_enabled", False)

    post = AsyncMock()
    monkeypatch.setattr(
        managed_http,
        "PaidAsyncTransport",
        lambda **_kwargs: PaidAsyncTransport(runtime=runtime, transport=httpx.MockTransport(post)),
    )

    class BillablePool(managed_http.ManagedHTTPPool):
        billable_provider = "deepinfra"

    pool = BillablePool(10, httpx.Limits())

    async def prepare(**kwargs: Any) -> None:
        await pool.post("https://api.deepinfra.com/v1/inference/model", json={"queries": ["probe"]})

    monkeypatch.setattr(require_present(infra.events.config_listener), "synchronize", prepare)
    try:
        with pytest.raises(BudgetError, match="paid call blocked"):
            await infra.bootstrap_configuration()
        post.assert_not_awaited()
        catalog.contract.assert_awaited_once_with("deepinfra", "model")
        cast("AsyncMock", require_present(require_present(infra.corpus_control).start)).assert_not_awaited()
    finally:
        await pool.close()
