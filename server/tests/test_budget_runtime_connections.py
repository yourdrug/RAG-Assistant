"""Synchronous budget reads use the configured database and fail closed."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from composition import infrastructure
from composition.infrastructure import InfrastructureContainer
from infrastructure.budget.runtime import BudgetRuntime
from infrastructure.database.database import DatabaseManager


@pytest.fixture
def runtime() -> BudgetRuntime:
    return BudgetRuntime()


def test_sync_policy_reads_through_database_manager(runtime: BudgetRuntime) -> None:
    runtime.bootstrapped = True
    database = MagicMock()
    runtime.database = database
    connection = database.master_connection.return_value.__enter__.return_value
    connection.execute.return_value.all.return_value = [
        ("cost_rate_limit_enabled", "true"),
        ("cost_hourly_limit", "2"),
        ("cost_daily_limit", "10"),
    ]
    policy = runtime.policy_sync()
    assert policy.enabled
    assert policy.hourly_usd == "2"
    assert policy.daily_usd == "10"
    database.master_connection.assert_called_once_with()
    assert str(connection.execute.call_args_list[0].args[0]) == "SET statement_timeout = '5s'"
    database.master_connection.return_value.__exit__.assert_called_once()


def test_sync_policy_database_failure_is_not_bypassed(runtime: BudgetRuntime) -> None:
    runtime.bootstrapped = True
    database = MagicMock()
    runtime.database = database
    database.master_connection.side_effect = ConnectionError
    with pytest.raises(ConnectionError):
        runtime.policy_sync()


@pytest.mark.asyncio
async def test_configure_uses_injected_database(runtime: BudgetRuntime) -> None:
    database = MagicMock()
    runtime.configure(AsyncMock(), MagicMock(), MagicMock(), database_manager=database)
    assert runtime.database is database


@pytest.mark.asyncio
async def test_container_runtimes_keep_their_resources_when_another_is_configured() -> None:
    first, second = InfrastructureContainer(), InfrastructureContainer()
    ledger, sessions, catalog, database = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    first.budget_runtime.configure(ledger, sessions, catalog, database_manager=database)
    first.budget_runtime.bootstrapped = True
    second.budget_runtime.configure(MagicMock(), MagicMock(), MagicMock(), database_manager=MagicMock())

    assert first.budget_runtime is not second.budget_runtime
    assert first.budget_runtime.service is not None
    assert first.budget_runtime.service.ledger is ledger
    assert first.budget_runtime.session_factory is sessions
    assert first.budget_runtime.catalog is catalog
    assert first.budget_runtime.database is database
    assert first.budget_runtime.bootstrapped
    assert not second.budget_runtime.bootstrapped


@pytest.mark.asyncio
async def test_container_init_wires_its_runtime_to_clients_and_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(infrastructure, "redis_client", MagicMock())
    monkeypatch.setattr(infrastructure.ServiceContainer, "init", lambda _self: None)
    monkeypatch.setattr(infrastructure, "PostgresCorpusControl", MagicMock(return_value=AsyncMock()))
    first, second = InfrastructureContainer(), InfrastructureContainer()
    first_db, second_db = MagicMock(spec=DatabaseManager), MagicMock(spec=DatabaseManager)
    try:
        first.init(first_db)
        assert first.budget_runtime.service is not None
        ledger = first.budget_runtime.service.ledger
        first.budget_runtime.bootstrapped = True
        second.init(second_db)
        assert first.ml.clients.budget_runtime is first.budget_runtime
        assert second.ml.clients.budget_runtime is second.budget_runtime
        assert first.budget_runtime.service.ledger is ledger
        assert first.budget_runtime.database is first_db
        assert second.budget_runtime.database is second_db
        assert first.budget_runtime.bootstrapped
        await second.dispose()
        assert first.budget_runtime.bootstrapped
    finally:
        await second.dispose()
        await first.dispose()
