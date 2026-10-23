"""Database connection ownership, driver-safe URLs and lock cleanup."""

import asyncio
from importlib import import_module
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from config import settings
from infrastructure.database.database import DatabaseManager, DatabaseNode
from infrastructure.database.dataclasses import DatabaseURL
from infrastructure.database.enums import DatabaseNodeRole
from infrastructure.uow_factory import UnitOfWorkFactory


@pytest.mark.parametrize("password", ["p@ss:word/with?special#characters%", "кириллица @:/%"])
def test_database_url_round_trip(password: str) -> None:
    components = DatabaseURL("user@name", password, "::1", "5433", "rag")
    url = make_url(str(components))
    assert url.username == "user@name"
    assert url.password == password
    assert url.host == "::1"
    assert url.port == 5433
    assert url == components.url


@pytest.mark.parametrize("failure", [False, True])
def test_standalone_sync_connection_disposes_engine(monkeypatch: pytest.MonkeyPatch, failure: bool) -> None:
    monkeypatch.setattr(settings, "db_password", "p@ss:/?#%")
    engine = MagicMock()
    create = MagicMock(return_value=engine)
    monkeypatch.setattr(import_module("infrastructure.database.database"), "create_engine", create)

    if failure:
        engine.connect.side_effect = ConnectionError
        with pytest.raises(ConnectionError), DatabaseManager().master_connection():
            pass
    else:
        with DatabaseManager().master_connection(connect_args={"connect_timeout": 10}) as connection:
            assert connection is engine.connect.return_value.__enter__.return_value
        assert create.call_args.kwargs["connect_args"]["connect_timeout"] == 10
        assert "statement_timeout" in create.call_args.kwargs["connect_args"]["options"]

    assert create.call_args.args[0].password == settings.db_password
    assert create.call_args.kwargs["poolclass"] is NullPool
    assert create.call_args.kwargs["isolation_level"] == "AUTOCOMMIT"
    engine.dispose.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_standalone_async_connection_disposes_engine(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    engine = MagicMock(dispose=AsyncMock())
    create = MagicMock(return_value=engine)
    monkeypatch.setattr(import_module("infrastructure.database.database"), "create_async_engine", create)

    if failure:
        engine.connect.side_effect = ConnectionError
        with pytest.raises(ConnectionError):
            async with DatabaseManager().async_master_connection():
                pass
    else:
        async with DatabaseManager().async_master_connection() as connection:
            assert connection is engine.connect.return_value.__aenter__.return_value

    assert create.call_args.kwargs["poolclass"] is NullPool
    engine.dispose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_initialized_manager_reuses_and_retains_dedicated_engines() -> None:
    database = DatabaseManager()
    sync_engine = MagicMock()
    async_engine = MagicMock(dispose=AsyncMock())
    database.master_connection_engine = sync_engine
    database.async_master_connection_engine = async_engine
    with database.master_connection():
        pass
    async with database.async_master_connection():
        pass
    sync_engine.dispose.assert_not_called()
    async_engine.dispose.assert_not_awaited()
    await database.disconnect()
    sync_engine.dispose.assert_called_once_with()
    async_engine.dispose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_ingestion_lock_cancellation_releases_session() -> None:
    database = MagicMock()
    connection = AsyncMock()
    context = database.async_master_connection.return_value
    context.__aenter__.return_value = connection
    factory = UnitOfWorkFactory(database)
    started = asyncio.Event()

    async def hold() -> None:
        async with factory.ingestion_lock():
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold())
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    statements = [str(call.args[0]) for call in connection.execute.await_args_list]
    assert statements == ["SELECT pg_advisory_lock(724193820146)", "SELECT pg_advisory_unlock(724193820146)"]
    context.__aexit__.assert_awaited_once()
    database.get_session.assert_not_called()


@pytest.mark.asyncio
async def test_ingestion_lock_acquisition_failure_closes_connection() -> None:
    database = MagicMock()
    connection = AsyncMock()
    connection.execute.side_effect = ConnectionError
    context = database.async_master_connection.return_value
    context.__aenter__.return_value = connection
    with pytest.raises(ConnectionError):
        async with UnitOfWorkFactory(database).ingestion_lock():
            pytest.fail("Lock acquisition must fail")
    context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_after_partial_initialization_is_safe() -> None:
    database = DatabaseManager()
    database.master_node = DatabaseNode(
        DatabaseNodeRole.MASTER, DatabaseURL("u", "p", "localhost", "5432", "rag"), 0
    )
    database.slave_nodes = [
        database.master_node,
        DatabaseNode(DatabaseNodeRole.SLAVE, database.master_node.node_url, 1),
    ]
    await database.disconnect()
    await database.disconnect()


def test_override_connection_retains_manager_database(monkeypatch: pytest.MonkeyPatch) -> None:
    database = DatabaseManager()
    configured = MagicMock(url=DatabaseURL("u", "p", "custom-host", "5433", "custom-db").url)
    database.master_connection_engine = configured
    temporary = MagicMock()
    create = MagicMock(return_value=temporary)
    monkeypatch.setattr(import_module("infrastructure.database.database"), "create_engine", create)
    with database.master_connection(connect_args={"connect_timeout": 10}):
        pass
    assert create.call_args.args[0] == configured.url
    configured.dispose.assert_not_called()
    temporary.dispose.assert_called_once_with()
