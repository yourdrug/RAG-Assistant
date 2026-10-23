"""Alembic uses the shared URL and manager without losing transaction boundaries."""

from importlib import import_module
from pathlib import Path
import runpy
from unittest.mock import AsyncMock, MagicMock

from alembic import context
import pytest

from config import settings


ENV_PATH = Path(__file__).parents[1] / "app/infrastructure/database/migrations/env.py"


def migration_context(monkeypatch: pytest.MonkeyPatch, *, offline: bool) -> MagicMock:
    config = MagicMock(config_file_name=None, attributes={})
    monkeypatch.setattr(context, "config", config, raising=False)
    monkeypatch.setattr(context, "configure", config.configure_context)
    monkeypatch.setattr(context, "begin_transaction", MagicMock())
    monkeypatch.setattr(context, "run_migrations", MagicMock())
    monkeypatch.setattr(context, "is_offline_mode", MagicMock(return_value=offline))
    return config


def test_offline_migrations_preserve_password_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    config = migration_context(monkeypatch, offline=True)
    monkeypatch.setattr(settings, "db_password", "p@ss:/?#%")
    runpy.run_path(str(ENV_PATH))
    assert config.configure_context.call_args.kwargs["url"].password == settings.db_password
    config.set_section_option.assert_not_called()


def test_online_migrations_use_transactional_manager_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    migration_context(monkeypatch, offline=False)
    database = MagicMock()
    connection = AsyncMock()
    database.async_master_connection.return_value.__aenter__.return_value = connection
    monkeypatch.setattr(
        import_module("infrastructure.database.database"), "DatabaseManager", lambda: database
    )
    runpy.run_path(str(ENV_PATH))
    database.async_master_connection.assert_called_once_with(
        connect_args={"options": "-c statement_timeout=0"}
    )
    connection.execution_options.assert_awaited_once_with(isolation_level="READ COMMITTED")
    connection.run_sync.assert_awaited_once()
    database.async_master_connection.return_value.__aexit__.assert_awaited_once()


def test_injected_migration_connection_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    config = migration_context(monkeypatch, offline=False)
    connection = MagicMock()
    config.attributes["connection"] = connection
    runpy.run_path(str(ENV_PATH))
    config.configure_context.assert_called_once()
    assert config.configure_context.call_args.kwargs["connection"] is connection
