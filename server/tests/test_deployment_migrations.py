"""Migrations use manager-owned connections and always release advisory locks."""

from unittest.mock import MagicMock

import pytest

from scripts import migrate


def test_migration_uses_manager_and_preserves_lock_timeout(monkeypatch):
    database = MagicMock()
    monkeypatch.setattr(migrate, "DatabaseManager", MagicMock(return_value=database))
    upgrade = MagicMock()
    monkeypatch.setattr(migrate.command, "upgrade", upgrade)
    migrate.main()
    database.master_connection.assert_called_once_with(
        connect_args={"connect_timeout": 10, "options": "-c statement_timeout=0"}
    )
    connection = database.master_connection.return_value.__enter__.return_value
    assert str(connection.execute.call_args_list[0].args[0]) == "SET lock_timeout = '60s'"
    assert str(connection.execute.call_args_list[1].args[0]) == "SELECT pg_advisory_lock(:key)"
    upgrade.assert_called_once()
    database.master_connection.return_value.__exit__.assert_called_once()


def test_migration_failure_releases_advisory_lock(monkeypatch):
    database = MagicMock()
    monkeypatch.setattr(migrate, "DatabaseManager", MagicMock(return_value=database))
    monkeypatch.setattr(migrate.command, "upgrade", MagicMock(side_effect=RuntimeError("migration failed")))
    with pytest.raises(RuntimeError, match="migration failed"):
        migrate.main()
    connection = database.master_connection.return_value.__enter__.return_value
    statement, parameters = connection.execute.call_args.args
    assert str(statement) == "SELECT pg_advisory_unlock(:key)"
    assert parameters == {"key": migrate.LOCK_ID}
    database.master_connection.return_value.__exit__.assert_called_once()
