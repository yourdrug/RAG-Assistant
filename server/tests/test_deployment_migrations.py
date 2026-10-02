"""Migration connection options preserve arbitrary passwords and release the advisory lock."""

from unittest.mock import MagicMock

import pytest

from scripts import migrate


def test_migration_does_not_interpolate_password_into_dsn(monkeypatch):
    connect = MagicMock()
    monkeypatch.setattr(migrate.psycopg, "connect", connect)
    monkeypatch.setattr(migrate.settings, "db_password", "p@ss:word/with?special#characters")
    upgrade = MagicMock()
    monkeypatch.setattr(migrate.command, "upgrade", upgrade)
    migrate.main()
    assert not connect.call_args.args
    assert connect.call_args.kwargs["password"] == "p@ss:word/with?special#characters"
    assert connect.call_args.kwargs["connect_timeout"] == 10
    upgrade.assert_called_once()


def test_migration_failure_releases_advisory_lock(monkeypatch):
    connect = MagicMock()
    monkeypatch.setattr(migrate.psycopg, "connect", connect)
    monkeypatch.setattr(migrate.command, "upgrade", MagicMock(side_effect=RuntimeError("migration failed")))
    with pytest.raises(RuntimeError, match="migration failed"):
        migrate.main()
    cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    assert cursor.execute.call_args.args == ("SELECT pg_advisory_unlock(%s)", (migrate.LOCK_ID,))
