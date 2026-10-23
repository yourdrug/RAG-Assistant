"""Baseline upgrade/check/downgrade; TEST_POSTGRES_URL needs CREATEDB."""

import asyncio
from collections.abc import Iterator
import json
import os
from pathlib import Path
from unittest.mock import MagicMock
import uuid

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import MetaData, Table, create_engine, func, inspect, select, text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from config import settings
from domain.ingestion.profiles.registry import DomainProfileRegistry
from infrastructure.domain_profile import register_all_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.initialization import build_defaults, seed_config_defaults, seed_domain_config_defaults
from infrastructure.uow_factory import UnitOfWorkFactory
from scripts import migrate


pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("error::sqlalchemy.exc.SAWarning")]


@pytest.fixture
def migration_connection() -> Iterator[Connection]:
    url = os.environ.get("TEST_POSTGRES_URL")

    if not url:
        pytest.skip("Set TEST_POSTGRES_URL for PostgreSQL integration tests")

    admin_url = make_url(url).set(drivername="postgresql+psycopg")
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    database = f"test_{uuid.uuid4().hex}"
    engine = create_engine(admin_url.set(database=database))
    try:
        with admin.connect() as control:
            control.execute(text(f'CREATE DATABASE "{database}"'))
            try:
                with engine.connect() as connection:
                    yield connection
            finally:
                engine.dispose()
                control.execute(text(f'DROP DATABASE "{database}"'))
    finally:
        admin.dispose()


def test_baseline_round_trip_matches_metadata(migration_connection: Connection) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = migration_connection
    command.upgrade(config, "head")
    command.check(config)

    inspector = inspect(migration_connection)
    tables = set(inspector.get_table_names()) - {"alembic_version"}
    assert len(tables) == 25

    for name in tables:
        assert migration_connection.scalar(select(func.count()).select_from(Table(name, MetaData()))) == 0

    expected_checks = json.loads((Path(__file__).parents[1] / "fixtures/baseline_checks.json").read_text())
    actual_checks = migration_connection.execute(
        text(
            "SELECT c.relname, x.conname, pg_get_constraintdef(x.oid) "
            "FROM pg_constraint x JOIN pg_class c ON c.oid=x.conrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND x.contype='c' ORDER BY c.relname,x.conname"
        )
    )
    assert [list(row) for row in actual_checks] == expected_checks
    gin = migration_connection.scalar(
        text(
            "SELECT indexdef FROM pg_indexes WHERE schemaname='public' AND indexname='ix_chunks_content_trgm'"
        )
    )
    assert gin is not None
    assert "gin_trgm_ops" in gin
    registry_id = migration_connection.scalar(
        text(
            "INSERT INTO ingestion_registry (ingestion_key, filename) VALUES ('a.txt', 'a.txt') RETURNING id"
        )
    )
    assert registry_id is not None
    timestamp = migration_connection.scalar(
        text("SELECT creation_date FROM ingestion_registry WHERE id=:id"), {"id": registry_id}
    )
    assert timestamp is not None
    with pytest.raises(IntegrityError), migration_connection.begin_nested():
        migration_connection.execute(
            text(
                "INSERT INTO ingestion_registry (ingestion_key, filename, creation_date) "
                "VALUES ('rejected.txt', 'rejected.txt', NULL)"
            )
        )
    migration_connection.commit()
    command.downgrade(config, "base")
    assert inspect(migration_connection).get_table_names() == ["alembic_version"]
    assert migration_connection.scalar(text("SELECT count(*) FROM pg_extension WHERE extname='pg_trgm'")) == 1
    migration_connection.commit()
    command.upgrade(config, "head")
    command.check(config)


@pytest.mark.asyncio
async def test_baseline_supports_idempotent_configuration_bootstrap(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    database = MagicMock()
    database.get_session.side_effect = lambda **_kwargs: sessions()
    factory = UnitOfWorkFactory(database)
    registry = DomainProfileRegistry()
    register_all_profiles(registry, DomainSettingsAdapter())
    defaults = build_defaults()
    expected = {(row.key, row.domain_key): row.value for row in defaults}

    for profile in registry.all():
        for row in profile.config_defaults():
            expected.setdefault((row.key, profile.key), row.value)

    async def bootstrap() -> None:
        await seed_config_defaults(factory)
        await seed_domain_config_defaults(factory, registry)

    await asyncio.gather(bootstrap(), bootstrap())
    async with factory.create(master=True) as uow:
        rows = await uow.config_parameters.get_all()
        assert {(row.key, row.domain_key): row.value for row in rows} == expected
        await uow.config_parameters.update_value("cache_enabled", "false")
    await bootstrap()
    async with factory.create(master=True) as uow:
        rows = await uow.config_parameters.get_all()
        assert len(rows) == len(expected)
        cache = await uow.config_parameters.get_by_key_and_domain("cache_enabled")
        assert cache is not None
        assert cache.value == "false"


def test_standalone_migration_uses_manager_with_special_password(
    migration_connection: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = migration_connection.engine.url
    monkeypatch.setattr(settings, "db_user", url.username)
    monkeypatch.setattr(settings, "db_password", "p@ss:/?#%")
    monkeypatch.setattr(settings, "db_host", url.host or url.query.get("host", "localhost"))
    monkeypatch.setattr(settings, "db_port", str(url.port or url.query.get("port", 5432)))
    monkeypatch.setattr(settings, "db_name", url.database)
    migrate.main()
    assert "config_parameters" in inspect(migration_connection).get_table_names()
