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
from pydantic import SecretStr
import pytest
from sqlalchemy import MetaData, Table, create_engine, func, inspect, select, text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from config import settings
from domain.ingestion.profiles.registry import DomainProfileRegistry
from infrastructure.domain_packs.belarus_law.pack import BELARUS_LAW_PACK
from infrastructure.domain_profile import register_pack_profiles
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


def test_knowledge_space_upgrade_preserves_existing_act_scope_identities(
    migration_connection: Connection,
) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = migration_connection
    command.upgrade(config, "0012")
    scopes = ["internal_public", "internal_private:7", "internal_group:4", "client_private:9"]

    for scope in scopes:
        migration_connection.execute(
            text(
                "INSERT INTO regulatory_acts (creation_date, act_type, act_number, title, visibility_scope) "
                "VALUES (NOW(), 'decree', '5', 'Test act', :scope)"
            ),
            {"scope": scope},
        )

    migration_connection.commit()
    command.upgrade(config, "head")
    space_id = migration_connection.scalar(text("SELECT id FROM knowledge_spaces"))
    assert set(migration_connection.scalars(text("SELECT visibility_scope FROM regulatory_acts"))) == {
        f"space:{space_id}:{scope}" for scope in scopes
    }
    migration_connection.commit()
    command.downgrade(config, "0012")
    assert set(migration_connection.scalars(text("SELECT visibility_scope FROM regulatory_acts"))) == set(
        scopes
    )


def test_baseline_round_trip_matches_metadata(migration_connection: Connection) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = migration_connection
    command.upgrade(config, "head")
    command.check(config)

    inspector = inspect(migration_connection)
    tables = set(inspector.get_table_names()) - {"alembic_version"}
    # 0016 adds the non-secret RAG configuration snapshot table.
    assert len(tables) == 27

    # Only the starting knowledge space exists in a new installation.
    assert [
        tuple(row)
        for row in migration_connection.execute(text("SELECT name, pack_key FROM knowledge_spaces"))
    ] == [("Общее", "generic")]

    for name in tables - {"knowledge_spaces"}:
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


def config_rows(connection: Connection) -> dict[str, str]:
    rows = connection.execute(text("SELECT key, value FROM config_parameters WHERE domain_key IS NULL"))
    return {row.key: row.value for row in rows}


def insert_config_rows(connection: Connection, values: dict[str, str]) -> None:
    for key, value in values.items():
        connection.execute(
            text("INSERT INTO config_parameters (key, value, category) VALUES (:key, :value, 'legacy')"),
            {"key": key, "value": value},
        )


@pytest.mark.parametrize(
    "legacy, expected",
    [
        (
            {
                "llm_provider": "openai_compatible",
                "openrouter_model": "cloud-model",
                "openai_compatible_base_url": "http://vllm:8000/v1",
                "openai_compatible_model": "local-model",
                "openai_compatible_auxiliary_model": "small-model",
            },
            {
                "llm_base_url": "http://vllm:8000/v1",
                "llm_model": "local-model",
                "llm_auxiliary_model": "small-model",
            },
        ),
        (
            {
                "llm_provider": "openrouter",
                "openrouter_base_url": "https://openrouter.ai/api/v1",
                "openrouter_model": "cloud-model",
                "openai_compatible_model": "local-model",
            },
            {"llm_base_url": "https://openrouter.ai/api/v1", "llm_model": "cloud-model"},
        ),
    ],
)
def test_unified_llm_endpoint_migration_keeps_active_provider(
    migration_connection: Connection, legacy: dict[str, str], expected: dict[str, str]
) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = migration_connection
    command.upgrade(config, "0009")
    insert_config_rows(migration_connection, legacy)
    command.upgrade(config, "0010")
    assert config_rows(migration_connection) == expected
    command.downgrade(config, "0009")
    restored = config_rows(migration_connection)
    assert restored["llm_provider"] == legacy["llm_provider"]
    assert not set(restored) & {"llm_base_url", "llm_model", "llm_auxiliary_model"}


def test_generic_ml_endpoint_migration_drops_provider_keys(migration_connection: Connection) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = migration_connection
    command.upgrade(config, "0010")
    insert_config_rows(
        migration_connection,
        {
            "ml_provider": "tei",
            "tei_embed_url": "http://tei-embed:8080",
            "deepinfra_rerank_model": "custom-reranker",
            "retriever_top_k": "5",
        },
    )
    command.upgrade(config, "0011")
    assert config_rows(migration_connection) == {"rerank_model": "custom-reranker", "retriever_top_k": "5"}
    command.downgrade(config, "0010")
    assert config_rows(migration_connection) == {"retriever_top_k": "5"}


@pytest.mark.asyncio
async def test_baseline_supports_idempotent_configuration_bootstrap(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    database = MagicMock()
    database.get_session.side_effect = lambda **_kwargs: sessions()
    factory = UnitOfWorkFactory(database)
    registry = DomainProfileRegistry()
    register_pack_profiles(registry, BELARUS_LAW_PACK, DomainSettingsAdapter())
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
    # The password must really authenticate: a role of its own owns the disposable database.
    role = f"migrator_{uuid.uuid4().hex[:12]}"
    password = "p@ss:/?#%"  # noqa: S105 - disposable test role
    migration_connection.execute(text(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}'"))
    migration_connection.execute(text(f'ALTER DATABASE "{url.database}" OWNER TO "{role}"'))
    migration_connection.commit()
    monkeypatch.setattr(settings, "db_user", role)
    monkeypatch.setattr(settings, "db_password", SecretStr(password))
    monkeypatch.setattr(settings, "db_host", url.host or url.query.get("host", "localhost"))
    monkeypatch.setattr(settings, "db_port", str(url.port or url.query.get("port", 5432)))
    monkeypatch.setattr(settings, "db_name", url.database)
    try:
        migrate.main()
        assert "config_parameters" in inspect(migration_connection).get_table_names()
    finally:
        migration_connection.rollback()
        migration_connection.execute(text(f'REASSIGN OWNED BY "{role}" TO CURRENT_USER'))
        migration_connection.execute(text(f'DROP OWNED BY "{role}"'))
        migration_connection.execute(text(f'DROP ROLE "{role}"'))
        migration_connection.commit()


def test_answer_attempt_migration_clears_legacy_logs_and_preserves_messages(
    migration_connection: Connection,
) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = migration_connection
    command.upgrade(config, "0015")
    migration_connection.execute(text("INSERT INTO chat_logs (question, answer) VALUES ('old', 'answer')"))
    migration_connection.execute(
        text("INSERT INTO messages (role, content) VALUES ('assistant', 'delivered')")
    )
    migration_connection.commit()
    command.upgrade(config, "head")
    assert migration_connection.scalar(text("SELECT count(*) FROM chat_logs")) == 0
    assert migration_connection.scalar(text("SELECT count(*) FROM messages")) == 1
    columns = {column["name"] for column in inspect(migration_connection).get_columns("chat_logs")}
    assert {"outcome", "space_id", "message_id", "config_hash", "sources_count"} <= columns
    assert not columns & {"retrieval_count", "reranker_score"}
    migration_connection.commit()
    command.downgrade(config, "0015")
    assert migration_connection.scalar(text("SELECT count(*) FROM messages")) == 1
