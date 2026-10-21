"""Service metadata matches the deployed migrations, including database constraints."""

from typing import Any

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
import pytest
from sqlalchemy import inspect
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from infrastructure.database.basemodel import BaseModel


pytestmark = pytest.mark.integration
SERVICE_TABLES = ("cost_usage", "cost_rates", "corpus_processes", "corpus_control")


def include_service_table(obj: Any, name: str | None, kind: str, reflected: bool, other: Any) -> bool:
    return kind != "table" or name in SERVICE_TABLES


def check_definitions(connection: Connection) -> dict[str, list[str]]:
    inspector = inspect(connection)
    return {
        name: sorted(constraint["sqltext"] for constraint in inspector.get_check_constraints(name))
        for name in SERVICE_TABLES
    }


def compare_migrated_schema(connection: Connection) -> None:
    expected_checks = check_definitions(connection)

    context = MigrationContext.configure(
        connection,
        opts={"include_object": include_service_table, "compare_server_default": True},
    )
    assert compare_metadata(context, BaseModel.metadata) == []
    assert check_definitions(connection) == expected_checks


@pytest.mark.asyncio
async def test_service_tables_match_migrations(pg_engine: AsyncEngine) -> None:
    async with pg_engine.begin() as connection:
        await connection.run_sync(compare_migrated_schema)
