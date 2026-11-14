"""Real transaction pooling keeps timeouts and routes session locks directly."""

import os

from pydantic import SecretStr
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from config import DatabaseSection
from infrastructure.database.database import DatabaseManager


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_transaction_pool_preserves_settings_and_direct_session_locks() -> None:
    proxy_url = os.environ.get("TEST_PGBOUNCER_URL")
    direct_url = os.environ.get("TEST_POSTGRES_URL")

    if not proxy_url or not direct_url:
        pytest.skip("Set TEST_PGBOUNCER_URL and TEST_POSTGRES_URL")

    proxy, direct = make_url(proxy_url), make_url(direct_url)
    assert proxy.host
    assert proxy.username
    assert proxy.database
    assert direct.host
    settings = DatabaseSection(
        db_host=proxy.host,
        db_port=str(proxy.port),
        db_user=proxy.username,
        db_password=SecretStr(proxy.password or ""),
        db_name=proxy.database,
        db_use_pgbouncer=True,
        db_direct_host=direct.host,
        db_direct_port=str(direct.port),
        db_statement_timeout_ms=2000,
    )
    database = DatabaseManager(settings)
    await database.connect()
    try:
        async with database.async_master_connection() as connection:
            pid = await connection.scalar(text("SELECT pg_backend_pid()"))
            await connection.execute(text("SELECT pg_advisory_lock(819273645)"))

            for _ in range(5):
                async with database.get_session(master=True) as session, session.begin():
                    assert await session.scalar(text("SHOW statement_timeout")) == "2s"
                    assert await session.scalar(text("SELECT pg_backend_pid()")) != pid

            assert await connection.scalar(text("SELECT pg_backend_pid()")) == pid
            assert await connection.scalar(text("SELECT pg_advisory_unlock(819273645)")) is True
    finally:
        await database.disconnect()
