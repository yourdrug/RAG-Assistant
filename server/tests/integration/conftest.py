"""PostgreSQL contract tests, isolated in a disposable schema per test."""

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from infrastructure.database.basemodel import BaseModel
import infrastructure.database.models  # noqa: F401


@pytest_asyncio.fixture
async def pg_engine():
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set TEST_POSTGRES_URL for PostgreSQL integration tests")
    schema = f"test_{uuid.uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with engine.begin() as conn:
            await conn.run_sync(BaseModel.metadata.create_all)
        yield engine
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
