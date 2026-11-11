"""Configure database metadata and run asynchronous Alembic migrations."""

from __future__ import annotations

import asyncio
import importlib
import logging
from logging.config import fileConfig
from typing import TYPE_CHECKING

from alembic import context

from config import DatabaseSection, process_section
from infrastructure.database.basemodel import _Base as Base
from infrastructure.database.database import DatabaseManager


if TYPE_CHECKING:
    from sqlalchemy.engine import Connection


config = context.config
target_metadata = Base.metadata


def load_model_modules() -> None:
    """Import ORM model modules so Alembic autogenerate can detect them."""
    model_modules: tuple[str, ...] = ("infrastructure.database.models",)

    for model_module in model_modules:
        try:
            importlib.import_module(model_module)
        except ModuleNotFoundError:
            logging.getLogger("default").debug("Module %s not found", model_module)


def init_alembic_config() -> None:
    """Configure migration logging."""
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = DatabaseManager(process_section(DatabaseSection)).master_url

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute all pending migrations with alembic."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run transactional migrations on a manager-owned connection."""
    database = DatabaseManager(process_section(DatabaseSection))
    async with database.async_master_connection(
        connect_args={"options": "-c statement_timeout=0"}
    ) as connection:
        configured_connection = await connection.execution_options(isolation_level="READ COMMITTED")
        await configured_connection.run_sync(do_run_migrations)


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connection = config.attributes.get("connection")

    if connection is not None:
        do_run_migrations(connection)
    else:
        asyncio.run(run_async_migrations())


load_model_modules()
init_alembic_config()
run_migrations_offline() if context.is_offline_mode() else run_migrations_online()
