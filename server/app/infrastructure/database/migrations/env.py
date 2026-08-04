"""Alembic env.py — async migrations (KinTree-style)."""

import asyncio
import importlib
from logging.config import fileConfig

from alembic import context
from config import settings
from infrastructure.database.basemodel import _Base as Base
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config
section = config.config_ini_section
target_metadata = Base.metadata


def load_model_modules() -> None:
    """Import ORM model modules so Alembic autogenerate can detect them."""

    model_modules: tuple = ("infrastructure.database.models",)

    for model_module in model_modules:
        try:
            importlib.import_module(model_module)
        except ModuleNotFoundError:
            print(f"Module {model_module} not found")


def init_alembic_config() -> None:
    """Inject DB settings from config into alembic.ini placeholders."""

    if config.config_file_name is not None:
        fileConfig(config.config_file_name)

    config.set_section_option(section, "DB_USER", settings.db_user)
    config.set_section_option(section, "DB_PASSWORD", settings.db_password)
    config.set_section_option(section, "DB_HOST", settings.db_host)
    config.set_section_option(section, "DB_PORT", settings.db_port)
    config.set_section_option(section, "DB_NAME", settings.db_name)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""

    url = config.get_main_option("sqlalchemy.url")

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
    """Create an async engine and run migrations."""

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


load_model_modules()
init_alembic_config()
run_migrations_offline() if context.is_offline_mode() else run_migrations_online()
