"""CLI command: one-shot resync of dynamic configuration from the database.

Use the same validated writer snapshot as API/worker/ML CLI. This command
applies configuration only in its own process; it does not force peer resync.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from infrastructure.database.database import database
from infrastructure.config_migration import migrate_env_configuration
from infrastructure.uow_factory import UnitOfWorkFactory
from presentation.cli.commands.ingest import run_with_infrastructure

logger = logging.getLogger("cli")

config_app = typer.Typer(help="Manage dynamic configuration")


async def run_env_migration(env_file: Path, apply: bool, trim_env: bool = False) -> None:
    try:
        await database.connect()
        await migrate_env_configuration(
            UnitOfWorkFactory(database=database), env_file, apply, trim_env=trim_env
        )
    finally:
        await database.disconnect()


@config_app.command("migrate-env")
def config_migrate_env(
    env_file: Path = typer.Option(..., "--env-file", exists=True, file_okay=True, dir_okay=False),
    apply: bool = typer.Option(False, "--apply", help="Commit migration; default is a read-only dry run"),
    trim_env: bool = typer.Option(False, "--trim-env", help="Remove migrated dynamic and legacy keys"),
) -> None:
    """Migrate legacy env offline, preserving existing DB values and secrets."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(run_env_migration(env_file, apply, trim_env))
    except ValueError as exc:
        logger.error("Config migration blocked: %s", exc)
        raise typer.Exit(1) from None
    except Exception:
        # DB and filesystem exception details can contain credentials/values.
        logger.error("Config migration failed; env backup retained. Fix the failure and retry.")
        raise typer.Exit(1) from None


async def run_config_resync():
    async def status(infra):
        return infra.configuration.status()

    return await run_with_infrastructure(status)


@config_app.command("resync")
def config_resync() -> None:
    """Validate and apply the DB snapshot in this CLI process."""
    try:
        status = asyncio.run(run_config_resync())
        logger.info("CLI configuration snapshot applied: %s", status.applied_version)
    except Exception:
        logger.error("Config resync failed; the database snapshot was not applied")
        raise typer.Exit(1) from None
