"""CLI command: one-shot resync of dynamic configuration from the database.

Use the same validated writer snapshot as API/worker/ML CLI. This command
applies configuration only in its own process; it does not force peer resync.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer

from application.admin.dto import ConfigurationStatus
from composition.cli.commands.ingest import run_with_infrastructure
from composition.infrastructure import InfrastructureContainer
from composition.utils import require
from infrastructure.config_migration import migrate_env_configuration
from infrastructure.database.database import database
from infrastructure.uow_factory import UnitOfWorkFactory


logger = logging.getLogger("cli")


def log_blocked_migration(error: ValueError) -> None:
    logger.error("Config migration blocked: %s", error)


def log_failure(message: str, error: Exception) -> None:
    logger.error("%s (%s)", message, type(error).__name__)


config_app = typer.Typer(help="Manage dynamic configuration")


async def run_env_migration(env_file: Path, *, apply: bool, trim_env: bool = False) -> None:
    try:
        await database.connect()
        await migrate_env_configuration(
            UnitOfWorkFactory(database=database), env_file, apply=apply, trim_env=trim_env
        )
    finally:
        await database.disconnect()


@config_app.command("migrate-env")
def config_migrate_env(
    *,
    env_file: Annotated[Path, typer.Option("--env-file", exists=True, file_okay=True, dir_okay=False)],
    apply: Annotated[
        bool, typer.Option("--apply", help="Commit migration; default is a read-only dry run")
    ] = False,
    trim_env: Annotated[
        bool, typer.Option("--trim-env", help="Remove migrated dynamic and legacy keys")
    ] = False,
) -> None:
    """Migrate legacy env offline, preserving existing DB values and secrets."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(run_env_migration(env_file, apply=apply, trim_env=trim_env))
    except ValueError as exc:
        log_blocked_migration(exc)
        raise typer.Exit(1) from None
    except Exception as exc:  # noqa: BLE001 — CLI boundary redacts arbitrary provider/DB errors.
        # DB and filesystem exception details can contain credentials/values.
        log_failure("Config migration failed; env backup retained. Fix the failure and retry.", exc)
        raise typer.Exit(1) from None


async def run_config_resync() -> ConfigurationStatus:
    async def status(infra: InfrastructureContainer) -> ConfigurationStatus:
        return require(infra.configuration, "configuration").status()

    return await run_with_infrastructure(status)


@config_app.command("resync")
def config_resync() -> None:
    """Validate and apply the DB snapshot in this CLI process."""
    try:
        status = asyncio.run(run_config_resync())
        logger.info("CLI configuration snapshot applied: %s", status.applied_version)
    except Exception as exc:  # noqa: BLE001 — CLI boundary redacts arbitrary provider/DB errors.
        log_failure("Config resync failed; the database snapshot was not applied", exc)
        raise typer.Exit(1) from None
