"""Explicit benchmark export and non-destructive legacy import."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

import typer

from application.benchmark import BenchmarkExportService, BenchmarkImportService
from infrastructure.benchmark.reporting.legacy_import import load_legacy_runs
from infrastructure.database.database import database
from infrastructure.uow_factory import UnitOfWorkFactory
from infrastructure.utils.async_tasks import run_sync_io


benchmark_app = typer.Typer(help="Export database runs or import legacy reports")


async def export_run(run_id: int, export_format: str, out: Path) -> None:
    try:
        await database.connect()
        service = BenchmarkExportService(UnitOfWorkFactory(database))
        await service.header(run_id)
        # --out is the only path that writes a local report. Exclusive creation preserves existing files.
        with out.open("xb") as stream:
            async for block in service.export(run_id, export_format):
                await run_sync_io(stream.write, block)
    finally:
        await database.disconnect()


@benchmark_app.command("export")
def benchmark_export(
    run_id: int,
    *,
    export_format: Annotated[Literal["csv", "json"], typer.Option("--format")] = "json",
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    asyncio.run(export_run(run_id, export_format, out))


async def import_runs(path: Path, dataset: str, *, apply: bool) -> None:
    runs, errors = await asyncio.to_thread(load_legacy_runs, path, dataset)

    for error in errors:
        typer.echo(error, err=True)

    if errors and apply:
        msg = "Import blocked by validation errors; files have not been changed"
        raise ValueError(msg)

    try:
        await database.connect()
        result = await BenchmarkImportService(UnitOfWorkFactory(database)).import_runs(runs, apply=apply)
        typer.echo(
            f"dry_run={result.dry_run} accepted={result.imported} "
            f"duplicates={result.duplicates} errors={len(errors)}"
        )
    finally:
        await database.disconnect()


@benchmark_app.command("import-legacy")
def benchmark_import(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    *,
    dataset: Annotated[
        str, typer.Option("--dataset", help="Explicit dataset; never infer corpus from a filename")
    ],
    apply: Annotated[bool, typer.Option("--apply")] = False,
) -> None:
    try:
        asyncio.run(import_runs(path, dataset, apply=apply))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
