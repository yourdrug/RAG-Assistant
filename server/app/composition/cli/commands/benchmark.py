"""Benchmark runs via the worker queue, explicit export and non-destructive legacy import."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

import typer

from application.benchmark import BenchmarkExportService, BenchmarkImportService, BenchmarkResultService
from application.benchmark.services import BenchmarkQuestionService
from application.budget import billing_principal
from application.jobs.service import JobService
from composition.cli.commands.ingest import run_with_infrastructure
from composition.infrastructure import InfrastructureContainer
from config import LLMSection, RetrievalSection, process_section
from domain.runtime.jobs import BackgroundJobStatus, JobType
from infrastructure.benchmark.reporting.legacy_import import load_legacy_runs
from infrastructure.database.database import database
from infrastructure.uow_factory import UnitOfWorkFactory
from infrastructure.utils.async_tasks import run_sync_io
from infrastructure.worker.queue import close_arq_pool, enqueue_benchmark


benchmark_app = typer.Typer(help="Run benchmarks, export database runs or import legacy reports")

POLL_INTERVAL_SEC = 15.0
SUMMARY_KEYS = (
    "total_questions",
    "hit_rate",
    "avg_mrr",
    "avg_correctness",
    "avg_faithfulness",
    "total_time_sec",
)


async def enqueue_runs(
    infra: InfrastructureContainer, datasets: list[str], top_k: int | None, judge_model: str | None
) -> dict[str, int]:
    """Create one durable benchmark job per dataset and hand it to the worker, like POST /benchmark."""
    selected = datasets or await BenchmarkQuestionService(infra.db.uow).list_datasets()

    if not selected:
        msg = "No benchmark datasets in the database; import questions in Admin -> Benchmark first"
        raise ValueError(msg)

    k = top_k or process_section(RetrievalSection).retriever_top_k
    judge = judge_model or process_section(LLMSection).active_llm_model
    jobs = JobService(infra.db.uow)
    job_ids: dict[str, int] = {}

    try:
        for dataset in selected:
            job_id = await jobs.create_job(JobType.BENCHMARK)
            await enqueue_benchmark(dataset=dataset, top_k=k, judge_model=judge, job_id=job_id)
            job_ids[dataset] = job_id
            typer.echo(f"queued dataset={dataset} job_id={job_id} top_k={k} judge={judge}")
    finally:
        await close_arq_pool()

    return job_ids


async def wait_for_runs(infra: InfrastructureContainer, job_ids: dict[str, int]) -> bool:
    """Poll job rows until every run is terminal; report summaries and return overall success."""
    jobs = JobService(infra.db.uow)
    results = BenchmarkResultService(infra.db.uow)
    pending = dict(job_ids)
    succeeded = True

    while pending:
        await asyncio.sleep(POLL_INTERVAL_SEC)

        for dataset, job_id in list(pending.items()):
            job = await jobs.get(job_id)

            if job.status not in {BackgroundJobStatus.DONE, BackgroundJobStatus.FAILED}:
                continue

            del pending[dataset]

            if job.status == BackgroundJobStatus.FAILED:
                succeeded = False
                typer.echo(f"FAILED dataset={dataset} job_id={job_id}: {job.error_message}", err=True)
                continue

            run = await results.get_job_result(job_id)
            metrics = run.summary_metrics if run is not None else {}
            summary = " ".join(f"{key}={metrics[key]}" for key in SUMMARY_KEYS if key in metrics)
            typer.echo(f"done dataset={dataset} job_id={job_id} run_id={run.id if run else None} {summary}")

    return succeeded


@benchmark_app.command("run")
def benchmark_run(
    *,
    datasets: Annotated[
        list[str] | None, typer.Option("--dataset", "-d", help="Dataset to run (repeatable); default: all")
    ] = None,
    top_k: Annotated[int | None, typer.Option("--top-k", help="Retriever top-k; default: settings")] = None,
    judge_model: Annotated[
        str | None, typer.Option("--judge-model", help="Judge model; default: active LLM model")
    ] = None,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the worker to finish")] = True,
) -> None:
    """Queue benchmark runs for the worker (same path as Admin -> Benchmark)."""

    async def run_action(infra: InfrastructureContainer) -> bool:
        billing_principal.set("system:benchmark-cli")
        job_ids = await enqueue_runs(infra, datasets or [], top_k, judge_model)
        return await wait_for_runs(infra, job_ids) if wait else True

    try:
        succeeded = asyncio.run(run_with_infrastructure(run_action))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if not succeeded:
        raise typer.Exit(1)


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
