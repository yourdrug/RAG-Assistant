"""Benchmark runs via the worker queue, explicit export and non-destructive legacy import."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

from application.benchmark import (
    BenchmarkExportService,
    BenchmarkImportService,
    BenchmarkQualityService,
    BenchmarkResultService,
)
from application.benchmark.dataset import load_benchmark_questions
from application.benchmark.dto import RegressionCheckOutput
from application.benchmark.identity import fingerprint
from application.benchmark.quality import dataset_pack, dataset_space
from application.benchmark.services import BenchmarkQuestionService
from application.jobs.principal import job_principal
from application.jobs.service import JobService
from application.spaces.readiness import corpus_readiness
from application.spaces.resolution import default_space_ref
from composition.cli.commands.ingest import run_with_infrastructure
from composition.domain_packs import PACK_REGISTRATIONS, selected_pack
from composition.infrastructure import InfrastructureContainer
from config import LLMSection, RetrievalSection, process_section
from domain.benchmark.policy import GateStatus
from domain.runtime.jobs import BackgroundJobStatus, JobType
from domain.shared.exceptions import EntityNotFoundError, ValidationError
from infrastructure.benchmark.chunk_documents import UowBenchmarkChunkDocuments
from infrastructure.benchmark.reporting.benchmark_history_adapter import BenchmarkHistoryAdapter
from infrastructure.benchmark.reporting.legacy_import import load_legacy_runs
from infrastructure.benchmark.seed import load_pack_seed, read_questions
from infrastructure.database.database import database
from infrastructure.uow_factory import UnitOfWorkFactory
from infrastructure.utils.async_tasks import run_sync_io
from infrastructure.worker.queue import close_arq_pool, enqueue_benchmark


if TYPE_CHECKING:
    from domain.knowledge.space import SpaceRef


logger = logging.getLogger(__name__)


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


async def run_space(infra: InfrastructureContainer, dataset: str, space_id: int | None) -> SpaceRef:
    """The explicit space, else the only space of the dataset's pack (`<pack>-<purpose>`)."""
    if space_id is None and dataset_pack(dataset) is None:
        return await default_space_ref(infra.db.uow)

    async with infra.db.uow.create() as uow:
        return await dataset_space(uow, dataset, space_id)


async def require_ready_corpus(infra: InfrastructureContainer, space_ids: set[int]) -> None:
    """Chunks reach the vector store through the outbox; a run before that searches a partial corpus."""
    for space_id in sorted(space_ids):
        readiness = await corpus_readiness(infra.db.uow, space_id)

        if not readiness.ready:
            message = (
                f"Corpus is not searchable yet ({readiness.describe()}); wait for indexing "
                "(`task seed` waits by default), fix failed documents, or pass --allow-unready"
            )
            raise ValueError(message)


async def enqueue_runs(
    infra: InfrastructureContainer,
    datasets: list[str],
    top_k: int | None,
    judge_model: str | None,
    space_id: int | None = None,
    *,
    require_ready: bool = True,
) -> dict[str, int]:
    """Create one durable benchmark job per dataset and hand it to the worker, like POST /benchmark.

    Each job carries the hash of the questions seen here; the worker refuses to run another set.
    """
    selected = datasets or await BenchmarkQuestionService(infra.db.uow).list_datasets()

    if not selected:
        msg = "No benchmark datasets in the database; import questions in Admin -> Benchmark first"
        raise ValueError(msg)

    k = top_k or process_section(RetrievalSection).retriever_top_k
    judge = judge_model or process_section(LLMSection).llm_model
    jobs = JobService(infra.db.uow)
    job_ids: dict[str, int] = {}

    # Resolve every space before queueing, so a bad dataset does not leave a partial batch.
    spaces = {dataset: await run_space(infra, dataset, space_id) for dataset in selected}

    if require_ready:
        await require_ready_corpus(infra, {space.id for space in spaces.values()})

    hashes = {
        dataset: fingerprint(await load_benchmark_questions(infra.db.uow, dataset)) for dataset in selected
    }

    try:
        for dataset in selected:
            space = spaces[dataset]
            payload = {
                "dataset": dataset,
                "top_k": k,
                "judge_model": judge,
                "space_id": space.id,
                "dataset_hash": hashes[dataset],
            }
            job_id = await jobs.create_job(
                JobType.BENCHMARK, cost_principal=job_principal.get(), dispatch_payload=payload
            )
            try:
                await enqueue_benchmark(
                    dataset=dataset,
                    top_k=k,
                    judge_model=judge,
                    job_id=job_id,
                    space_id=space.id,
                    dataset_hash=hashes[dataset],
                )
            except Exception:  # noqa: BLE001 — committed work remains available to the relay
                logger.warning("Benchmark delivery deferred: job_id=%d", job_id)
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


async def require_synced_questions(infra: InfrastructureContainer, datasets: list[str]) -> None:
    """A run must be reproducible from git: its packs' questions must match the committed snapshot."""
    service = BenchmarkQuestionService(infra.db.uow)
    selected = datasets or await service.list_datasets()
    packs = sorted({pack for dataset in selected if (pack := dataset_pack(dataset)) in PACK_REGISTRATIONS})

    for pack in packs:
        seed = load_pack_seed(selected_pack(pack))

        if seed is None:
            continue

        snapshot = await run_sync_io(read_questions, seed.questions_path)

        if not (await service.plan_sync(snapshot, seed.dataset_prefix)).in_sync:
            message = (
                f"Questions of pack {pack} differ from {seed.questions_path.name}; run `task seed:check`, "
                "then `task seed:export` or `task seed`, or pass --allow-unsynced"
            )
            raise ValueError(message)


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
    space_id: Annotated[int | None, typer.Option("--space-id", min=1)] = None,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the worker to finish")] = True,
    allow_unsynced: Annotated[
        bool,
        typer.Option(
            "--allow-unsynced", help="Run although database questions differ from the committed seed"
        ),
    ] = False,
    allow_unready: Annotated[
        bool,
        typer.Option("--allow-unready", help="Run although documents of the space are not indexed yet"),
    ] = False,
) -> None:
    """Queue benchmark runs for the worker (same path as Admin -> Benchmark)."""

    async def run_action(infra: InfrastructureContainer) -> bool:
        job_principal.set("system:benchmark-cli")

        if not allow_unsynced:
            await require_synced_questions(infra, datasets or [])

        job_ids = await enqueue_runs(
            infra, datasets or [], top_k, judge_model, space_id, require_ready=not allow_unready
        )
        return await wait_for_runs(infra, job_ids) if wait else True

    try:
        succeeded = asyncio.run(run_with_infrastructure(run_action))
    except (ValueError, ValidationError, EntityNotFoundError) as exc:
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


def print_regression(output: RegressionCheckOutput) -> None:
    verdict = {
        GateStatus.PASSED: "PASS",
        GateStatus.FAILED: "FAIL",
        GateStatus.INSUFFICIENT_DATA: "INSUFFICIENT DATA",
    }[output.status]
    typer.echo(
        f"{verdict} dataset={output.dataset} run_id={output.run_id} baseline_run_id={output.baseline_run_id} "
        f"compared_questions={output.compared_questions}"
    )

    for result in output.results:
        mark = "FAIL" if result.failed else "ok"
        typer.echo(
            f"  [{mark}] {result.kind}:{result.metric} current={result.current} baseline={result.baseline} "
            f"delta={result.delta} threshold={result.threshold}"
            + (f" ({result.note})" if result.note else "")
        )

    for item in output.excluded_questions:
        typer.echo(f"  excluded {item['reason']}: {item['key']} {item.get('question') or ''}")

    for warning in output.warnings:
        typer.echo(f"  warning: {warning}")


def gate_datasets() -> list[str]:
    """The PR gate checks the quick set of every registered pack; a pack without one is reported.

    Full `-main` sets are checked on request; offline-only `-chunking-*` subsets never have online runs.
    """
    return [f"{pack}-quick" for pack in PACK_REGISTRATIONS]


def overall_status(outputs: list[RegressionCheckOutput]) -> GateStatus:
    """Any failure fails the gate; otherwise any dataset without enough data leaves it undecided."""
    statuses = {output.status for output in outputs}

    if GateStatus.FAILED in statuses:
        return GateStatus.FAILED

    if not outputs or GateStatus.INSUFFICIENT_DATA in statuses:
        return GateStatus.INSUFFICIENT_DATA

    return GateStatus.PASSED


@benchmark_app.command("regression")
def benchmark_regression(
    *,
    run_id: Annotated[
        int | None, typer.Option("--run-id", help="Run to check; default: latest per dataset")
    ] = None,
    datasets: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset whose latest run is checked (repeatable); default: `<pack>-quick` of every pack",
        ),
    ] = None,
) -> None:
    """Compare runs with their baselines and dataset acceptance thresholds.

    Checks stored runs only: run the benchmark first (`task bench -- -d <dataset>`).
    Exit 1 on regression, 2 when a run, a comparable baseline or a thresholded metric is missing.
    """

    async def check(infra: InfrastructureContainer) -> GateStatus:
        service = BenchmarkResultService(infra.db.uow)
        history = BenchmarkHistoryAdapter()

        if run_id is not None:
            outputs = [await service.check_regression(run_id, history)]
        else:
            selected = datasets or gate_datasets()
            outputs = [await service.check_regression(None, history, dataset=name) for name in selected]

        for output in outputs:
            print_regression(output)

        if not outputs:
            typer.echo("no datasets to check", err=True)

        return overall_status(outputs)

    status = asyncio.run(run_with_infrastructure(check))

    if status is GateStatus.FAILED:
        raise typer.Exit(1)

    if status is GateStatus.INSUFFICIENT_DATA:
        raise typer.Exit(2)


@benchmark_app.command("reachability")
def benchmark_reachability(
    *,
    datasets: Annotated[list[str], typer.Option("--dataset", "-d", help="Dataset to check (repeatable)")],
    space_id: Annotated[
        int | None, typer.Option("--space-id", min=1, help="Default: the only space of the dataset's pack")
    ] = None,
) -> None:
    """Check that expected fragments lie whole inside one chunk of the corpus; no model is called.

    Exit 1 when any fragment is unreachable or a dataset has no labelled fragments.
    """

    async def check(infra: InfrastructureContainer) -> bool:
        service = BenchmarkQualityService(infra.db.uow, UowBenchmarkChunkDocuments(infra.db.uow))
        complete = True

        for dataset in datasets:
            space, report = await service.fragment_reachability(dataset, space_id)
            typer.echo(
                f"{dataset} space={space.id} ({space.pack_key}): questions={report.questions} "
                f"fragments={report.fragments} reachable={report.reachable} share={report.share}"
            )

            for item in report.unreachable:
                typer.echo(f"  {item.reason.value}: q{item.question_id} {item.source}: {item.text}")

            complete = complete and report.fragments > 0 and not report.unreachable

        return complete

    if not asyncio.run(run_with_infrastructure(check)):
        raise typer.Exit(1)


@benchmark_app.command("noise")
def benchmark_noise(
    *,
    dataset: Annotated[str, typer.Option("--dataset", "-d")],
    run_ids: Annotated[
        list[int] | None, typer.Option("--run-id", help="Finished repeated runs to measure (repeatable)")
    ] = None,
    repeat: Annotated[
        int,
        typer.Option("--repeat", min=0, max=10, help="Queue N new runs first (paid: calls RAG and judge)"),
    ] = 0,
    top_k: Annotated[int | None, typer.Option("--top-k")] = None,
    judge_model: Annotated[str | None, typer.Option("--judge-model")] = None,
    space_id: Annotated[int | None, typer.Option("--space-id", min=1)] = None,
) -> None:
    """Store sigma of each compared metric for a dataset from repeated runs of one configuration."""

    async def measure(infra: InfrastructureContainer) -> list[int]:
        job_principal.set("system:benchmark-cli")
        selected = list(run_ids or [])
        results = BenchmarkResultService(infra.db.uow)

        for _ in range(repeat):
            jobs = await enqueue_runs(infra, [dataset], top_k, judge_model, space_id)

            if not await wait_for_runs(infra, jobs):
                msg = "A repeated run failed; noise was not stored"
                raise ValueError(msg)

            run = await results.get_job_result(jobs[dataset])

            if run is not None and run.id is not None:
                selected.append(run.id)

        policy = await BenchmarkQualityService(infra.db.uow).measure_noise(dataset, selected, None)

        if policy.noise is not None:
            for metric, sigma in sorted(policy.noise.sigma.items()):
                typer.echo(f"sigma {metric}={sigma}")

        return selected

    try:
        measured = asyncio.run(run_with_infrastructure(measure))
    except (ValueError, ValidationError, EntityNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"noise stored for dataset={dataset} runs={measured}")
