"""Benchmark seed: fill a fresh installation with the packs' corpora and question sets.

Every registered pack that ships a seed (``infrastructure.benchmark.seed``) is processed:
its space is created when absent, corpus files are uploaded under ``benchmark-corpus/<pack>/``
and indexed, and the committed question snapshot is reconciled with the database.
Re-running is safe: unchanged files are neither re-uploaded nor re-indexed, questions are
matched by dataset and case id, and nothing is deleted.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import time
from typing import TYPE_CHECKING, Annotated

import typer

from application.benchmark import BenchmarkQualityService
from application.benchmark.services import BenchmarkQuestionService
from application.jobs.principal import job_principal
from application.spaces.readiness import corpus_readiness
from composition.cli.commands.ingest import run_with_infrastructure
from composition.domain_packs import PACK_REGISTRATIONS, selected_pack
from composition.service_providers import create_ingestion_service
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.space import KnowledgeSpace
from domain.shared.exceptions import ValidationError
from infrastructure.benchmark.chunk_documents import UowBenchmarkChunkDocuments
from infrastructure.benchmark.seed import PackSeed, load_pack_seed, read_questions, write_questions
from infrastructure.database.database import database
from infrastructure.uow_factory import UnitOfWorkFactory
from infrastructure.utils.async_tasks import run_sync_io


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.ingestion.space_router import SpaceIngestionService
    from composition.infrastructure import InfrastructureContainer
    from domain.benchmark.question_snapshot import QuestionSyncPlan
    from domain.knowledge.document import Document


logger = logging.getLogger(__name__)

seed_app = typer.Typer(help="Benchmark seed: corpora and question snapshots of the domain packs")

PackOption = Annotated[
    list[str] | None, typer.Option("--pack", "-p", help="Pack to process (repeatable); default: all")
]
MAX_LISTED_DIFFERENCES = 10
WAIT_POLL_SECONDS = 15.0


def pack_seeds(packs: list[str] | None) -> list[PackSeed]:
    keys = packs or list(PACK_REGISTRATIONS)
    unknown = sorted(set(keys) - set(PACK_REGISTRATIONS))

    if unknown:
        message = f"Unknown packs: {', '.join(unknown)}; known: {', '.join(PACK_REGISTRATIONS)}"
        raise typer.BadParameter(message)

    seeds = [seed for key in keys if (seed := load_pack_seed(selected_pack(key))) is not None]

    if not seeds:
        message = "No pack ships a benchmark seed"
        raise typer.BadParameter(message)

    return seeds


async def ensure_space(infra: InfrastructureContainer, seed: PackSeed) -> int:
    """The only space of the pack, else the one named in the seed; created when the pack has none."""
    async with infra.db.uow.create(master=True) as uow:
        spaces = [space for space in await uow.knowledge_spaces.list_all() if space.pack_key == seed.pack_key]
        named = [space for space in spaces if space.name == seed.space_name]
        chosen = spaces[0] if len(spaces) == 1 else (named[0] if named else None)

        if chosen is None and spaces:
            message = f"{len(spaces)} spaces use pack {seed.pack_key} and none is named {seed.space_name!r}"
            raise ValidationError(message)

        if chosen is None:
            chosen = await uow.knowledge_spaces.create(
                KnowledgeSpace(name=seed.space_name, pack_key=seed.pack_key)
            )
            typer.echo(f"[{seed.pack_key}] created space {chosen.id} {seed.space_name!r}")

    if chosen.id is None:
        message = f"Space of pack {seed.pack_key} is not persisted"
        raise ValidationError(message)

    return chosen.id


async def stored_bytes(infra: InfrastructureContainer, key: str) -> bytes | None:
    if await infra.ml.storage.get_file_info(key) is None:
        return None

    path = await infra.ml.storage.download_to_temp(key)

    try:
        return await run_sync_io(path.read_bytes)
    finally:
        await run_sync_io(path.unlink, missing_ok=True)


@dataclass
class CorpusStats:
    uploaded: int = 0
    skipped: int = 0
    conflicts: list[str] = field(default_factory=list)


async def manual_copy_conflict(infra: InfrastructureContainer, document: Document, data: bytes) -> str | None:
    """Why a same-name document uploaded another way cannot stand in for the seed file, or None."""
    if document.status != DocumentStatus.DONE:
        return f"its status is {document.status}"

    if await stored_bytes(infra, document.source_path) != data:
        return "its content differs from the repository file"

    return None


async def seed_corpus(
    infra: InfrastructureContainer, ingestion: SpaceIngestionService, seed: PackSeed, space_id: int
) -> CorpusStats:
    stats = CorpusStats()

    async with infra.db.uow.create() as uow:
        manual = await uow.documents.find_active_slots_by_filenames(
            [document.file for document in seed.documents], space_id=space_id
        )
    uploaded_manually = {document.filename: document for document in manual}

    for document in seed.documents:
        data = await run_sync_io(document.path.read_bytes)

        if (existing := uploaded_manually.get(document.file)) is not None:
            # Indexing the seed copy as well would put the same text twice into search,
            # so the other copy is accepted only when it is indexed and identical.
            reason = await manual_copy_conflict(infra, existing, data)

            if reason is None:
                stats.skipped += 1
            else:
                stats.conflicts.append(
                    f"{document.file}: document {existing.id} with this name was uploaded another way and "
                    f"{reason}; delete it in Admin -> Documents and run seed again"
                )

            continue

        key = seed.storage_key(document)

        # Re-uploading identical bytes changes the modification time and forces re-indexing.
        if await stored_bytes(infra, key) != data:
            await infra.ml.storage.upload_file(key, data)
            stats.uploaded += 1

        # The registry skips a file whose stored size and modification time are already indexed.
        await ingestion.run_single(
            key, domain=document.domain, visibility=DocumentVisibility.INTERNAL_PUBLIC, space_id=space_id
        )

    return stats


async def wait_until_searchable(
    infra: InfrastructureContainer, spaces: dict[str, int], wait_seconds: float
) -> bool:
    """Wait for the outbox to bring every document of the spaces into the vector store."""
    deadline = time.monotonic() + wait_seconds

    while True:
        states = {pack: await corpus_readiness(infra.db.uow, space) for pack, space in spaces.items()}
        waiting = {pack: state for pack, state in states.items() if not state.ready}
        failed = {pack: state for pack, state in waiting.items() if state.failed}

        if not waiting:
            typer.echo("Corpora are indexed and searchable")
            return True

        if failed or time.monotonic() >= deadline:
            for pack, state in waiting.items():
                typer.echo(f"[{pack}] not searchable: {state.describe()}")

            return False

        typer.echo(
            "indexing: " + "; ".join(f"[{pack}] {state.describe()}" for pack, state in waiting.items())
        )
        await asyncio.sleep(WAIT_POLL_SECONDS)


def describe_plan(pack_key: str, plan: QuestionSyncPlan) -> None:
    typer.echo(
        f"[{pack_key}] questions: create={len(plan.create)} update={len(plan.update)} "
        f"unchanged={plan.unchanged} database_only={len(plan.database_only)}"
    )
    lines = [f"  + {q.dataset}/{q.case_id or q.question[:60]}" for q in plan.create]
    lines += [
        f"  ~ {item.key[0]}/{item.key[1][:60]}: {', '.join(sorted(item.changes))}" for item in plan.update
    ]
    lines += [
        f"  ? {q.dataset}/{q.case_id or q.question[:60]} (only in database)" for q in plan.database_only
    ]

    for line in lines[:MAX_LISTED_DIFFERENCES]:
        typer.echo(line)

    if len(lines) > MAX_LISTED_DIFFERENCES:
        typer.echo(f"  … and {len(lines) - MAX_LISTED_DIFFERENCES} more")


async def report_reachability(infra: InfrastructureContainer, seed: PackSeed, space_id: int) -> None:
    service = BenchmarkQualityService(infra.db.uow, UowBenchmarkChunkDocuments(infra.db.uow))
    datasets = sorted(
        {question.dataset for question in await run_sync_io(read_questions, seed.questions_path)}
    )

    for dataset in datasets:
        _, report = await service.fragment_reachability(dataset, space_id)

        if report.fragments:
            typer.echo(
                f"[{seed.pack_key}] reachability {dataset}: {report.reachable}/{report.fragments} "
                f"fragments ({report.share})"
            )


async def with_questions[T](action: Callable[[BenchmarkQuestionService], Awaitable[T]]) -> T:
    """Database only: export and check run in a commit hook and must not reach model providers."""
    try:
        await database.connect()
        return await action(BenchmarkQuestionService(UnitOfWorkFactory(database)))
    finally:
        await database.disconnect()


@seed_app.command("run")
def seed_run(
    *,
    packs: PackOption = None,
    skip_corpus: Annotated[bool, typer.Option("--skip-corpus", help="Only reconcile questions")] = False,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Wait until the corpora are in the vector store")
    ] = True,
    timeout: Annotated[float, typer.Option("--timeout", min=0, help="Seconds to wait for indexing")] = 1800,
) -> None:
    """Create spaces, upload and index corpora, apply question snapshots; idempotent.

    Exit 1 when a corpus file conflicts with a same-name document, or indexing fails or times out.
    """
    seeds = pack_seeds(packs)

    async def action(infra: InfrastructureContainer) -> bool:
        job_principal.set("system:seed-cli")
        ingestion = create_ingestion_service(infra)
        questions = BenchmarkQuestionService(infra.db.uow)
        spaces: dict[str, int] = {}
        conflicts: list[str] = []

        for seed in seeds:
            space_id = spaces[seed.pack_key] = await ensure_space(infra, seed)

            if not skip_corpus:
                stats = await seed_corpus(infra, ingestion, seed, space_id)
                conflicts += [f"[{seed.pack_key}] {conflict}" for conflict in stats.conflicts]
                typer.echo(
                    f"[{seed.pack_key}] corpus: {len(seed.documents)} files, uploaded={stats.uploaded} "
                    f"skipped={stats.skipped} conflicts={len(stats.conflicts)} (space {space_id})"
                )

            snapshot = await run_sync_io(read_questions, seed.questions_path)
            describe_plan(seed.pack_key, await questions.apply_sync(snapshot, seed.dataset_prefix))
            # PostgreSQL chunks only: says nothing about the vector store, see the wait below.
            await report_reachability(infra, seed, space_id)

        for conflict in conflicts:
            typer.echo(f"CONFLICT {conflict}")

        searchable = await wait_until_searchable(infra, spaces, timeout) if wait else True
        return searchable and not conflicts

    if not asyncio.run(run_with_infrastructure(action)):
        raise typer.Exit(1)


@seed_app.command("export")
def seed_export(*, packs: PackOption = None) -> None:
    """Write the database datasets `<pack>-*` into each pack's questions.json."""
    seeds = pack_seeds(packs)

    async def action(questions: BenchmarkQuestionService) -> None:
        for seed in seeds:
            stored = await questions.snapshot(seed.dataset_prefix)
            await run_sync_io(write_questions, seed.questions_path, stored)
            typer.echo(f"[{seed.pack_key}] exported {len(stored)} questions to {seed.questions_path.name}")

    asyncio.run(with_questions(action))


@seed_app.command("check")
def seed_check(*, packs: PackOption = None) -> None:
    """Compare the database with the committed snapshots; exit 1 when they differ."""
    seeds = pack_seeds(packs)

    async def action(questions: BenchmarkQuestionService) -> bool:
        in_sync = True

        for seed in seeds:
            snapshot = await run_sync_io(read_questions, seed.questions_path)
            plan = await questions.plan_sync(snapshot, seed.dataset_prefix)

            if not plan.in_sync:
                describe_plan(seed.pack_key, plan)
                in_sync = False

        if in_sync:
            typer.echo("Benchmark questions match the committed snapshots")
        else:
            typer.echo("Run `task seed:export` to save database edits, or `task seed` to apply the files")

        return in_sync

    if not asyncio.run(with_questions(action)):
        raise typer.Exit(1)
