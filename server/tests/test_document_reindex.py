"""Resumable re-chunking of stored documents (CLI ``reindex run``)."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import TYPE_CHECKING

import pytest

from application.documents.dto import ProcessDocumentCommand
from application.documents.persistence import ManualChunkEditsError
from application.documents.reindex import (
    DocumentReindexService,
    ReindexEvent,
    ReindexOutcome,
    ReindexRequest,
)
from composition.cli.commands.reindex import CheckpointWriter, read_checkpoint
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DocumentVisibility, SourceType
from tests.fakes import FakeUnitOfWork, FakeUnitOfWorkFactory
from tests.test_document_processor_characterization import build_harness, seed_document


if TYPE_CHECKING:
    from pathlib import Path
    from types import SimpleNamespace


class RecordingProcessor:
    def __init__(self, fail_ids: frozenset[int] = frozenset()) -> None:
        self.commands: list[ProcessDocumentCommand] = []
        self.fail_ids = fail_ids

    async def process(self, command: ProcessDocumentCommand) -> None:
        self.commands.append(command)

        if command.document_id in self.fail_ids:
            message = "parse failed"
            raise RuntimeError(message)


async def seed(uow: FakeUnitOfWork, **overrides: object) -> Document:
    values: dict[str, object] = {
        "filename": "act.docx",
        "source_path": "uploads/act.docx",
        "status": DocumentStatus.DONE,
        "doc_domain": "decree",
        "visibility": DocumentVisibility.INTERNAL_GROUP,
        "owner_id": 7,
        "group_id": 3,
    }
    values.update(overrides)
    return await uow.documents.save(Document(**values))  # type: ignore[arg-type]


@pytest.fixture
def uow() -> FakeUnitOfWork:
    return FakeUnitOfWork()


@pytest.mark.asyncio
async def test_reindex_keeps_identity_acl_and_domain_and_skips_unsafe_documents(uow: FakeUnitOfWork) -> None:
    done = await seed(uow)
    manual = await seed(uow, has_manual_edits=True)
    processing = await seed(uow, status=DocumentStatus.PROCESSING)
    other_domain = await seed(uow, doc_domain="legal")
    no_file = await seed(uow, source_type=SourceType.MANUAL.value, source_path="")
    failed_before = await seed(uow, status=DocumentStatus.FAILED)
    processor = RecordingProcessor()

    report = await DocumentReindexService(FakeUnitOfWorkFactory(uow), processor).run(
        ReindexRequest(doc_domain="decree")
    )

    assert report.ids(ReindexOutcome.REINDEXED) == [done.id, failed_before.id]
    assert report.ids(ReindexOutcome.SKIPPED_MANUAL) == [manual.id]
    assert report.ids(ReindexOutcome.SKIPPED_STATUS) == [processing.id]
    assert report.ids(ReindexOutcome.SKIPPED_SOURCE) == [no_file.id]
    assert other_domain.id not in {event.document_id for event in report.events}
    command = processor.commands[0]
    assert (command.document_id, command.storage_key, command.replace_id) == (done.id, done.source_path, None)
    assert (command.visibility, command.owner_id, command.group_id, command.doc_domain) == (
        DocumentVisibility.INTERNAL_GROUP,
        7,
        3,
        "decree",
    )


@pytest.mark.asyncio
async def test_batches_pause_resume_and_limit(uow: FakeUnitOfWork) -> None:
    documents = [await seed(uow) for _ in range(5)]
    pauses: list[float] = []

    async def sleep(seconds: float) -> None:
        pauses.append(seconds)

    service = DocumentReindexService(FakeUnitOfWorkFactory(uow), RecordingProcessor(), sleep=sleep)
    first = await service.run(ReindexRequest(batch_size=2, pause_seconds=1.5, limit=3))
    assert first.ids(ReindexOutcome.REINDEXED) == [doc.id for doc in documents[:3]]
    assert pauses == [1.5]

    resumed = await service.run(ReindexRequest(batch_size=2, after_id=first.last_id or 0))
    assert resumed.ids(ReindexOutcome.REINDEXED) == [doc.id for doc in documents[3:]]


@pytest.mark.asyncio
async def test_failure_is_reported_and_batch_continues(uow: FakeUnitOfWork) -> None:
    first, second = await seed(uow), await seed(uow)
    processor = RecordingProcessor(fail_ids=frozenset({first.id or 0}))

    report = await DocumentReindexService(FakeUnitOfWorkFactory(uow), processor).run(ReindexRequest())

    assert report.ids(ReindexOutcome.FAILED) == [first.id]
    assert report.ids(ReindexOutcome.REINDEXED) == [second.id]


@pytest.mark.asyncio
async def test_manual_edit_after_listing_is_not_overwritten(uow: FakeUnitOfWork) -> None:
    first, second = await seed(uow), await seed(uow)
    processor = RecordingProcessor()

    class EditingProcessor:
        async def process(self, command: ProcessDocumentCommand) -> None:
            await processor.process(command)
            # An administrator edits the next document while this one is processed.
            await uow.documents.set_has_manual_edits(second.id or 0, value=True)

    report = await DocumentReindexService(FakeUnitOfWorkFactory(uow), EditingProcessor()).run(
        ReindexRequest(batch_size=10)
    )

    assert report.ids(ReindexOutcome.REINDEXED) == [first.id]
    assert report.ids(ReindexOutcome.SKIPPED_MANUAL) == [second.id]


@pytest.mark.asyncio
async def test_dry_run_and_explicit_ids_do_not_process(uow: FakeUnitOfWork) -> None:
    documents = [await seed(uow) for _ in range(3)]
    processor = RecordingProcessor()

    report = await DocumentReindexService(FakeUnitOfWorkFactory(uow), processor).run(
        ReindexRequest(document_ids=(documents[2].id or 0, documents[0].id or 0), dry_run=True)
    )

    assert report.ids(ReindexOutcome.SELECTED) == [documents[0].id, documents[2].id]
    assert processor.commands == []


def test_checkpoint_round_trip_keeps_failures(tmp_path: Path) -> None:
    path = tmp_path / "reindex.json"
    assert read_checkpoint(path) == (0, [])
    writer = CheckpointWriter(path, {"doc_domain": "decree"}, [2])
    writer(ReindexEvent(5, "a.docx", ReindexOutcome.FAILED))
    assert read_checkpoint(path) == (5, [2, 5])
    assert json.loads(path.read_text(encoding="utf-8"))["doc_domain"] == "decree"


@pytest.mark.parametrize("values", [{"batch_size": 0}, {"pause_seconds": -1}, {"limit": 0}])
def test_request_validation(values: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        ReindexRequest(**values)  # type: ignore[arg-type]


# --- Processor behaviour for re-chunking (``reindex_snapshot``) -------------------


def chunk_outbox(harness: SimpleNamespace) -> list[object]:
    return list(harness.uow.fake_vector_outbox._entries.values())


async def rechunk(harness: SimpleNamespace, document: Document, *, snapshot: Document | None) -> None:
    await harness.processor.process(
        ProcessDocumentCommand(
            document_id=document.id or 0,
            storage_key=document.source_path,
            original_filename=document.filename,
            visibility=document.visibility,
            owner_id=document.owner_id,
            group_id=document.group_id,
            replace_id=None,
            doc_domain=str(document.doc_domain),
            reindex_snapshot=snapshot,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "before,snapshot,expected",
    [
        (DocumentStatus.DONE, True, DocumentStatus.DONE),
        (DocumentStatus.FAILED, True, DocumentStatus.FAILED),
        (DocumentStatus.DONE, False, DocumentStatus.FAILED),  # Upload path is unchanged.
    ],
)
async def test_failed_rechunk_restores_done_document_with_old_chunks(
    tmp_path: Path, before: DocumentStatus, *, snapshot: bool, expected: DocumentStatus
) -> None:
    harness = build_harness(tmp_path, parse_error=RuntimeError("parser crashed"))
    document = await seed_document(
        harness.repo, status=before, chunks=5, warning_message="old warning", quality_score=0.1
    )

    with pytest.raises(RuntimeError, match="parser crashed"):
        await rechunk(harness, document, snapshot=replace(document) if snapshot else None)

    assert document.status == expected
    assert chunk_outbox(harness) == []  # Old chunks and their vectors are untouched.

    if expected == DocumentStatus.DONE:
        assert (document.chunks, document.warning_message, document.quality_score) == (5, "old warning", 0.1)
    else:
        assert document.error_message == "parser crashed"


@pytest.mark.asyncio
async def test_manual_edit_during_rechunk_aborts_commit_and_keeps_edits(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    document = await seed_document(harness.repo, status=DocumentStatus.DONE, chunks=2)
    snapshot = replace(document)
    parse = harness.parser.parse

    def parse_while_admin_edits(*args: object) -> object:
        # The edit commits after the reindex re-read, before the persist transaction.
        document.has_manual_edits = True
        return parse(*args)

    harness.parser.parse = parse_while_admin_edits

    with pytest.raises(ManualChunkEditsError):
        await rechunk(harness, document, snapshot=snapshot)

    # The rolled-back transaction restores repository state from copies; re-read.
    stored = await harness.repo.get_by_id(document.id or 0)
    assert stored is not None
    assert (stored.status, stored.has_manual_edits) == (DocumentStatus.DONE, True)
    assert chunk_outbox(harness) == []
    assert harness.uow.chunks._chunks == []


@pytest.mark.asyncio
async def test_reindex_reports_edit_during_processing_as_skipped(uow: FakeUnitOfWork) -> None:
    document = await seed(uow)

    class EditedDuringProcessing:
        async def process(self, command: ProcessDocumentCommand) -> None:
            assert command.reindex_snapshot is not None
            assert command.reindex_snapshot is not await uow.documents.get_by_id(document.id or 0)
            message = "edited"
            raise ManualChunkEditsError(message)

    report = await DocumentReindexService(FakeUnitOfWorkFactory(uow), EditedDuringProcessing()).run(
        ReindexRequest()
    )

    assert report.ids(ReindexOutcome.SKIPPED_MANUAL) == [document.id]
    assert report.ids(ReindexOutcome.FAILED) == []
