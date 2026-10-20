"""Public command contracts pinned before splitting DocumentCommandService.

Keep ACL decisions, conflict outcomes, transaction ordering, outbox payloads
and storage compensation observable through the existing facade methods.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.documents.delete import DocumentDeleteCommand
from application.documents.rename import DocumentRenameCommand
from application.documents.upload import DocumentUploadCommand
from application.documents import DocumentCommandService
from application.identity import UserContextFactory
from domain.knowledge.document import Document
from domain.knowledge.acts.models import ActVersion
from domain.shared.exceptions import BusinessRuleViolation, ValidationError
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.enums import DocumentVisibility
from domain.identity.roles import UserRole
from fakes import FakeUnitOfWorkFactory


class TrackingFactory(FakeUnitOfWorkFactory):
    def __init__(self):
        super().__init__()
        self.events = []
        self.active = False
        self.commit = MagicMock()

    @asynccontextmanager
    async def create(self, master=False):
        self.events.append(("begin", master))
        self.active = True
        try:
            async with super().create(master=master) as uow:
                yield uow
                self.commit()
        except BaseException:
            self.events.append("rollback")
            raise
        else:
            self.events.append("commit")
        finally:
            self.active = False


def harness():
    factory = TrackingFactory()

    async def upload(key, data):
        assert factory.active
        factory.events.append("upload")

    async def copy(old, new):
        assert factory.active
        factory.events.append("copy")

    async def delete(key):
        assert not factory.active
        factory.events.append("storage_delete")

    storage = MagicMock(supported_extensions=(".pdf", ".txt"))
    storage.upload_file = AsyncMock(side_effect=upload)
    storage.copy_file = AsyncMock(side_effect=copy)
    storage.delete_file = AsyncMock(side_effect=delete)
    bm25 = AsyncMock()
    context_factory = UserContextFactory()
    command = DocumentCommandService(
        DocumentUploadCommand(factory, storage, context_factory),
        DocumentDeleteCommand(factory, storage, context_factory),
        DocumentRenameCommand(factory, storage, context_factory),
    )
    return SimpleNamespace(command=command, factory=factory, uow=factory._uow, storage=storage, bm25=bm25)


async def seed(h, **kwargs):
    return await h.uow.documents.save(
        Document(
            filename="report.pdf",
            status=DocumentStatus.DONE,
            source_path="old-key",
            visibility="internal_private",
            owner_id=kwargs.pop("owner_id", 1),
            **kwargs,
        )
    )


async def upload(h, **kwargs):
    inputs = {
        "filename": "report.pdf",
        "file_data": b"content",
        "visibility": "internal_private",
        "group_id": None,
        "user_id": 1,
        "user_kind": "internal",
        "user_role": "user",
    }
    inputs.update(kwargs)
    return await h.command.upload(**inputs)


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", [None, "general", "legal"])
async def test_upload_implicit_conflict_and_cleanup_order(domain):
    h = harness()
    old = await seed(h, version_group_id=42)
    dto = await upload(h, doc_domain=domain, rename_on_conflict=False)
    assert dto.filename == "report(1).pdf"
    assert dto.version_group_id == 42
    assert dto.storage_key == f"uploads/users/1/{dto.id}_report(1).pdf"
    assert dto.source_path == dto.storage_key
    assert dto.replace_id == old.id
    assert h.factory.events == [("begin", True), "upload", "commit"]
    assert await h.uow.documents.get_by_id(old.id) is not None
    assert h.uow.vector_outbox._entries == {}
    h.storage.delete_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [DocumentStatus.DONE, DocumentStatus.PROCESSING])
async def test_add_new_preserves_existing_document_and_starts_independent_chain(status):
    h = harness()
    old = await seed(h, version_group_id=42)
    old.status = status
    dto = await upload(h, doc_domain="general", rename_on_conflict=True)
    assert dto.filename == "report(1).pdf"
    assert dto.replace_id is None
    assert dto.version_group_id is None
    assert await h.uow.documents.get_by_id(old.id) is not None
    assert h.uow.vector_outbox._entries == {}
    h.storage.delete_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_id", [1, 2])
async def test_explicit_replacement_requires_ownership_and_is_deferred(owner_id):
    h = harness()
    old = await seed(h, owner_id=owner_id)
    if owner_id != 1:
        with pytest.raises(BusinessRuleViolation, match="your own documents"):
            await upload(h, replaces_document_id=old.id)
        h.storage.upload_file.assert_not_awaited()
    else:
        dto = await upload(h, replaces_document_id=old.id)
        assert dto.replace_id == old.id
        assert dto.version_group_id == old.id
    assert await h.uow.documents.get_by_id(old.id) is not None
    h.storage.delete_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_replacement_cannot_change_visibility_scope():
    h = harness()
    old = await seed(h)
    with pytest.raises(BusinessRuleViolation, match="same visibility scope"):
        await upload(h, replaces_document_id=old.id, visibility="internal_public", user_role="admin")
    h.storage.upload_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_replacement_rejects_add_new_flag_and_active_target():
    h = harness()
    old = await seed(h)
    with pytest.raises(ValidationError, match="rename_on_conflict"):
        await upload(h, replaces_document_id=old.id, rename_on_conflict=True)
    old.status = DocumentStatus.PROCESSING
    with pytest.raises(BusinessRuleViolation, match="already being processed"):
        await upload(h, filename="other.pdf", replaces_document_id=old.id)
    h.storage.upload_file.assert_not_awaited()


async def seed_act_timeline(h, deleted_index):
    docs = [await seed(h) for _ in range(3)]
    starts = [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)]
    versions = []
    for index, doc in enumerate(docs):
        version = await h.uow.act_versions.create(
            ActVersion(
                id=None,
                act_id=7,
                document_id=doc.id,
                effective_from=starts[index],
                effective_to=starts[index + 1] if index < 2 else None,
                is_current=index == 2,
            )
        )
        versions.append(version)
        await h.uow.chunks.bulk_insert(
            document_id=doc.id,
            filename=doc.filename,
            visibility="internal_private",
            owner_id=1,
            group_id=None,
            doc_domain="legal",
            chunks=["text"],
            content_hashes=["hash"],
            act_version_id=version.id,
            effective_from=version.effective_from,
            effective_to=version.effective_to,
            is_current=version.is_current,
        )

    original_delete = h.uow.documents.delete

    async def cascade_delete(document_id):
        await original_delete(document_id)
        # Match the PostgreSQL foreign key cascade before a survivor is promoted.
        h.uow.act_versions._versions.pop(versions[deleted_index].id)

    h.uow.documents.delete = cascade_delete
    h.uow.act_versions.list_by_act = AsyncMock(wraps=h.uow.act_versions.list_by_act)
    return docs, versions


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted_index", [1, 2])
async def test_delete_reconnects_act_timeline_and_search_projections(deleted_index):
    h = harness()
    docs, _versions = await seed_act_timeline(h, deleted_index)

    await h.command.delete_document(docs[deleted_index].id, 1, "user")
    h.uow.act_versions.list_by_act.assert_awaited_once_with(7, for_update=True)
    survivors = await h.uow.act_versions.list_by_act(7)
    assert len(survivors) == 2
    assert survivors[-1].is_current is True
    assert survivors[-1].effective_to is None
    assert survivors[0].effective_to == survivors[1].effective_from
    changed = survivors[0] if deleted_index == 1 else survivors[-1]
    chunk = next(c for c in h.uow.chunks._chunks if c["act_version_id"] == changed.id)
    assert chunk["is_current"] == changed.is_current
    assert chunk["effective_to"] == changed.effective_to
    entries = list(h.uow.vector_outbox._entries.values())
    assert entries[0].operation == "delete_by_document"
    assert entries[1].operation == "update_metadata"
    assert entries[1].payload["act_version_id"] == changed.id
    h.storage.delete_file.assert_awaited_once_with("old-key")


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted_index", [1, 2])
async def test_failed_timeline_update_rolls_back_deletion(deleted_index):
    h = harness()
    docs, _versions = await seed_act_timeline(h, deleted_index)
    h.uow.act_versions.update = AsyncMock(side_effect=RuntimeError("projection failure"))
    with pytest.raises(RuntimeError, match="projection failure"):
        await h.command.delete_document(docs[deleted_index].id, 1, "user")
    assert await h.uow.documents.get_by_id(docs[deleted_index].id) is not None
    assert len(await h.uow.act_versions.list_by_act(7)) == 3
    assert h.uow.vector_outbox._entries == {}
    h.storage.delete_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete", "rename"])
@pytest.mark.parametrize("owner", [1, 2])
async def test_write_ownership_is_checked_before_side_effects(operation, owner):
    h = harness()
    doc = await seed(h, owner_id=owner)
    if operation == "delete":
        action = h.command.delete_document(doc.id, 1, "user")
    else:
        action = h.command.rename_document(doc.id, "new.pdf", 1, "user")
    if owner != 1:
        with pytest.raises(BusinessRuleViolation, match="your own documents"):
            await action
        assert h.factory.events == [("begin", True), "rollback"]
        assert h.uow.vector_outbox._entries == {}
        assert (await h.uow.documents.get_by_id(doc.id)).filename == "report.pdf"
        h.storage.copy_file.assert_not_awaited()
        h.storage.delete_file.assert_not_awaited()
        h.bm25.remove.assert_not_awaited()
    else:
        await action
        assert h.factory.events[-2:] == ["commit", "storage_delete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_id,kind", [(None, None), (9, None), (9, "internal"), (9, "client")])
async def test_curator_upload_rejects_missing_invalid_or_unassigned_client(client_id, kind):
    h = harness()
    if kind:
        h.uow.users.add_user(9, kind=kind)
    error = BusinessRuleViolation if kind == "client" else ValidationError
    with pytest.raises(error):
        await upload(h, visibility="client_private", user_role="curator", client_id=client_id)
    assert h.uow.documents._documents == {}
    h.storage.upload_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,kind,user_id", [("admin", "internal", 1), ("curator", "internal", 1), ("user", "client", 9)]
)
async def test_client_upload_uses_effective_owner(role, kind, user_id):
    h = harness()
    h.uow.users.add_user(9, kind="client")
    h.uow.assignments.set_user_kind(9, "client")
    await h.uow.assignments.assign_user(curator_id=1, target_user_id=9, assigned_by=99)
    dto = await upload(
        h, visibility="client_private", user_role=role, user_kind=kind, user_id=user_id, client_id=9
    )
    assert dto.owner_id == 9
    assert dto.group_id is None
    assert dto.storage_key == f"uploads/users/9/{dto.id}_report.pdf"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("write failed"), asyncio.CancelledError()])
@pytest.mark.parametrize("operation", ["upload", "rename"])
async def test_storage_compensation_after_transaction_rollback(operation, failure):
    h = harness()
    # Upload compensation runs inside its transaction, rename compensation after rollback.
    h.storage.delete_file = AsyncMock()
    if operation == "upload":
        h.uow.documents.set_source_path = AsyncMock(side_effect=failure)
        action = upload(h)
    else:
        doc = await seed(h)
        h.uow.documents.update_filename = AsyncMock(side_effect=failure)
        action = h.command.rename_document(doc.id, "new.pdf", 1, "user")
    with pytest.raises(type(failure)) as caught:
        await action
    assert caught.value is failure
    assert h.factory.events[-1] == "rollback"
    h.storage.delete_file.assert_awaited_once()
    if operation == "upload":
        assert h.uow.documents._documents == {}
        key = h.storage.upload_file.call_args.args[0]
    else:
        assert (await h.uow.documents.get_by_id(doc.id)).filename == "report.pdf"
        key = h.storage.copy_file.call_args.args[1]
    h.storage.delete_file.assert_awaited_once_with(key)
    assert h.uow.vector_outbox._entries == {}


@pytest.mark.asyncio
async def test_rename_updates_chunks_and_exact_outbox_metadata_before_commit():
    h = harness()
    doc = await seed(h)
    await h.uow.chunks.bulk_insert(
        document_id=doc.id,
        filename=doc.filename,
        visibility="internal_private",
        owner_id=1,
        group_id=None,
        doc_domain="legal",
        chunks=["text"],
        content_hashes=["hash"],
    )
    h.uow.documents.find_active_slot = AsyncMock(wraps=h.uow.documents.find_active_slot)
    dto = await h.command.rename_document(doc.id, "new.pdf", 1, "user")
    h.uow.documents.find_active_slot.assert_awaited_once_with(1, "new.pdf", None, for_update=True)
    assert dto.source_path == f"uploads/users/1/{doc.id}_new.pdf"
    assert h.uow.chunks._chunks[0]["filename"] == "new.pdf"
    entry = next(iter(h.uow.vector_outbox._entries.values()))
    assert entry.operation == "rename_document"
    assert entry.payload == {"document_id": doc.id, "filename": "new.pdf"}
    assert h.factory.events == [("begin", True), "copy", "commit", "storage_delete"]


@pytest.mark.asyncio
async def test_delete_db_failure_rolls_back_outbox_and_summaries_without_deleting_storage():
    h = harness()
    doc = await seed(h)
    conversation = await h.uow.conversations.create(user_id=1)
    await h.uow.conversations.update_summary(conversation.id, "summary")
    h.uow.documents.delete = AsyncMock(side_effect=RuntimeError("DB failure"))
    with pytest.raises(RuntimeError, match="DB failure"):
        await h.command.delete_document(doc.id, 1, "user")
    assert await h.uow.documents.get_by_id(doc.id) is not None
    assert h.uow.vector_outbox._entries == {}
    assert (await h.uow.conversations.get_by_id(conversation.id)).summary == "summary"
    h.storage.delete_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("db_fails", [False, True])
async def test_delete_defers_bm25_to_outbox_and_storage_until_commit(db_fails):
    h = harness()
    doc = await seed(h)
    await h.uow.chunks.bulk_insert(
        document_id=doc.id,
        filename=doc.filename,
        visibility=DocumentVisibility.INTERNAL_PRIVATE.value,
        owner_id=1,
        group_id=None,
        doc_domain="general",
        chunks=["text"],
        content_hashes=["hash"],
    )

    h.uow.chunks.list_for_document = AsyncMock(wraps=h.uow.chunks.list_for_document)
    if db_fails:
        h.uow.documents.delete = AsyncMock(side_effect=RuntimeError("DB failure"))
        with pytest.raises(RuntimeError, match="DB failure"):
            await h.command.delete_document(doc.id, 1, UserRole.USER)
        assert h.factory.events == [("begin", True), "rollback"]
        h.storage.delete_file.assert_not_awaited()
    else:
        await h.command.delete_document(doc.id, 1, UserRole.USER)
        assert h.factory.events == [("begin", True), "commit", "storage_delete"]
        h.storage.delete_file.assert_awaited_once_with("old-key")

    h.bm25.remove.assert_not_awaited()
    h.uow.chunks.list_for_document.assert_not_awaited()
    if db_fails:
        assert h.uow.vector_outbox._entries == {}
    else:
        entry = next(iter(h.uow.vector_outbox._entries.values()))
        assert entry.operation == "delete_by_document"
        assert entry.payload == {"document_id": doc.id}


@pytest.mark.asyncio
async def test_rename_enqueues_one_payload_patch_for_large_document():
    h = harness()
    doc = await seed(h)
    count = 10001
    await h.uow.chunks.bulk_insert(
        document_id=doc.id,
        filename=doc.filename,
        visibility=DocumentVisibility.INTERNAL_PRIVATE.value,
        owner_id=1,
        chunks=[f"text {i}" for i in range(count)],
    )
    await h.command.rename_document(doc.id, "new.pdf", 1, UserRole.USER)
    entries = list(h.uow.vector_outbox._entries.values())
    assert len(entries) == 1
    assert entries[0].operation == "rename_document"
    assert entries[0].payload == {"document_id": doc.id, "filename": "new.pdf"}


@pytest.mark.asyncio
async def test_delete_commit_failure_preserves_document_and_sparse_index():
    h = harness()
    doc = await seed(h)
    h.factory.commit.side_effect = RuntimeError("commit failed")
    with pytest.raises(RuntimeError, match="commit failed"):
        await h.command.delete_document(doc.id, 1, UserRole.USER)
    assert await h.uow.documents.get_by_id(doc.id) is not None
    assert h.uow.vector_outbox._entries == {}
    h.bm25.remove.assert_not_awaited()
    h.storage.delete_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["rename", "delete"])
async def test_assignment_mutation_preserves_shared_ingestion_source(operation):
    h = harness()
    first = await seed(h, ingestion_key="docs/shared.pdf")
    second = await seed(h, owner_id=2, ingestion_key="docs/shared.pdf")
    if operation == "rename":
        dto = await h.command.rename_document(first.id, "display.pdf", 1, "user")
        assert dto.source_path == "old-key"
        assert dto.filename == "display.pdf"
    else:
        await h.command.delete_document(first.id, 1, "user")
        assert await h.uow.documents.get_by_id(first.id) is None
    assert (await h.uow.documents.get_by_id(second.id)).source_path == "old-key"
    h.storage.copy_file.assert_not_awaited()
    h.storage.delete_file.assert_not_awaited()
