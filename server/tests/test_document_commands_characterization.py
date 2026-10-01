"""Public command contracts pinned before splitting DocumentCommandService.

Keep ACL decisions, conflict outcomes, transaction ordering, outbox payloads
and storage compensation observable through the existing facade methods.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.services.document_command_service import DocumentCommandService
from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, ValidationError
from domain.value_objects.document_status import DocumentStatus
from fakes import FakeUnitOfWorkFactory


class TrackingFactory(FakeUnitOfWorkFactory):
    def __init__(self):
        super().__init__()
        self.events = []
        self.active = False

    @asynccontextmanager
    async def create(self, master=False):
        self.events.append(("begin", master))
        self.active = True
        try:
            async with super().create(master=master) as uow:
                yield uow
        except BaseException:
            self.events.append("rollback")
            raise
        else:
            self.events.append("commit")
        finally:
            self.active = False


def harness(*, versioned=False):
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
    bm25 = MagicMock()
    profile = SimpleNamespace(key="legal" if versioned else "general", is_versioned=versioned)
    registry = SimpleNamespace(get=lambda key: profile)
    command = DocumentCommandService(factory, MagicMock(), storage, bm25, domain_registry=registry)
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
    h = harness(versioned=domain == "legal")
    old = await seed(h, version_group_id=42)
    dto = await upload(h, doc_domain=domain, rename_on_conflict=False)
    assert dto.filename == "report(1).pdf"
    assert dto.version_group_id == 42
    assert dto.storage_key == f"uploads/users/1/{dto.id}_report(1).pdf"
    assert dto.source_path == dto.storage_key
    assert dto.replace_id == (old.id if domain is None else None)
    assert h.factory.events[:3] == [("begin", True), "upload", "commit"]
    if domain == "general":
        assert await h.uow.documents.get_by_id(old.id) is None
        entries = list(h.uow.vector_outbox._entries.values())
        assert [(e.operation.value, e.payload) for e in entries] == [
            ("delete_by_document", {"document_id": old.id})
        ]
        h.storage.delete_file.assert_awaited_once_with("old-key")
        assert h.factory.events[-1] == "storage_delete"
    else:
        assert await h.uow.documents.get_by_id(old.id) is not None
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
        h.bm25.remove.assert_not_called()
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
    assert entry.payload == {
        "points": [
            {
                "chunk_id": 1,
                "page_content": "text",
                "metadata": {
                    "document_id": doc.id,
                    "visibility": "internal_private",
                    "owner_id": 1,
                    "group_id": None,
                    "source": "new.pdf",
                    "doc_domain": "legal",
                    "content_hash": "hash",
                },
            }
        ]
    }
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
    assert (await h.uow.conversations.get(conversation.id)).summary == "summary"
    h.storage.delete_file.assert_not_awaited()
