"""Tests for DocumentService upload refactoring."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from application.services.document_service import DocumentService
from application.services.user_context_factory import UserContextFactory
from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility

from fakes import FakeUnitOfWork, FakeUnitOfWorkFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CTX = UserContext(
    user_id=1,
    user_kind="internal",
    user_role="admin",
    group_ids=[],
    managed_client_ids=[],
    managed_internal_ids=[],
    managed_group_ids=[],
)


def _make_doc(
    *,
    doc_id: int | None = None,
    filename: str = "test.pdf",
    status: DocumentStatus = DocumentStatus.DONE,
    owner_id: int = 1,
    group_id: int | None = None,
    version_group_id: int | None = None,
    source_path: str | None = None,
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
) -> Document:
    doc = Document(
        filename=filename,
        visibility=visibility,
        owner_id=owner_id,
        group_id=group_id,
        status=status,
        version_group_id=version_group_id,
    )
    if doc_id is not None:
        doc.id = doc_id
    if source_path is not None:
        doc.source_path = source_path
    return doc


def _service(uow: FakeUnitOfWork | None = None, file_storage=None):
    uow = uow or FakeUnitOfWork()
    factory = FakeUnitOfWorkFactory(uow)
    fs = file_storage or MagicMock()
    fs.supported_extensions = (".pdf", ".docx", ".doc", ".txt", ".md")
    fs.upload_file = AsyncMock()
    fs.delete_file = AsyncMock()
    return (
        DocumentService(
            uow_factory=factory,
            vector_store_repo=MagicMock(),
            file_storage=fs,
            bm25_index=MagicMock(),
        ),
        uow,
        fs,
    )


@pytest.fixture(autouse=True)
def _patch_user_context_build(monkeypatch):
    """Replace UserContextFactory.build with a stub returning a fixed context."""

    async def _fake_build(self, uow, user_id, user_kind, user_role):
        return _FAKE_CTX

    monkeypatch.setattr(UserContextFactory, "build", _fake_build)


# ---------------------------------------------------------------------------
# _resolve_version_group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_version_group_explicit_replaces():
    """replaces_document_id set → version group from that doc."""
    svc, uow, _ = _service()
    replaces_doc = _make_doc(doc_id=42, version_group_id=10)
    # FakeDocumentRepository.save auto-assigns id=1, ignoring doc_id.
    # We need the doc in the repo with id=1.
    saved = await uow.documents.save(replaces_doc)
    assert saved.id == 1

    vgid, pending, existing = await svc._resolve_version_group(
        uow, existing=None, replaces_document_id=1, pending_replace_id=None
    )
    assert vgid == 10
    assert pending is None
    assert existing is saved


@pytest.mark.asyncio
async def test_resolve_version_group_explicit_replaces_no_version_group():
    """replaces_document_id set but no version_group → falls back to doc id."""
    svc, uow, _ = _service()
    replaces_doc = _make_doc(doc_id=7, version_group_id=None)
    saved = await uow.documents.save(replaces_doc)

    vgid, _, _ = await svc._resolve_version_group(
        uow, existing=None, replaces_document_id=saved.id, pending_replace_id=None
    )
    assert vgid == saved.id


@pytest.mark.asyncio
async def test_resolve_version_group_explicit_not_found():
    """replaces_document_id points to nonexistent doc → EntityNotFound."""
    svc, uow, _ = _service()
    with pytest.raises(EntityNotFound):
        await svc._resolve_version_group(
            uow, existing=None, replaces_document_id=999, pending_replace_id=None
        )


@pytest.mark.asyncio
async def test_resolve_version_group_implicit_conflict():
    """No explicit replaces but existing DONE → implicit version group."""
    svc, uow, _ = _service()
    existing = _make_doc(doc_id=5, version_group_id=3, status=DocumentStatus.DONE)
    saved = await uow.documents.save(existing)

    vgid, pending, returned = await svc._resolve_version_group(
        uow, existing=saved, replaces_document_id=None, pending_replace_id=None
    )
    assert vgid == 3
    assert pending == saved.id
    assert returned is saved


@pytest.mark.asyncio
async def test_resolve_version_group_implicit_no_version_group():
    """Existing DONE doc without version_group → falls back to doc id."""
    svc, uow, _ = _service()
    existing = _make_doc(doc_id=9, version_group_id=None, status=DocumentStatus.DONE)
    saved = await uow.documents.save(existing)

    vgid, pending, _ = await svc._resolve_version_group(
        uow, existing=saved, replaces_document_id=None, pending_replace_id=None
    )
    assert vgid == saved.id
    assert pending == saved.id


@pytest.mark.asyncio
async def test_resolve_version_group_no_conflict():
    """No replaces_document_id and no existing → all None."""
    svc, uow, _ = _service()
    vgid, pending, returned = await svc._resolve_version_group(
        uow, existing=None, replaces_document_id=None, pending_replace_id=None
    )
    assert vgid is None
    assert pending is None
    assert returned is None


@pytest.mark.asyncio
async def test_resolve_version_group_existing_pending():
    """Existing is PENDING (not DONE/FAILED) → no implicit conflict."""
    svc, uow, _ = _service()
    existing = _make_doc(doc_id=3, status=DocumentStatus.PENDING)
    await uow.documents.save(existing)

    vgid, pending, _ = await svc._resolve_version_group(
        uow, existing=existing, replaces_document_id=None, pending_replace_id=None
    )
    assert vgid is None
    assert pending is None


# ---------------------------------------------------------------------------
# _persist_upload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_upload_success():
    """Happy path: save + S3 upload + set_source_path."""
    svc, uow, fs = _service()
    doc = _make_doc()

    saved_doc, key = await svc._persist_upload(
        uow, doc, b"file-bytes", owner_id=1, effective_group_id=None, filename="test.pdf"
    )
    assert saved_doc.id is not None
    assert "uploads/users/1/" in key
    assert "test.pdf" in key
    fs.upload_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_persist_upload_s3_failure_compensates():
    """S3 upload fails → compensating delete of orphaned object."""
    svc, uow, fs = _service()
    fs.upload_file = AsyncMock(side_effect=RuntimeError("S3 down"))

    doc = _make_doc()
    with pytest.raises(RuntimeError, match="S3 down"):
        await svc._persist_upload(uow, doc, b"data", owner_id=1, effective_group_id=None, filename="test.pdf")
    fs.delete_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_persist_upload_s3_failure_delete_also_fails():
    """S3 upload fails and compensating delete also fails → original error raised."""
    svc, uow, fs = _service()
    fs.upload_file = AsyncMock(side_effect=RuntimeError("S3 down"))
    fs.delete_file = AsyncMock(side_effect=RuntimeError("delete failed"))

    doc = _make_doc()
    with pytest.raises(RuntimeError, match="S3 down"):
        await svc._persist_upload(uow, doc, b"data", owner_id=1, effective_group_id=None, filename="test.pdf")
    fs.delete_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# _maybe_resolve_conflict_sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_resolve_conflict_sync_no_domain():
    """doc_domain is None → no conflict resolution."""
    svc, uow, _ = _service()
    storage_deletes = []
    result = await svc._maybe_resolve_conflict_sync(
        doc=MagicMock(),
        existing=MagicMock(),
        pending_replace_id=5,
        doc_domain=None,
        uow=uow,
        storage_deletes=storage_deletes,
    )
    assert result == 5
    assert storage_deletes == []


@pytest.mark.asyncio
async def test_maybe_resolve_conflict_sync_no_existing():
    """existing is None → no conflict resolution."""
    svc, uow, _ = _service()
    result = await svc._maybe_resolve_conflict_sync(
        doc=MagicMock(),
        existing=None,
        pending_replace_id=5,
        doc_domain="general",
        uow=uow,
        storage_deletes=[],
    )
    assert result == 5


@pytest.mark.asyncio
async def test_maybe_resolve_conflict_sync_no_pending():
    """pending_replace_id is None → no conflict resolution."""
    svc, uow, _ = _service()
    result = await svc._maybe_resolve_conflict_sync(
        doc=MagicMock(),
        existing=MagicMock(),
        pending_replace_id=None,
        doc_domain="general",
        uow=uow,
        storage_deletes=[],
    )
    assert result is None


# ---------------------------------------------------------------------------
# upload (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_basic():
    """Basic upload with no conflicts → doc created, DTO returned."""
    svc, uow, fs = _service()
    dto = await svc.upload(
        filename="report.pdf",
        file_data=b"content",
        visibility="internal_private",
        group_id=None,
        user_id=1,
        user_kind="internal",
        user_role="user",
    )
    assert dto.filename == "report.pdf"
    assert dto.id > 0
    assert dto.replace_id is None
    fs.upload_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_unsupported_extension():
    """Unsupported file extension → ValidationError."""
    svc, _, _ = _service()
    with pytest.raises(ValidationError, match="Unsupported file format"):
        await svc.upload(
            filename="data.xyz",
            file_data=b"x",
            visibility="internal_public",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )


@pytest.mark.asyncio
async def test_upload_internal_group_no_group_id():
    """INTERNAL_GROUP visibility without group_id → ValidationError."""
    svc, _, _ = _service()
    with pytest.raises(ValidationError, match="group_id required"):
        await svc.upload(
            filename="doc.pdf",
            file_data=b"data",
            visibility="internal_group",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )


@pytest.mark.asyncio
async def test_upload_active_processing_rejected():
    """Existing doc in PENDING/PROCESSING/INDEXING → BusinessRuleViolation."""
    svc, uow, _ = _service()
    pending_doc = _make_doc(doc_id=10, status=DocumentStatus.PENDING, filename="doc.pdf")
    await uow.documents.save(pending_doc)

    with pytest.raises(BusinessRuleViolation, match="already being processed"):
        await svc.upload(
            filename="doc.pdf",
            file_data=b"data",
            visibility="internal_private",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )


@pytest.mark.asyncio
async def test_upload_concurrent_duplicate_rejected():
    """Concurrent upload of same filename → BusinessRuleViolation."""
    svc, uow, fs = _service()
    # Make save raise UniqueConstraintViolation
    from domain.exceptions import UniqueConstraintViolation

    original_save = uow.documents.save

    call_count = 0

    async def failing_save(doc):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise UniqueConstraintViolation("dup")
        return await original_save(doc)

    uow.documents.save = failing_save

    with pytest.raises(BusinessRuleViolation, match="already being uploaded"):
        await svc.upload(
            filename="doc.pdf",
            file_data=b"data",
            visibility="internal_private",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )


# ---------------------------------------------------------------------------
# delete_document (characterization)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_happy_path():
    """delete_document: outbox enqueued, BM25 removal, summary clear, doc deleted, storage deleted."""
    svc, uow, fs = _service()
    fs.copy_file = AsyncMock()
    doc = _make_doc(doc_id=1, source_path="s3://bucket/docs/1/test.pdf")
    await uow.documents.save(doc)

    # Add a chunk with content_hash for BM25 removal verification
    await uow.chunks.bulk_insert(
        document_id=1, filename="test.pdf", visibility="internal_public",
        chunks=["chunk content"], content_hashes=["hash_abc"],
    )

    await svc.delete_document(document_id=1, user_id=1, user_role="admin")

    # 1. Outbox entry enqueued with DELETE_BY_DOCUMENT
    outbox_entries = list(uow.vector_outbox._entries.values())
    assert len(outbox_entries) == 1
    assert outbox_entries[0].operation.value == "delete_by_document"
    assert outbox_entries[0].payload == {"document_id": 1}

    # 2. BM25 index: remove called for the chunk's content_hash
    svc._bm25_index.remove.assert_called_once_with("hash_abc")

    # 3. Summaries cleared (all in the fake)
    for conv in uow.conversations._convs.values():
        assert conv.get("summary") is None

    # 4. Document deleted from repository
    assert await uow.documents.get_by_id(1) is None

    # 5. Storage file deleted (after transaction commit)
    fs.delete_file.assert_awaited_once_with("s3://bucket/docs/1/test.pdf")


@pytest.mark.asyncio
async def test_delete_document_not_found():
    """delete_document: non-existent doc → EntityNotFound."""
    svc, _, _ = _service()
    with pytest.raises(EntityNotFound):
        await svc.delete_document(document_id=999, user_id=1, user_role="admin")


@pytest.mark.asyncio
async def test_delete_document_no_source_path():
    """delete_document: doc without source_path → storage delete skipped."""
    svc, uow, fs = _service()
    doc = _make_doc(doc_id=1, source_path=None)
    await uow.documents.save(doc)

    await svc.delete_document(document_id=1, user_id=1, user_role="admin")

    fs.delete_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_document_storage_delete_failure_logs_warning():
    """delete_document: S3 delete fails → warning logged, no exception raised."""
    svc, uow, fs = _service()
    doc = _make_doc(doc_id=1, source_path="s3://bucket/docs/1/test.pdf")
    await uow.documents.save(doc)
    fs.delete_file = AsyncMock(side_effect=RuntimeError("S3 down"))

    # Should not raise — storage delete failure is best-effort
    await svc.delete_document(document_id=1, user_id=1, user_role="admin")

    # Document still deleted from DB
    assert await uow.documents.get_by_id(1) is None
    fs.delete_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# rename_document (characterization)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rename_document_different_path_copies_and_deletes_old():
    """rename_document: different path → S3 copy, DB updated, outbox enqueued, old file deleted."""
    svc, uow, fs = _service()
    fs.copy_file = AsyncMock()
    doc = _make_doc(doc_id=1, source_path="s3://bucket/u/1/1/old.pdf", filename="old.pdf")
    await uow.documents.save(doc)
    await uow.chunks.bulk_insert(
        document_id=1, filename="old.pdf", visibility="internal_public",
        chunks=["content"], content_hashes=["h1"],
    )

    dto = await svc.rename_document(
        document_id=1, new_filename="new.pdf", user_id=1, user_role="admin"
    )

    # 1. S3 copy called (old → new)
    fs.copy_file.assert_awaited_once()
    src, dst = fs.copy_file.call_args[0]
    assert "old.pdf" in src
    assert "new.pdf" in dst

    # 2. Document filename updated in DB
    assert dto.filename == "new.pdf"

    # 3. Outbox entry enqueued with UPSERT_CHUNKS
    outbox_entries = list(uow.vector_outbox._entries.values())
    assert len(outbox_entries) == 1
    assert outbox_entries[0].operation.value == "upsert_chunks"

    # 4. Old source file deleted (after transaction commit)
    fs.delete_file.assert_awaited_once_with("s3://bucket/u/1/1/old.pdf")


@pytest.mark.asyncio
async def test_rename_document_same_path_no_copy():
    """rename_document: doc without source_path → no S3 copy needed."""
    svc, uow, fs = _service()
    fs.copy_file = AsyncMock()
    doc = _make_doc(doc_id=1, source_path=None, filename="test.pdf")
    await uow.documents.save(doc)

    await svc.rename_document(
        document_id=1, new_filename="test.pdf", user_id=1, user_role="admin"
    )

    # No S3 copy (source_path is None → condition skips copy)
    fs.copy_file.assert_not_awaited()
    # No old file deletion (nothing changed)
    fs.delete_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_rename_document_unsupported_extension():
    """rename_document: unsupported extension → ValidationError."""
    svc, _, _ = _service()
    with pytest.raises(ValidationError, match="Unsupported file format"):
        await svc.rename_document(
            document_id=1, new_filename="data.xyz", user_id=1, user_role="admin"
        )


@pytest.mark.asyncio
async def test_rename_document_not_found():
    """rename_document: non-existent doc → EntityNotFound."""
    svc, _, _ = _service()
    with pytest.raises(EntityNotFound):
        await svc.rename_document(
            document_id=999, new_filename="new.pdf", user_id=1, user_role="admin"
        )


@pytest.mark.asyncio
async def test_rename_document_conflict_with_pending_doc():
    """rename_document: existing doc with same name in PENDING → BusinessRuleViolation."""
    svc, uow, _ = _service()
    # Create two docs with different IDs but same name-slot target
    doc1 = _make_doc(doc_id=1, filename="a.pdf", owner_id=1, group_id=None)
    doc2 = _make_doc(doc_id=2, filename="b.pdf", status=DocumentStatus.PENDING, owner_id=1, group_id=None)
    await uow.documents.save(doc1)
    await uow.documents.save(doc2)

    with pytest.raises(BusinessRuleViolation, match="already being processed"):
        await svc.rename_document(
            document_id=1, new_filename="b.pdf", user_id=1, user_role="admin"
        )


@pytest.mark.asyncio
async def test_rename_document_conflict_with_done_doc_renames():
    """rename_document: existing DONE doc with same name → auto-resolves to unique name."""
    svc, uow, fs = _service()
    fs.copy_file = AsyncMock()
    doc1 = _make_doc(doc_id=1, filename="a.pdf", source_path="s3://bucket/u/1/1/a.pdf", owner_id=1)
    doc2 = _make_doc(doc_id=2, filename="report.pdf", status=DocumentStatus.DONE, owner_id=1)
    await uow.documents.save(doc1)
    await uow.documents.save(doc2)

    dto = await svc.rename_document(
        document_id=1, new_filename="report.pdf", user_id=1, user_role="admin"
    )

    # Filename should have been auto-resolved to avoid collision
    assert dto.filename != "report.pdf"


# ---------------------------------------------------------------------------
# rename_document — compensation tests (DDD-003)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rename_document_copy_succeeds_db_fails_compensates():
    """rename_document: S3 copy OK, DB update fails → compensating delete of orphaned copy."""
    svc, uow, fs = _service()
    fs.copy_file = AsyncMock()

    doc = _make_doc(doc_id=1, source_path="s3://bucket/u/1/1/old.pdf", filename="old.pdf")
    await uow.documents.save(doc)

    # Make update_filename raise to simulate DB failure after S3 copy
    async def failing_update(*args, **kwargs):
        raise RuntimeError("DB failure")

    uow.documents.update_filename = failing_update

    with pytest.raises(RuntimeError, match="DB failure"):
        await svc.rename_document(
            document_id=1, new_filename="new.pdf", user_id=1, user_role="admin"
        )

    # S3 copy was attempted
    fs.copy_file.assert_awaited_once()
    # Compensating: orphaned copy deleted (uses the computed storage key, not the hardcoded path)
    assert fs.delete_file.await_count == 1
    deleted_path = fs.delete_file.call_args[0][0]
    assert "new.pdf" in deleted_path


@pytest.mark.asyncio
async def test_rename_document_copy_fails_no_compensation():
    """rename_document: S3 copy fails → no compensation delete, exception propagates."""
    svc, uow, fs = _service()
    fs.copy_file = AsyncMock(side_effect=RuntimeError("S3 down"))

    doc = _make_doc(doc_id=1, source_path="s3://bucket/u/1/1/old.pdf", filename="old.pdf")
    await uow.documents.save(doc)

    with pytest.raises(RuntimeError, match="S3 down"):
        await svc.rename_document(
            document_id=1, new_filename="new.pdf", user_id=1, user_role="admin"
        )

    # No compensation delete (copy failed, nothing to clean up)
    fs.delete_file.assert_not_awaited()


# ---------------------------------------------------------------------------
# list_documents (characterization)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_documents_admin_sees_all():
    """Admin sees all documents."""
    svc, uow, _ = _service()
    await uow.documents.save(_make_doc(doc_id=1, filename="a.pdf"))
    await uow.documents.save(_make_doc(doc_id=2, filename="b.pdf"))

    result = await svc.list_documents(user_id=99, user_kind="internal", user_role="admin")

    assert len(result) == 2
    filenames = {d.filename for d in result}
    assert filenames == {"a.pdf", "b.pdf"}


@pytest.mark.asyncio
async def test_list_documents_empty():
    """No documents → empty list."""
    svc, _, _ = _service()
    result = await svc.list_documents(user_id=1, user_kind="internal", user_role="user")
    assert result == []


@pytest.mark.asyncio
async def test_list_documents_user_sees_only_own():
    """Regular user only sees own PRIVATE documents (via ACL)."""
    svc, uow, _ = _service()
    await uow.documents.save(_make_doc(
        doc_id=1, filename="own.pdf", owner_id=1,
        visibility=DocumentVisibility.INTERNAL_PRIVATE,
    ))
    await uow.documents.save(_make_doc(
        doc_id=2, filename="other.pdf", owner_id=2,
        visibility=DocumentVisibility.INTERNAL_PRIVATE,
    ))

    from unittest.mock import patch

    async def _user_ctx(self_factory, uow, user_id, user_kind, user_role):
        return UserContext(
            user_id=user_id, user_kind=user_kind, user_role=user_role,
            group_ids=[], managed_client_ids=[], managed_internal_ids=[], managed_group_ids=[],
        )

    with patch.object(UserContextFactory, "build", _user_ctx):
        result = await svc.list_documents(user_id=1, user_kind="internal", user_role="user")

    assert len(result) == 1
    assert result[0].filename == "own.pdf"


@pytest.mark.asyncio
async def test_list_documents_client_sees_only_own():
    """Client user only sees own CLIENT_PRIVATE documents."""
    svc, uow, _ = _service()
    await uow.documents.save(_make_doc(
        doc_id=1, filename="own.pdf", owner_id=10,
        visibility=DocumentVisibility.CLIENT_PRIVATE,
    ))
    await uow.documents.save(_make_doc(
        doc_id=2, filename="other.pdf", owner_id=20,
        visibility=DocumentVisibility.CLIENT_PRIVATE,
    ))

    from unittest.mock import patch

    async def _user_ctx(self_factory, uow, user_id, user_kind, user_role):
        return UserContext(
            user_id=user_id, user_kind=user_kind, user_role=user_role,
            group_ids=[], managed_client_ids=[], managed_internal_ids=[], managed_group_ids=[],
        )

    with patch.object(UserContextFactory, "build", _user_ctx):
        result = await svc.list_documents(user_id=10, user_kind="client", user_role="user")

    assert len(result) == 1
    assert result[0].filename == "own.pdf"


@pytest.mark.asyncio
async def test_list_documents_with_limit_offset():
    """Pagination works."""
    svc, uow, _ = _service()
    for i in range(5):
        await uow.documents.save(_make_doc(filename=f"doc{i}.pdf", owner_id=1))

    result = await svc.list_documents(
        user_id=99, user_kind="internal", user_role="user", limit=2, offset=0
    )
    assert len(result) == 2

    result2 = await svc.list_documents(
        user_id=99, user_kind="internal", user_role="user", limit=2, offset=2
    )
    assert len(result2) == 2


# ---------------------------------------------------------------------------
# get_document (characterization)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_document_happy_path():
    """get_document: existing doc + ACL OK → returns DTO."""
    svc, uow, _ = _service()
    doc = _make_doc(doc_id=1, filename="report.pdf")
    await uow.documents.save(doc)

    dto = await svc.get_document(document_id=1, user_id=1, user_kind="internal", user_role="admin")

    assert dto.id == 1
    assert dto.filename == "report.pdf"


@pytest.mark.asyncio
async def test_get_document_not_found():
    """get_document: non-existent doc → EntityNotFound."""
    svc, _, _ = _service()
    with pytest.raises(EntityNotFound):
        await svc.get_document(document_id=999, user_id=1, user_kind="internal", user_role="admin")


# ---------------------------------------------------------------------------
# list_uploadable_clients (characterization)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_uploadable_clients_admin():
    """Admin sees all client users."""
    svc, uow, _ = _service()
    uow.users.add_user(10, email="client1@test.com", kind="client")
    uow.users.add_user(20, email="client2@test.com", kind="client")
    uow.users.add_user(30, email="internal@test.com", kind="internal")

    result = await svc.list_uploadable_clients(user_id=99, user_kind="internal", user_role="admin")

    assert len(result) == 2
    emails = {c.email for c in result}
    assert emails == {"client1@test.com", "client2@test.com"}


@pytest.mark.asyncio
async def test_list_uploadable_clients_curator():
    """Curator sees only managed clients."""
    svc, uow, _ = _service()
    uow.users.add_user(10, email="managed@test.com", kind="client")
    uow.users.add_user(20, email="unmanaged@test.com", kind="client")
    uow.users.add_user(30, email="other@test.com", kind="client")
    uow.assignments.set_user_kind(10, "client")
    uow.assignments.set_user_kind(20, "client")
    await uow.assignments.assign_user(curator_id=1, target_user_id=10, assigned_by=99)

    result = await svc.list_uploadable_clients(user_id=1, user_kind="internal", user_role="curator")

    assert len(result) == 1
    assert result[0].id == 10


@pytest.mark.asyncio
async def test_list_uploadable_clients_client_returns_empty():
    """Client users get empty list."""
    svc, _, _ = _service()
    result = await svc.list_uploadable_clients(user_id=10, user_kind="client", user_role="user")
    assert result == []


@pytest.mark.asyncio
async def test_list_uploadable_clients_user_returns_empty():
    """Regular users get empty list."""
    svc, _, _ = _service()
    result = await svc.list_uploadable_clients(user_id=1, user_kind="internal", user_role="user")
    assert result == []


# ---------------------------------------------------------------------------
# list_source_files (characterization)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_source_files_basic():
    """Returns distinct filenames."""
    svc, uow, _ = _service()
    await uow.documents.save(_make_doc(filename="report.pdf"))
    await uow.documents.save(_make_doc(filename="report.pdf"))
    await uow.documents.save(_make_doc(filename="other.pdf"))

    result = await svc.list_source_files()

    assert sorted(result) == ["other.pdf", "report.pdf"]


@pytest.mark.asyncio
async def test_list_source_files_with_search():
    """Search filters filenames."""
    svc, uow, _ = _service()
    await uow.documents.save(_make_doc(filename="report.pdf"))
    await uow.documents.save(_make_doc(filename="data.csv"))
    await uow.documents.save(_make_doc(filename="report_v2.pdf"))

    result = await svc.list_source_files(search="report")

    assert sorted(result) == ["report.pdf", "report_v2.pdf"]


@pytest.mark.asyncio
async def test_list_source_files_empty():
    """No documents → empty list."""
    svc, _, _ = _service()
    result = await svc.list_source_files()
    assert result == []


# ---------------------------------------------------------------------------
# delete_document — permission checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_not_owner_not_admin():
    """Non-owner non-admin → BusinessRuleViolation (via check_ownership)."""
    svc, uow, _ = _service()
    doc = _make_doc(doc_id=1, owner_id=10)
    await uow.documents.save(doc)

    from unittest.mock import patch

    async def _user_ctx(self_factory, uow, user_id, user_kind, user_role):
        return UserContext(
            user_id=user_id, user_kind=user_kind, user_role=user_role,
            group_ids=[], managed_client_ids=[], managed_internal_ids=[], managed_group_ids=[],
        )

    with patch.object(UserContextFactory, "build", _user_ctx):
        with pytest.raises(BusinessRuleViolation):
            await svc.delete_document(document_id=1, user_id=99, user_role="user")
