"""Tests for DocumentCommandService and DocumentQueryService."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.documents import DocumentCommandService, DocumentQueryService
from application.documents.conflicts import resolve_upload_version_group
from application.documents.delete import DocumentDeleteCommand
from application.documents.rename import DocumentRenameCommand
from application.documents.upload import DocumentUploadCommand
from application.identity import UserContextFactory
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.shared.exceptions import (
    BusinessRuleViolationError,
    EntityNotFoundError,
    UniqueConstraintError,
    ValidationError,
)
from tests.fakes import FakeUnitOfWork, FakeUnitOfWorkFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CTX = UserContext(
    user_id=1,
    user_kind=UserKind.INTERNAL,
    user_role=UserRole.ADMIN,
    group_ids=(),
    managed_client_ids=(),
    managed_internal_ids=(),
    managed_group_ids=(),
)


def _make_doc(  # noqa: PLR0913 - Explicit inputs for the parametrized scenario/factory.
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


def _service(
    uow: FakeUnitOfWork | None = None, file_storage: MagicMock | None = None
) -> tuple[DocumentCommandService, DocumentQueryService, FakeUnitOfWork, MagicMock]:
    uow = uow or FakeUnitOfWork()
    factory = FakeUnitOfWorkFactory(uow)
    fs = file_storage or MagicMock()
    fs.supported_extensions = (".pdf", ".docx", ".txt", ".md")
    fs.upload_file = AsyncMock()
    fs.delete_file = AsyncMock()
    context_factory = UserContextFactory()
    cmd = DocumentCommandService(
        DocumentUploadCommand(factory, fs, context_factory),
        DocumentDeleteCommand(factory, fs, context_factory),
        DocumentRenameCommand(factory, fs, context_factory),
    )
    query = DocumentQueryService(uow_factory=factory)
    return cmd, query, uow, fs


@pytest.fixture(autouse=True)
def _patch_user_context_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace UserContextFactory.build with a stub returning a fixed context."""

    async def _fake_build(
        self: object, uow: object, user_id: object, user_kind: object, user_role: object
    ) -> UserContext:
        return _FAKE_CTX

    monkeypatch.setattr(UserContextFactory, "build", _fake_build)


# ---------------------------------------------------------------------------
# _resolve_version_group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_version_group_explicit_replaces() -> None:
    """replaces_document_id set → version group from that doc."""
    _cmd, _query, uow, _ = _service()
    replaces_doc = _make_doc(doc_id=42, version_group_id=10)
    # FakeDocumentRepository.save auto-assigns id=1, ignoring doc_id.
    # We need the doc in the repo with id=1.
    saved = await uow.documents.save(replaces_doc)
    assert saved.id == 1

    vgid, pending, existing = await resolve_upload_version_group(
        uow, existing=None, replaces_document_id=1, pending_replace_id=None
    )
    assert vgid == 10
    assert pending == saved.id
    assert existing is saved


@pytest.mark.asyncio
async def test_resolve_version_group_explicit_replaces_no_version_group() -> None:
    """replaces_document_id set but no version_group → falls back to doc id."""
    _cmd, _query, uow, _ = _service()
    replaces_doc = _make_doc(doc_id=7, version_group_id=None)
    saved = await uow.documents.save(replaces_doc)

    vgid, _, _ = await resolve_upload_version_group(
        uow, existing=None, replaces_document_id=saved.id, pending_replace_id=None
    )
    assert vgid == saved.id


@pytest.mark.asyncio
async def test_resolve_version_group_explicit_not_found() -> None:
    """replaces_document_id points to nonexistent doc → EntityNotFoundError."""
    _cmd, _query, uow, _ = _service()
    with pytest.raises(EntityNotFoundError):
        await resolve_upload_version_group(
            uow, existing=None, replaces_document_id=999, pending_replace_id=None
        )


@pytest.mark.asyncio
async def test_resolve_version_group_implicit_conflict() -> None:
    """No explicit replaces but existing DONE → implicit version group."""
    _cmd, _query, uow, _ = _service()
    existing = _make_doc(doc_id=5, version_group_id=3, status=DocumentStatus.DONE)
    saved = await uow.documents.save(existing)

    vgid, pending, returned = await resolve_upload_version_group(
        uow, existing=saved, replaces_document_id=None, pending_replace_id=None
    )
    assert vgid == 3
    assert pending == saved.id
    assert returned is saved


@pytest.mark.asyncio
async def test_resolve_version_group_implicit_no_version_group() -> None:
    """Existing DONE doc without version_group → falls back to doc id."""
    _cmd, _query, uow, _ = _service()
    existing = _make_doc(doc_id=9, version_group_id=None, status=DocumentStatus.DONE)
    saved = await uow.documents.save(existing)

    vgid, pending, _ = await resolve_upload_version_group(
        uow, existing=saved, replaces_document_id=None, pending_replace_id=None
    )
    assert vgid == saved.id
    assert pending == saved.id


@pytest.mark.asyncio
async def test_resolve_version_group_no_conflict() -> None:
    """No replaces_document_id and no existing → all None."""
    _cmd, _query, uow, _ = _service()
    vgid, pending, returned = await resolve_upload_version_group(
        uow, existing=None, replaces_document_id=None, pending_replace_id=None
    )
    assert vgid is None
    assert pending is None
    assert returned is None


@pytest.mark.asyncio
async def test_resolve_version_group_existing_pending() -> None:
    """Existing is PENDING (not DONE/FAILED) → no implicit conflict."""
    _cmd, _query, uow, _ = _service()
    existing = _make_doc(doc_id=3, status=DocumentStatus.PENDING)
    await uow.documents.save(existing)

    vgid, pending, _ = await resolve_upload_version_group(
        uow, existing=existing, replaces_document_id=None, pending_replace_id=None
    )
    assert vgid is None
    assert pending is None


# ---------------------------------------------------------------------------
# persist_upload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_upload_success() -> None:
    """Happy path: save + S3 upload + set_source_path."""
    cmd, _query, uow, fs = _service()
    doc = _make_doc()

    saved_doc, key = await cmd._upload.persist_upload(
        uow, doc, b"file-bytes", owner_id=1, effective_group_id=None, filename="test.pdf"
    )
    assert saved_doc.id is not None
    assert "uploads/users/1/" in key
    assert "test.pdf" in key
    fs.upload_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_persist_upload_s3_failure_compensates() -> None:
    """S3 upload fails → compensating delete of orphaned object."""
    cmd, _query, uow, fs = _service()
    fs.upload_file = AsyncMock(side_effect=RuntimeError("S3 down"))

    doc = _make_doc()
    with pytest.raises(RuntimeError, match="S3 down"):
        await cmd._upload.persist_upload(
            uow, doc, b"data", owner_id=1, effective_group_id=None, filename="test.pdf"
        )
    fs.delete_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_persist_upload_s3_failure_delete_also_fails() -> None:
    """S3 upload fails and compensating delete also fails → original error raised."""
    cmd, _query, uow, fs = _service()
    fs.upload_file = AsyncMock(side_effect=RuntimeError("S3 down"))
    fs.delete_file = AsyncMock(side_effect=RuntimeError("delete failed"))

    doc = _make_doc()
    with pytest.raises(RuntimeError, match="S3 down"):
        await cmd._upload.persist_upload(
            uow, doc, b"data", owner_id=1, effective_group_id=None, filename="test.pdf"
        )
    fs.delete_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# Checks for upload: end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_basic() -> None:
    """Basic upload with no conflicts → doc created, DTO returned."""
    cmd, _query, _uow, fs = _service()
    dto = await cmd.upload(
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
async def test_upload_unsupported_extension() -> None:
    """Unsupported file extension → ValidationError."""
    cmd, _, _, _ = _service()
    with pytest.raises(ValidationError, match="Unsupported file format"):
        await cmd.upload(
            filename="data.xyz",
            file_data=b"x",
            visibility="internal_public",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )


@pytest.mark.asyncio
async def test_upload_internal_group_no_group_id() -> None:
    """INTERNAL_GROUP visibility without group_id → ValidationError."""
    cmd, _, _, _ = _service()
    with pytest.raises(ValidationError, match="group_id required"):
        await cmd.upload(
            filename="doc.pdf",
            file_data=b"data",
            visibility="internal_group",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )


@pytest.mark.asyncio
async def test_upload_active_processing_rejected() -> None:
    """Existing doc in PENDING/PROCESSING/INDEXING → BusinessRuleViolationError."""
    cmd, _query, uow, _ = _service()
    pending_doc = _make_doc(doc_id=10, status=DocumentStatus.PENDING, filename="doc.pdf")
    await uow.documents.save(pending_doc)

    with pytest.raises(BusinessRuleViolationError, match="already being processed"):
        await cmd.upload(
            filename="doc.pdf",
            file_data=b"data",
            visibility="internal_private",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )


@pytest.mark.asyncio
async def test_upload_unique_constraint_violation_translates_to_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database uniqueness violation is translated to an application conflict."""
    cmd, _query, uow, _fs = _service()

    original_save = uow.documents.save

    call_count = 0

    async def failing_save(doc: Any) -> Document:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            msg = "dup"
            raise UniqueConstraintError(msg)

        return await original_save(doc)

    monkeypatch.setattr(uow.documents, "save", failing_save)

    with pytest.raises(BusinessRuleViolationError, match="already being uploaded"):
        await cmd.upload(
            filename="doc.pdf",
            file_data=b"data",
            visibility="internal_private",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )


# ---------------------------------------------------------------------------
# Checks for delete_document: characterization.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_happy_path() -> None:
    """delete_document: outbox enqueued, summary clear, doc deleted, storage deleted."""
    cmd, _query, uow, fs = _service()
    fs.copy_file = AsyncMock()
    doc = _make_doc(doc_id=1, source_path="s3://bucket/docs/1/test.pdf")
    await uow.documents.save(doc)

    # Sparse cleanup is deferred to the committed outbox.
    await uow.chunks.bulk_insert(
        document_id=1,
        filename="test.pdf",
        visibility="internal_public",
        chunks=[ChunkInsert(content="chunk content", content_hash="hash_abc")],
    )

    await cmd.delete_document(document_id=1, user_id=1, user_role="admin")

    # 1. Outbox entry enqueued with DELETE_BY_DOCUMENT
    outbox_entries = list(uow.fake_vector_outbox._entries.values())
    assert len(outbox_entries) == 1
    assert outbox_entries[0].operation.value == "delete_by_document"
    assert outbox_entries[0].payload == {"document_id": 1}

    # 3. Summaries cleared (all in the fake)
    for conv in uow.fake_conversations._convs.values():
        assert conv.get("summary") is None

    # 4. Document deleted from repository
    assert await uow.documents.get_by_id(1) is None

    # 5. Storage file deleted (after transaction commit)
    fs.delete_file.assert_awaited_once_with("s3://bucket/docs/1/test.pdf")


@pytest.mark.asyncio
async def test_delete_document_not_found() -> None:
    """delete_document: non-existent doc → EntityNotFoundError."""
    cmd, _, _, _ = _service()
    with pytest.raises(EntityNotFoundError):
        await cmd.delete_document(document_id=999, user_id=1, user_role="admin")


@pytest.mark.asyncio
async def test_delete_document_no_source_path() -> None:
    """delete_document: doc without source_path → storage delete skipped."""
    cmd, _query, uow, fs = _service()
    doc = _make_doc(doc_id=1, source_path=None)
    await uow.documents.save(doc)

    await cmd.delete_document(document_id=1, user_id=1, user_role="admin")

    fs.delete_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_document_storage_delete_failure_logs_warning() -> None:
    """delete_document: S3 delete fails → warning logged, no exception raised."""
    cmd, _query, uow, fs = _service()
    doc = _make_doc(doc_id=1, source_path="s3://bucket/docs/1/test.pdf")
    await uow.documents.save(doc)
    fs.delete_file = AsyncMock(side_effect=RuntimeError("S3 down"))

    # Should not raise — storage delete failure is best-effort
    await cmd.delete_document(document_id=1, user_id=1, user_role="admin")

    # Document still deleted from DB
    assert await uow.documents.get_by_id(1) is None
    fs.delete_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# Checks for rename_document: characterization.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rename_document_different_path_copies_and_deletes_old() -> None:
    """rename_document: different path → S3 copy, DB updated, outbox enqueued, old file deleted."""
    cmd, _query, uow, fs = _service()
    fs.copy_file = AsyncMock()
    doc = _make_doc(doc_id=1, source_path="s3://bucket/u/1/1/old.pdf", filename="old.pdf")
    await uow.documents.save(doc)
    await uow.chunks.bulk_insert(
        document_id=1,
        filename="old.pdf",
        visibility="internal_public",
        chunks=[ChunkInsert(content="content", content_hash="h1")],
    )

    dto = await cmd.rename_document(document_id=1, new_filename="new.pdf", user_id=1, user_role="admin")

    # 1. S3 copy called (old → new)
    fs.copy_file.assert_awaited_once()
    src, dst = fs.copy_file.call_args[0]
    assert "old.pdf" in src
    assert "new.pdf" in dst

    # 2. Document filename updated in DB
    assert dto.filename == "new.pdf"

    # 3. Outbox entry patches source metadata without recomputing vectors
    outbox_entries = list(uow.fake_vector_outbox._entries.values())
    assert len(outbox_entries) == 1
    assert outbox_entries[0].operation.value == "rename_document"

    # 4. Old source file deleted (after transaction commit)
    fs.delete_file.assert_awaited_once_with("s3://bucket/u/1/1/old.pdf")


@pytest.mark.asyncio
async def test_rename_document_same_path_no_copy() -> None:
    """rename_document: doc without source_path → no S3 copy needed."""
    cmd, _query, uow, fs = _service()
    fs.copy_file = AsyncMock()
    doc = _make_doc(doc_id=1, source_path=None, filename="test.pdf")
    await uow.documents.save(doc)

    await cmd.rename_document(document_id=1, new_filename="test.pdf", user_id=1, user_role="admin")

    # No S3 copy (source_path is None → condition skips copy)
    fs.copy_file.assert_not_awaited()
    # No old file deletion (nothing changed)
    fs.delete_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_rename_document_unsupported_extension() -> None:
    """rename_document: unsupported extension → ValidationError."""
    cmd, _, _, _ = _service()
    with pytest.raises(ValidationError, match="Unsupported file format"):
        await cmd.rename_document(document_id=1, new_filename="data.xyz", user_id=1, user_role="admin")


@pytest.mark.asyncio
async def test_rename_document_not_found() -> None:
    """rename_document: non-existent doc → EntityNotFoundError."""
    cmd, _, _, _ = _service()
    with pytest.raises(EntityNotFoundError):
        await cmd.rename_document(document_id=999, new_filename="new.pdf", user_id=1, user_role="admin")


@pytest.mark.asyncio
async def test_rename_document_conflict_with_pending_doc() -> None:
    """rename_document: existing doc with same name in PENDING → BusinessRuleViolationError."""
    cmd, _query, uow, _ = _service()
    # Create two docs with different IDs but same name-slot target
    doc1 = _make_doc(doc_id=1, filename="a.pdf", owner_id=1, group_id=None)
    doc2 = _make_doc(doc_id=2, filename="b.pdf", status=DocumentStatus.PENDING, owner_id=1, group_id=None)
    await uow.documents.save(doc1)
    await uow.documents.save(doc2)

    with pytest.raises(BusinessRuleViolationError, match="already being processed"):
        await cmd.rename_document(document_id=1, new_filename="b.pdf", user_id=1, user_role="admin")


@pytest.mark.asyncio
async def test_rename_document_conflict_with_done_doc_renames() -> None:
    """rename_document: existing DONE doc with same name → auto-resolves to unique name."""
    cmd, _query, uow, fs = _service()
    fs.copy_file = AsyncMock()
    doc1 = _make_doc(doc_id=1, filename="a.pdf", source_path="s3://bucket/u/1/1/a.pdf", owner_id=1)
    doc2 = _make_doc(doc_id=2, filename="report.pdf", status=DocumentStatus.DONE, owner_id=1)
    await uow.documents.save(doc1)
    await uow.documents.save(doc2)

    dto = await cmd.rename_document(document_id=1, new_filename="report.pdf", user_id=1, user_role="admin")

    # Filename should have been auto-resolved to avoid collision
    assert dto.filename != "report.pdf"


# ---------------------------------------------------------------------------
# rename_document — compensation tests (DDD-003)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rename_document_copy_succeeds_db_fails_compensates(monkeypatch: pytest.MonkeyPatch) -> None:
    """rename_document: S3 copy OK, DB update fails → compensating delete of orphaned copy."""
    cmd, _query, uow, fs = _service()
    fs.copy_file = AsyncMock()

    doc = _make_doc(doc_id=1, source_path="s3://bucket/u/1/1/old.pdf", filename="old.pdf")
    await uow.documents.save(doc)

    # Make update_filename raise to simulate DB failure after S3 copy
    async def failing_update(*args: Any, **kwargs: Any) -> None:
        msg = "DB failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(uow.documents, "update_filename", failing_update)

    with pytest.raises(RuntimeError, match="DB failure"):
        await cmd.rename_document(document_id=1, new_filename="new.pdf", user_id=1, user_role="admin")

    # S3 copy was attempted
    fs.copy_file.assert_awaited_once()
    # Compensating: orphaned copy deleted (uses the computed storage key, not the hardcoded path)
    assert fs.delete_file.await_count == 1
    deleted_path = fs.delete_file.call_args[0][0]
    assert "new.pdf" in deleted_path


@pytest.mark.asyncio
async def test_rename_document_copy_fails_no_compensation() -> None:
    """rename_document: S3 copy fails → no compensation delete, exception propagates."""
    cmd, _query, uow, fs = _service()
    fs.copy_file = AsyncMock(side_effect=RuntimeError("S3 down"))

    doc = _make_doc(doc_id=1, source_path="s3://bucket/u/1/1/old.pdf", filename="old.pdf")
    await uow.documents.save(doc)

    with pytest.raises(RuntimeError, match="S3 down"):
        await cmd.rename_document(document_id=1, new_filename="new.pdf", user_id=1, user_role="admin")

    # No compensation delete (copy failed, nothing to clean up)
    fs.delete_file.assert_not_awaited()


# ---------------------------------------------------------------------------
# Checks for list_documents: characterization.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_documents_admin_sees_all() -> None:
    """Admin sees all documents."""
    _cmd, query, uow, _ = _service()
    await uow.documents.save(_make_doc(doc_id=1, filename="a.pdf"))
    await uow.documents.save(_make_doc(doc_id=2, filename="b.pdf"))

    result = await query.list_documents(user_id=99, user_kind="internal", user_role="admin")

    assert len(result) == 2
    filenames = {d.filename for d in result}
    assert filenames == {"a.pdf", "b.pdf"}


@pytest.mark.asyncio
async def test_list_documents_empty() -> None:
    """No documents → empty list."""
    _, query, _, _ = _service()
    result = await query.list_documents(user_id=1, user_kind="internal", user_role="user")
    assert result == []


@pytest.mark.asyncio
async def test_list_documents_user_sees_only_own() -> None:
    """Regular user only sees own PRIVATE documents (via ACL)."""
    _cmd, query, uow, _ = _service()
    await uow.documents.save(
        _make_doc(
            doc_id=1,
            filename="own.pdf",
            owner_id=1,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
        )
    )
    await uow.documents.save(
        _make_doc(
            doc_id=2,
            filename="other.pdf",
            owner_id=2,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
        )
    )

    async def _user_ctx(
        self_factory: Any, uow: Any, user_id: Any, user_kind: Any, user_role: Any
    ) -> UserContext:
        return UserContext(
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            group_ids=(),
            managed_client_ids=(),
            managed_internal_ids=(),
            managed_group_ids=(),
        )

    with patch.object(UserContextFactory, "build", _user_ctx):
        result = await query.list_documents(user_id=1, user_kind="internal", user_role="user")

    assert len(result) == 1
    assert result[0].filename == "own.pdf"


@pytest.mark.asyncio
async def test_list_documents_client_sees_only_own() -> None:
    """Client user only sees own CLIENT_PRIVATE documents."""
    _cmd, query, uow, _ = _service()
    await uow.documents.save(
        _make_doc(
            doc_id=1,
            filename="own.pdf",
            owner_id=10,
            visibility=DocumentVisibility.CLIENT_PRIVATE,
        )
    )
    await uow.documents.save(
        _make_doc(
            doc_id=2,
            filename="other.pdf",
            owner_id=20,
            visibility=DocumentVisibility.CLIENT_PRIVATE,
        )
    )

    async def _user_ctx(
        self_factory: Any, uow: Any, user_id: Any, user_kind: Any, user_role: Any
    ) -> UserContext:
        return UserContext(
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            group_ids=(),
            managed_client_ids=(),
            managed_internal_ids=(),
            managed_group_ids=(),
        )

    with patch.object(UserContextFactory, "build", _user_ctx):
        result = await query.list_documents(user_id=10, user_kind="client", user_role="user")

    assert len(result) == 1
    assert result[0].filename == "own.pdf"


@pytest.mark.asyncio
async def test_list_documents_with_limit_offset() -> None:
    """Pagination works."""
    _cmd, query, uow, _ = _service()

    for i in range(5):
        await uow.documents.save(_make_doc(filename=f"doc{i}.pdf", owner_id=1))

    result = await query.list_documents(user_id=99, user_kind="internal", user_role="user", limit=2, offset=0)
    assert len(result) == 2

    result2 = await query.list_documents(
        user_id=99, user_kind="internal", user_role="user", limit=2, offset=2
    )
    assert len(result2) == 2


# ---------------------------------------------------------------------------
# Checks for get_document: characterization.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_document_happy_path() -> None:
    """get_document: existing doc + ACL OK → returns DTO."""
    _cmd, query, uow, _ = _service()
    doc = _make_doc(doc_id=1, filename="report.pdf")
    await uow.documents.save(doc)

    dto = await query.get_document(document_id=1, user_id=1, user_kind="internal", user_role="admin")

    assert dto.id == 1
    assert dto.filename == "report.pdf"


@pytest.mark.asyncio
async def test_get_document_not_found() -> None:
    """get_document: non-existent doc → EntityNotFoundError."""
    _, query, _, _ = _service()
    with pytest.raises(EntityNotFoundError):
        await query.get_document(document_id=999, user_id=1, user_kind="internal", user_role="admin")


# ---------------------------------------------------------------------------
# Checks for list_uploadable_clients: characterization.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_uploadable_clients_admin() -> None:
    """Admin sees all client users."""
    _cmd, query, uow, _ = _service()
    uow.fake_users.add_user(10, email="client1@test.com", kind="client")
    uow.fake_users.add_user(20, email="client2@test.com", kind="client")
    uow.fake_users.add_user(30, email="internal@test.com", kind="internal")

    result = await query.list_uploadable_clients(user_id=99, user_kind="internal", user_role="admin")

    assert len(result) == 2
    emails = {c.email for c in result}
    assert emails == {"client1@test.com", "client2@test.com"}


@pytest.mark.asyncio
async def test_list_uploadable_clients_curator() -> None:
    """Curator sees only managed clients."""
    _cmd, query, uow, _ = _service()
    uow.fake_users.add_user(10, email="managed@test.com", kind="client")
    uow.fake_users.add_user(20, email="unmanaged@test.com", kind="client")
    uow.fake_users.add_user(30, email="other@test.com", kind="client")
    uow.fake_assignments.set_user_kind(10, "client")
    uow.fake_assignments.set_user_kind(20, "client")
    await uow.assignments.assign_user(manager_id=1, target_user_id=10, assigned_by=99)

    result = await query.list_uploadable_clients(user_id=1, user_kind="internal", user_role="curator")

    assert len(result) == 1
    assert result[0].id == 10


@pytest.mark.asyncio
async def test_list_uploadable_clients_client_returns_empty() -> None:
    """Client users get empty list."""
    _, query, _, _ = _service()
    result = await query.list_uploadable_clients(user_id=10, user_kind="client", user_role="user")
    assert result == []


@pytest.mark.asyncio
async def test_list_uploadable_clients_user_returns_empty() -> None:
    """Regular users get empty list."""
    _, query, _, _ = _service()
    result = await query.list_uploadable_clients(user_id=1, user_kind="internal", user_role="user")
    assert result == []


# ---------------------------------------------------------------------------
# Checks for list_source_files: characterization.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_source_files_basic() -> None:
    """Returns distinct filenames."""
    _cmd, query, uow, _ = _service()
    await uow.documents.save(_make_doc(filename="report.pdf"))
    await uow.documents.save(_make_doc(filename="report.pdf"))
    await uow.documents.save(_make_doc(filename="other.pdf"))

    result = await query.list_source_files()

    assert sorted(result) == ["other.pdf", "report.pdf"]


@pytest.mark.asyncio
async def test_list_source_files_with_search() -> None:
    """Search filters filenames."""
    _cmd, query, uow, _ = _service()
    await uow.documents.save(_make_doc(filename="report.pdf"))
    await uow.documents.save(_make_doc(filename="data.csv"))
    await uow.documents.save(_make_doc(filename="report_v2.pdf"))

    result = await query.list_source_files(search="report")

    assert sorted(result) == ["report.pdf", "report_v2.pdf"]


@pytest.mark.asyncio
async def test_list_source_files_empty() -> None:
    """No documents → empty list."""
    _, query, _, _ = _service()
    result = await query.list_source_files()
    assert result == []


# ---------------------------------------------------------------------------
# delete_document — permission checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_not_owner_not_admin() -> None:
    """Non-owner non-admin → BusinessRuleViolationError (via check_ownership)."""
    cmd, _query, uow, _ = _service()
    doc = _make_doc(doc_id=1, owner_id=10)
    await uow.documents.save(doc)

    async def _user_ctx(
        self_factory: object, uow: object, user_id: int, user_kind: UserKind, user_role: UserRole
    ) -> UserContext:
        return UserContext(
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            group_ids=(),
            managed_client_ids=(),
            managed_internal_ids=(),
            managed_group_ids=(),
        )

    with patch.object(UserContextFactory, "build", _user_ctx), pytest.raises(BusinessRuleViolationError):
        await cmd.delete_document(document_id=1, user_id=99, user_role="user")
