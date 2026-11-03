"""Transaction and persistence contracts of shared application test doubles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from domain.chat.conversation import Conversation
from domain.chat.message import Message
from domain.identity.roles import UserKind
from domain.identity.user_context import UserContext
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus
from tests.fakes import (
    FakeChunkRepository,
    FakeConversationRepository,
    FakeDocumentRepository,
    FakeMessageRepository,
    FakeUnitOfWorkFactory,
)


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory


@pytest.mark.asyncio
async def test_factory_commits_and_rolls_back_shared_repositories() -> None:
    factory: UnitOfWorkFactory = FakeUnitOfWorkFactory()
    async with factory.create(master=True) as uow:
        saved = await uow.documents.save(Document(filename="committed.pdf"))
    assert saved.id is not None

    async def abort(document_id: int) -> None:
        async with factory.create(master=True) as uow:
            await uow.documents.delete(document_id)
            await uow.documents.save(Document(filename="rolled-back.pdf"))
            msg = "abort"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="abort"):
        await abort(saved.id)

    async with factory.create() as uow:
        restored = await uow.documents.get_by_id(saved.id)
        assert restored is not None
        assert restored.filename == "committed.pdf"
        assert len(await uow.documents.list_all()) == 1


@pytest.mark.asyncio
async def test_factory_keeps_event_handlers_local_to_each_transaction() -> None:
    factory: UnitOfWorkFactory = FakeUnitOfWorkFactory()
    events: list[object] = []

    async def record(event: object) -> None:
        events.append(event)

    async with factory.create() as uow:
        uow.on_event(record)
        await uow.publish_event("first")
    async with factory.create() as uow:
        await uow.publish_event("second")
    assert events == ["first"]


@pytest.mark.asyncio
async def test_conversation_save_returns_entity_with_identity() -> None:
    repo = FakeConversationRepository()
    conversation = await repo.save(Conversation(user_id=7, summary="summary"))
    assert conversation.id is not None
    restored = await repo.get_by_id(conversation.id)
    assert restored is not None
    assert (restored.user_id, restored.summary) == (7, "summary")
    assert not await repo.update_summary_if_unchanged(conversation.id, "stale", "changed")
    assert await repo.update_summary_if_unchanged(conversation.id, "summary", "changed")


@pytest.mark.asyncio
async def test_document_status_update_observes_expected_version() -> None:
    repo = FakeDocumentRepository()
    document = await repo.save(Document(status=DocumentStatus.INDEXING))
    assert document.id is not None
    assert not await repo.update_status(document.id, status=DocumentStatus.DONE, expected_version=1)
    assert document.status == DocumentStatus.INDEXING
    assert await repo.update_status(
        document.id, status=DocumentStatus.DONE, chunks=2, chars=9, expected_version=0
    )
    assert (document.status, document.version, document.chunks, document.chars) == (
        DocumentStatus.DONE,
        1,
        2,
        9,
    )
    assert not await repo.update_status(document.id + 1, status=DocumentStatus.DONE)


@pytest.mark.asyncio
async def test_message_pagination_handles_unsaved_message_ids() -> None:
    repo = FakeMessageRepository()
    persisted = Message(id=2, conversation_id=1)
    await repo.save(Message(conversation_id=1))
    await repo.save(persisted)
    assert await repo.list_page(1, limit=2, before_id=3) == [persisted]


@pytest.mark.asyncio
@pytest.mark.parametrize("table", [False, True])
async def test_unsupported_context_reads_cannot_bypass_acl(table: bool) -> None:
    repo = FakeChunkRepository()
    user = UserContext(user_id=7, user_kind=UserKind.INTERNAL)
    read = repo.get_table_batches(1, "table", user=user) if table else repo.get_neighbors(1, 0, user=user)
    with pytest.raises(NotImplementedError, match="ACL-aware"):
        await read
