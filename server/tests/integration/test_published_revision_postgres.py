"""Staged revisions on PostgreSQL: hidden until published, ids kept, superseded tokens refused."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.fakes import DEFAULT_SPACE_ID

from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.document_nodes import DocumentNode
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import OutboxOperation, VectorOutboxEntry
from domain.knowledge.repositories.chunk import ChunkInsertOptions
from infrastructure.database.models import ChunkModel, DocumentModel, DocumentNodeModel, StagedChunkModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.repositories.config.sqlalchemy_vector_outbox_repository import (
    SQLAlchemyVectorOutboxRepository,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

PUBLIC = DocumentVisibility.INTERNAL_PUBLIC.value
OPTIONS = ChunkInsertOptions(space_id=DEFAULT_SPACE_ID)


def node(node_id: str, chunk_end: int) -> DocumentNode:
    return DocumentNode(node_id, None, "article", node_id, None, 0, 0, chunk_end, (1, 2), 10, "hash")


async def seed(sessions: async_sessionmaker[AsyncSession]) -> int:
    async with sessions.begin() as session:
        doc = DocumentModel(
            space_id=DEFAULT_SPACE_ID, filename="law.txt", visibility=PUBLIC, status=DocumentStatus.DONE
        )
        session.add(doc)
        await session.flush()
        chunks = SQLAlchemyChunkRepository(session)
        await chunks.bulk_insert(
            doc.id, "law.txt", PUBLIC, [ChunkInsert("old 1"), ChunkInsert("old 2")], OPTIONS
        )
        await chunks.replace_document_nodes(doc.id, [node("article:old", 1)])
        return doc.id


async def contents(session: AsyncSession, document_id: int) -> list[tuple[int, str]]:
    rows = await session.execute(
        select(ChunkModel.id, ChunkModel.content)
        .where(ChunkModel.document_id == document_id)
        .order_by(ChunkModel.chunk_index)
    )
    return [(row.id, row.content) for row in rows]


async def test_staged_revision_is_hidden_until_published_and_keeps_its_ids(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    document_id = await seed(sessions)

    async with sessions.begin() as session:
        published_before = await contents(session, document_id)
        staged = await SQLAlchemyChunkRepository(session).stage_revision(
            document_id,
            "law.txt",
            PUBLIC,
            [ChunkInsert("new 1", embed_text="ctx new 1"), ChunkInsert("new 2")],
            nodes=[node("article:new", 1)],
            options=ChunkInsertOptions(space_id=DEFAULT_SPACE_ID, pack_metadata={"edition": 1}),
        )

    # Staged ids come from the chunk sequence: no collision with published chunks.
    assert not set(staged.chunk_ids) & {chunk_id for chunk_id, _ in published_before}

    async with sessions.begin() as session:
        chunks = SQLAlchemyChunkRepository(session)
        assert await contents(session, document_id) == published_before
        assert await chunks.get_staged_revision(document_id) == staged
        await chunks.set_pack_metadata(document_id, {"edition": 2})
        await chunks.update_filename_by_document_id(document_id, "renamed.txt")
        records = await chunks.get_staged_by_ids(document_id, staged.token, list(staged.chunk_ids))
        assert [(r.content, r.embed_text, r.filename, r.pack_metadata) for r in records] == [
            ("new 1", "ctx new 1", "renamed.txt", {"edition": 2}),
            ("new 2", None, "renamed.txt", {"edition": 2}),
        ]
        assert await chunks.get_staged_by_ids(document_id, "other-token", list(staged.chunk_ids)) == []
        assert await chunks.publish_staged_revision(document_id, "other-token") is None

    async with sessions.begin() as session:
        switch = await SQLAlchemyChunkRepository(session).publish_staged_revision(document_id, staged.token)

    assert switch is not None
    assert switch.published_ids == staged.chunk_ids
    assert switch.replaced_ids == tuple(chunk_id for chunk_id, _ in published_before)

    async with sessions.begin() as session:
        assert await contents(session, document_id) == list(
            zip(staged.chunk_ids, ["new 1", "new 2"], strict=True)
        )
        published = await SQLAlchemyChunkRepository(session).get_by_ids(document_id, list(staged.chunk_ids))
        assert {(r.filename, r.manual) for r in published} == {("renamed.txt", False)}
        nodes = await session.scalars(
            select(DocumentNodeModel.node_id).where(DocumentNodeModel.document_id == document_id)
        )
        assert list(nodes) == ["article:new"]
        assert await session.scalar(select(StagedChunkModel.id).limit(1)) is None
        assert await SQLAlchemyChunkRepository(session).get_staged_revision(document_id) is None


async def test_discarded_revision_leaves_published_chunks(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    document_id = await seed(sessions)

    async with sessions.begin() as session:
        chunks = SQLAlchemyChunkRepository(session)
        first = await chunks.stage_revision(
            document_id, "law.txt", PUBLIC, [ChunkInsert("A")], nodes=[], options=OPTIONS
        )
        assert await chunks.discard_staged_revision(document_id) == list(first.chunk_ids)
        second = await chunks.stage_revision(
            document_id, "law.txt", PUBLIC, [ChunkInsert("B")], nodes=[], options=OPTIONS
        )

    async with sessions.begin() as session:
        chunks = SQLAlchemyChunkRepository(session)
        assert await chunks.publish_staged_revision(document_id, first.token) is None
        assert [content for _, content in await contents(session, document_id)] == ["old 1", "old 2"]
        assert await chunks.get_staged_revision(document_id) == second

    # Deleting the document removes its staged revision with it.
    async with sessions.begin() as session:
        await session.delete(await session.get(DocumentModel, document_id))

    async with sessions.begin() as session:
        assert await session.scalar(select(StagedChunkModel.id).limit(1)) is None


@pytest.mark.parametrize("operation", [OutboxOperation.PUBLISH_REVISION, OutboxOperation.RENAME_DOCUMENT])
async def test_outbox_accepts_revision_and_rename_operations(
    pg_engine: AsyncEngine, operation: OutboxOperation
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    document_id = await seed(sessions)

    async with sessions.begin() as session:
        entry = await SQLAlchemyVectorOutboxRepository(session).enqueue(
            VectorOutboxEntry(
                operation=operation,
                aggregate_id=document_id,
                payload={"published_ids": [], "replaced_ids": []},
            )
        )

    assert entry.id is not None
