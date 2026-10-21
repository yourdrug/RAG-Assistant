"""Chunk positions remain unique across concurrent writes and schema upgrades."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from application.chunks import ChunkMutationService
from domain.identity.roles import UserRole
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.database.models import ChunkModel, DocumentModel, VectorStoreOutboxModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.uow_factory import UnitOfWorkFactory


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from application.chunks.dto import AddChunkResult


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def seed_document(sessions: async_sessionmaker[AsyncSession], initial_index: int = 3) -> int:
    async with sessions.begin() as session:
        doc = DocumentModel(
            filename="ordinary.txt",
            visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
            status=DocumentStatus.DONE,
        )
        session.add(doc)
        await session.flush()

        if initial_index >= 0:
            session.add(
                ChunkModel(
                    document_id=doc.id,
                    chunk_index=initial_index,
                    content="Existing",
                    visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
                )
            )

        return doc.id


async def wait_for_document_lock(
    sessions: async_sessionmaker[AsyncSession], second_read: asyncio.Event
) -> None:
    async with sessions() as observer:
        while True:
            assert not second_read.is_set(), "Second writer read MAX before first writer committed"
            blocked = await observer.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                    "WHERE cardinality(pg_blocking_pids(pid)) > 0 "
                    "AND query LIKE 'SELECT%documents%FOR UPDATE%')"
                )
            )

            if blocked:
                return

            await asyncio.sleep(0.01)


async def check_committed_chunks(
    sessions: async_sessionmaker[AsyncSession], document_id: int, initial_index: int
) -> None:
    async with sessions() as session:
        indexes = list(
            (await session.scalars(select(ChunkModel.chunk_index).order_by(ChunkModel.chunk_index))).all()
        )
        assert indexes == ([initial_index] if initial_index >= 0 else []) + [
            initial_index + 1,
            initial_index + 2,
        ]
        doc = await session.get(DocumentModel, document_id)
        assert doc is not None
        assert doc.chunks == len(indexes)
        entries = (await session.scalars(select(VectorStoreOutboxModel))).all()
        assert sorted(entry.payload["points"][0]["metadata"]["chunk_index"] for entry in entries) == [
            initial_index + 1,
            initial_index + 2,
        ]


@pytest.mark.parametrize("initial_index", [-1, 3])
async def test_concurrent_add_serializes_before_reading_max(
    pg_engine: AsyncEngine, initial_index: int
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    document_id = await seed_document(sessions, initial_index)
    database = MagicMock()
    database.get_session.side_effect = lambda **_kwargs: sessions()
    service = ChunkMutationService(UnitOfWorkFactory(database), SimpleNamespace(chunk_size=100))
    first_read, release, second_read = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = SQLAlchemyChunkRepository.get_max_chunk_index

    async def pause_after_max(repo: SQLAlchemyChunkRepository, doc_id: int) -> int:
        value = await original(repo, doc_id)

        if first_read.is_set():
            second_read.set()
        else:
            first_read.set()
            await release.wait()

        return value

    async def add(content: str) -> AddChunkResult:
        return await service.add_chunk(document_id, content * 4, 1, UserRole.ADMIN)

    with patch.object(SQLAlchemyChunkRepository, "get_max_chunk_index", pause_after_max):
        first = asyncio.create_task(add("First manual passage. "))
        second = None
        try:
            await asyncio.wait_for(first_read.wait(), 5)
            second = asyncio.create_task(add("Second manual passage. "))
            await asyncio.wait_for(wait_for_document_lock(sessions, second_read), 5)
        finally:
            release.set()
            results = await asyncio.wait_for(
                asyncio.gather(first, *([second] if second is not None else [])), 5
            )

    assert [result.chunk_index for result in results] == [initial_index + 1, initial_index + 2]
    await check_committed_chunks(sessions, document_id, initial_index)


async def test_duplicate_index_is_conflict_and_unrelated_integrity_errors_propagate(
    pg_engine: AsyncEngine,
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    document_id = await seed_document(sessions)
    async with sessions() as session:
        repo = SQLAlchemyChunkRepository(session)
        with pytest.raises(BusinessRuleViolationError, match="Chunk index already exists"):
            await repo.insert_one(
                document_id, 3, "Duplicate", "ordinary.txt", "internal_public", doc_domain="general"
            )
        await session.rollback()
        with pytest.raises(IntegrityError):
            await repo.insert_one(
                -1, 4, "Missing parent", "ordinary.txt", "internal_public", doc_domain="general"
            )
        await session.rollback()
        assert list((await session.scalars(select(ChunkModel.chunk_index))).all()) == [3]
