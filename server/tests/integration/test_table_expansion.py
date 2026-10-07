"""Real SQL table boundaries and request ACL, including backward row continuations."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.value_objects.chunk_context import TABLE_CONTEXT_MAX_CHUNKS
from domain.value_objects.page_content_type import PageContentType
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.database.models import DocumentModel, UserModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata
from infrastructure.ml.rag.rag_postprocess import enrich_with_neighbors
from langchain.schema import Document

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_two_tables_text_limit_row_range_and_denied_document(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    header = "| Field | Format | Condition |\n|---|---|---|"
    contents = [
        header + "\n| 23 | an..19 | only after approval |",
        header + "\n| 23 | an..19 | except revoked approval |",
        header + "\n| 24 | n..8 | unrelated row |",
        "INTERMEDIATE TEXT",
        "| Other | Value |\n|---|---|\n| SECOND TABLE | forbidden |",
    ]
    contexts = [
        {"table_id": "table-1", "table_header": header, "table_row_start": n, "table_row_end": n}
        for n in [1, 1, 2]
    ] + [{}, {"table_id": "table-2", "table_row_start": 1, "table_row_end": 1}]
    async with sessions.begin() as session:
        users = [
            UserModel(
                email=f"table-user-{n}@example.org",
                hashed_password="test",
                role=UserRole.USER.value,
                kind=UserKind.INTERNAL.value,
            )
            for n in range(2)
        ]
        session.add_all(users)
        await session.flush()
        docs = [
            DocumentModel(
                filename=f"tables-{n}.md",
                visibility=DocumentVisibility.INTERNAL_PRIVATE.value,
                owner_id=users[n].id,
            )
            for n in range(2)
        ]
        session.add_all(docs)
        await session.flush()
        repository = SQLAlchemyChunkRepository(session)
        for doc in docs:
            await repository.bulk_insert(
                doc.id,
                doc.filename,
                doc.visibility,
                contents,
                owner_id=doc.owner_id,
                context_metadata=contexts,
                content_hashes=[f"doc-{doc.id}-{n}" for n in range(len(contents))],
                content_types=[PageContentType.TABLE.value] * 3 + [None, PageContentType.TABLE.value],
            )
    allowed = UserContext(users[0].id, UserKind.INTERNAL, UserRole.USER)
    denied = UserContext(users[1].id, UserKind.INTERNAL, UserRole.USER)
    async with sessions() as session:
        repository = SQLAlchemyChunkRepository(session)
        batches = await repository.get_table_batches(docs[0].id, "table-1", user=allowed)
        assert [item.content for item in batches] == contents[:3]
        assert await repository.get_table_batches(docs[0].id, "table-1", user=denied) == []
        assert await repository.get_table_batches(docs[1].id, "table-1", user=allowed) == []
        assert await repository.get_table_batches(docs[0].id, "missing", user=allowed) == []
        assert len(await repository.get_table_batches(docs[0].id, "table-1", user=allowed, limit=1)) == 1
        for limit in [0, -1, TABLE_CONTEXT_MAX_CHUNKS + 1]:
            with pytest.raises(ValueError):
                await repository.get_table_batches(docs[0].id, "table-1", user=allowed, limit=limit)
        parts = await repository.get_table_batches(
            docs[0].id,
            "table-1",
            user=allowed,
            row_start=1,
            row_end=1,
            exclude_hashes={batches[1].content_hash},
        )
        assert [part.content for part in parts] == contents[:1]
        # Hit the second part: continuation lookup must find the preceding part too.
        seed = Document(page_content=batches[1].content, metadata=chunk_result_metadata(batches[1]))
        focused = await enrich_with_neighbors(
            [(seed, 0.9)],
            False,
            repository,
            user=allowed,
            query="Формат поля 23?",
        )
        text = "\n".join(doc.page_content for doc, _ in focused)
        assert "only after approval" in text and "except revoked approval" in text
        assert "unrelated row" not in text and "INTERMEDIATE" not in text and "SECOND TABLE" not in text
        full = await enrich_with_neighbors(
            [(seed, 0.9)],
            True,
            repository,
            user=allowed,
            query="Перечисли все поля таблицы",
        )
        assert len(full) == 3
        assert len(text) < sum(len(doc.page_content) for doc, _ in full)
