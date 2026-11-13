"""Run the production parent-scope query on SQLite with the mapped chunk and node tables.

Only storage types are copied (JSONB → JSON); authorization and the query stay production code.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

from sqlalchemy import JSON, Column, MetaData, Table, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from domain.knowledge.document_nodes import PARENT_NODE_ID_KEY
from infrastructure.database.models import ChunkModel, DocumentNodeModel
from infrastructure.repositories.chunk.chunk_context import SQLAlchemyChunkContextRepository


if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.ext.asyncio import AsyncSession


DOCUMENT_ID = 7
FOREIGN_DOCUMENT_ID = 8
ACT_VERSION_ID = 9
OWNER_ID = 12
CHAPTER = "chapter:1"
ARTICLE = "chapter:1/article:1"
POINT = "chapter:1/article:1/point:2"
SIBLING_ARTICLE = "chapter:1/article:2"
# Rows: chunk index, parent node id, body.
CHUNKS = (
    (0, ARTICLE, "Статья 1. Вступление"),
    (1, ARTICLE, "1. Первый пункт"),
    (2, POINT, "а) подпункт второго пункта"),
    (3, SIBLING_ARTICLE, "Статья 2. Другая статья"),
    (4, None, "Текст без раздела"),
)
NODES = (
    (CHAPTER, None, "chapter", 1, 0, 3, 400),
    (ARTICLE, CHAPTER, "article", 2, 0, 2, 300),
    (POINT, ARTICLE, "point", 3, 2, 2, 100),
    (SIBLING_ARTICLE, CHAPTER, "article", 2, 3, 3, 100),
)


def sqlite_table(model: type[ChunkModel | DocumentNodeModel], schema: MetaData) -> Table:
    return Table(
        model.__tablename__,
        schema,
        *(
            Column(
                column.name,
                JSON() if isinstance(column.type, JSONB) else column.type,
                primary_key=column.primary_key,
            )
            for column in model.__table__.columns
        ),
    )


@contextmanager
def parent_scope_repository(
    visibility: str,
    *,
    owner_id: int | None = OWNER_ID,
    nodes: tuple[tuple[str, str | None, str, int, int, int, int], ...] = NODES,
) -> Iterator[SQLAlchemyChunkContextRepository]:
    engine = create_engine("sqlite://")
    schema = MetaData()
    sqlite_table(ChunkModel, schema)
    sqlite_table(DocumentNodeModel, schema)
    schema.create_all(engine)

    with Session(engine) as session:
        session.add_all(
            [
                ChunkModel(
                    id=index + 1,
                    document_id=DOCUMENT_ID,
                    chunk_index=index,
                    filename="law.docx",
                    content=content,
                    visibility=visibility,
                    owner_id=owner_id,
                    act_version_id=ACT_VERSION_ID,
                    context_metadata={PARENT_NODE_ID_KEY: node_id} if node_id else {},
                )
                for index, node_id, content in CHUNKS
            ]
            + [
                # Same node ids in a public foreign document must never be mixed in.
                ChunkModel(
                    id=100,
                    document_id=FOREIGN_DOCUMENT_ID,
                    chunk_index=1,
                    filename="foreign.docx",
                    content="foreign",
                    visibility="internal_public",
                    owner_id=None,
                    context_metadata={PARENT_NODE_ID_KEY: ARTICLE},
                )
            ]
            + [
                DocumentNodeModel(
                    id=position + 1,
                    document_id=DOCUMENT_ID,
                    act_version_id=ACT_VERSION_ID,
                    node_id=node_id,
                    parent_id=parent_id,
                    kind=kind,
                    label=node_id,
                    depth=depth,
                    chunk_start=start,
                    chunk_end=end,
                    pages=[],
                    char_count=chars,
                    text_hash="0" * 16,
                )
                for position, (node_id, parent_id, kind, depth, start, end, chars) in enumerate(nodes)
            ]
            + [
                DocumentNodeModel(
                    id=100,
                    document_id=FOREIGN_DOCUMENT_ID,
                    node_id=ARTICLE,
                    kind="article",
                    label=ARTICLE,
                    depth=2,
                    chunk_start=0,
                    chunk_end=5,
                    pages=[],
                    char_count=10,
                    text_hash="0" * 16,
                )
            ]
        )
        session.flush()
        async_session = SimpleNamespace(execute=AsyncMock(side_effect=session.execute))
        yield SQLAlchemyChunkContextRepository(cast("AsyncSession", async_session))

    engine.dispose()
