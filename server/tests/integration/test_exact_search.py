"""Exact search treats boundaries and metacharacters literally in PostgreSQL."""

import secrets

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.enums import DocumentVisibility
from infrastructure.database.models import ChunkModel, DocumentModel, UserModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,other",
    [
        ("слово", "словосочетание"),
        ("rate%", "rate-any"),
        ("doc_id", "docXid"),
        ("[x]", "x"),
        ("foo.bar", "fooXbar"),
        (r"a\b", "ab"),
        ("two words", "twoXwords"),
    ],
)
async def test_exact_matches_boundaries_and_literals_with_acl(
    pg_engine: AsyncEngine, query: str, other: str
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        owner = UserModel(email="owner@example.org", hashed_password=secrets.token_hex(32))
        session.add(owner)
        await session.flush()
        doc = DocumentModel(
            filename="literal.txt", visibility=DocumentVisibility.INTERNAL_PRIVATE.value, owner_id=owner.id
        )
        session.add(doc)
        await session.flush()
        contents = [query, f"{query.upper()} tail", f"head {query}", f"({query})", f"a{query}b", other]
        session.add_all(
            ChunkModel(
                document_id=doc.id,
                chunk_index=i,
                content=content,
                visibility=doc.visibility,
                owner_id=owner.id,
            )
            for i, content in enumerate(contents)
        )
    allowed = UserContext(owner.id, UserKind.INTERNAL, UserRole.USER)
    denied = UserContext(owner.id + 1, UserKind.INTERNAL, UserRole.USER)
    async with sessions() as session:
        repository = SQLAlchemyChunkRepository(session)
        matches, total = await repository.search_substring_page(query, allowed, document_id=doc.id)
        assert total == 4
        assert [match.chunk_index for match in matches] == [0, 1, 2, 3]
        assert await repository.search_substring_page(query, denied, document_id=doc.id) == ([], 0)
        assert await repository.search_substring_page(query, allowed, document_id=doc.id + 1) == ([], 0)
