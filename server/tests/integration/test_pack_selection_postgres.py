"""A pack selection selects the same chunks in PostgreSQL as in Qdrant and the reference semantics."""

import secrets

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from tests.pack_selection_cases import PACK_METADATA, SELECTIONS

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.enums import DocumentVisibility
from domain.knowledge.pack_metadata import matches
from infrastructure.database.models import ChunkModel, DocumentModel, UserModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_sql_selection_matches_reference_semantics_under_acl(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        owner = UserModel(email=f"{secrets.token_hex(4)}@example.org", hashed_password=secrets.token_hex(32))
        stranger = UserModel(
            email=f"{secrets.token_hex(4)}@example.org", hashed_password=secrets.token_hex(32)
        )
        session.add_all([owner, stranger])
        await session.flush()
        public = DocumentModel(filename="public.txt", visibility=DocumentVisibility.INTERNAL_PUBLIC.value)
        private = DocumentModel(
            filename="private.txt", visibility=DocumentVisibility.INTERNAL_PRIVATE.value, owner_id=stranger.id
        )
        session.add_all([public, private])
        await session.flush()
        session.add_all(
            ChunkModel(
                document_id=public.id,
                chunk_index=index,
                content=f"marker {index}",
                visibility=public.visibility,
                pack_metadata=metadata,
            )
            for index, metadata in enumerate(PACK_METADATA)
        )
        # Matches every selection by its pack metadata but is private to another user.
        session.add(
            ChunkModel(
                document_id=private.id,
                chunk_index=0,
                content="marker private",
                visibility=private.visibility,
                owner_id=stranger.id,
                pack_metadata=PACK_METADATA[2],
            )
        )

    user = UserContext(owner.id, UserKind.INTERNAL, UserRole.USER)
    async with sessions() as session:
        repository = SQLAlchemyChunkRepository(session)

        for name, selection in SELECTIONS.items():
            found = await repository.search_substring(
                "marker", user, limit=100, mode="icontains", selection=selection
            )
            expected = {index for index, metadata in enumerate(PACK_METADATA) if matches(selection, metadata)}
            assert {chunk.chunk_index for chunk in found if chunk.document_id == public.id} == expected, name
            assert all(chunk.document_id != private.id for chunk in found), name

        everything = await repository.search_substring("marker", user, limit=100, mode="icontains")
        assert len(everything) == len(PACK_METADATA)
        assert {chunk.chunk_index: chunk.pack_metadata for chunk in everything} == dict(
            enumerate(PACK_METADATA)
        )
