"""Execute real search SQL against PostgreSQL to protect pagination and scoped totals."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
import secrets
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from application.search import SearchService
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.enums import DocDomain, DocumentVisibility, SearchMode
from infrastructure.database.models import ChunkModel, DocumentModel, UserModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository


pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def repository(pg_engine: AsyncEngine) -> AsyncIterator[SQLAlchemyChunkRepository]:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        session.add(UserModel(id=99, email="other@example.org", hashed_password=secrets.token_hex(32)))
        await session.flush()
        session.add_all(
            DocumentModel(
                id=document_id,
                filename=f"test-{document_id}.txt",
                visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
            )
            for document_id in (10, 20)
        )
        await session.flush()
        rows = [
            ChunkModel(
                id=chunk_id,
                document_id=10 if chunk_id < 4 else 20,
                filename="test.txt",
                chunk_index=chunk_id,
                content="a needle appears here" if chunk_id != 7 else "no matching text",
                visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
                doc_domain=DocDomain.GENERAL.value,
                is_current=True,
            )
            for chunk_id in range(1, 8)
        ]
        rows[3].visibility = DocumentVisibility.INTERNAL_PRIVATE.value
        rows[3].owner_id = 99
        rows[4].effective_from = datetime.now(UTC).astimezone().date() + timedelta(days=1)
        rows[5].is_current = False
        session.add_all(rows)
    async with sessions() as session:
        yield SQLAlchemyChunkRepository(session)


@pytest.fixture
def user() -> UserContext:
    return UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(SearchMode))
async def test_pages_cover_all_accessible_matches_with_same_total(
    repository: SQLAlchemyChunkRepository, user: UserContext, mode: SearchMode
) -> None:
    first, total = await repository.search_substring_page("needle", user, limit=2, offset=0, mode=mode.value)
    second, second_total = await repository.search_substring_page(
        "needle", user, limit=2, offset=2, mode=mode.value
    )
    assert [row.chunk_id for row in first] == [1, 2]
    assert [row.chunk_id for row in second] == [3]
    assert total == second_total == 3


@pytest.mark.asyncio
async def test_out_of_range_page_still_reports_scoped_total(
    repository: SQLAlchemyChunkRepository, user: UserContext
) -> None:
    rows, total = await repository.search_substring_page("needle", user, limit=2, offset=100)
    assert rows == []
    assert total == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("document_id,total", [(10, 3), (20, 0)])
async def test_document_filter_applies_to_total_without_bypassing_acl(
    repository: SQLAlchemyChunkRepository, user: UserContext, document_id: int, total: int
) -> None:
    rows, actual_total = await repository.search_substring_page("needle", user, document_id=document_id)
    assert actual_total == total
    assert len(rows) == total


@pytest.mark.asyncio
async def test_no_matches_and_short_queries_have_zero_total(
    repository: SQLAlchemyChunkRepository, user: UserContext
) -> None:
    for query in ["missing", "  "]:
        assert await repository.search_substring_page(query, user) == ([], 0)


@pytest.mark.asyncio
async def test_application_service_preserves_requested_page_and_total(
    repository: SQLAlchemyChunkRepository, user: UserContext
) -> None:
    uow = AsyncMock()
    uow.chunks = repository
    uow.groups.get_user_group_ids.return_value = []
    factory = Mock()
    factory.create.return_value.__aenter__ = AsyncMock(return_value=uow)
    factory.create.return_value.__aexit__ = AsyncMock(return_value=False)
    service = SearchService(factory)
    rows, total = await service.exact_search("needle", user, limit=2, offset=2)
    assert [row.chunk_id for row in rows] == [3]
    assert total == 3
