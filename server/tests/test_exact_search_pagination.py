"""Execute real search SQL against SQLite to protect pagination and scoped totals."""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import Column, Integer, JSON, MetaData, Table, create_engine
from sqlalchemy.dialects.postgresql import JSONB

from application.search import SearchService
from domain.value_objects.doc_domain import DocDomain
from domain.identity.roles import UserKind, UserRole
from domain.value_objects.search_mode import SearchMode
from domain.identity.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.database.models import ChunkModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from fakes import FakeUnitOfWorkFactory


class LocalSQLSession:
    def __init__(self, connection):
        self.connection = connection

    async def execute(self, stmt):
        return self.connection.execute(stmt)


@pytest.fixture
def repository():
    metadata = MetaData()
    chunks = Table(
        "chunks",
        metadata,
        *[
            Column(column.name, JSON() if isinstance(column.type, JSONB) else column.type)
            for column in ChunkModel.__table__.columns
        ],
    )
    Table("act_versions", metadata, Column("id", Integer), Column("act_id", Integer))
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    with engine.connect() as connection:
        rows = []
        for chunk_id in range(1, 8):
            rows.append(
                {
                    "id": chunk_id,
                    "document_id": 10 if chunk_id < 4 else 20,
                    "filename": "test.txt",
                    "chunk_index": chunk_id,
                    "content": "a needle appears here" if chunk_id != 7 else "no matching text",
                    "visibility": DocumentVisibility.INTERNAL_PUBLIC.value,
                    "doc_domain": DocDomain.GENERAL.value,
                    "owner_id": None,
                    "is_current": True,
                    "effective_from": None,
                    "effective_to": None,
                }
            )
        rows[3].update(visibility=DocumentVisibility.INTERNAL_PRIVATE.value, owner_id=99)
        rows[4].update(effective_from=date.today() + timedelta(days=1))
        rows[5].update(is_current=False)
        connection.execute(chunks.insert(), rows)
        yield SQLAlchemyChunkRepository(LocalSQLSession(connection))
    engine.dispose()


@pytest.fixture
def user():
    return UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(SearchMode))
async def test_pages_cover_all_accessible_matches_with_same_total(repository, user, mode):
    first, total = await repository.search_substring_page("needle", user, limit=2, offset=0, mode=mode.value)
    second, second_total = await repository.search_substring_page(
        "needle", user, limit=2, offset=2, mode=mode.value
    )
    assert [row.chunk_id for row in first] == [1, 2]
    assert [row.chunk_id for row in second] == [3]
    assert total == second_total == 3


@pytest.mark.asyncio
async def test_out_of_range_page_still_reports_scoped_total(repository, user):
    rows, total = await repository.search_substring_page("needle", user, limit=2, offset=100)
    assert rows == []
    assert total == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("document_id,total", [(10, 3), (20, 0)])
async def test_document_filter_applies_to_total_without_bypassing_acl(repository, user, document_id, total):
    rows, actual_total = await repository.search_substring_page("needle", user, document_id=document_id)
    assert actual_total == total
    assert len(rows) == total


@pytest.mark.asyncio
async def test_no_matches_and_short_queries_have_zero_total(repository, user):
    for query in ["missing", "  "]:
        assert await repository.search_substring_page(query, user) == ([], 0)


@pytest.mark.asyncio
async def test_application_service_preserves_requested_page_and_total(repository, user):
    factory = FakeUnitOfWorkFactory()
    factory._uow.chunks = SimpleNamespace(search_substring_page=repository.search_substring_page)
    service = SearchService(factory)
    rows, total = await service.exact_search("needle", user, limit=2, offset=2)
    assert [row.chunk_id for row in rows] == [3]
    assert total == 3
