"""Filename slots remain reserved throughout indexing and for completed documents."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex, CreateTable

from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.database.models import DocumentModel
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository


class LocalDocumentSession:
    """Execute the repository's real SQL in a synchronous in-memory database."""

    def __init__(self, session: Session):
        self.session = session

    async def execute(self, statement):
        return self.session.execute(statement)


@pytest.fixture
def document_session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(CreateTable(DocumentModel.__table__))
    with Session(engine) as session:
        yield session
    engine.dispose()


def add_document(session: Session, status: DocumentStatus, owner_id=None, group_id=None):
    doc = DocumentModel(
        filename="report.pdf",
        source_path="report.pdf",
        visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
        status=status.value,
        owner_id=owner_id,
        group_id=group_id,
    )
    session.add(doc)
    session.flush()
    return doc


@pytest.mark.asyncio
@pytest.mark.parametrize("status", list(DocumentStatus))
@pytest.mark.parametrize("owner_id,group_id", [(None, None), (1, None), (None, 2)])
async def test_find_active_slot_preserves_each_lifecycle_status(document_session, status, owner_id, group_id):
    doc = add_document(document_session, status, owner_id, group_id)
    repository = SQLAlchemyDocumentRepository(LocalDocumentSession(document_session))
    found = await repository.find_active_slot(owner_id, "report.pdf", group_id)
    assert found is not None
    assert found.id == doc.id
    assert await repository.find_active_slot(owner_id, "other.pdf", group_id) is None
    assert await repository.find_active_slot(99, "report.pdf", group_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", list(DocumentStatus))
@pytest.mark.parametrize("owner_id,group_id", [(None, None), (1, None), (None, 2), (1, 2)])
async def test_batch_slots_include_indexing_and_exclude_other_scopes(
    document_session, status, owner_id, group_id
):
    doc = add_document(document_session, status, owner_id=owner_id, group_id=group_id)
    add_document(document_session, status, owner_id=99, group_id=group_id)
    add_document(document_session, status, owner_id=owner_id, group_id=99)
    repository = SQLAlchemyDocumentRepository(LocalDocumentSession(document_session))
    found = await repository.find_active_slots_by_filenames(
        ["report.pdf", "missing.pdf"], owner_id=owner_id, group_id=group_id
    )
    assert [item.id for item in found] == [doc.id]
    assert await repository.find_active_slots_by_filenames([]) == []


@pytest.mark.parametrize("owner_id,group_id", [(None, None), (1, None), (None, 2)])
def test_postgresql_slot_index_rejects_upload_while_indexing(document_session, owner_id, group_id):
    # PostgreSQL's emitted COALESCE partial-index SQL also executes in SQLite.
    index = next(
        index for index in DocumentModel.__table__.indexes if index.name == "ux_documents_active_slot"
    )
    document_session.connection().exec_driver_sql(
        str(CreateIndex(index).compile(dialect=postgresql.dialect()))
    )
    add_document(document_session, DocumentStatus.INDEXING, owner_id, group_id)
    with pytest.raises(IntegrityError):
        add_document(document_session, DocumentStatus.PENDING, owner_id, group_id)
