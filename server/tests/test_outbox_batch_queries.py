"""Execute batch inspector SQL and verify constant query counts."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from domain.entities.vector_outbox_entry import OutboxStatus
from infrastructure.database.models import VectorStoreOutboxModel
from infrastructure.repositories.config.sqlalchemy_vector_outbox_repository import (
    SQLAlchemyVectorOutboxRepository,
)


if TYPE_CHECKING:
    from sqlalchemy import Table


@pytest.mark.asyncio
async def test_batch_outbox_counts_and_failure_details_execute_two_queries() -> None:
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(CreateTable(cast("Table", VectorStoreOutboxModel.__table__)))
    with Session(engine) as session:
        for document_id in [1, 2, 99]:
            for status in OutboxStatus:
                session.add(
                    VectorStoreOutboxModel(
                        operation="upsert_chunks",
                        aggregate_id=document_id,
                        payload={},
                        status=status.value,
                        attempts=2,
                        max_attempts=8,
                        last_error=status.value,
                        next_attempt_at=datetime.now(tz=UTC),
                    )
                )

        session.flush()
        db = AsyncMock()
        db.execute.side_effect = session.execute
        repository = SQLAlchemyVectorOutboxRepository(db)
        assert await repository.count_by_documents([1, 2, 3]) == {
            1: {"pending": 3, "failed": 2},
            2: {"pending": 3, "failed": 2},
            3: {"pending": 0, "failed": 0},
        }
        details = await repository.get_failed_details_by_documents([1, 2, 3])
        assert set(details) == {1, 2}

        for entries in details.values():
            assert {entry["last_error"] for entry in entries} == {"failed", "dead_letter"}
            assert all(entry["attempts"] == 2 and entry["max_attempts"] == 8 for entry in entries)

        assert db.execute.await_count == 2
        assert await repository.count_by_documents([]) == {}
        assert await repository.get_failed_details_by_documents([]) == {}
        assert db.execute.await_count == 2
        assert await repository.count_by_document(1) == {"pending": 3, "failed": 2}
        assert await repository.get_failed_details(1) == details[1]
        assert await repository.get_failed_details(3) == []
    engine.dispose()
