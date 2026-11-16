"""Manual chunks inherit edition state in both transactional projections."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from application.domain_packs.belarus_law.act_versioning.document_hooks import ActEditionHooks
from domain.domain_packs.belarus_law.acts.edition_metadata import edition_metadata
from domain.domain_packs.belarus_law.acts.models import ActVersion
from domain.domain_packs.belarus_law.acts.repositories import ActRepositories
from domain.identity.roles import UserRole
from domain.knowledge.index_sync import OutboxOperation
from infrastructure.domain_packs.belarus_law.persistence.act_versions import SQLAlchemyActVersionRepository
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from tests.test_chunk_service import _make_service, _make_uow


if TYPE_CHECKING:
    from application.chunks import ChunkMutationService


def edition_uow(version: ActVersion | None = None, error: Exception | None = None) -> AsyncMock:
    uow = _make_uow()
    act_versions = SimpleNamespace(get_by_document_id=AsyncMock(return_value=version, side_effect=error))
    uow.pack_repositories = ActRepositories(regulatory_acts=MagicMock(), act_versions=act_versions)
    return uow


def edition_service(uow: AsyncMock) -> ChunkMutationService:
    return _make_service(uow, pack_hooks=ActEditionHooks(MagicMock(), MagicMock()))


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import ClauseElement


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version",
    [
        pytest.param(ActVersion(9, 3, 1, date(2020, 1, 1), date(2024, 1, 1), False), id="archived"),
        pytest.param(ActVersion(9, 3, 1, date(2024, 1, 1)), id="current"),
        pytest.param(ActVersion(9, None, 1), id="undated-pending-linkage"),
        pytest.param(None, id="unversioned"),
    ],
)
async def test_add_chunk_inherits_edition_in_sql_model_and_outbox(version: ActVersion | None) -> None:
    models: list[Any] = []

    async def flush() -> None:
        models[0].id = 20

    session = SimpleNamespace(add=models.append, flush=AsyncMock(side_effect=flush))
    uow = edition_uow(version)
    # Exercise the composed repository and actual ORM construction, not just
    # the arguments sent by the service to a mocked insert_one.
    uow.chunks.insert_one = SQLAlchemyChunkRepository(cast("AsyncSession", session)).insert_one
    result = await edition_service(uow).add_chunk(
        1, "A manually added regulatory passage. " * 8, 100, UserRole.ADMIN, page=3, section="Rules"
    )

    uow.pack_repositories.act_versions.get_by_document_id.assert_awaited_once_with(1, for_update=True)
    expected = edition_metadata(version) if version else {}
    model = models[0]
    assert model.pack_metadata == expected
    assert model.manual is True
    assert model.context_metadata == {"page": 3, "section": "Rules"}
    entry = uow.vector_outbox.enqueue.call_args.args[0]
    assert entry.operation == OutboxOperation.UPSERT_CHUNKS
    point = entry.payload["points"][0]
    assert point["chunk_id"] == result.id == model.id
    metadata = point["metadata"]
    assert metadata["pack_metadata"] == expected
    assert metadata["page"] == 3
    assert metadata["section"] == "Rules"


@pytest.mark.asyncio
async def test_edition_read_failure_aborts_manual_chunk_creation() -> None:
    uow = edition_uow(error=OSError("edition read failed"))
    service = edition_service(uow)
    with pytest.raises(OSError, match="edition read failed"):
        await service.add_chunk(1, "A manual passage. " * 20, 100, UserRole.ADMIN)
    uow.chunks.insert_one.assert_not_awaited()
    uow.vector_outbox.enqueue.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("for_update", [False, True])
async def test_for_update_locks_selected_edition_as_well_as_document(for_update: bool) -> None:
    session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None))
    )
    await SQLAlchemyActVersionRepository(cast("AsyncSession", session)).get_by_document_id(
        1, for_update=for_update
    )
    statements = [
        str(cast("ClauseElement", call.args[0]).compile(dialect=postgresql.dialect()))  # type: ignore[no-untyped-call]  # SQLAlchemy exposes an untyped dialect constructor.
        for call in session.execute.await_args_list
    ]
    assert len(statements) == (2 if for_update else 1)
    assert ("FOR UPDATE" in statements[-1]) is for_update

    if for_update:
        assert "FOR UPDATE" in statements[0]
