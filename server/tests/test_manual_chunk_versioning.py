"""Manual chunks inherit edition state in both transactional projections."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from domain.entities.act_version import ActVersion
from domain.entities.vector_outbox_entry import OutboxOperation
from domain.value_objects.roles import UserRole
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.repositories.misc.sqlalchemy_act_version_repository import SQLAlchemyActVersionRepository
from test_chunk_service import _make_service, _make_uow


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
async def test_add_chunk_inherits_edition_in_sql_model_and_outbox(version):
    models = []

    async def flush():
        models[0].id = 20

    session = SimpleNamespace(add=models.append, flush=AsyncMock(side_effect=flush))
    uow = _make_uow()
    uow.act_versions.get_by_document_id.return_value = version
    # Exercise the composed repository and actual ORM construction, not just
    # the arguments sent by the service to a mocked insert_one.
    uow.chunks.insert_one = SQLAlchemyChunkRepository(session).insert_one
    result = await _make_service(uow).add_chunk(
        1, "A manually added regulatory passage. " * 8, 100, UserRole.ADMIN.value, page=3, section="Rules"
    )

    uow.act_versions.get_by_document_id.assert_awaited_once_with(1, for_update=True)
    expected = {
        "act_version_id": version.id if version else None,
        "effective_from": version.effective_from if version else None,
        "effective_to": version.effective_to if version else None,
        "is_current": version.is_current if version else True,
    }
    model = models[0]
    assert {key: getattr(model, key) for key in expected} == expected
    assert model.manual is True
    assert model.context_metadata == {"page": 3, "section": "Rules"}
    entry = uow.vector_outbox.enqueue.call_args.args[0]
    assert entry.operation == OutboxOperation.UPSERT_CHUNKS
    point = entry.payload["points"][0]
    assert point["chunk_id"] == result.id == model.id
    metadata = point["metadata"]
    assert {key: metadata[key] for key in expected} == {
        key: value.isoformat() if isinstance(value, date) else value for key, value in expected.items()
    }
    assert metadata["act_id"] == (version.act_id if version else None)
    assert metadata["page"] == 3 and metadata["section"] == "Rules"


@pytest.mark.asyncio
async def test_edition_read_failure_aborts_manual_chunk_creation():
    uow = _make_uow()
    uow.act_versions.get_by_document_id.side_effect = OSError("edition read failed")
    service = _make_service(uow)
    with pytest.raises(OSError, match="edition read failed"):
        await service.add_chunk(1, "A manual passage. " * 20, 100, UserRole.ADMIN.value)
    uow.chunks.insert_one.assert_not_awaited()
    uow.vector_outbox.enqueue.assert_not_awaited()
    service._mutation._bm25_index.add.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("for_update", [False, True])
async def test_for_update_locks_selected_edition_as_well_as_document(for_update):
    session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None))
    )
    await SQLAlchemyActVersionRepository(session).get_by_document_id(1, for_update=for_update)
    statements = [
        str(call.args[0].compile(dialect=postgresql.dialect())) for call in session.execute.await_args_list
    ]
    assert len(statements) == (2 if for_update else 1)
    assert ("FOR UPDATE" in statements[-1]) is for_update
    if for_update:
        assert "FOR UPDATE" in statements[0]
