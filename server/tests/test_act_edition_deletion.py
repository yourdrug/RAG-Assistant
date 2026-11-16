"""Deleting a document reconnects its act timeline: the law pack's deletion hook."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.domain_packs.belarus_law.act_versioning.document_hooks import ActEditionHooks
from domain.domain_packs.belarus_law.acts.edition_metadata import edition_metadata
from domain.domain_packs.belarus_law.acts.models import ActVersion
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.repositories.chunk import ChunkInsertOptions
from tests.fakes import require_present
from tests.law_pack import LawFakeUnitOfWork
from tests.test_document_commands_characterization import harness, seed


if TYPE_CHECKING:
    from types import SimpleNamespace

    from domain.knowledge.document import Document


def edition_harness() -> SimpleNamespace:
    return harness(LawFakeUnitOfWork(), ActEditionHooks(MagicMock(), MagicMock()))


async def seed_act_timeline(
    h: SimpleNamespace, deleted_index: int
) -> tuple[list[Document], list[ActVersion]]:
    docs = [await seed(h) for _ in range(3)]
    starts = [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)]
    versions = []

    for index, doc in enumerate(docs):
        version = await h.uow.act_versions.create(
            ActVersion(
                id=None,
                act_id=7,
                document_id=require_present(doc.id),
                effective_from=starts[index],
                effective_to=starts[index + 1] if index < 2 else None,
                is_current=index == 2,
            )
        )
        versions.append(version)
        await h.uow.chunks.bulk_insert(
            document_id=doc.id,
            filename=doc.filename,
            visibility="internal_private",
            chunks=[ChunkInsert(content="text", content_hash="hash")],
            options=ChunkInsertOptions(
                owner_id=1,
                group_id=None,
                doc_domain="legal",
                pack_metadata=edition_metadata(version),
            ),
        )

    original_delete = h.uow.documents.delete

    async def cascade_delete(document_id: int) -> None:
        await original_delete(document_id)
        # Match the PostgreSQL foreign key cascade before a survivor is promoted.
        h.uow.act_versions._versions.pop(versions[deleted_index].id)

    h.uow.documents.delete = cascade_delete
    h.uow.act_versions.list_by_act = AsyncMock(wraps=h.uow.act_versions.list_by_act)
    return docs, versions


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted_index", [1, 2])
async def test_delete_reconnects_act_timeline_and_search_projections(deleted_index: int) -> None:
    h = edition_harness()
    docs, _versions = await seed_act_timeline(h, deleted_index)

    await h.command.delete_document(docs[deleted_index].id, 1, "user")
    h.uow.act_versions.list_by_act.assert_awaited_once_with(7, for_update=True)
    survivors = await h.uow.act_versions.list_by_act(7)
    assert len(survivors) == 2
    assert survivors[-1].is_current is True
    assert survivors[-1].effective_to is None
    assert survivors[0].effective_to == survivors[1].effective_from
    changed = survivors[0] if deleted_index == 1 else survivors[-1]
    chunk = next(c for c in h.uow.chunks._chunks if c["document_id"] == changed.document_id)
    assert chunk["pack_metadata"] == edition_metadata(changed)
    entries = list(h.uow.vector_outbox._entries.values())
    assert entries[0].operation == "delete_by_document"
    assert entries[1].operation == "update_metadata"
    assert (entries[1].aggregate_type, entries[1].aggregate_id) == ("document", changed.document_id)
    h.storage.delete_file.assert_awaited_once_with("old-key")


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted_index", [1, 2])
async def test_failed_timeline_update_rolls_back_deletion(deleted_index: int) -> None:
    h = edition_harness()
    docs, _versions = await seed_act_timeline(h, deleted_index)
    h.uow.act_versions.update = AsyncMock(side_effect=RuntimeError("projection failure"))
    with pytest.raises(RuntimeError, match="projection failure"):
        await h.command.delete_document(docs[deleted_index].id, 1, "user")
    assert await h.uow.documents.get_by_id(docs[deleted_index].id) is not None
    assert len(await h.uow.act_versions.list_by_act(7)) == 3
    assert h.uow.vector_outbox._entries == {}
    h.storage.delete_file.assert_not_awaited()
