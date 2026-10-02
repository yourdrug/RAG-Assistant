"""Interpretation context survives persistence, exact search, neighbors and edits."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.schema import Document

from application.services.document_pipeline import process_chunks
from domain.entities.raw_document import RawDocument
from domain.repositories.chunk_repository import ChunkSearchResult
from domain.value_objects.chunk_context import extract_chunk_context
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata
from infrastructure.ml.rag.helpers import apply_exact_search
from infrastructure.ml.rag.rag_formatting import format_docs
from infrastructure.ml.rag.rag_postprocess import enrich_with_neighbors
from infrastructure.repositories.chunk.chunk_mapping import to_chunk_search_result
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository


def make_context():
    return {
        "parent_units": [{"content": "Only applications submitted after January 2025 are eligible."}],
        "page": 4,
        "pages": [4],
        "page_start": 4,
        "page_end": 4,
        "unit_kind": "subpoint_num",
        "point_number": "21",
        "subpoint_num_number": "21.2",
        "section": "point 21 > subpoint_num 21.2",
        "doc_date": "2026-01-01",
        "domain_metadata": {"point": "21"},
    }


@pytest.mark.asyncio
async def test_ingestion_context_reaches_sql_search_and_neighbor_prompts():
    models = []

    async def flush():
        for number, model in enumerate(models, 1):
            model.id = number

    session = SimpleNamespace(
        execute=AsyncMock(), add_all=lambda rows: models.extend(rows), flush=AsyncMock(side_effect=flush)
    )
    repository = SQLAlchemyChunkRepository(session)
    uow = SimpleNamespace(
        chunks=repository,
        vector_outbox=SimpleNamespace(enqueue=AsyncMock()),
        documents=SimpleNamespace(update_status=AsyncMock()),
    )
    context = make_context()
    chunk = RawDocument("21.2. Submit the passport.", {**context, "document_id": 7})
    await process_chunks(
        None,
        7,
        "rules.pdf",
        [chunk],
        DocumentVisibility.INTERNAL_PRIVATE.value,
        12,
        None,
        DocDomain.DECREE.value,
        _existing_uow=uow,
    )
    assert models[0].context_metadata == context
    assert "document_id" not in models[0].context_metadata
    assert models[0].visibility == DocumentVisibility.INTERNAL_PRIVATE.value
    assert models[0].owner_id == 12
    assert (
        uow.vector_outbox.enqueue.call_args.args[0].payload["points"][0]["metadata"]["parent_units"]
        == (context["parent_units"])
    )

    # Return columns from the actual SQL projection, rather than hand-ordered DTO fields.
    async def execute(stmt):
        row = tuple(
            getattr(models[0], column.key) if column.table.name == "chunks" else None
            for column in stmt.selected_columns
        )
        return SimpleNamespace(all=lambda: [row])

    session.execute = AsyncMock(side_effect=execute)
    user = UserContext(user_id=12, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
    results = await repository.search_substring("passport", user)
    assert results[0].context_metadata == context
    candidates = []
    search = SimpleNamespace(search_substring=AsyncMock(return_value=results))
    await apply_exact_search("passport", candidates, user, SimpleNamespace(as_of_date=None), search)
    assert len(candidates) == 1
    assert context["parent_units"][0]["content"] in format_docs(candidates)
    assert candidates[0].metadata["page"] == 4
    assert candidates[0].metadata["subpoint_num_number"] == "21.2"
    search.search_substring.assert_awaited_once()
    assert search.search_substring.call_args.kwargs["user"] is user

    neighbor = to_chunk_search_result(models[0])
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=[neighbor]))
    seed = Document(page_content="A separate related passage.", metadata={"document_id": 7, "chunk_index": 2})
    enriched = await enrich_with_neighbors([(seed, 0.9)], True, search, user=user)
    assert len(enriched) == 2
    assert context["parent_units"][0]["content"] in format_docs([enriched[1]])
    assert enriched[1][0].metadata["pages"] == [4]
    assert search.get_neighbors.call_args.kwargs["user"] is user


def test_stale_context_cannot_override_acl_identity_or_version_state():
    poisoned = {
        **make_context(),
        "source": "other.pdf",
        "document_id": 999,
        "visibility": DocumentVisibility.INTERNAL_PUBLIC.value,
        "owner_id": 99,
        "is_current": True,
        "effective_from": "1990-01-01",
        "act_version_id": 999,
    }
    result = ChunkSearchResult(
        chunk_id=1,
        document_id=7,
        filename="rules.pdf",
        content="Submit the passport.",
        chunk_index=3,
        visibility=DocumentVisibility.INTERNAL_PRIVATE.value,
        owner_id=12,
        act_version_id=9,
        effective_from=date(2025, 1, 1),
        is_current=False,
        context_metadata=poisoned,
    )
    metadata = chunk_result_metadata(result)
    assert metadata["visibility"] == DocumentVisibility.INTERNAL_PRIVATE.value
    assert metadata["owner_id"] == 12
    assert metadata["document_id"] == 7 and metadata["source"] == "rules.pdf"
    assert metadata["act_version_id"] == 9 and metadata["effective_from"] == "2025-01-01"
    assert metadata["is_current"] is False
    assert metadata["parent_units"] == poisoned["parent_units"]
    poisoned["parent_units"][0]["content"] = "Modified later"
    assert metadata["parent_units"][0]["content"] != "Modified later"


@pytest.mark.asyncio
async def test_missing_context_entries_fail_before_replacing_chunks():
    session = SimpleNamespace(execute=AsyncMock())
    with pytest.raises(ValueError, match="one entry per chunk"):
        await SQLAlchemyChunkRepository(session).bulk_insert(
            7,
            "rules.pdf",
            DocumentVisibility.INTERNAL_PRIVATE.value,
            ["first", "second"],
            context_metadata=[make_context()],
        )
    session.execute.assert_not_awaited()


def test_context_selection_excludes_transient_parser_offsets():
    context = make_context()
    selected = extract_chunk_context({**context, "_page_spans": [(0, 4, 1)], "_table_bbox": (1, 2, 3, 4)})
    assert selected == context


@pytest.mark.asyncio
async def test_edit_keeps_scope_and_current_acl_in_vector_outbox():
    from test_chunk_service import _make_chunk, _make_doc, _make_service, _make_uow

    context = make_context()
    chunk = _make_chunk(
        context_metadata=context,
        section=context["section"],
        act_version_id=9,
        effective_from=date(2025, 1, 1),
    )
    doc = _make_doc(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=12)
    uow = _make_uow(doc=doc, chunk=chunk)
    await _make_service(uow).edit_chunk(
        1, 10, "Updated passport submission rule. " * 8, 12, UserRole.USER.value
    )
    metadata = uow.vector_outbox.enqueue.call_args.args[0].payload["points"][0]["metadata"]
    assert metadata["parent_units"] == context["parent_units"]
    assert metadata["page"] == 4
    assert metadata["visibility"] == DocumentVisibility.INTERNAL_PRIVATE.value
    assert metadata["owner_id"] == 12
    assert metadata["act_version_id"] == 9 and metadata["effective_from"] == "2025-01-01"


@pytest.mark.asyncio
async def test_manual_chunk_persists_page_and_section_context():
    from test_chunk_service import _make_service, _make_uow

    uow = _make_uow()
    await _make_service(uow).add_chunk(
        1,
        "A complete manually added passage. " * 8,
        1,
        UserRole.ADMIN.value,
        page=5,
        section="Manual additions",
    )
    assert uow.chunks.insert_one.call_args.kwargs["context_metadata"] == {
        "page": 5,
        "section": "Manual additions",
    }
