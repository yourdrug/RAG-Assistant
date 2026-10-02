"""Date/versioning regressions through the real services and local Qdrant engine."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.schema import Document as SearchDocument
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams

from application.dto.versioning_dto import VersioningResult
from application.services.act_versioning_service import ActVersioningService
from application.services.document_persistence import persist_document_result
from domain.domain_profile.date_parsing import parse_date_guess
from domain.domain_profile.profiles.decree import DecreeDomainProfile
from domain.domain_profile.profiles.legal import LegalDomainProfile
from domain.domain_profile.protocol import ReferenceMatch
from domain.entities.document import Document
from domain.entities.raw_document import RawDocument
from domain.repositories.chunk_repository import ChunkSearchResult
from domain.exceptions import ValidationError
from domain.value_objects.stream_events import MetaEvent
from fakes import FakeUnitOfWorkFactory
from infrastructure.ml.rag.rag_postprocess import resolve_temporal_conflicts
from infrastructure.ml.rag.helpers import apply_exact_search
from infrastructure.repositories.vector.acl import with_temporal_filter
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher
from presentation.api.routes.chat import _format_sse_event
from presentation.api.schemas.chat import ChatRequest
from test_chunk_search_acl import _capture, _eval, _where


class Settings:
    def get(self, key, domain_key=""):
        return "0.85" if key == "effective_date_auto_trust_threshold" else "0"


def system():
    factory = FakeUnitOfWorkFactory()
    cache = SimpleNamespace(invalidate_by_document_ids=AsyncMock(return_value=1))
    service = ActVersioningService(factory, Settings(), cache)
    return factory, service, cache


def put_document(factory, document_id, visibility="internal_public", owner_id=None, group_id=None):
    factory._uow.documents._documents[document_id] = Document(
        id=document_id,
        filename=f"{document_id}.rtf",
        visibility=visibility,
        owner_id=owner_id,
        group_id=group_id,
    )


async def upload(service, document_id, effective_date, signing_year=2020):
    return await service.handle_versioned_upload(
        DecreeDomainProfile(settings=Settings()),
        document_id,
        [ReferenceMatch("decree_number", "5"), ReferenceMatch("decree_date", f"1 января {signing_year}")],
        effective_date,
        0.9,
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-01-06", date(2026, 1, 6)),
        ("в силу с 01.06.2026.", date(2026, 6, 1)),
        ("1/6/2026", date(2026, 6, 1)),
        ("1 января 2026 г.", date(2026, 1, 1)),
        ("01", None),
        ("2026", None),
        ("31.02.2026", None),
    ],
)
def test_complete_dates_only(text, expected):
    assert parse_date_guess(text) == expected


@pytest.mark.parametrize("profile", [DecreeDomainProfile, LegalDomainProfile])
@pytest.mark.parametrize(
    "text, expected",
    [
        ("вступает в силу с 01.06.2026.", date(2026, 6, 1)),
        ("вступает в силу с 2026-01-06.", date(2026, 1, 6)),
    ],
)
def test_numeric_effective_dates_are_not_truncated(profile, text, expected):
    assert profile(settings=Settings()).extract_effective_date(text).effective_from == expected


def test_calendar_date_has_priority_and_question_date_is_reported_in_sse():
    question = "Указ от 01.01.2020: правила по состоянию на 2024-06-01?"
    assert ChatRequest(question=question).as_of_date == date(2024, 6, 1)
    request = ChatRequest(question=question, as_of_date="2025-01-01")
    assert request.as_of_date == date(2025, 1, 1)
    event = _format_sse_event(MetaEvent(7, [], as_of_date=request.as_of_date), "request")
    assert '"as_of_date": "2025-01-01"' in event
    with pytest.raises(ValueError, match="Invalid date"):
        ChatRequest(question="Правила по состоянию на 31.02.2024?")


def test_legal_act_identity_includes_number_and_signing_date():
    refs = LegalDomainProfile(settings=Settings()).extract_references(
        "Федеральный закон от 1 января 2020 г. N 5-ФЗ\nСтатья 1. Правила."
    )
    assert ReferenceMatch("act_number", "5-ФЗ") in refs
    assert parse_date_guess(next(ref.value for ref in refs if ref.kind == "act_date")) == date(2020, 1, 1)


@pytest.mark.asyncio
async def test_backfilled_and_future_versions_keep_chronological_intervals_and_sql_flags():
    factory, service, cache = system()
    for document_id in range(1, 5):
        put_document(factory, document_id)
    old = await upload(service, 1, date(2020, 1, 1))
    factory._uow.chunks._chunks.append({"act_version_id": old.id, "is_current": True})
    newest = await upload(service, 2, date(2024, 1, 1))
    factory._uow.chunks._chunks.append({"act_version_id": newest.id, "is_current": True})
    backfilled = await upload(service, 3, date(2022, 1, 1))
    future = await upload(service, 4, date(2099, 1, 1))
    assert old.effective_to == date(2022, 1, 1)
    assert backfilled.effective_to == date(2024, 1, 1)
    assert newest.effective_to == date(2099, 1, 1)
    assert newest.is_current and not backfilled.is_current and not future.is_current
    assert factory._uow.chunks._chunks[0]["is_current"] is False
    assert set(cache.invalidate_by_document_ids.call_args.args[0]) == {1, 2, 3, 4}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "visibility,owner,group",
    [
        ("internal_private", 9, None),
        ("client_private", 9, None),
        ("internal_group", None, 4),
    ],
)
async def test_same_number_private_upload_cannot_change_public_version(visibility, owner, group):
    factory, service, _ = system()
    put_document(factory, 1)
    put_document(factory, 2, visibility, owner, group)
    public = await upload(service, 1, date(2020, 1, 1))
    other = await upload(service, 2, date(2024, 1, 1))
    assert public.is_current and public.effective_to is None
    assert other.act_id != public.act_id
    with pytest.raises(ValidationError, match="visibility scope"):
        await service.update_version(other.id, act_id=public.act_id)
    assert (await factory._uow.act_versions.get_by_id(public.id)).is_current


@pytest.mark.asyncio
async def test_different_years_and_owners_have_independent_act_identities():
    factory, service, _ = system()
    for document_id, owner in ((1, 10), (2, 10), (3, 11)):
        put_document(factory, document_id, "internal_private", owner)
    one = await upload(service, 1, date(2020, 1, 1), 2020)
    two = await upload(service, 2, date(2021, 1, 1), 2021)
    three = await upload(service, 3, date(2020, 1, 1), 2020)
    assert len({one.act_id, two.act_id, three.act_id}) == 3
    assert all(v.is_current for v in (one, two, three))


@pytest.mark.asyncio
async def test_reindexing_does_not_duplicate_version_or_overwrite_manual_dates():
    factory, service, _ = system()
    put_document(factory, 1)
    initial = await upload(service, 1, date(2020, 1, 1))
    await service.update_version(initial.id, effective_from=date(2021, 1, 1))
    repeated = await upload(service, 1, date(2024, 1, 1))
    assert repeated.id == initial.id
    assert repeated.effective_from == date(2021, 1, 1)
    assert len(factory._uow.act_versions._versions) == 1


@pytest.mark.asyncio
async def test_explicit_null_clears_dates_and_unlink_invalidates_document_cache():
    factory, service, cache = system()
    put_document(factory, 1)
    version = await upload(service, 1, date(2020, 1, 1))
    await service.update_version(version.id, effective_to=date(2025, 1, 1))
    await service.update_version(
        version.id,
        effective_from_provided=True,
        effective_to_provided=True,
        act_id_provided=True,
    )
    stored = await factory._uow.act_versions.get_by_id(version.id)
    assert stored.effective_from is None and stored.effective_to is None and stored.act_id is None
    entry = list(factory._uow.vector_outbox._entries.values())[-1]
    assert entry.payload["effective_from"] is None and entry.payload["effective_to"] is None
    cache.invalidate_by_document_ids.assert_any_await([1])


@pytest.mark.asyncio
async def test_invalid_interval_is_rejected():
    factory, service, _ = system()
    put_document(factory, 1)
    version = await upload(service, 1, date(2020, 1, 1))
    with pytest.raises(ValidationError, match="later"):
        await service.update_version(version.id, effective_to=date(2019, 1, 1))


async def persist(factory, service, document_id, text):
    profile = DecreeDomainProfile(settings=Settings())
    plan = await service.prepare_document_versioning(profile, text)
    return await persist_document_result(
        factory,
        document_id=document_id,
        original_filename=f"{document_id}.rtf",
        raw_chunks=[RawDocument(page_content=text, metadata={})],
        visibility="internal_public",
        owner_id=None,
        group_id=None,
        doc_domain="decree",
        replace_id=None,
        warning_message=None,
        quality=None,
        storage_deletes=[],
        domain_registry=None,
        versioning=VersioningResult(None, None, None, None, None),
        versioning_plan=plan,
        versioning_profile=profile,
        act_versioning_service=service,
    )


@pytest.mark.asyncio
async def test_persistence_uses_trusted_dates_and_propagates_backfilled_interval():
    factory, service, _ = system()
    for document_id in (1, 2, 3):
        put_document(factory, document_id)
    await upload(service, 1, date(2024, 1, 1))
    result = await persist(factory, service, 2, "УКАЗ №5 от 1 января 2020 г.\nВступает в силу с 01.01.2022.")
    assert result.effective_to == date(2024, 1, 1) and result.is_current is False
    chunk = factory._uow.chunks._chunks[-1]
    assert chunk["effective_to"] == date(2024, 1, 1) and chunk["is_current"] is False
    result = await persist(factory, service, 3, "УКАЗ №9 от 1 января 2020 г.\nПОСТАНОВЛЯЮ:\n1. Пункт.")
    version = await factory._uow.act_versions.get_by_id(result.act_version_id)
    assert version.effective_from is None
    assert factory._uow.chunks._chunks[-1]["effective_from"] is None


def test_real_qdrant_engine_handles_missing_null_and_half_open_date_bounds():
    client = QdrantClient(":memory:")
    try:
        client.create_collection("versions", vectors_config=VectorParams(size=1, distance=Distance.COSINE))
        metadata = [
            {"is_current": True},
            {"is_current": True, "effective_from": None, "effective_to": None},
            {"is_current": False},
            {"is_current": False, "effective_from": "2020-01-01", "effective_to": "2024-01-01"},
            {"is_current": True, "effective_from": "2024-01-01", "effective_to": "2099-01-01"},
            {"is_current": False, "effective_from": "2099-01-01"},
        ]
        client.upsert(
            "versions",
            points=[
                PointStruct(id=i, vector=[1.0], payload={"metadata": meta})
                for i, meta in enumerate(metadata, 1)
            ],
        )
        for as_of_date, expected in [
            (date(2023, 12, 31), {1, 2, 4}),
            (date(2024, 1, 1), {1, 2, 5}),
            (date(2099, 1, 1), {1, 2, 6}),
            (None, {1, 2, 5}),
        ]:
            points, _ = client.scroll("versions", scroll_filter=with_temporal_filter(Filter(), as_of_date))
            assert {point.id for point in points} == expected
    finally:
        client.close()


def test_all_chunks_of_selected_edition_survive_with_mixed_date_types():
    old = SearchDocument(
        page_content="old", metadata={"act_id": 1, "act_version_id": 1, "effective_from": date(2020, 1, 1)}
    )
    newer = [
        SearchDocument(
            page_content=f"new {i}",
            metadata={"act_id": 1, "act_version_id": 2, "effective_from": "2024-01-01"},
        )
        for i in range(3)
    ]
    result = resolve_temporal_conflicts([(old, 0.99)] + [(doc, 0.8) for doc in newer])
    assert [doc.page_content for doc, _ in result] == [doc.page_content for doc in newer]


@pytest.mark.asyncio
async def test_delayed_outbox_metadata_uses_latest_state_and_invalidates_cache():
    factory, service, cache = system()
    put_document(factory, 1)
    version = await upload(service, 1, date(2020, 1, 1))
    await service.update_version(version.id, effective_from=date(2024, 1, 1))
    vector_store = SimpleNamespace(update_metadata_by_act_version=AsyncMock())
    dispatcher = OutboxDispatcher(factory, vector_store, cache)
    await dispatcher._apply_version_metadata(version.id, {"effective_from": "2020-01-01"})
    metadata = vector_store.update_metadata_by_act_version.call_args.args[1]
    assert metadata["effective_from"] == "2024-01-01"
    cache.invalidate_by_document_ids.assert_any_await([1])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["exact", "icontains"])
async def test_sql_search_enforces_dates_without_weakening_acl(mode):
    stmt = await _capture("rules", {"id": 1, "kind": "internal"}, [], mode, None, as_of_date=date(2024, 1, 1))
    tree = _where(stmt)
    row = {
        "content": " rules ",
        "visibility": "internal_public",
        "is_current": False,
        "effective_from": date(2020, 1, 1),
        "effective_to": date(2024, 1, 1),
    }
    assert not _eval(tree, row, "rules")
    row.update(effective_from=date(2024, 1, 1), effective_to=None)
    assert _eval(tree, row, "rules")
    row.update(visibility="internal_private", owner_id=9)
    assert not _eval(tree, row, "rules")
    sql = str(stmt.compile())
    assert "LEFT OUTER JOIN act_versions" in sql
    assert "act_versions.act_id" in sql


@pytest.mark.asyncio
async def test_exact_candidates_include_temporal_metadata_and_request_date():
    result = ChunkSearchResult(
        chunk_id=1,
        document_id=2,
        filename="rules.rtf",
        content="rules",
        chunk_index=0,
        act_id=4,
        act_version_id=6,
        effective_from=date(2024, 1, 1),
        is_current=False,
    )
    search = SimpleNamespace(search_substring=AsyncMock(return_value=[result]))
    candidates = []
    await apply_exact_search("rules", candidates, {}, SimpleNamespace(as_of_date=date(2024, 1, 1)), search)
    assert search.search_substring.call_args.kwargs["as_of_date"] == date(2024, 1, 1)
    assert candidates[0].metadata["act_id"] == 4
    assert candidates[0].metadata["act_version_id"] == 6
    assert candidates[0].metadata["effective_from"] == "2024-01-01"


@pytest.mark.asyncio
async def test_cache_key_changes_each_day_for_current_state_but_not_for_explicit_date(monkeypatch):
    from infrastructure.ml.rag import rag_steps
    from dataclasses import replace
    from test_rag_pipeline import _make_rag

    class Clock(date):
        current = date(2024, 1, 1)

        @classmethod
        def today(cls):
            return cls.current

    monkeypatch.setattr(rag_steps, "date", Clock)
    monkeypatch.setattr(rag_steps, "check_cache", AsyncMock(return_value=None))
    monkeypatch.setattr(rag_steps, "get_corpus_revision", AsyncMock(return_value="0"))
    ctx = SimpleNamespace(
        user_kind="internal",
        user_id=1,
        user_group_ids=[],
        user_role="user",
        curator_scope=None,
        as_of_date=None,
        depth=None,
        summary=None,
    )
    rag = _make_rag()
    rag = replace(rag, features=replace(rag.features, cache_enabled=True))
    state = SimpleNamespace(ctx=ctx, query_for_search="rules", question="rules", history_messages=[], rag=rag)
    await rag_steps.step_check_cache(state)
    first_hash = state.q_hash
    Clock.current = date(2024, 1, 2)
    await rag_steps.step_check_cache(state)
    assert state.q_hash != first_hash
    ctx.as_of_date = date(2024, 1, 1)
    await rag_steps.step_check_cache(state)
    assert state.q_hash == first_hash


@pytest.mark.asyncio
async def test_admin_all_versions_pagination_includes_reviewed_entries_and_referenced_acts():
    from presentation.api.routes.admin_act_versions import list_act_versions

    factory, service, _ = system()
    for document_id in (1, 2):
        put_document(factory, document_id)
        version = await upload(service, document_id, date(2020 + document_id, 1, 1))
        await service.update_version(version.id, verified_by=10)
    response = await list_act_versions(
        review="all", limit=1, offset=0, admin=SimpleNamespace(id=10), service=service
    )
    assert response.total == 2 and len(response.versions) == 1 and response.next_offset == 1
    assert response.versions[0].date_source == "manual"
    assert response.versions[0].act_id in {act.id for act in response.acts}
    response = await list_act_versions(
        review="all", limit=1, offset=1, admin=SimpleNamespace(id=10), service=service
    )
    assert response.next_offset is None


@pytest.mark.asyncio
async def test_same_effective_date_supersedes_old_edition():
    factory, service, _ = system()
    for document_id in (1, 2):
        put_document(factory, document_id)
    original = await upload(service, 1, date(2024, 1, 1))
    corrected = await upload(service, 2, date(2024, 1, 1))
    assert original.effective_to == original.effective_from
    assert not original.is_current and corrected.is_current


@pytest.mark.asyncio
async def test_backfill_after_future_edition_takes_effect_cannot_become_latest(monkeypatch):
    from application.services import act_versioning_service

    class Clock(date):
        current = date(2026, 1, 1)

        @classmethod
        def today(cls):
            return cls.current

    monkeypatch.setattr(act_versioning_service, "date", Clock)
    factory, service, _ = system()
    for document_id in (1, 2, 3):
        put_document(factory, document_id)
    await upload(service, 1, date(2024, 1, 1))
    await upload(service, 2, date(2099, 1, 1))
    Clock.current = date(2100, 1, 1)
    backfilled = await upload(service, 3, date(2098, 1, 1))
    assert not backfilled.is_current and backfilled.effective_to == date(2099, 1, 1)
