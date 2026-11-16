"""Date/versioning regressions through the real services and local Qdrant engine."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

from langchain.schema import Document as SearchDocument
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams
from sqlalchemy.dialects import postgresql

from application.documents.dto import PersistDocumentCommand
from application.documents.persistence import persist_document_result
from application.domain_packs.belarus_law.act_versioning import (
    ActVersioningService,
    service as act_versioning_service,
)
from application.domain_packs.belarus_law.act_versioning.document_hooks import ActEditionHooks
from application.ingestion import DocumentSyncService
from application.rag.stream_events import MetaEvent
from domain.domain_packs.belarus_law.acts.edition_metadata import edition_metadata
from domain.identity.roles import UserKind
from domain.identity.user_context import UserContext
from domain.ingestion.profiles.types import ReferenceMatch
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility
from domain.knowledge.index_sync import OutboxOperation
from domain.shared.dates import parse_date_guess
from domain.shared.exceptions import ValidationError
from infrastructure.domain_packs.belarus_law.domain.decree import DecreeDomainProfile
from infrastructure.domain_packs.belarus_law.domain.legal import LegalDomainProfile
from infrastructure.domain_packs.belarus_law.evidence.editions import (
    edition_selection,
    resolve_edition_conflicts,
)
from infrastructure.ml.rag import rag_steps
from infrastructure.ml.rag.retrieval_pipeline import apply_exact_search
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher
from infrastructure.repositories.vector.pack_selection import pack_selection_condition
from presentation.api.routes.chat import format_sse_event
from presentation.api.schemas.chat import ChatRequest
from presentation.domain_packs.belarus_law.act_versions_routes import list_act_versions
from tests.fakes import require_present
from tests.law_pack import LawFakeUnitOfWorkFactory
from tests.rag_stage_fixtures import check_cache
from tests.test_chunk_search_acl import capture
from tests.test_rag_pipeline import make_rag


if TYPE_CHECKING:
    from domain.domain_packs.belarus_law.acts.models import ActVersion
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.knowledge.index_sync import VectorOutboxEntry
    from domain.rag.context import RagContext
    from presentation.api.schemas.auth import CurrentUser


class Settings:
    def get(self, key: str, domain_key: str = "") -> str:
        _ = domain_key
        return "0.85" if key == "effective_date_auto_trust_threshold" else "0"


def system() -> tuple[LawFakeUnitOfWorkFactory, ActVersioningService, SimpleNamespace]:
    factory = LawFakeUnitOfWorkFactory()
    cache = SimpleNamespace(invalidate_by_document_ids=AsyncMock(return_value=1))
    service = ActVersioningService(factory, Settings(), cache)
    return factory, service, cache


def put_document(
    factory: LawFakeUnitOfWorkFactory,
    document_id: int,
    visibility: str = "internal_public",
    owner_id: int | None = None,
    group_id: int | None = None,
) -> None:
    factory._uow.fake_documents._documents[document_id] = Document(
        id=document_id,
        filename=f"{document_id}.rtf",
        visibility=DocumentVisibility(visibility),
        owner_id=owner_id,
        group_id=group_id,
    )


async def upload(
    service: ActVersioningService, document_id: int, effective_date: Any, signing_year: int = 2020
) -> ActVersion:
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
def test_complete_dates_only(text: str, expected: Any) -> None:
    assert parse_date_guess(text) == expected


@pytest.mark.parametrize("profile", [DecreeDomainProfile, LegalDomainProfile])
@pytest.mark.parametrize(
    "text, expected",
    [
        ("вступает в силу с 01.06.2026.", date(2026, 6, 1)),
        ("вступает в силу с 2026-01-06.", date(2026, 1, 6)),
    ],
)
def test_numeric_effective_dates_are_not_truncated(profile: Any, text: str, expected: Any) -> None:
    assert profile(settings=Settings()).extract_effective_date(text).effective_from == expected


def test_calendar_date_has_priority_and_question_date_is_reported_in_sse() -> None:
    question = "Указ от 01.01.2020: правила по состоянию на 2024-06-01?"
    assert ChatRequest(question=question).as_of_date == date(2024, 6, 1)
    request = ChatRequest.model_validate({"question": question, "as_of_date": "2025-01-01"})
    assert request.as_of_date == date(2025, 1, 1)
    event = format_sse_event(MetaEvent(7, [], as_of_date=request.as_of_date), "request")
    assert '"as_of_date": "2025-01-01"' in require_present(event)
    with pytest.raises(ValueError, match="Invalid date"):
        ChatRequest(question="Правила по состоянию на 31.02.2024?")


def test_legal_act_identity_includes_number_and_signing_date() -> None:
    refs = LegalDomainProfile(settings=Settings()).extract_references(
        "Федеральный закон от 1 января 2020 г. N 5-ФЗ\nСтатья 1. Правила."
    )
    assert ReferenceMatch("act_number", "5-ФЗ") in refs
    assert parse_date_guess(next(ref.value for ref in refs if ref.kind == "act_date")) == date(2020, 1, 1)


@pytest.mark.asyncio
async def test_backfilled_and_future_versions_keep_chronological_intervals_and_sql_flags() -> None:
    factory, service, cache = system()

    for document_id in range(1, 5):
        put_document(factory, document_id)

    old = await upload(service, 1, date(2020, 1, 1))
    factory._uow.fake_chunks._chunks.append({"document_id": 1, "pack_metadata": edition_metadata(old)})
    newest = await upload(service, 2, date(2024, 1, 1))
    factory._uow.fake_chunks._chunks.append({"document_id": 2, "pack_metadata": edition_metadata(newest)})
    backfilled = await upload(service, 3, date(2022, 1, 1))
    future = await upload(service, 4, date(2099, 1, 1))
    old = require_present(await factory._uow.fake_act_versions.get_by_id(require_present(old.id)))
    newest = require_present(await factory._uow.fake_act_versions.get_by_id(require_present(newest.id)))
    backfilled = require_present(
        await factory._uow.fake_act_versions.get_by_id(require_present(backfilled.id))
    )
    assert old.effective_to == date(2022, 1, 1)
    assert backfilled.effective_to == date(2024, 1, 1)
    assert newest.effective_to == date(2099, 1, 1)
    assert newest.is_current
    assert not backfilled.is_current
    assert not future.is_current
    assert factory._uow.fake_chunks._chunks[0]["pack_metadata"]["is_current"] is False
    assert factory._uow.fake_chunks._chunks[1]["pack_metadata"]["effective_to"] == "2099-01-01"
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
async def test_same_number_private_upload_cannot_change_public_version(
    visibility: str, owner: int | None, group: int | None
) -> None:
    factory, service, _ = system()
    put_document(factory, 1)
    put_document(factory, 2, visibility, owner, group)
    public = await upload(service, 1, date(2020, 1, 1))
    other = await upload(service, 2, date(2024, 1, 1))
    assert public.is_current
    assert public.effective_to is None
    assert other.act_id != public.act_id
    with pytest.raises(ValidationError, match="visibility scope"):
        await service.update_version(require_present(other.id), act_id=public.act_id)
    assert require_present(
        await factory._uow.fake_act_versions.get_by_id(require_present(public.id))
    ).is_current


@pytest.mark.asyncio
async def test_different_years_and_owners_have_independent_act_identities() -> None:
    factory, service, _ = system()

    for document_id, owner in ((1, 10), (2, 10), (3, 11)):
        put_document(factory, document_id, "internal_private", owner)

    one = await upload(service, 1, date(2020, 1, 1), 2020)
    two = await upload(service, 2, date(2021, 1, 1), 2021)
    three = await upload(service, 3, date(2020, 1, 1), 2020)
    assert len({one.act_id, two.act_id, three.act_id}) == 3
    assert all(v.is_current for v in (one, two, three))


@pytest.mark.asyncio
async def test_each_space_keeps_its_own_timeline_for_the_same_act() -> None:
    factory, service, _ = system()

    for document_id, space_id in ((1, 1), (2, 2), (3, 2)):
        put_document(factory, document_id)
        factory._uow.fake_documents._documents[document_id].space_id = space_id

    first_space = await upload(service, 1, date(2020, 1, 1))
    second_space_old = await upload(service, 2, date(2021, 1, 1))
    second_space_new = await upload(service, 3, date(2024, 1, 1))
    assert second_space_old.act_id is not None
    assert second_space_old.act_id == second_space_new.act_id
    assert second_space_old.act_id != first_space.act_id
    versions = factory._uow.fake_act_versions
    old = require_present(await versions.get_by_id(require_present(second_space_old.id)))
    first = require_present(await versions.get_by_id(require_present(first_space.id)))
    assert old.effective_to == date(2024, 1, 1)
    assert not old.is_current
    assert first.is_current
    assert first.effective_to is None

    with pytest.raises(ValidationError, match="visibility scope"):
        await service.update_version(require_present(second_space_new.id), act_id=first_space.act_id)


@pytest.mark.asyncio
async def test_reindexing_does_not_duplicate_version_or_overwrite_manual_dates() -> None:
    factory, service, _ = system()
    put_document(factory, 1)
    initial = await upload(service, 1, date(2020, 1, 1))
    await service.update_version(require_present(initial.id), effective_from=date(2021, 1, 1))
    repeated = await upload(service, 1, date(2024, 1, 1))
    assert repeated.id == initial.id
    assert repeated.effective_from == date(2021, 1, 1)
    assert len(factory._uow.fake_act_versions._versions) == 1


@pytest.mark.asyncio
async def test_explicit_null_clears_dates_and_unlink_invalidates_document_cache() -> None:
    factory, service, cache = system()
    put_document(factory, 1)
    version = await upload(service, 1, date(2020, 1, 1))
    factory._uow.fake_chunks._chunks.append({"document_id": 1, "pack_metadata": edition_metadata(version)})
    await service.update_version(require_present(version.id), effective_to=date(2025, 1, 1))
    await service.update_version(
        require_present(version.id),
        effective_from_provided=True,
        effective_to_provided=True,
        act_id_provided=True,
    )
    stored = await factory._uow.fake_act_versions.get_by_id(require_present(version.id))
    assert require_present(stored).effective_from is None
    assert require_present(stored).effective_to is None
    assert require_present(stored).act_id is None
    entry = list(factory._uow.fake_vector_outbox._entries.values())[-1]
    assert (entry.operation, entry.aggregate_type, entry.aggregate_id) == (
        OutboxOperation.UPDATE_METADATA,
        "document",
        1,
    )
    chunk_metadata = factory._uow.fake_chunks._chunks[-1]["pack_metadata"]
    assert chunk_metadata["effective_from"] is None
    assert chunk_metadata["effective_to"] is None
    assert chunk_metadata["act_id"] is None
    cache.invalidate_by_document_ids.assert_any_await([1])


@pytest.mark.asyncio
async def test_invalid_interval_is_rejected() -> None:
    factory, service, _ = system()
    put_document(factory, 1)
    version = await upload(service, 1, date(2020, 1, 1))
    with pytest.raises(ValidationError, match="later"):
        await service.update_version(require_present(version.id), effective_to=date(2019, 1, 1))


def decree_hooks(service: ActVersioningService) -> ActEditionHooks:
    profile = DecreeDomainProfile(settings=Settings())
    return ActEditionHooks(service, cast("DomainProfileRegistry", SimpleNamespace(get=lambda _key: profile)))


async def persist(
    factory: LawFakeUnitOfWorkFactory, service: ActVersioningService, document_id: int, text: str
) -> Any:
    hooks = decree_hooks(service)
    return await persist_document_result(
        factory,
        PersistDocumentCommand(
            document_id=document_id,
            original_filename=f"{document_id}.rtf",
            raw_chunks=[RawDocument(page_content=text, metadata={})],
            visibility=DocumentVisibility.INTERNAL_PUBLIC,
            owner_id=None,
            group_id=None,
            doc_domain="decree",
            replace_id=None,
            warning_message=None,
            quality=None,
            storage_deletes=[],
            pack_document=await hooks.prepare("decree", text),
        ),
        domain_registry=None,
        pack_hooks=hooks,
    )


@pytest.mark.asyncio
async def test_persistence_uses_trusted_dates_and_propagates_backfilled_interval() -> None:
    factory, service, _ = system()

    for document_id in (1, 2, 3):
        put_document(factory, document_id)

    await upload(service, 1, date(2024, 1, 1))
    result = await persist(factory, service, 2, "УКАЗ №5 от 1 января 2020 г.\nВступает в силу с 01.01.2022.")
    assert result.pack_metadata["effective_to"] == "2024-01-01"
    assert result.pack_metadata["is_current"] is False
    chunk = factory._uow.fake_chunks._chunks[-1]
    assert chunk["pack_metadata"] == result.pack_metadata
    result = await persist(factory, service, 3, "УКАЗ №9 от 1 января 2020 г.\nПОСТАНОВЛЯЮ:\n1. Пункт.")
    version = await factory._uow.fake_act_versions.get_by_id(result.pack_metadata["act_version_id"])
    assert require_present(version).effective_from is None
    assert factory._uow.fake_chunks._chunks[-1]["pack_metadata"]["effective_from"] is None


def test_real_qdrant_engine_handles_missing_null_and_half_open_date_bounds() -> None:
    client = QdrantClient(":memory:")
    try:
        client.create_collection("versions", vectors_config=VectorParams(size=1, distance=Distance.COSINE))
        metadata: list[dict[str, Any]] = [
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
                PointStruct(
                    id=i, vector=[1.0], payload={"metadata": {"pack_metadata": {"act_version_id": i, **meta}}}
                )
                for i, meta in enumerate(metadata, 1)
            ],
        )

        for as_of_date, expected in [
            (date(2023, 12, 31), {1, 2, 4}),
            (date(2024, 1, 1), {1, 2, 5}),
            (date(2099, 1, 1), {1, 2, 6}),
            (None, {1, 2, 5}),
        ]:
            selection = Filter(must=[pack_selection_condition(edition_selection(as_of_date))])
            points, _ = client.scroll("versions", scroll_filter=selection)
            assert {point.id for point in points} == expected
    finally:
        client.close()


def test_all_chunks_of_selected_edition_survive_with_mixed_date_types() -> None:
    old = SearchDocument(
        page_content="old",
        metadata={"pack_metadata": {"act_id": 1, "act_version_id": 1, "effective_from": date(2020, 1, 1)}},
    )
    newer = [
        SearchDocument(
            page_content=f"new {i}",
            metadata={"pack_metadata": {"act_id": 1, "act_version_id": 2, "effective_from": "2024-01-01"}},
        )
        for i in range(3)
    ]
    result = resolve_edition_conflicts([(old, 0.99)] + [(doc, 0.8) for doc in newer])
    assert [doc.page_content for doc, _ in result] == [doc.page_content for doc in newer]


@pytest.mark.asyncio
async def test_delayed_outbox_metadata_uses_latest_state_and_invalidates_cache() -> None:
    factory, service, cache = system()
    put_document(factory, 1)
    put_document(factory, 2)
    version = await upload(service, 1, date(2020, 1, 1))
    factory._uow.fake_chunks._chunks.append({"document_id": 1, "pack_metadata": edition_metadata(version)})
    await service.update_version(require_present(version.id), effective_from=date(2024, 1, 1))
    vector_store = SimpleNamespace(set_pack_metadata_by_document_id=AsyncMock())
    dispatcher = OutboxDispatcher(factory, vector_store, cache)
    cache.invalidate_by_document_ids.reset_mock()
    # A late retry of the projection still applies the committed SQL state.
    await dispatcher._apply_pack_metadata(1, [1, 2])
    document_id, metadata = vector_store.set_pack_metadata_by_document_id.call_args.args
    assert document_id == 1
    assert metadata["effective_from"] == "2024-01-01"
    cache.invalidate_by_document_ids.assert_awaited_once_with([2], raise_on_error=True)
    entries = [
        entry
        for entry in factory._uow.fake_vector_outbox._entries.values()
        if entry.operation == "update_metadata"
    ]
    assert entries
    assert all(entry.aggregate_type == "document" and entry.aggregate_id == 1 for entry in entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["exact", "icontains"])
async def test_sql_search_applies_edition_selection_with_acl(mode: str) -> None:
    stmt = await capture(
        "rules", {"id": 1, "kind": "internal"}, [], mode, None, selection=edition_selection(date(2024, 1, 1))
    )
    dialect = postgresql.dialect()  # type: ignore[no-untyped-call]  # SQLAlchemy exposes an untyped dialect constructor.
    sql = str(stmt.compile(dialect=dialect))
    assert "chunks.pack_metadata" in sql
    assert "chunks.visibility" in sql
    assert "act_versions" not in sql


@pytest.mark.asyncio
async def test_exact_candidates_include_temporal_metadata_and_request_date() -> None:
    result = ChunkRecord(
        chunk_id=1,
        document_id=2,
        filename="rules.rtf",
        content="rules",
        chunk_index=0,
        pack_metadata={"act_id": 4, "act_version_id": 6, "effective_from": "2024-01-01", "is_current": False},
    )
    search = SimpleNamespace(search_substring=AsyncMock(return_value=[result]))
    candidates: list[Any] = [SearchDocument(page_content="seed", metadata={"document_id": 2})]
    await apply_exact_search(
        "Что предусматривает пункт 5?",
        candidates,
        UserContext(1, UserKind.INTERNAL),
        cast("RagContext", SimpleNamespace(as_of_date=date(2024, 1, 1), space=None)),
        search,
    )
    assert search.search_substring.call_args.kwargs["selection"] == edition_selection(date(2024, 1, 1))
    assert candidates[-1].metadata["pack_metadata"]["act_id"] == 4
    assert candidates[-1].metadata["pack_metadata"]["act_version_id"] == 6
    assert candidates[-1].metadata["pack_metadata"]["effective_from"] == "2024-01-01"


@pytest.mark.asyncio
async def test_cache_key_changes_each_day_for_current_state_but_not_for_explicit_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class Clock(date):
        current = date(2024, 1, 1)

        @classmethod
        def today(cls) -> Any:
            return cls.current

    monkeypatch.setattr("infrastructure.ml.cache.keys.cache_today", Clock.today)
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
        space=None,
    )
    rag = make_rag()
    rag = replace(rag, features=replace(rag.features, cache_enabled=True))
    state = SimpleNamespace(ctx=ctx, query_for_search="rules", question="rules", history_messages=[], rag=rag)
    await check_cache(state)
    first_hash = state.q_hash
    Clock.current = date(2024, 1, 2)
    await check_cache(state)
    assert state.q_hash != first_hash
    ctx.as_of_date = date(2024, 1, 1)
    await check_cache(state)
    assert state.q_hash == first_hash


@pytest.mark.asyncio
async def test_admin_all_versions_pagination_includes_reviewed_entries_and_referenced_acts() -> None:

    factory, service, _ = system()

    for document_id in (1, 2):
        put_document(factory, document_id)
        version = await upload(service, document_id, date(2020 + document_id, 1, 1))
        await service.update_version(require_present(version.id), verified_by=10)

    response = await list_act_versions(
        review="all", limit=1, offset=0, _admin=cast("CurrentUser", SimpleNamespace(id=10)), service=service
    )
    assert response.total == 2
    assert len(response.versions) == 1
    assert response.next_offset == 1
    assert response.versions[0].date_source == "manual"
    assert response.versions[0].act_id in {act.id for act in response.acts}
    response = await list_act_versions(
        review="all", limit=1, offset=1, _admin=cast("CurrentUser", SimpleNamespace(id=10)), service=service
    )
    assert response.next_offset is None


@pytest.mark.asyncio
async def test_same_effective_date_supersedes_old_edition() -> None:
    factory, service, _ = system()

    for document_id in (1, 2):
        put_document(factory, document_id)

    original = await upload(service, 1, date(2024, 1, 1))
    corrected = await upload(service, 2, date(2024, 1, 1))
    original = require_present(await factory._uow.fake_act_versions.get_by_id(require_present(original.id)))
    assert original.effective_to == original.effective_from
    assert not original.is_current
    assert corrected.is_current


@pytest.mark.asyncio
async def test_backfill_after_future_edition_takes_effect_cannot_become_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class Clock(date):
        current = date(2026, 1, 1)

        @classmethod
        def today(cls) -> Any:
            return cls.current

    monkeypatch.setattr(act_versioning_service, "today", Clock.today)
    factory, service, _ = system()

    for document_id in (1, 2, 3):
        put_document(factory, document_id)

    await upload(service, 1, date(2024, 1, 1))
    await upload(service, 2, date(2099, 1, 1))
    Clock.current = date(2100, 1, 1)
    backfilled = await upload(service, 3, date(2098, 1, 1))
    assert not backfilled.is_current
    assert backfilled.effective_to == date(2099, 1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["upload", "cli"])
@pytest.mark.parametrize("outbox_failure", [False, True])
async def test_prepared_edition_and_chunks_share_transaction(  # noqa: PLR0915 - One scenario verifies the complete transaction/lifecycle.
    path: object, outbox_failure: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory, service, cache = system()
    put_document(factory, 1)
    put_document(factory, 2)
    original = await upload(service, 1, date(2020, 1, 1))
    cache.invalidate_by_document_ids.reset_mock()
    text = "УКАЗ №5 от 1 января 2020 г.\nВступает в силу с 01.01.2024."
    uow = factory._uow
    entries_before = dict(uow.fake_vector_outbox._entries)
    enqueue = uow.fake_vector_outbox.enqueue

    async def enqueue_with_failure(entry: VectorOutboxEntry) -> None:
        await enqueue(entry)

        if entry.operation == OutboxOperation.UPSERT_CHUNKS:
            # The edition already exists when chunk projection fails.
            assert len(await uow.fake_act_versions.list_by_act(require_present(original.act_id))) == 2
            msg = "chunk outbox failed"
            raise RuntimeError(msg)

    if outbox_failure:
        monkeypatch.setattr(uow.fake_vector_outbox, "enqueue", enqueue_with_failure)

    async def execute() -> None:
        if path == "upload":
            await persist(factory, service, 2, text)
            return

        hooks = decree_hooks(service)
        sync = DocumentSyncService(factory, hooks)
        await sync.sync_one_document(
            "2.rtf",
            2,
            "s3://docs/2.rtf",
            [RawDocument(page_content=text, metadata={"doc_domain": "decree"})],
            pack_document=await sync.prepare_pack_document("2.rtf", "decree", text),
            visibility=DocumentVisibility.INTERNAL_PUBLIC,
            owner_id=None,
            group_id=None,
        )

    if outbox_failure:
        with pytest.raises(RuntimeError, match="chunk outbox failed"):
            await execute()
        assert uow._rolled_back
        assert len(await uow.fake_act_versions.list_by_act(require_present(original.act_id))) == 1
        restored = await uow.fake_act_versions.get_by_id(require_present(original.id))
        assert require_present(restored).effective_to is None
        assert require_present(restored).is_current
        assert uow.fake_chunks._chunks == []
        assert uow.fake_vector_outbox._entries == entries_before
        cache.invalidate_by_document_ids.assert_not_awaited()
    else:
        await execute()
        assert uow._committed
        versions = await uow.fake_act_versions.list_by_act(require_present(original.act_id))
        created = next(v for v in versions if v.document_id == 2)
        assert created.effective_from == date(2024, 1, 1)
        assert uow.fake_chunks._chunks[-1]["pack_metadata"]["act_version_id"] == created.id
        cache.invalidate_by_document_ids.assert_awaited_once()
