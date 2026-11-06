"""Stale cache/index data must not bypass the primary database ACL."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document as Chunk
import pytest

from application.documents import DocumentAccessService
from application.rag.dto import (
    CompletedAnswer,
    ContextBudget,
    PreparedContext,
    SanitizedAnswer,
)
from composition.service_providers import create_rag_service
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import CuratorScope, UserContext
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from infrastructure.ml.rag import rag_steps, retrieval_pipeline
from infrastructure.ml.rag.document_access import cache_is_accessible, filter_scored_documents
from infrastructure.ml.rag.retrieval_work import RetrievalWork
from tests.fakes import FakeUnitOfWorkFactory
from tests.rag_stage_fixtures import check_cache as step_check_cache, retrieval_result
from tests.test_rag_improvements import state_for
from tests.test_rag_pipeline import make_rag


if TYPE_CHECKING:
    from application.rag.dto import (
        PreparedQuery,
        RetrievalResult,
    )
    from application.rag.service import RagService


@pytest.fixture
def database_access(monkeypatch: pytest.MonkeyPatch) -> tuple[DocumentAccessService, FakeUnitOfWorkFactory]:
    factory = FakeUnitOfWorkFactory()
    factory._uow.fake_documents._documents = {
        1: Document(id=1, visibility=DocumentVisibility.INTERNAL_PUBLIC),
        2: Document(id=2, visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=99),
        3: Document(id=3, visibility=DocumentVisibility.INTERNAL_GROUP, group_id=10),
        4: Document(id=4, visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=99),
    }
    monkeypatch.setattr(factory, "create", MagicMock(wraps=factory.create))
    return DocumentAccessService(factory), factory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user,expected",
    [
        (UserContext(1, UserKind.INTERNAL, UserRole.USER), {1}),
        (UserContext(1, UserKind.INTERNAL, UserRole.USER, group_ids=(10,)), {1, 3}),
        (UserContext(99, UserKind.INTERNAL, UserRole.USER), {1, 2}),
        (UserContext(99, UserKind.CLIENT, UserRole.USER), {4}),
        (UserContext(1, UserKind.INTERNAL, UserRole.ADMIN), {1, 2, 3}),
        (
            UserContext(
                1,
                UserKind.INTERNAL,
                UserRole.CURATOR,
                managed_internal_ids=(99,),
                managed_client_ids=(99,),
                managed_group_ids=(10,),
            ),
            {1, 2, 3, 4},
        ),
    ],
)
async def test_primary_database_search_acl(
    database_access: tuple[DocumentAccessService, FakeUnitOfWorkFactory],
    user: UserContext,
    expected: set[int],
) -> None:
    access, factory = database_access
    assert await access.allowed_document_ids([1, 2, 3, 4, 999], user) == expected
    cast("MagicMock", factory.create).assert_called_once_with(master=True)


@pytest.mark.asyncio
async def test_stale_index_ignores_deleted_and_inaccessible_documents(
    database_access: tuple[DocumentAccessService, FakeUnitOfWorkFactory],
) -> None:
    access, _ = database_access
    user = UserContext(1, UserKind.INTERNAL, UserRole.USER)
    # Index claims all documents are public; DB ACL is authoritative.
    docs = [
        (
            Chunk(
                page_content=str(did),
                metadata={"document_id": did, "doc_visibility": DocumentVisibility.INTERNAL_PUBLIC.value},
            ),
            0.9,
        )
        for did in [1, 2, 3, 4, 999]
    ]
    docs += [(Chunk(page_content="legacy without ID"), 0.8)]
    permitted = await filter_scored_documents(docs, access, user)
    assert [doc.metadata["document_id"] for doc, _ in permitted] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dependencies,source_ids,expected",
    [
        ([1], [1], True),
        ([1, 2], [1], False),
        ([1, 999], [1], False),
        ([1], [2], False),
        ([], [], False),
        (None, [1], False),
        ([True], [], False),
    ],
)
async def test_cache_checks_all_dependencies_not_only_citations(
    database_access: tuple[DocumentAccessService, FakeUnitOfWorkFactory],
    dependencies: list[int] | list[bool] | None,
    source_ids: list[int],
    expected: bool,
) -> None:
    access, _ = database_access
    cached = {
        "answer": "sensitive",
        "document_ids": dependencies,
        "sources": [{"document_id": did} for did in source_ids],
    }
    user = UserContext(1, UserKind.INTERNAL, UserRole.USER)
    assert await cache_is_accessible(cached, access, user) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", [False, True])
async def test_cache_hit_becomes_miss_without_redis_invalidation(
    database_access: tuple[DocumentAccessService, FakeUnitOfWorkFactory],
    monkeypatch: pytest.MonkeyPatch,
    deleted: bool,
) -> None:

    access, factory = database_access

    if deleted:
        del factory._uow.fake_documents._documents[1]
    else:
        factory._uow.fake_documents._documents[1].visibility = DocumentVisibility.INTERNAL_PRIVATE
        factory._uow.fake_documents._documents[1].owner_id = 99

    cached = {"answer": "must not escape", "document_ids": [1], "sources": [{"document_id": 1}]}
    monkeypatch.setattr(rag_steps, "get_corpus_revision", AsyncMock(return_value="unchanged"))
    monkeypatch.setattr(rag_steps, "check_cache", AsyncMock(return_value=cached))
    rag = make_rag()
    rag = replace(rag, features=replace(rag.features, cache_enabled=True))
    state = state_for([], rag=rag)
    _, events = await step_check_cache(state, document_access=access)
    assert events is None
    assert not state.terminal


@pytest.mark.asyncio
async def test_database_failure_never_emits_cached_answer(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(rag_steps, "get_corpus_revision", AsyncMock(return_value="0"))
    monkeypatch.setattr(
        rag_steps,
        "check_cache",
        AsyncMock(return_value={"answer": "secret", "document_ids": [1], "sources": [{"document_id": 1}]}),
    )
    access = SimpleNamespace(allowed_document_ids=AsyncMock(side_effect=RuntimeError("DB unavailable")))
    rag = make_rag()
    state = state_for([], rag=replace(rag, features=replace(rag.features, cache_enabled=True)))
    with pytest.raises(RuntimeError, match="DB unavailable"):
        await step_check_cache(state, document_access=access)
    assert not state.terminal


@pytest.mark.asyncio
async def test_neighbor_expansion_cannot_reintroduce_forbidden_evidence(
    database_access: tuple[DocumentAccessService, FakeUnitOfWorkFactory], monkeypatch: pytest.MonkeyPatch
) -> None:

    access, _ = database_access
    visible = Chunk(page_content="allowed", metadata={"document_id": 1})
    hidden = Chunk(page_content="secret", metadata={"document_id": 2})
    reranker = AsyncMock(return_value=[(visible, 0.9)])
    monkeypatch.setattr(retrieval_pipeline, "rerank_documents", reranker)
    monkeypatch.setattr(
        retrieval_pipeline, "enrich_with_neighbors", AsyncMock(return_value=[(visible, 0.9), (hidden, 0.9)])
    )
    ml = SimpleNamespace(reranker_semaphore=__import__("asyncio").Semaphore(1), reranker=lambda: None)
    ctx = RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER))
    docs, _, _ = await retrieval_pipeline.rerank_and_enrich(
        "question",
        [visible, hidden],
        RetrievalWork(
            rag=make_rag(),
            breadth=Breadth.NARROW,
            query_domain="general",
            rerank_top_n=4,
            retrieval_filter=None,
            ctx=ctx,
            history_messages=[],
            enumerate_cases=False,
            visibility_conditions=[],
            t_pipeline_start=0,
            question="question",
            access_filter=None,
            user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER),
        ),
        ml,
        None,
        document_access=access,
    )
    assert reranker.call_args.args[1] == [visible]
    assert docs == [(visible, 0.9)]


@pytest.mark.asyncio
async def test_self_rag_retry_reuses_authorized_pipeline_and_merges_query_evidence(
    database_access: tuple[DocumentAccessService, FakeUnitOfWorkFactory], monkeypatch: pytest.MonkeyPatch
) -> None:

    service, ctx, searches, exact, captured = retry_fixture(database_access, monkeypatch)
    _ = [event async for event in service.stream("initial", [], ctx)]
    assert searches == ["initial", "refined"]
    assert [call.args[0] for call in exact.await_args_list] == searches
    assert captured[-1].sub_queries == ("initial", "probe initial", "refined", "probe refined")
    assert captured[-1].prepared == captured[0].prepared
    assert [d.chunk.content for d in captured[-1].documents] == ["allowed"]


def retry_fixture(
    database_access: tuple[DocumentAccessService, FakeUnitOfWorkFactory], monkeypatch: pytest.MonkeyPatch
) -> tuple[RagService, RagContext, list[str], AsyncMock, list[RetrievalResult]]:
    access, _ = database_access
    visible = Chunk(page_content="allowed", metadata={"document_id": 1})
    hidden = Chunk(page_content="secret", metadata={"document_id": 2})

    ctx = RagContext(
        user=UserContext(
            user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.CURATOR, group_ids=(10,)
        ).with_curator_scope(CuratorScope(managed_internal_ids=(88,))),
        as_of_date=date(2025, 1, 1),
    )
    searches = []

    async def retrieve(query: str, request: RetrievalWork, clients: object) -> tuple[list[Chunk], list[str]]:
        assert request.ctx == ctx
        assert request.user == ctx.to_user_context()
        assert request.retrieval_filter is not None
        searches.append(query)
        return [visible, hidden], [query, "probe " + query]

    async def rerank(
        query: str,
        candidates: list[Chunk],
        request: RetrievalWork,
        clients: object,
        chunk_search: object,
        **kwargs: Any,
    ) -> tuple[list[tuple[Chunk, float]], list[Chunk], float]:
        assert candidates == [visible]
        return [(visible, 0.9)], [visible], 0.9

    exact = AsyncMock()
    monkeypatch.setattr(retrieval_pipeline, "retrieve_with_decomposition", retrieve)
    monkeypatch.setattr(retrieval_pipeline, "rerank_and_enrich", rerank)
    monkeypatch.setattr(retrieval_pipeline, "apply_exact_search", exact)
    monkeypatch.setattr(rag_steps, "handle_relevance_gate", AsyncMock(side_effect=[False, True]))
    monkeypatch.setattr(
        rag_steps,
        "assess_sufficiency",
        AsyncMock(return_value=SimpleNamespace(is_sufficient=False, suggested_refinement="refined")),
    )
    rag = replace(make_rag(), features=replace(make_rag().features, condense_enabled=False))
    monkeypatch.setattr(rag_steps, "build_rag_settings", lambda: rag)
    service = create_rag_service(MagicMock(), document_access=access)
    captured = capture_retry(service, ctx, monkeypatch)
    return service, ctx, searches, exact, captured


def capture_retry(
    service: RagService, ctx: RagContext, monkeypatch: pytest.MonkeyPatch
) -> list[RetrievalResult]:
    captured = []
    original_retrieve = service.retrieval.retrieve

    async def with_exact(query: PreparedQuery, previous: RetrievalResult | None = None) -> RetrievalResult:
        if previous is None:
            seed = retrieval_result(state_for([], ctx=ctx))
            previous = replace(seed, exact_ref_boost=True)

        result = await original_retrieve(query, previous)
        captured.append(result)
        return result

    monkeypatch.setattr(service.retrieval, "retrieve", with_exact)

    def build(result: RetrievalResult, *, capture: bool = True) -> PreparedContext:
        del capture
        return PreparedContext(result, (), (), result.documents, ContextBudget(0, 0, 0, 0), "answer", None)

    monkeypatch.setattr(service.context, "build", build)
    monkeypatch.setattr(
        service.generation,
        "generate",
        AsyncMock(side_effect=lambda context: SanitizedAnswer(context, "answer", None, 0)),
    )
    monkeypatch.setattr(service.finalization, "finalize", lambda answer: CompletedAnswer(answer, (), 0.9))
    return captured
