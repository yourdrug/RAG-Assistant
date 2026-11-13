"""Tests for ACL filter propagation to Qdrant calls (F-017 regression).

Verifies that must-style filters (from with_temporal_filter / with_domain_filter)
are correctly passed to qdrant_dense_search and resolve_hashes_batch,
rather than being silently dropped.
"""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.schema import Document
import pytest
from sqlalchemy import JSON, Column, MetaData, Table, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.access_control import get_visibility_conditions
from domain.knowledge.chunk import ChunkRecord
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.ranking import same_text_run
from infrastructure.database.models import ChunkModel
from infrastructure.ml.rag.rag_postprocess import enrich_with_neighbors
from infrastructure.ml.rag.rag_retrieval import qdrant_dense_search, resolve_hashes_batch
from infrastructure.ml.rag.retrieval_pipeline import (
    rerank_and_enrich,
    retrieve_with_decomposition,
    run_retrieval,
)
from infrastructure.ml.rag.retrieval_work import RetrievalWork
from infrastructure.repositories.chunk.chunk_context import (
    MAX_PARENT_CHUNKS,
    SQLAlchemyChunkContextRepository,
)
from infrastructure.repositories.vector.acl import (
    build_qdrant_filter,
    with_domain_filter,
    with_temporal_filter,
)
from tests.document_node_sql import (
    ACT_VERSION_ID,
    ARTICLE,
    CHAPTER,
    CHUNKS,
    DOCUMENT_ID,
    FOREIGN_DOCUMENT_ID,
    NODES,
    OWNER_ID,
    POINT,
    SIBLING_ARTICLE,
    parent_scope_repository,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(kind: str = "internal", uid: int = 5) -> dict[str, str | int]:
    return {"id": uid, "kind": kind, "role": "user"}


def _make_ml_clients(
    qdrant_client: MagicMock | None = None,
    embeddings: MagicMock | None = None,
    bm25_index: MagicMock | None = None,
) -> MagicMock:
    ml = MagicMock()
    ml.qdrant_client.return_value = qdrant_client or MagicMock()
    embs = embeddings or MagicMock()
    embs.embed_query = AsyncMock(return_value=[0.1] * 384)
    ml.embeddings.return_value = embs
    ml.ensure_bm25_loaded = AsyncMock(return_value=bm25_index)
    return ml


# ---------------------------------------------------------------------------
# qdrant_dense_search: must-style filter must be passed to client.search
# ---------------------------------------------------------------------------


class TestDenseSearchACLFilter:
    def test_normal_acl_filter_passed_to_search(self) -> None:

        af = build_qdrant_filter(_user(), [1, 2])
        mock_client = MagicMock()
        mock_client.search.return_value = []

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(qdrant_dense_search("test", 10, af, ml))

        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs["query_filter"] is not None

    def test_must_style_filter_after_temporal_not_dropped(self) -> None:

        af = build_qdrant_filter(_user(), [1, 2])
        temporal_af = with_temporal_filter(af, None)

        mock_client = MagicMock()
        mock_client.search.return_value = []

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(qdrant_dense_search("test", 10, temporal_af, ml))

        call_kwargs = mock_client.search.call_args
        applied_filter = call_kwargs.kwargs["query_filter"]
        assert applied_filter is not None, (
            "Must-style filter from with_temporal_filter was dropped — dense search ran without ACL!"
        )
        assert applied_filter is temporal_af

    def test_must_style_filter_after_domain_not_dropped(self) -> None:

        af = build_qdrant_filter(_user(), [1, 2])
        legal_af = with_domain_filter(af, "legal")

        mock_client = MagicMock()
        mock_client.search.return_value = []

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(qdrant_dense_search("test", 10, legal_af, ml))

        call_kwargs = mock_client.search.call_args
        applied_filter = call_kwargs.kwargs["query_filter"]
        assert applied_filter is not None
        assert applied_filter is legal_af

    def test_none_filter_rejected_before_io(self) -> None:

        mock_client = MagicMock()
        mock_client.search.return_value = []

        ml = _make_ml_clients(qdrant_client=mock_client)
        with pytest.raises(RuntimeError, match="requires access_filter"):
            asyncio.run(qdrant_dense_search("test", 10, None, ml))

        ml.qdrant_client.assert_not_called()
        ml.embeddings.assert_not_called()
        ml.embeddings.return_value.embed_query.assert_not_awaited()
        mock_client.search.assert_not_called()


# ---------------------------------------------------------------------------
# resolve_hashes_batch: must-style filter must wrap scroll_filter
# ---------------------------------------------------------------------------


class TestResolveHashesACLFilter:
    def test_must_style_filter_includes_acl_in_scroll(self) -> None:

        af = build_qdrant_filter(_user(), [1, 2])
        temporal_af = with_temporal_filter(af, None)
        # temporal_af = Filter(must=[...]) → should=None

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(resolve_hashes_batch(["hash1", "hash2"], temporal_af, ml))

        call_kwargs = mock_client.scroll.call_args
        scroll_filter = call_kwargs.kwargs["scroll_filter"]
        assert scroll_filter is not None
        assert scroll_filter.must is not None
        assert temporal_af in scroll_filter.must

    def test_none_filter_raises_runtime_error(self) -> None:
        """access_filter=None must raise RuntimeError (FINDING-020 security guard)."""

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        ml = _make_ml_clients(qdrant_client=mock_client)
        with pytest.raises(RuntimeError, match="requires access_filter"):
            asyncio.run(resolve_hashes_batch(["hash1"], None, ml))


@pytest.mark.asyncio
@pytest.mark.parametrize("decomposition", [False, True])
@pytest.mark.parametrize("sub_queries", [["query"], ["first query", "second query"]])
async def test_decomposition_and_legal_fallback_preserve_acl(
    decomposition: bool, sub_queries: list[str]
) -> None:

    af = with_temporal_filter(build_qdrant_filter(_user(), [1, 2]), None)
    conditions = get_visibility_conditions(UserKind.INTERNAL, [1, 2], for_list=False)
    rag = MagicMock()
    rag.features.decomposition_enabled = decomposition
    ml = _make_ml_clients()
    with (
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.decompose_question",
            AsyncMock(return_value=sub_queries),
        ),
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.run_hybrid_search",
            AsyncMock(return_value=[]),
        ) as search,
    ):
        candidates, actual_queries = await retrieve_with_decomposition(
            "query",
            RetrievalWork(
                fetch_k=10,
                retrieval_filter=af,
                rag=rag,
                breadth=Breadth.NARROW,
                query_domain="legal",
                effective_dense_weight=1.0,
                effective_sparse_weight=1.0,
                visibility_conditions=conditions,
                t_pipeline_start=0,
                question="query",
                access_filter=af,
                ctx=RagContext(
                    user=UserContext(
                        user_id=5, user_kind=UserKind.INTERNAL, group_ids=(1, 2), user_role=UserRole.USER
                    )
                ),
                user=UserContext(user_id=5, user_kind=UserKind.INTERNAL, user_role=UserRole.USER),
            ),
            ml,
        )

    assert candidates == []
    assert actual_queries == (sub_queries if decomposition else ["query"])
    assert search.await_count == 2 * len(actual_queries)

    for domain_call, fallback_call in zip(
        search.await_args_list[::2],
        search.await_args_list[1::2],
        strict=True,
    ):
        assert af in domain_call.args[2].must
        assert fallback_call.args[2] is af

        for call in (domain_call, fallback_call):
            assert call.kwargs["visibility_conditions"] is conditions
            assert call.kwargs["user_id"] == 5
            assert call.kwargs["user_group_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_legal_rerank_fallback_preserves_acl_and_temporal_filter() -> None:

    af = with_temporal_filter(build_qdrant_filter(_user(), [1, 2]), None)
    conditions = get_visibility_conditions(UserKind.INTERNAL, [1, 2], for_list=False)
    ctx = RagContext(
        user=UserContext(user_id=5, user_kind=UserKind.INTERNAL, user_role=UserRole.USER, group_ids=(1, 2)),
        as_of_date=None,
    )
    rag = MagicMock()
    ml = _make_ml_clients()
    with (
        patch("infrastructure.ml.rag.retrieval_pipeline.rerank_documents", AsyncMock(return_value=[])),
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.run_hybrid_search",
            AsyncMock(return_value=[]),
        ) as search,
        patch("infrastructure.ml.rag.retrieval_pipeline.enrich_with_neighbors", AsyncMock(return_value=[])),
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.prepare_generation_budget",
            return_value=(MagicMock(), [], 4096, 400, 1000),
        ),
    ):
        await rerank_and_enrich(
            "query",
            [],
            RetrievalWork(
                rag=rag,
                breadth=Breadth.NARROW,
                query_domain="legal",
                rerank_top_n=5,
                retrieval_filter=af,
                ctx=ctx,
                history_messages=[],
                enumerate_cases=False,
                visibility_conditions=conditions,
                t_pipeline_start=0,
                question="query",
                access_filter=af,
                user=UserContext(user_id=5, user_kind=UserKind.INTERNAL, user_role=UserRole.USER),
            ),
            ml,
            None,
        )

    search.assert_awaited_once_with(
        "query",
        rag.retriever.fetch_k,
        af,
        rag,
        ml_clients=ml,
        visibility_conditions=conditions,
        user_id=5,
        user_group_ids=[1, 2],
    )


@pytest.fixture(autouse=True)
def mock_embedding_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "infrastructure.ml.rag.rag_retrieval.ensure_embedding_identity", lambda *_args: "test:model:v1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain,first_hit,expected_calls",
    [("general", False, 1), ("legal", True, 1), ("legal", False, 2)],
)
async def test_retrieval_domain_fallback_only_on_empty_legal_search(
    domain: str, first_hit: bool, expected_calls: int
) -> None:

    af = with_temporal_filter(build_qdrant_filter(_user(), [1, 2]), None)
    ctx = RagContext(
        user=UserContext(user_id=5, user_kind=UserKind.INTERNAL, group_ids=(1, 2), user_role=UserRole.USER)
    )
    work = RetrievalWork(
        rag=MagicMock(),
        t_pipeline_start=0,
        question="query",
        ctx=ctx,
        user=ctx.to_user_context(),
        access_filter=af,
        retrieval_filter=af,
        query_domain=domain,
        fetch_k=10,
        effective_dense_weight=0.7,
        effective_sparse_weight=0.3,
        visibility_conditions=[MagicMock()],
    )
    hit = MagicMock()
    results = [[hit] if first_hit else []]

    if expected_calls == 2:
        results.append([hit])

    with patch(
        "infrastructure.ml.rag.retrieval_pipeline.run_hybrid_search", AsyncMock(side_effect=results)
    ) as search:
        candidates = await run_retrieval("query", work, MagicMock())

    assert candidates == ([hit] if first_hit or expected_calls == 2 else [])
    assert search.await_count == expected_calls
    full_search = ("query", 10, 0.7, 0.3)
    assert work.full_corpus_searches == ({full_search} if domain == "general" or not first_hit else set())

    for call in search.await_args_list:
        assert call.args[:2] == ("query", 10)
        assert call.kwargs["dense_weight"] == 0.7
        assert call.kwargs["sparse_weight"] == 0.3
        assert call.kwargs["visibility_conditions"] is work.visibility_conditions
        assert call.kwargs["user_id"] == 5
        assert call.kwargs["user_group_ids"] == [1, 2]

    if domain == "legal":
        assert af in search.await_args_list[0].args[2].must
    else:
        assert search.await_args_list[0].args[2] is af

    if expected_calls == 2:
        assert search.await_args_list[1].args[2] is af


def _fallback_work(af: object, **overrides: object) -> RetrievalWork:
    ctx = RagContext(
        user=UserContext(user_id=5, user_kind=UserKind.INTERNAL, user_role=UserRole.USER, group_ids=(1, 2)),
    )
    work = RetrievalWork(
        rag=MagicMock(),
        breadth=Breadth.NARROW,
        query_domain="legal",
        rerank_top_n=5,
        retrieval_filter=af,
        ctx=ctx,
        visibility_conditions=get_visibility_conditions(UserKind.INTERNAL, [1, 2], for_list=False),
        t_pipeline_start=0,
        question="query",
        access_filter=af,
        user=ctx.to_user_context(),
    )

    for key, value in overrides.items():
        setattr(work, key, value)

    return work


async def _rerank_with_fallback(
    work: RetrievalWork, fallback_docs: list[tuple[Document, float]]
) -> tuple[list[tuple[Document, float]], AsyncMock]:
    with (
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.rerank_documents",
            AsyncMock(side_effect=[[], fallback_docs]),
        ),
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.run_hybrid_search",
            AsyncMock(return_value=[MagicMock()]),
        ) as search,
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.enrich_with_neighbors",
            AsyncMock(side_effect=lambda docs, **_kwargs: docs),
        ),
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.prepare_generation_budget",
            return_value=(MagicMock(), [], 4096, 400, 1000),
        ),
    ):
        docs, _, _ = await rerank_and_enrich("query", [], work, _make_ml_clients(), None)

    return docs, search


@pytest.mark.asyncio
async def test_legal_rerank_fallback_resolves_temporal_conflicts() -> None:
    old = Document(
        page_content="old", metadata={"act_id": 1, "act_version_id": 1, "effective_from": "2020-01-01"}
    )
    new = Document(
        page_content="new", metadata={"act_id": 1, "act_version_id": 2, "effective_from": "2024-01-01"}
    )
    af = with_temporal_filter(build_qdrant_filter(_user(), [1, 2]), None)

    docs, search = await _rerank_with_fallback(_fallback_work(af), [(old, 0.9), (new, 0.8)])

    search.assert_awaited_once()
    assert [doc.metadata["act_version_id"] for doc, _ in docs] == [2]


def _fallback_rag() -> MagicMock:
    rag = MagicMock()
    rag.retriever.fetch_k = 10
    rag.hybrid_search.dense_weight = 0.7
    rag.hybrid_search.sparse_weight = 0.3
    return rag


@pytest.mark.asyncio
async def test_legal_rerank_fallback_skipped_when_same_full_corpus_search_already_ran() -> None:
    af = with_temporal_filter(build_qdrant_filter(_user(), [1, 2]), None)
    work = _fallback_work(af, rag=_fallback_rag(), full_corpus_searches={("query", 10, 0.7, 0.3)})

    docs, search = await _rerank_with_fallback(work, [(Document(page_content="x"), 0.9)])

    search.assert_not_awaited()
    assert docs == []


@pytest.mark.asyncio
async def test_legal_rerank_fallback_runs_when_full_corpus_search_used_other_params() -> None:
    af = with_temporal_filter(build_qdrant_filter(_user(), [1, 2]), None)
    work = _fallback_work(af, rag=_fallback_rag(), full_corpus_searches={("query", 40, 0.7, 0.3)})
    found = Document(page_content="x")

    docs, search = await _rerank_with_fallback(work, [(found, 0.9)])

    search.assert_awaited_once()
    assert [doc for doc, _ in docs] == [found]


@pytest.mark.asyncio
async def test_legal_rerank_fallback_searches_full_question_after_decomposition() -> None:
    """Sub-queries fell back to the full corpus, but the original question never ran there."""
    af = with_temporal_filter(build_qdrant_filter(_user(), [1, 2]), None)
    rag = _fallback_rag()
    rag.features.decomposition_enabled = True
    work = _fallback_work(af, rag=rag, fetch_k=10, effective_dense_weight=0.7, effective_sparse_weight=0.3)
    sub_hit = Document(page_content="sub", metadata={"chunk_id": "s"})
    found = Document(page_content="answer", metadata={"chunk_id": "a"})

    async def search(
        query: str, _fetch_k: int, access_filter: object, *_args: object, **_kwargs: object
    ) -> list[Document]:
        return [sub_hit] if access_filter is af else []

    with (
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.decompose_question",
            AsyncMock(return_value=["part one", "part two"]),
        ),
        patch("infrastructure.ml.rag.retrieval_pipeline.run_hybrid_search", AsyncMock(side_effect=search)),
    ):
        _, sub_queries = await retrieve_with_decomposition("query", work, _make_ml_clients())

    assert sub_queries[:2] == ["part one", "part two"]
    assert ("query", 10, 0.7, 0.3) not in work.full_corpus_searches
    assert ("part one", 10, 0.7, 0.3) in work.full_corpus_searches

    docs, fallback_search = await _rerank_with_fallback(work, [(found, 0.9)])

    fallback_search.assert_awaited_once()
    fallback_call = fallback_search.await_args
    assert fallback_call is not None
    assert fallback_call.args[0] == "query"
    assert fallback_call.args[2] is af
    assert [doc for doc, _ in docs] == [found]


@pytest.mark.parametrize(
    "candidate, expected",
    [
        ({"chunk_index": 1, "page": 4}, True),
        ({"chunk_index": 3, "pages": [5]}, True),
        ({"chunk_index": 3, "page_start": 5, "page_end": 6}, True),
        ({"chunk_index": 2}, False),
        ({"chunk_index": 4}, False),
        ({"document_id": 8}, False),
        ({"act_version_id": 10}, False),
        ({"page": 9}, False),
        ({"section": "Another section"}, False),
        ({"content_type": "table"}, False),
        ({"parent_units": [{"content": "Different conditions"}]}, False),
    ],
)
def test_plain_text_run_keeps_document_version_position_and_pages(
    candidate: dict[str, object],
    expected: bool,
) -> None:
    anchor = Document(
        page_content="anchor", metadata={"document_id": 7, "act_version_id": 9, "chunk_index": 2, "page": 4}
    )
    other = Document(page_content="neighbor", metadata={**anchor.metadata, "chunk_index": 3, **candidate})
    assert same_text_run(anchor, other) is expected


@pytest.mark.parametrize("enumerate_cases", [False, True])
@pytest.mark.asyncio
async def test_general_expansion_accepts_only_immediate_same_document_neighbors(
    enumerate_cases: bool,
) -> None:
    user = UserContext(user_id=12, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
    anchor = Document(
        page_content="anchor", metadata={"document_id": 7, "act_version_id": 9, "chunk_index": 2, "page": 4}
    )
    neighbors = [
        ChunkRecord(
            chunk_id=1,
            document_id=7,
            filename="guide.pdf",
            content="permitted neighbor",
            chunk_index=3,
            act_version_id=9,
            context_metadata={"page": 5},
        ),
        ChunkRecord(
            chunk_id=2,
            document_id=8,
            filename="other.pdf",
            content="foreign neighbor",
            chunk_index=3,
            act_version_id=9,
            context_metadata={"page": 5},
        ),
        ChunkRecord(
            chunk_id=3,
            document_id=7,
            filename="guide.pdf",
            content="other edition",
            chunk_index=1,
            act_version_id=8,
            context_metadata={"page": 4},
        ),
        ChunkRecord(
            chunk_id=4,
            document_id=7,
            filename="guide.pdf",
            content="distant page",
            chunk_index=1,
            act_version_id=9,
            context_metadata={"page": 9},
        ),
    ]
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=neighbors))
    enriched = await enrich_with_neighbors(
        [(anchor, 1.0)], enumerate_cases=enumerate_cases, chunk_search=search, user=user
    )
    assert [doc.page_content for doc, _ in enriched] == ["anchor", "permitted neighbor"]
    assert enriched[1][1] == 0.9
    assert search.get_neighbors.call_args.kwargs["window"] == 1
    assert search.get_neighbors.call_args.kwargs["user"] is user


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "visibility, role, viewer, expected",
    [
        ("internal_private", UserRole.USER, 12, 2),
        ("internal_private", UserRole.USER, 99, 0),
        ("client_private", UserRole.ADMIN, 99, 0),
        ("internal_public", UserRole.USER, 99, 2),
    ],
)
async def test_neighbor_sql_applies_search_acl_and_document_scope(
    visibility: str,
    role: UserRole,
    viewer: int,
    expected: int,
) -> None:
    # Execute the actual ORM query locally. Copy only storage types to SQLite;
    # authorization and the mapped query remain the production implementations.
    engine = create_engine("sqlite://")
    schema = MetaData()
    Table(
        "chunks",
        schema,
        *(
            Column(
                column.name,
                JSON() if isinstance(column.type, JSONB) else column.type,
                primary_key=column.primary_key,
            )
            for column in ChunkModel.__table__.columns
        ),
    )
    schema.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                ChunkModel(
                    id=index,
                    document_id=7,
                    chunk_index=index,
                    filename="guide.pdf",
                    content=f"neighbor {index}",
                    visibility=visibility,
                    owner_id=12,
                    act_version_id=9,
                    effective_from=date(2025, 1, 1),
                )
                for index in (1, 3)
            ]
            + [
                ChunkModel(
                    id=4,
                    document_id=8,
                    chunk_index=1,
                    filename="foreign.pdf",
                    content="foreign",
                    visibility="internal_public",
                    owner_id=None,
                )
            ]
        )
        session.flush()
        async_session = SimpleNamespace(execute=AsyncMock(side_effect=session.execute))
        repository = SQLAlchemyChunkContextRepository(cast("AsyncSession", async_session))
        user = UserContext(user_id=viewer, user_kind=UserKind.INTERNAL, user_role=role)
        results = await repository.get_neighbors(7, 2, user=user)
        assert len(results) == expected
        assert all(row.document_id == 7 for row in results)
    engine.dispose()


# ---------------------------------------------------------------------------
# Parent-child expansion (6.4): a new read path over document_nodes + chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "visibility, role, viewer, allowed",
    [
        ("internal_private", UserRole.USER, OWNER_ID, True),
        ("internal_private", UserRole.USER, 99, False),
        # Listing may show CLIENT_PRIVATE to ADMIN; RAG search scope must not.
        ("client_private", UserRole.ADMIN, 99, False),
        ("internal_public", UserRole.USER, 99, True),
    ],
)
async def test_parent_scope_sql_applies_search_acl(
    visibility: str, role: UserRole, viewer: int, allowed: bool
) -> None:
    user = UserContext(user_id=viewer, user_kind=UserKind.INTERNAL, user_role=role)

    with parent_scope_repository(visibility) as repository:
        results = await repository.get_parent_context(
            DOCUMENT_ID, POINT, user=user, act_version_id=ACT_VERSION_ID, max_chars=70
        )

    if allowed:
        assert [row.chunk_index for row in results] == [0, 1, 2]
        assert all(row.document_id == DOCUMENT_ID for row in results)
    else:
        assert results == []


@pytest.mark.asyncio
async def test_parent_scope_sql_climbs_while_the_ancestor_fits() -> None:
    user = UserContext(user_id=OWNER_ID, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)

    with parent_scope_repository("internal_private") as repository:

        async def indexes(node_id: str, max_chars: int) -> list[int]:
            rows = await repository.get_parent_context(
                DOCUMENT_ID, node_id, user=user, act_version_id=ACT_VERSION_ID, max_chars=max_chars
            )
            return [row.chunk_index for row in rows]

        # Real body sizes: point 26, article 61, chapter 84 characters.
        assert await indexes(POINT, 30) == [2]
        assert await indexes(POINT, 70) == [0, 1, 2]
        assert await indexes(POINT, 100) == [0, 1, 2, 3]
        # The nearest parent does not fit: callers keep the bounded neighbour heuristics.
        assert await indexes(POINT, 20) == []
        assert await indexes(SIBLING_ARTICLE, 30) == [3]


@pytest.mark.asyncio
async def test_parent_scope_sql_measures_current_bodies_not_stored_size() -> None:
    # Manual edits change bodies without touching document_nodes.char_count.
    user = UserContext(user_id=OWNER_ID, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
    stale = {ARTICLE: 10_000, CHAPTER: 1}
    nodes = tuple((*node[:6], stale.get(node[0], node[6])) for node in NODES)

    with parent_scope_repository("internal_private", nodes=nodes) as repository:
        results = await repository.get_parent_context(
            DOCUMENT_ID, POINT, user=user, act_version_id=ACT_VERSION_ID, max_chars=70
        )

    assert [row.chunk_index for row in results] == [0, 1, 2]


@pytest.mark.asyncio
async def test_parent_scope_sql_never_returns_a_truncated_parent() -> None:
    # An article spanning more chunks than one read allows is skipped, not cut to its start.
    user = UserContext(user_id=OWNER_ID, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
    nodes = tuple(
        (*node[:5], MAX_PARENT_CHUNKS + 11, node[6]) if node[0] in {ARTICLE, CHAPTER} else node
        for node in NODES
    )

    with parent_scope_repository("internal_private", nodes=nodes) as repository:
        nearest = await repository.get_parent_context(
            DOCUMENT_ID, POINT, user=user, act_version_id=ACT_VERSION_ID, max_chars=1000
        )
        oversized = await repository.get_parent_context(
            DOCUMENT_ID, ARTICLE, user=user, act_version_id=ACT_VERSION_ID, max_chars=1000
        )

    assert [row.chunk_index for row in nearest] == [2]
    assert oversized == []


@pytest.mark.asyncio
async def test_parent_scope_sql_stays_in_document_and_edition() -> None:
    user = UserContext(user_id=OWNER_ID, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)

    with parent_scope_repository("internal_private") as repository:
        other_edition = await repository.get_parent_context(
            DOCUMENT_ID, ARTICLE, user=user, act_version_id=ACT_VERSION_ID + 1, max_chars=1000
        )
        foreign = await repository.get_parent_context(
            FOREIGN_DOCUMENT_ID, ARTICLE, user=user, act_version_id=None, max_chars=1000
        )

    assert other_edition == []
    assert [row.document_id for row in foreign] == [FOREIGN_DOCUMENT_ID]


@pytest.mark.asyncio
async def test_parent_scope_sql_stops_at_a_missing_ancestor() -> None:
    user = UserContext(user_id=OWNER_ID, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
    nodes = tuple(node for node in NODES if node[0] != ARTICLE)

    with parent_scope_repository("internal_private", nodes=nodes) as repository:
        results = await repository.get_parent_context(
            DOCUMENT_ID, POINT, user=user, act_version_id=ACT_VERSION_ID, max_chars=1000
        )

    assert [row.chunk_index for row in results] == [2]


@pytest.mark.asyncio
async def test_parent_expansion_passes_request_user_and_keeps_order_near_anchor() -> None:
    user = UserContext(user_id=OWNER_ID, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
    anchor = Document(
        page_content="а) подпункт второго пункта",
        metadata={
            "document_id": DOCUMENT_ID,
            "chunk_index": 2,
            "act_version_id": ACT_VERSION_ID,
            "section": "Статья 1 > 2.",
            "parent_node_id": POINT,
        },
    )
    records = [
        ChunkRecord(index + 1, DOCUMENT_ID, "law.docx", content, index, act_version_id=ACT_VERSION_ID)
        for index, _, content in CHUNKS[:4]
    ]
    search = SimpleNamespace(
        get_parent_context=AsyncMock(return_value=records),
        get_neighbors=AsyncMock(return_value=[]),
    )

    result = await enrich_with_neighbors(
        [(anchor, 0.9)], enumerate_cases=False, chunk_search=search, user=user, max_context_tokens=6000
    )

    kwargs = search.get_parent_context.call_args.kwargs
    assert kwargs["user"] is user
    assert kwargs["act_version_id"] == ACT_VERSION_ID
    assert kwargs["max_chars"] == 4500
    search.get_neighbors.assert_not_called()
    assert [doc.metadata["chunk_index"] for doc, _ in result] == [2, 1, 3, 0]


@pytest.mark.asyncio
async def test_parent_expansion_falls_back_to_neighbors_without_a_fitting_parent() -> None:
    user = UserContext(user_id=OWNER_ID, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
    anchor = Document(
        page_content="body",
        metadata={"document_id": DOCUMENT_ID, "chunk_index": 2, "section": "S", "parent_node_id": POINT},
    )
    search = SimpleNamespace(
        get_parent_context=AsyncMock(return_value=[]),
        get_neighbors=AsyncMock(return_value=[]),
    )

    await enrich_with_neighbors([(anchor, 0.9)], enumerate_cases=False, chunk_search=search, user=user)

    assert search.get_neighbors.call_args.kwargs["user"] is user
