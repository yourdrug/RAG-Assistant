"""Reranker cache reuse, failure handling and sparse-candidate ACL contracts."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.schema import Document
from qdrant_client.models import Filter

from domain.value_objects.visibility import DocumentVisibility
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.benchmark.sweep_data import SweepDataSource
from infrastructure.benchmark.sweep_reranking import cache_reranker_scores
from infrastructure.benchmark.sweep_scoring import cache_candidates
from infrastructure.benchmark.sweep_strategies import SweepCancelled


@pytest.mark.asyncio
async def test_sweep_candidates_forward_acl_to_real_bm25():
    visibilities = list(DocumentVisibility)
    hashes = [visibility.value for visibility in visibilities]
    bm25 = BM25Index(
        texts=["benchmark evidence"] * len(visibilities),
        hashes=hashes,
        doc_visibility=hashes,
        doc_owner_id=[99] * len(visibilities),
        doc_group_id=[99] * len(visibilities),
    )
    allowed = set(visibilities) - {DocumentVisibility.CLIENT_PRIVATE}
    points = [
        SimpleNamespace(
            score=1.0,
            payload={
                "page_content": "benchmark evidence",
                "metadata": {"content_hash": visibility.value, "visibility": visibility.value},
            },
        )
        for visibility in allowed
    ]
    client = SimpleNamespace(search=MagicMock(return_value=points))
    clients = SimpleNamespace(
        qdrant_client=lambda: client,
        embeddings=lambda: SimpleNamespace(embed_query_sync=lambda query: [0.1]),
        _ensure_bm25_loaded=AsyncMock(return_value=bm25),
    )
    data = SweepDataSource(None, clients)

    _, sparse, _ = await data.cache_candidates([{"question": "benchmark evidence"}], 10)

    assert {hash_value for hash_value, _ in sparse["benchmark evidence"]} == {
        visibility.value for visibility in allowed
    }
    assert client.search.call_args.kwargs["query_filter"] is not None


@pytest.mark.asyncio
async def test_cache_scores_each_query_once_and_keeps_query_specific_scores():
    doc = Document(page_content="text", metadata={"source": "law.pdf", "heading": "Article 17"})
    predict = AsyncMock(side_effect=[[0.4], [0.9]])
    scores = await cache_reranker_scores(
        [{"question": "q1"}, {"question": "q1"}, {"question": "q2"}],
        {"q1": [("hash", 1, doc)], "q2": []},
        {"q1": [("hash", 1)], "q2": [("hash", 1)]},
        {"hash": doc},
        SimpleNamespace(predict=predict),
    )
    assert scores == {"q1": {"hash": 0.4}, "q2": {"hash": 0.9}}
    assert predict.await_count == 2
    assert predict.call_args_list[0].args[0] == [("q1", "[law.pdf] (Article 17) text")]
    assert predict.call_args_list[1].args[0] == [("q2", "[law.pdf] (Article 17) text")]


@pytest.mark.asyncio
async def test_cache_can_be_cancelled_between_queries():
    doc = Document(page_content="text")
    predict = AsyncMock(return_value=[0.4])
    cancel = AsyncMock(side_effect=[False, True])
    with pytest.raises(SweepCancelled):
        await cache_reranker_scores(
            [{"question": "q1"}, {"question": "q2"}],
            {"q1": [("hash", 1, doc)], "q2": [("hash", 1, doc)]},
            {},
            {"hash": doc},
            SimpleNamespace(predict=predict),
            cancel,
        )
    predict.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("reranker unavailable"), asyncio.CancelledError()])
async def test_owned_reranker_is_closed_and_failures_propagate(monkeypatch, failure):
    from infrastructure.ml.clients import factories

    reranker = SimpleNamespace(predict=AsyncMock(side_effect=failure), close=AsyncMock())
    monkeypatch.setattr(factories, "create_reranker", lambda: reranker)
    doc = Document(page_content="text")
    data = SweepDataSource(None, None)
    with pytest.raises(type(failure)):
        await data.cache_reranker_scores(
            [{"question": "q"}], {"q": [("hash", 1, doc)]}, {}, {"hash": doc}, None
        )
    reranker.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("injected_clients", [False, True])
async def test_sparse_only_chunks_are_resolved_with_acl_before_reranking(monkeypatch, injected_clients):
    from infrastructure.ml.clients import factories

    dense_point = SimpleNamespace(
        score=0.9,
        payload={"page_content": "article 29", "metadata": {"content_hash": "dense", "source": "law.pdf"}},
    )
    sparse_point = SimpleNamespace(
        payload={"page_content": "article 17", "metadata": {"content_hash": "sparse", "source": "law.pdf"}}
    )
    client = SimpleNamespace(
        search=MagicMock(return_value=[dense_point]), scroll=MagicMock(return_value=([sparse_point], None))
    )
    embeddings = SimpleNamespace(embed_query_sync=lambda query: [0.1])
    bm25 = SimpleNamespace(search_with_hashes=MagicMock(return_value=[("sparse", 1)]))
    acl = Filter(must_not=[])
    clients = None
    if injected_clients:
        clients = SimpleNamespace(
            qdrant_client=lambda: client,
            embeddings=lambda: embeddings,
            _ensure_bm25_loaded=AsyncMock(return_value=bm25),
            qdrant_search_semaphore=asyncio.Semaphore(1),
        )
    else:
        monkeypatch.setattr(factories, "create_qdrant_client", lambda: client)
        monkeypatch.setattr(factories, "create_embeddings", lambda: embeddings)
        monkeypatch.setattr(factories, "load_bm25_index", lambda: bm25)
    dense, sparse, candidates = await cache_candidates(
        [{"question": "q"}],
        2,
        ml_clients=clients,
        access_filter=acl,
    )
    assert client.search.call_args.kwargs["query_filter"] is acl
    assert acl in client.scroll.call_args.kwargs["scroll_filter"].must
    assert candidates["sparse"].page_content == "article 17"
    predict = AsyncMock(return_value=[0.1, 0.9])
    scores = await cache_reranker_scores(
        [{"question": "q"}], dense, sparse, candidates, SimpleNamespace(predict=predict)
    )
    assert scores == {"q": {"dense": 0.1, "sparse": 0.9}}


@pytest.mark.asyncio
async def test_sparse_resolution_without_acl_fails_closed(monkeypatch):
    from infrastructure.ml.clients import factories

    client = SimpleNamespace(search=MagicMock(return_value=[]), scroll=MagicMock())
    monkeypatch.setattr(factories, "create_qdrant_client", lambda: client)
    monkeypatch.setattr(
        factories, "create_embeddings", lambda: SimpleNamespace(embed_query_sync=lambda q: [])
    )
    monkeypatch.setattr(
        factories,
        "load_bm25_index",
        lambda: SimpleNamespace(search_with_hashes=lambda *a, **k: [("hash", 1)]),
    )
    with pytest.raises(RuntimeError, match="ACL filter"):
        await cache_candidates([{"question": "q"}], 2)
    client.scroll.assert_not_called()
