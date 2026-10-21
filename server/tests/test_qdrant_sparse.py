"""Incremental sparse search shares point identity and complete ACL scope with dense."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from langchain.schema import Document
import pytest
from qdrant_client import QdrantClient, models

from config import settings
from domain.identity.roles import UserKind
from domain.knowledge.access_control import get_visibility_conditions
from infrastructure.bm25.stemmer import stem_token
from infrastructure.bm25.tokenizer import tokenize, tokenize_raw
from infrastructure.ml.rag import rag_retrieval
from infrastructure.repositories.vector.acl import build_qdrant_filter, with_temporal_filter
from infrastructure.repositories.vector.qdrant_vector_store_repository import QdrantVectorStoreRepository
from infrastructure.repositories.vector.sparse_vectors import (
    SPARSE_VECTOR_NAME,
    encode_sparse,
    point_vectors,
    sparse_query,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture
def sparse_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[QdrantClient]:
    client = QdrantClient(":memory:")
    monkeypatch.setattr(settings, "collection_name", "sparse_tests")
    client.create_collection(
        settings.collection_name,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    yield client
    client.close()


def point(
    chunk_id: int,
    document_id: int,
    *,
    visibility: str = "internal_public",
    owner_id: int | None = None,
    group_id: int | None = None,
    is_current: bool = True,
    text: str = "порядок маркировки товаров",
) -> models.PointStruct:
    return models.PointStruct(
        id=chunk_id,
        vector=point_vectors(text, [1.0, 0.0]),
        payload={
            "page_content": text,
            "metadata": {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "visibility": visibility,
                "owner_id": owner_id,
                "group_id": group_id,
                "is_current": is_current,
                "source": "old.pdf",
                "act_version_id": 17,
                "heading": "Порядок",
            },
        },
    )


def test_sparse_encoder_preserves_russian_stems_and_numbers() -> None:
    document = encode_sparse("маркировки товаров 2026")
    query = sparse_query("маркировка 2026")
    assert set(query.vector.indices) <= set(document.indices)
    assert len(document.indices) == len(set(document.indices))
    assert all(value > 0 for value in document.values)
    assert encode_sparse("a !").indices == []
    assert encode_sparse("маркировки товаров 2026") == document


@pytest.mark.parametrize(
    "kind,role,uid,groups,expected",
    [
        ("internal", "user", 5, [7], {1, 2, 4}),
        ("internal", "user", 6, [], {1, 3}),
        ("internal", "admin", 5, [], {1, 2, 3, 4}),
        ("client", "user", 5, [], {5}),
    ],
)
def test_sparse_filters_before_top_k(
    sparse_client: QdrantClient, kind: str, role: str, uid: int, groups: list[int], *, expected: set[int]
) -> None:
    sparse_client.upsert(
        settings.collection_name,
        [
            point(1, 10),
            point(2, 20, visibility="internal_private", owner_id=5),
            point(3, 30, visibility="internal_private", owner_id=6),
            point(4, 40, visibility="internal_group", group_id=7),
            point(5, 50, visibility="client_private", owner_id=5),
            point(6, 60, is_current=False),
        ],
    )
    scope = with_temporal_filter(build_qdrant_filter({"id": uid, "kind": kind, "role": role}, groups), None)
    hits = rag_retrieval.search_sparse_points(sparse_client, "маркировка", 20, scope)
    assert {hit.id for hit in hits} == expected
    assert all((hit.payload or {})["metadata"]["is_current"] for hit in hits)
    assert len(rag_retrieval.search_sparse_points(sparse_client, "маркировка", 1, scope)) == 1


def test_delete_preserves_same_text_in_other_document_and_retry_is_safe(sparse_client: QdrantClient) -> None:
    sparse_client.upsert(settings.collection_name, [point(1, 10), point(2, 20)])
    selector = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchValue(value=10),
            )
        ]
    )

    for _ in range(2):
        sparse_client.delete(settings.collection_name, selector, wait=True)

    sparse = sparse_client.search(settings.collection_name, sparse_query("маркировка"))
    dense = sparse_client.search(settings.collection_name, [1.0, 0.0])
    assert [hit.id for hit in sparse] == [2]
    assert [hit.id for hit in dense] == [2]


def test_edit_replaces_sparse_terms_without_rebuilding(sparse_client: QdrantClient) -> None:
    sparse_client.upsert(settings.collection_name, [point(1, 10), point(2, 20)])
    sparse_client.upsert(settings.collection_name, [point(1, 10, text="налоговый вычет")])
    assert [p.id for p in sparse_client.search(settings.collection_name, sparse_query("маркировка"))] == [2]
    assert [p.id for p in sparse_client.search(settings.collection_name, sparse_query("налоговый"))] == [1]


@pytest.mark.asyncio
async def test_rename_patches_payload_without_replacing_vectors(sparse_client: QdrantClient) -> None:
    sparse_client.upsert(settings.collection_name, [point(1, 10), point(2, 20)])
    before = sparse_client.retrieve(settings.collection_name, [1], with_vectors=True)[0]
    repository = QdrantVectorStoreRepository(SimpleNamespace(qdrant_client=lambda: sparse_client))
    await repository.update_filename_by_document_id(10, "new.pdf")
    after = sparse_client.retrieve(settings.collection_name, [1], with_vectors=True)[0]
    assert after.vector == before.vector
    assert (after.payload or {})["metadata"] == {
        **(before.payload or {})["metadata"],
        "filename": "new.pdf",
        "source": "new.pdf",
    }
    assert (sparse_client.retrieve(settings.collection_name, [2])[0].payload or {})["metadata"][
        "source"
    ] == "old.pdf"


@pytest.mark.asyncio
async def test_hybrid_uses_point_ids_and_sparse_payload_without_hash_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_doc = Document(page_content="dense", metadata={"chunk_id": 1})
    sparse_doc = Document(page_content="sparse", metadata={"chunk_id": 2})
    dense = AsyncMock(return_value=[("same-hash", 0.9, dense_doc)])
    sparse = AsyncMock(return_value=[("2", 1.2, sparse_doc)])
    monkeypatch.setattr(rag_retrieval, "qdrant_dense_search", dense)
    monkeypatch.setattr(rag_retrieval, "qdrant_sparse_search", sparse)
    resolve = AsyncMock(side_effect=AssertionError("hash resolution is obsolete"))
    monkeypatch.setattr(rag_retrieval, "resolve_hashes_batch", resolve)
    rag = MagicMock(
        hybrid_search=SimpleNamespace(
            enabled=True, bm25_fetch_k=3, rrf_k=60, dense_weight=1.0, sparse_weight=1.0
        )
    )
    scope = build_qdrant_filter({"id": 5, "kind": "internal", "role": "user"}, [])
    docs = await rag_retrieval.run_hybrid_search(
        "query",
        10,
        scope,
        rag,
        MagicMock(),
        visibility_conditions=get_visibility_conditions(UserKind.INTERNAL, [], for_list=False),
        user_id=5,
        user_group_ids=[],
    )
    assert {doc.metadata["chunk_id"] for doc in docs} == {1, 2}
    assert dense.call_args.args[1] == 10
    assert sparse.call_args.args[1] == 3
    assert dense.call_args.args[2] is scope
    assert sparse.call_args.args[2] is scope
    resolve.assert_not_awaited()


def test_sparse_refuses_missing_scope(sparse_client: QdrantClient) -> None:
    with pytest.raises(RuntimeError, match="ACL"):
        rag_retrieval.search_sparse_points(sparse_client, "маркировка", 10, None)


@pytest.mark.parametrize("tokenizer", [tokenize, tokenize_raw])
def test_tokenizers_keep_numbers_and_normalize_yo(tokenizer: Callable[[str], list[str]]) -> None:
    assert tokenizer("я ю a x 1 2 5.1 14.04.2026 учёт") == [
        "1",
        "2",
        "5",
        "1",
        "14",
        "04",
        "2026",
        "учет",
    ]


@pytest.mark.parametrize(
    "base,inflected",
    [
        ("документ", "документами"),
        ("поставщик", "поставщика"),
        ("учет", "учёт"),
        ("маркировка", "маркировки"),
    ],
)
def test_sparse_word_forms_match(base: str, inflected: str) -> None:
    assert tokenize(base) == tokenize(inflected)
    assert stem_token(base) == stem_token(inflected)
    assert encode_sparse(base).indices == encode_sparse(inflected).indices


@pytest.mark.parametrize("left,right", [("пункт 1", "пункт 2"), ("статья 5.1", "статья 5.2")])
def test_sparse_distinguishes_numbers(left: str, right: str) -> None:
    assert encode_sparse(left).indices != encode_sparse(right).indices


@pytest.mark.asyncio
async def test_hybrid_limits_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    dense = AsyncMock(
        return_value=[
            (str(i), 1.0, Document(page_content=str(i), metadata={"chunk_id": i})) for i in range(5)
        ]
    )
    sparse = AsyncMock(return_value=[])
    monkeypatch.setattr(rag_retrieval, "qdrant_dense_search", dense)
    monkeypatch.setattr(rag_retrieval, "qdrant_sparse_search", sparse)
    rag = SimpleNamespace(
        hybrid_search=SimpleNamespace(
            enabled=True,
            bm25_fetch_k=7,
            rrf_k=60,
            dense_weight=1.0,
            sparse_weight=1.0,
        )
    )
    scope, clients = object(), MagicMock()
    kwargs = {"visibility_conditions": [object()], "user_id": 5, "user_group_ids": []}
    docs = await rag_retrieval.run_hybrid_search("query", 2, scope, rag, clients, **kwargs)
    assert len(docs) == 2
    dense.assert_awaited_once_with("query", 2, scope, clients)
    sparse.assert_awaited_once_with("query", 7, scope, clients)
    sparse.side_effect = ValueError("sparse failed")
    with pytest.raises(ValueError, match="sparse failed"):
        await rag_retrieval.run_hybrid_search("query", 2, scope, rag, clients, **kwargs)
