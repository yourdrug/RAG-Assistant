"""Published document revision: re-processing never removes a document from search.

A re-processed document is written as a staged revision whose points stay hidden
until every one is indexed; then it replaces the published revision. Unchanged
chunks reuse their dense vectors instead of paying for new embeddings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

from langchain.schema import Document as LCDocument
import pytest
from qdrant_client import QdrantClient, models

from application.indexing import process_chunks
from config import MLProviderSection, VectorStoreSection, process_section
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.document import Document
from domain.knowledge.enums import DocDomain, DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import OutboxOperation
from domain.knowledge.revision import (
    EMBED_HASH_KEY,
    REUSE_POINT_KEY,
    STAGED_KEY,
    STAGED_REVISION_KEY,
    StagedRevision,
    match_reusable_chunks,
)
from domain.shared.exceptions import BusinessRuleViolationError
from domain.shared.hashing import content_hash
from infrastructure.repositories.vector import qdrant_ops
from infrastructure.repositories.vector.acl import build_qdrant_filter
from infrastructure.repositories.vector.outbox_dispatcher import (
    STAGED_UPDATE_FAILED_WARNING,
    OutboxDispatcher,
)
from infrastructure.repositories.vector.qdrant_vector_store_repository import QdrantVectorStoreRepository
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME
from tests.fakes import FakeUnitOfWorkFactory
from tests.test_chunk_service import _make_service, _make_uow


if TYPE_CHECKING:
    from domain.knowledge.chunk import Chunk
    from domain.knowledge.index_sync import VectorIndexWriter


class RecordingIndex:
    """Vector index double: points by id with their content and staged flag."""

    def __init__(self, *, fail_uploads: bool = False) -> None:
        self.points: dict[int, dict[str, Any]] = {}
        self.fail_uploads = fail_uploads
        self.ensure_collection = AsyncMock()
        self.update_filename_by_document_id = AsyncMock()
        self.set_pack_metadata_by_document_id = AsyncMock()

    async def upload_documents(self, chunks: list[Chunk], **kwargs: object) -> None:
        del kwargs

        if self.fail_uploads:
            message = "embedding provider unavailable"
            raise ConnectionError(message)

        for chunk in chunks:
            self.points[chunk.metadata["chunk_id"]] = {
                "content": chunk.content,
                STAGED_KEY: bool(chunk.metadata.get(STAGED_KEY)),
            }

    async def delete_by_ids(self, ids: list[int]) -> None:
        for point_id in ids:
            self.points.pop(point_id, None)

    async def reveal_points(self, ids: list[int]) -> None:
        for point_id in ids:
            if point_id in self.points:
                self.points[point_id][STAGED_KEY] = False

    def searchable(self) -> dict[int, str]:
        return {pid: point["content"] for pid, point in self.points.items() if not point[STAGED_KEY]}


async def index(factory: FakeUnitOfWorkFactory, *texts: str) -> None:
    await process_chunks(
        factory,
        1,
        "report.txt",
        [RawDocument(page_content=text, metadata={"document_id": 1}) for text in texts],
        DocumentVisibility.INTERNAL_PUBLIC.value,
        owner_id=None,
        group_id=None,
        doc_domain=DocDomain.GENERAL.value,
    )


async def dispatch_pending(factory: FakeUnitOfWorkFactory, dispatcher: OutboxDispatcher) -> list[str]:
    operations = []

    for entry in await factory._uow.vector_outbox.claim_batch("test-worker"):
        operations.append(entry.operation.value)
        await dispatcher._apply_one(entry)

    return operations


async def document_state(factory: FakeUnitOfWorkFactory) -> Document:
    document = await factory._uow.documents.get_by_id(1)
    assert document is not None
    return document


async def published_texts(factory: FakeUnitOfWorkFactory) -> list[str]:
    rows, _ = await factory._uow.chunks.list_for_document(1)
    return [row.content for row in sorted(rows, key=lambda row: row.chunk_index)]


async def indexed_document(vector: RecordingIndex) -> tuple[FakeUnitOfWorkFactory, OutboxDispatcher]:
    factory = FakeUnitOfWorkFactory()
    await factory._uow.documents.save(Document(filename="report.txt", status=DocumentStatus.PROCESSING))
    await index(factory, "Статья 1. Старая.", "Статья 2. Без изменений.")
    dispatcher = OutboxDispatcher(factory, cast("VectorIndexWriter", vector), cache_invalidator=AsyncMock())
    await dispatch_pending(factory, dispatcher)
    assert vector.searchable() == {1: "Статья 1. Старая.", 2: "Статья 2. Без изменений."}
    return factory, dispatcher


@pytest.mark.asyncio
async def test_document_stays_searchable_until_staged_revision_is_published() -> None:
    vector = RecordingIndex()
    factory, dispatcher = await indexed_document(vector)
    await index(factory, "Статья 1. Новая.", "Статья 2. Без изменений.")

    # Committed but unpublished: SQL and the index still serve the previous text.
    assert await published_texts(factory) == ["Статья 1. Старая.", "Статья 2. Без изменений."]
    upsert = next(
        entry
        for entry in factory._uow.fake_vector_outbox._entries.values()
        if entry.payload.get(STAGED_REVISION_KEY) is not None
    )
    assert [point.get(REUSE_POINT_KEY) for point in upsert.payload["points"]] == [None, 2]
    assert not [
        entry
        for entry in factory._uow.fake_vector_outbox._entries.values()
        if entry.operation == OutboxOperation.DELETE_CHUNKS
    ]

    assert await dispatch_pending(factory, dispatcher) == [OutboxOperation.UPSERT_CHUNKS.value]
    # Every staged point is indexed and the revision is published in SQL; its points
    # are still hidden and the old ones still searchable until the publish entry runs.
    assert await published_texts(factory) == ["Статья 1. Новая.", "Статья 2. Без изменений."]
    assert vector.searchable() == {1: "Статья 1. Старая.", 2: "Статья 2. Без изменений."}
    assert {pid for pid, point in vector.points.items() if point[STAGED_KEY]} == {3, 4}
    assert (await document_state(factory)).status == DocumentStatus.INDEXING

    assert await dispatch_pending(factory, dispatcher) == [OutboxOperation.PUBLISH_REVISION.value]
    assert vector.searchable() == {3: "Статья 1. Новая.", 4: "Статья 2. Без изменений."}
    assert (await document_state(factory)).status == DocumentStatus.DONE
    assert await factory._uow.chunks.get_staged_revision(1) is None
    # Answers that cited the old revision are invalidated on publication.
    invalidator = cast("AsyncMock", dispatcher._cache_invalidator)
    invalidator.invalidate_by_document_ids.assert_awaited_with([1], raise_on_error=True)


@pytest.mark.asyncio
async def test_superseded_staged_revision_is_never_published() -> None:
    vector = RecordingIndex()
    factory, dispatcher = await indexed_document(vector)

    await index(factory, "Версия A.")
    await index(factory, "Версия B.")
    # The newer run discarded revision A and queued removal of its points.
    staged = await factory._uow.chunks.get_staged_revision(1)
    assert staged is not None
    assert staged.chunk_ids == (4,)

    while await dispatch_pending(factory, dispatcher):
        pass

    assert await published_texts(factory) == ["Версия B."]
    assert vector.searchable() == {4: "Версия B."}
    assert 3 not in vector.points


@pytest.mark.asyncio
async def test_failed_upload_keeps_previous_revision_and_marks_update_error() -> None:
    vector = RecordingIndex()
    factory, dispatcher = await indexed_document(vector)
    await index(factory, "Статья 1. Новая.")
    vector.fail_uploads = True

    for _ in range(8):
        await dispatch_pending(factory, dispatcher)
        assert vector.searchable() == {1: "Статья 1. Старая.", 2: "Статья 2. Без изменений."}

    vector.fail_uploads = False

    while await dispatch_pending(factory, dispatcher):
        pass

    assert await published_texts(factory) == ["Статья 1. Старая.", "Статья 2. Без изменений."]
    assert vector.searchable() == {1: "Статья 1. Старая.", 2: "Статья 2. Без изменений."}
    assert await factory._uow.chunks.get_staged_revision(1) is None
    document = await document_state(factory)
    assert (document.status, document.chunks, document.warning_message) == (
        DocumentStatus.DONE,
        2,
        STAGED_UPDATE_FAILED_WARNING,
    )


@pytest.mark.asyncio
async def test_acl_change_replaces_chunks_without_staging() -> None:
    """Narrowed access applies to the old content at once, not after publication."""
    vector = RecordingIndex()
    factory, _dispatcher = await indexed_document(vector)
    await process_chunks(
        factory,
        1,
        "report.txt",
        [RawDocument(page_content="Закрытый текст.", metadata={"document_id": 1})],
        DocumentVisibility.INTERNAL_PRIVATE.value,
        owner_id=7,
        group_id=None,
        doc_domain=DocDomain.GENERAL.value,
    )
    assert await factory._uow.chunks.get_staged_revision(1) is None
    rows, _ = await factory._uow.chunks.list_for_document(1)
    assert [(row.content, row.visibility, row.owner_id) for row in rows] == [
        ("Закрытый текст.", DocumentVisibility.INTERNAL_PRIVATE.value, 7)
    ]
    deletes = [
        entry.payload["chunk_ids"]
        for entry in factory._uow.fake_vector_outbox._entries.values()
        if entry.operation == OutboxOperation.DELETE_CHUNKS
    ]
    assert deletes == [[1, 2]]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["edit", "add"])
async def test_chunk_edits_wait_for_staged_revision(operation: str) -> None:
    uow = _make_uow()
    service = _make_service(uow)
    content = "Ручная правка текста документа для проверки ограничения. " * 4

    async def run() -> object:
        if operation == "edit":
            return await service.edit_chunk(1, 10, content, 1, UserRole.ADMIN)

        return await service.add_chunk(1, content, 1, UserRole.ADMIN)

    uow.chunks.get_staged_revision = AsyncMock(return_value=StagedRevision(1, "token", (30,)))
    with pytest.raises(BusinessRuleViolationError, match="re-indexed"):
        await run()

    uow.chunks.update_content.assert_not_awaited()
    uow.chunks.insert_one.assert_not_awaited()

    uow.chunks.get_staged_revision = AsyncMock(return_value=None)
    await run()


def test_reuse_pairs_repeated_inputs_by_occurrence() -> None:
    published = [(1, "a"), (2, "b"), (3, "a")]
    assert match_reusable_chunks(published, ["a", "a", "a", "c", "b"]) == [1, 3, None, None, 2]


QDRANT_POINTS = {1: None, 2: True, 3: False}


@pytest.fixture
def staged_collection() -> QdrantClient:
    client = QdrantClient(":memory:")
    name = process_section(VectorStoreSection).collection_name
    client.create_collection(
        name, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    client.upsert(
        name,
        [
            models.PointStruct(
                id=point_id,
                vector=[1.0, 0.0],
                payload={
                    "metadata": {
                        "visibility": DocumentVisibility.INTERNAL_PUBLIC.value,
                        **({STAGED_KEY: staged} if staged is not None else {}),
                    }
                },
            )
            for point_id, staged in QDRANT_POINTS.items()
        ],
    )
    return client


def scope_ids(client: QdrantClient, ctx: UserContext) -> set[int]:
    points, _ = client.scroll(
        process_section(VectorStoreSection).collection_name,
        scroll_filter=build_qdrant_filter(ctx, list(ctx.group_ids)),
        limit=10,
    )
    return {int(point.id) for point in points}


@pytest.mark.parametrize("role", list(UserRole))
def test_search_scope_hides_staged_points_for_every_role(
    staged_collection: QdrantClient, role: UserRole
) -> None:
    ctx = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=role)
    # Points written before revisions existed carry no flag and stay searchable.
    assert scope_ids(staged_collection, ctx) == {1, 3}

    QdrantVectorStoreRepository.patch_metadata(
        staged_collection, models.Filter(must=[models.HasIdCondition(has_id=[2, 99])]), {STAGED_KEY: False}
    )
    assert scope_ids(staged_collection, ctx) == {1, 2, 3}


class CountingEmbeddings:
    embedding_identity = "model-x"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        dim = process_section(MLProviderSection).embed_dim
        return [[float(len(text) % 7 + 1)] + [0.5] * (dim - 1) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class DenseCollection:
    """In-memory corpus whose embedding model the test can switch."""

    def __init__(self, client: QdrantClient) -> None:
        self.client = client
        self.identity = "model-x"


@pytest.fixture
def dense_collection(monkeypatch: pytest.MonkeyPatch) -> DenseCollection:
    client = QdrantClient(":memory:")
    client.create_collection(
        process_section(VectorStoreSection).collection_name,
        vectors_config=models.VectorParams(
            size=process_section(MLProviderSection).embed_dim, distance=models.Distance.COSINE
        ),
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    corpus = DenseCollection(client)
    monkeypatch.setattr(qdrant_ops, "ensure_embedding_identity", lambda *_args: corpus.identity)
    return corpus


def upload_docs(texts: dict[int, str], reuse: dict[int, int] | None = None) -> list[LCDocument]:
    return [
        LCDocument(
            page_content=text,
            metadata={
                "chunk_id": chunk_id,
                **({REUSE_POINT_KEY: reuse[chunk_id]} if reuse and chunk_id in reuse else {}),
            },
        )
        for chunk_id, text in texts.items()
    ]


@pytest.mark.asyncio
async def test_unchanged_chunks_reuse_dense_vectors(dense_collection: DenseCollection) -> None:
    embeddings = CountingEmbeddings()
    client = dense_collection.client
    await qdrant_ops.upload_document_batches(
        upload_docs({1: "первый абзац", 2: "второй абзац", 3: "третий абзац"}), embeddings, client
    )
    assert len(embeddings.calls) == 3
    embeddings.calls.clear()

    # One edited chunk out of three: only it is embedded again.
    await qdrant_ops.upload_document_batches(
        upload_docs(
            {11: "первый абзац", 12: "второй абзац изменён", 13: "третий абзац"}, {11: 1, 12: 2, 13: 3}
        ),
        embeddings,
        client,
    )
    assert embeddings.calls == ["второй абзац изменён"]
    records = {
        record.id: (record.payload or {}, record.vector)
        for record in client.retrieve(
            process_section(VectorStoreSection).collection_name,
            ids=[1, 11, 13],
            with_vectors=True,
            with_payload=True,
        )
    }
    vectors = {point_id: vector for point_id, (_, vector) in records.items() if isinstance(vector, dict)}
    assert vectors[11][""] == vectors[1][""]
    assert records[13][0]["metadata"][EMBED_HASH_KEY] == content_hash("третий абзац")
    assert REUSE_POINT_KEY not in records[11][0]["metadata"]
    # The sparse vector is always computed from the chunk itself.
    assert SPARSE_VECTOR_NAME in vectors[11]


@pytest.mark.asyncio
async def test_vectors_of_another_model_are_not_reused(dense_collection: DenseCollection) -> None:
    embeddings = CountingEmbeddings()
    client = dense_collection.client
    await qdrant_ops.upload_document_batches(upload_docs({1: "первый абзац"}), embeddings, client)
    dense_collection.identity = "model-y"
    embeddings.calls.clear()
    await qdrant_ops.upload_document_batches(upload_docs({11: "первый абзац"}, {11: 1}), embeddings, client)
    assert embeddings.calls == ["первый абзац"]
