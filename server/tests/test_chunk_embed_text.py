"""Dense context stays separate through SQL, retries, vector upload and prompts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from langchain.schema import Document
import pytest

from application.indexing import process_chunks
from config import MLProviderSection, process_section
from domain.ingestion.embed_text import build_embed_text
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.enums import DocumentVisibility
from domain.shared.hashing import content_hash
from infrastructure.ml.langchain_document_parser import LangchainDocumentSplitter
from infrastructure.ml.rag.context_rendering import chunk_body
from infrastructure.ml.rag.rag_formatting import format_docs
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher
from infrastructure.repositories.vector.qdrant_ops import prepare_upload_texts, upload_document_batches
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME, encode_sparse
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from domain.knowledge.chunk import Chunk


@pytest.mark.asyncio
async def test_split_sql_outbox_dense_payload_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: PLR0915 - Complete storage-to-retrieval scenario.
    body = "Заявитель подаёт заявление через личный кабинет."
    metadata = {"doc_title": "Порядок обмена", "section": "Подача заявления", "source": "guide.txt"}
    raw = LangchainDocumentSplitter().split([RawDocument(body, metadata)])[0]
    assert raw.page_content == body
    assert raw.embed_text == build_embed_text(raw.metadata, body)
    assert raw.embed_text != body
    assert "body_text" not in raw.metadata
    assert "embed_text" not in raw.metadata

    factory = FakeUnitOfWorkFactory()
    await process_chunks(
        factory,
        7,
        "guide.txt",
        [raw],
        DocumentVisibility.INTERNAL_PUBLIC.value,
        owner_id=None,
        group_id=None,
        doc_domain="general",
    )
    stored = await factory._uow.chunks.get_by_id(1)
    assert stored is not None
    assert stored.content == body
    assert stored.embed_text == raw.embed_text
    assert stored.content_hash == content_hash(body)
    entry = (await factory._uow.vector_outbox.claim_batch("test"))[0]
    point = entry.payload["points"][0]
    assert point["embed_text"] == raw.embed_text
    assert point["page_content"] == body

    embeddings = MagicMock()
    embeddings.embed_documents = AsyncMock(
        return_value=[[0.1] * process_section(MLProviderSection).embed_dim]
    )
    client = MagicMock()
    monkeypatch.setattr(
        "infrastructure.repositories.vector.qdrant_ops.ensure_embedding_identity", lambda *_args: "test:v2"
    )
    await upload_document_batches(
        [Document(page_content=body, metadata={**point["metadata"], "embed_text": point["embed_text"]})],
        embeddings,
        client,
    )
    embeddings.embed_documents.assert_awaited_once_with([raw.embed_text])
    uploaded = client.upsert.call_args.kwargs["points"][0]
    assert uploaded.payload["page_content"] == body
    assert "embed_text" not in uploaded.payload["metadata"]
    assert uploaded.payload["metadata"]["content_hash"] == content_hash(body)
    assert uploaded.vector[SPARSE_VECTOR_NAME] == encode_sparse(body)
    prompt = format_docs([Document(page_content=body, metadata=uploaded.payload["metadata"])])
    assert prompt.count(metadata["doc_title"]) == 1
    assert prompt.count(metadata["section"]) == 1
    assert prompt.count(body) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_during_upload", [False, True])
async def test_retry_reads_committed_dense_text_and_detects_context_only_edits(
    changed_during_upload: bool,
) -> None:
    factory = FakeUnitOfWorkFactory()
    await factory._uow.chunks.bulk_insert(
        7,
        "guide.txt",
        DocumentVisibility.INTERNAL_PUBLIC.value,
        [ChunkInsert(content="same body", embed_text="new context\n\nsame body")],
    )
    writer = AsyncMock()
    dispatcher = OutboxDispatcher(factory, writer)
    payload = {
        "points": [
            {
                "chunk_id": 1,
                "page_content": "same body",
                "embed_text": "stale context",
                "metadata": {"document_id": 7},
            }
        ]
    }

    async def upload(chunks: list[Chunk], **_kwargs: object) -> None:
        assert chunks[0].embed_text == "new context\n\nsame body"

        if changed_during_upload:
            await factory._uow.chunks.update_content(
                1,
                "same body",
                content_hash("same body"),
                datetime.now(UTC),
                12,
                embed_text="concurrent context",
            )

    writer.upload_documents.side_effect = upload

    if changed_during_upload:
        with pytest.raises(RuntimeError, match="retry required"):
            await dispatcher.apply_upsert(payload, document_id=7)
    else:
        await dispatcher.apply_upsert(payload, document_id=7)
        writer.upload_documents.assert_awaited_once()


def test_null_dense_text_uses_body_and_does_not_reconstruct_unbounded_parents() -> None:
    doc = Document(
        page_content="legacy body",
        metadata={"embed_text": None, "parent_units": [{"content": "Large parent " * 10000}]},
    )
    assert prepare_upload_texts([doc]) == ["legacy body"]
    assert doc.metadata["content_hash"] == content_hash("legacy body")


def test_original_body_starting_with_title_is_preserved_in_prompt() -> None:
    doc = Document(page_content="Название\n\nИсходный абзац.", metadata={"doc_title": "Название"})
    assert chunk_body(doc) == doc.page_content


@pytest.mark.parametrize("maximum", [80, 180, 400])
def test_dense_context_respects_budget_without_truncating_body_or_parent(maximum: int) -> None:
    body = "Полный текст правила."
    intro = "Вводное условие. " * 20
    metadata: dict[str, Any] = {
        "doc_title": "Заголовок " * 50,
        "section": "Раздел > Пункт 1",
        "parent_units": [{"content": intro}],
    }
    dense = build_embed_text(metadata, body, max_tokens=maximum, measure=len)
    assert len(dense) <= maximum
    assert dense.endswith(body)
    assert intro.strip() in dense or "Вводное условие" not in dense
    assert metadata["parent_units"][0]["content"] == intro


def test_dense_keeps_all_inherited_conditions_within_separate_budget() -> None:
    metadata = {
        "embed_prefix": "Инструкция\n\n",
        "parent_units": [
            {"content": "Условие всей статьи."},
            {"content": "Условие ближайшего пункта."},
        ],
    }
    dense = build_embed_text(metadata, "Тело нормы.", max_tokens=200, measure=len)
    assert dense == "Условие всей статьи.\nУсловие ближайшего пункта.\nИнструкция\n\nТело нормы."
    assert len(dense) <= 200
