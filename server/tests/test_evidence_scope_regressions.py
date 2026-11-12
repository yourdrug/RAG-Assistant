"""Evidence scope survives early retrieval, model inputs and context expansion."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document
import pytest

from application.search import HybridRetriever
from domain.identity.roles import UserKind
from domain.identity.user_context import UserContext
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.enums import PageContentType
from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.ranking import deduplicate_docs
from domain.shared.hashing import content_hash
from infrastructure.ml.ingestion.markdown import parse_markdown_sections
from infrastructure.ml.ingestion.sections import Section
from infrastructure.ml.ingestion.splitting import split_documents
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata
from infrastructure.ml.rag.context_rendering import render_selected_docs
from infrastructure.ml.rag.context_selection import select_context
from infrastructure.ml.rag.prompt_budget import estimate_text_tokens
from infrastructure.ml.rag.rag_postprocess import enrich_with_neighbors
from infrastructure.ml.rag.rag_reranking import (
    deduplicate_docs as deduplicate_candidates,
    rerank_documents,
    reranker_text,
)
from infrastructure.ml.rag.retrieval_pipeline import merge_evidence_candidates
from infrastructure.repositories.vector.qdrant_ops import build_points, prepare_upload_texts
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME, encode_sparse


@pytest.mark.parametrize(
    "changed",
    [
        {"document_id": 2},
        {"act_version_id": 2},
        {"section": "Existing clients"},
        {"parent_units": [{"content": "Only for existing clients."}]},
        {"parent_units": [{"content": "Only for new clients.", "heading": "Other condition"}]},
        {"table_row_key": "2"},
    ],
)
def test_early_deduplication_keeps_distinct_evidence(changed: dict[str, Any]) -> None:
    metadata = {
        "document_id": 1,
        "act_version_id": 1,
        "section": "New clients",
        "parent_units": [{"content": "Only for new clients."}],
    }
    first = Document(page_content="Submit the application.", metadata=metadata)
    other = Document(page_content=first.page_content, metadata={**metadata, **changed})
    duplicate = Document(page_content=first.page_content, metadata=dict(metadata))

    for deduplicate in (deduplicate_docs, deduplicate_candidates, merge_evidence_candidates):
        selected = deduplicate([first, other, duplicate])
        assert selected == [first, other]
        assert selected[0] is first

    _, packed = select_context([first, other, duplicate], 6000, estimate_text_tokens, [])
    assert len(packed) == 2

    assert HybridRetriever().merge_and_dedup(
        [("1", 1.0), ("3", 0.8)],
        [("2", 1.0)],
        {"1": (1.0, first), "2": (1.0, other), "3": (0.8, duplicate)},
        fetch_k=10,
        rrf_k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
    ) == [first, other]


def test_equal_filenames_in_different_paths_are_different_sources() -> None:
    docs = [
        Document(page_content="Same requirement.", metadata={"source": source})
        for source in ("new/rules.md", "existing/rules.md")
    ]
    assert deduplicate_docs(docs) == deduplicate_candidates(docs) == merge_evidence_candidates(docs) == docs


@pytest.mark.asyncio
async def test_dense_and_reranker_keep_parent_conditions_while_sparse_indexes_body() -> None:
    condition = ("Only for dangerous goods transportation. " + "A complete prerequisite. " * 60).strip()
    doc = Document(
        page_content="Submit the application.",
        metadata={"chunk_id": 1, "source": "rules.md", "parent_units": [{"content": condition}]},
    )
    original = doc.page_content
    scoped = content_with_parent_scope(doc)
    assert prepare_upload_texts([doc]) == [scoped]
    assert doc.metadata["content_hash"] == content_hash(original)
    point = build_points([doc], [[1.0, 0.0]])[0]
    assert isinstance(point.vector, dict)
    assert point.vector[SPARSE_VECTOR_NAME] == encode_sparse(original)
    assert point.payload is not None
    assert point.payload["page_content"] == original
    assert condition in reranker_text(doc)
    reranker = MagicMock(predict=MagicMock(return_value=[0.9]))
    assert await rerank_documents("dangerous goods", [doc], 1, reranker) == [(doc, 0.9)]
    assert condition in reranker.predict.call_args.args[0][0][1]
    assert doc.page_content == original


@pytest.mark.parametrize("heading", ["", "# Tariffs\n\n"])
def test_table_keeps_introduction_caption_and_notes_through_chunking(tmp_path: Path, heading: str) -> None:
    path = tmp_path / "tariffs.md"
    condition = "Тарифы применяются только к новым клиентам."

    if heading:
        condition += " Дополнительное обязательное условие." * 60

    caption = "Таблица 1. Стоимость обслуживания"
    note = "Примечания:\n\n1. Цена 50 действует при годовом договоре.\n\n2. Скидки не применяются."
    rows = "\n".join(f"| Основной {number} | 50 |" for number in range(25))
    path.write_text(
        f"{heading}{condition}\n\n{caption}\n\n| Тариф | Цена |\n|---|---|\n{rows}\n\n{note}"
        "\n\n# Other scope\n\nДля действующих клиентов цена 100.",
        encoding="utf-8",
    )
    parsed = parse_markdown_sections(path)
    table = next(section for section in parsed if section.kind == PageContentType.TABLE)
    assert condition in "\n".join(parent["content"] for parent in table.metadata["parent_units"])
    docs = LangchainDocumentParser().parse(path)
    chunks = split_documents([Document(page_content=doc.page_content, metadata=doc.metadata) for doc in docs])
    tables = [doc for doc in chunks if doc.metadata.get("content_type") == PageContentType.TABLE.value]
    assert len(tables) > 1

    for doc in tables:
        scoped = content_with_parent_scope(doc)
        assert all(text in scoped for text in (condition, caption, note, "50"))
        assert "цена 100" not in scoped
        assert prepare_upload_texts([doc]) == [scoped]
        assert condition in reranker_text(doc)
        assert scoped.count(condition) == 1


def test_table_context_respects_table_boundaries_and_preserves_existing_parents(tmp_path: Path) -> None:
    sections = [
        Section("Rules", "New clients only."),
        Section("Rules", "| Price |\n|---|\n| 50 |", PageContentType.TABLE),
        Section("Rules", "Note: first table only.\n\nExisting clients only."),
        Section(
            "Rules",
            "| Price |\n|---|\n| 100 |",
            PageContentType.TABLE,
            {"parent_units": [{"content": "Common parent condition."}]},
        ),
        Section("Elsewhere", "Note: unrelated."),
    ]
    docs = LangchainDocumentParser.sections_to_documents(sections, tmp_path / "rules.docx")
    first = content_with_parent_scope(docs[1])
    second = content_with_parent_scope(docs[3])
    assert "New clients only." in first
    assert "first table only" in first
    assert "Existing clients only." not in first
    assert "Existing clients only." in second
    assert "Common parent condition." in second
    assert "first table only" not in second
    assert "New clients only." not in second
    assert "unrelated" not in second


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_overfull_candidate_pool_does_not_disable_useful_neighbors(*, structured: bool) -> None:
    context = {"section": "Point 21"} if structured else {}
    seed = Document(
        page_content="Main requirement.",
        metadata={"document_id": 1, "source": "rules.md", "chunk_index": 1, **context},
    )
    tail = ChunkRecord(
        2,
        1,
        "rules.md",
        "Exception: only with approval.",
        2,
        content_hash=content_hash("Exception: only with approval."),
        context_metadata=context,
    )
    large = Document(page_content="Unrelated evidence. " * 100, metadata={"document_id": 2})
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=[tail]))
    user = UserContext(user_id=1, user_kind=UserKind.INTERNAL)
    candidates = [(seed, 1.0), (large, 0.95)]
    assert estimate_text_tokens(render_selected_docs(candidates)[0]) > 300
    assert (
        estimate_text_tokens(
            render_selected_docs(
                [
                    (seed, 1.0),
                    (Document(page_content=tail.content, metadata=chunk_result_metadata(tail)), 0.9),
                ]
            )[0]
        )
        < 150
    )
    result = await enrich_with_neighbors(
        candidates, enumerate_cases=False, chunk_search=search, user=user, max_context_tokens=300
    )
    assert tail.content in [doc.page_content for doc, _ in result]
    search.get_neighbors.assert_awaited_once()
    assert search.get_neighbors.call_args.kwargs["user"] is user
    context_text, _ = select_context(result, 300, estimate_text_tokens, [])
    assert seed.page_content in context_text
    assert tail.content in context_text
    assert estimate_text_tokens(context_text) <= 300


@pytest.mark.asyncio
async def test_plain_text_neighbors_cannot_cross_source_edition_or_parent_scope() -> None:
    seed = Document(
        page_content="Requirement.",
        metadata={"document_id": 1, "chunk_index": 1, "act_version_id": 5},
    )
    neighbors = [
        ChunkRecord(2, 2, "other.md", "Other source", 2, act_version_id=5),
        ChunkRecord(3, 1, "rules.md", "Other edition", 2, act_version_id=4),
        ChunkRecord(4, 1, "rules.md", "Distant chunk", 4, act_version_id=5),
        ChunkRecord(
            5,
            1,
            "rules.md",
            "Other scope",
            2,
            act_version_id=5,
            context_metadata={"parent_units": [{"content": "Other condition."}]},
        ),
    ]
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=neighbors))
    assert await enrich_with_neighbors(
        [(seed, 1.0)],
        enumerate_cases=False,
        chunk_search=search,
        user=UserContext(user_id=1, user_kind=UserKind.INTERNAL),
    ) == [(seed, 1.0)]
    search.get_neighbors.assert_awaited_once()


@pytest.mark.asyncio
async def test_large_stronger_anchor_keeps_expansion_priority_over_small_weak_candidate() -> None:
    seed = Document(
        page_content="Primary evidence. " * 30,
        metadata={"document_id": 1, "chunk_index": 1, "section": "Point 21"},
    )
    weak = Document(page_content="Weak evidence.", metadata={"document_id": 2, "chunk_index": 1})
    tail = ChunkRecord(2, 1, "rules.md", "Required continuation.", 2, section="Point 21")
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=[tail]))
    assert 150 < estimate_text_tokens(render_selected_docs([(seed, 1.0)])[0]) < 250
    result = await enrich_with_neighbors(
        [(seed, 1.0), (weak, 0.8)],
        enumerate_cases=False,
        chunk_search=search,
        user=UserContext(user_id=1, user_kind=UserKind.INTERNAL),
        max_context_tokens=300,
    )
    search.get_neighbors.assert_awaited_once()
    assert search.get_neighbors.call_args.args[0] == 1
    assert tail.content in [doc.page_content for doc, _ in result]
