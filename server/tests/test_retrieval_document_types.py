"""Retrieval policies preserve concrete document and scored-pair types."""

from typing import assert_type

from langchain_core.documents import Document

from domain.ingestion.raw_document import RawDocument
from domain.rag.retrieval import deduplicate_docs, prioritize_distinct_provisions
from infrastructure.ml.rag.rag_reranking import deduplicate_docs as deduplicate_candidates


def test_deduplication_preserves_document_types_and_first_instance() -> None:
    first = Document(page_content="same")
    duplicate = Document(page_content="same")
    other = Document(page_content="other")
    documents = [first, duplicate, other]
    selected = deduplicate_docs(documents)
    assert_type(selected, list[Document])
    assert selected == [first, other]
    assert selected[0] is first
    candidates = deduplicate_candidates(documents)
    assert_type(candidates, list[Document])
    assert candidates[0] is first
    assert candidates == selected

    raw = RawDocument(page_content="raw")
    selected_raw = deduplicate_docs([raw, raw])
    assert_type(selected_raw, list[RawDocument])
    assert selected_raw == [raw]
    assert deduplicate_docs([]) == []


def test_diversity_preserves_documents_pairs_and_scores() -> None:
    first = Document(page_content="first", metadata={"source": "act", "section": "point 1"})
    repeated = Document(page_content="repeated", metadata={"source": "act", "section": "point 1"})
    other = Document(page_content="other", metadata={"source": "act", "section": "point 2"})
    selected = prioritize_distinct_provisions([first, repeated, other])
    assert_type(selected, list[Document])
    assert selected == [first, other, repeated]

    pairs = [(first, 0.9), (repeated, 0.8), (other, 0.7)]
    selected_pairs = prioritize_distinct_provisions(pairs)
    assert_type(selected_pairs, list[tuple[Document, float]])
    assert selected_pairs == [pairs[0], pairs[2], pairs[1]]
    assert selected_pairs[0] is pairs[0]
    raw = RawDocument(page_content="raw")
    selected_raw = prioritize_distinct_provisions([raw])
    assert_type(selected_raw, list[RawDocument])
    assert selected_raw == [raw]
    assert prioritize_distinct_provisions([]) == []
