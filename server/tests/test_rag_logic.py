"""
Tests for domain/rag.py — pure RAG logic: formatting, sources, history, reranking.
Complements test_rag_chain.py with additional edge cases.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import domain.rag as rag  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(content: str, source: str = "a.pdf", page: int | None = 1):
    metadata = {"source": source}
    if page is not None:
        metadata["page"] = page
    return SimpleNamespace(page_content=content, metadata=metadata)


# ---------------------------------------------------------------------------
# format_docs
# ---------------------------------------------------------------------------


class TestFormatDocs:
    def test_single_doc_with_page(self):
        docs = [_doc("hello", "report.pdf", page=3)]
        result = rag.format_docs(docs)
        assert "[1] report.pdf (стр. 3)" in result
        assert "hello" in result

    def test_single_doc_without_page(self):
        docs = [_doc("content", "readme.md", page=None)]
        result = rag.format_docs(docs)
        assert "[1] readme.md" in result
        assert "(стр." not in result

    def test_multiple_docs_separated_by_separator(self):
        docs = [_doc("first", "a.pdf"), _doc("second", "b.pdf")]
        result = rag.format_docs(docs)
        # Separator is "\n\n---\n\n"
        assert "\n\n---\n\n" in result
        assert "[1]" in result
        assert "[2]" in result

    def test_empty_list_returns_empty_string(self):
        assert rag.format_docs([]) == ""

    def test_doc_with_unknown_source(self):
        doc = SimpleNamespace(page_content="text", metadata={})
        result = rag.format_docs([doc])
        assert "[1] unknown" in result

    def test_docs_numbering_starts_at_one(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        result = rag.format_docs(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_metadata_keys_preserved_in_output(self):
        doc = SimpleNamespace(page_content="data", metadata={"source": "x.pdf", "page": 7})
        result = rag.format_docs([doc])
        assert "x.pdf" in result
        assert "7" in result


# ---------------------------------------------------------------------------
# extract_sources
# ---------------------------------------------------------------------------


class TestExtractSources:
    def test_single_source_single_page(self):
        docs = [_doc("t", "a.pdf", page=1)]
        sources = rag.extract_sources(docs)
        assert len(sources) == 1
        assert sources[0]["source"] == "a.pdf"
        assert sources[0]["pages"] == [1]

    def test_same_source_multiple_pages_deduplicated(self):
        docs = [_doc("t1", "a.pdf", 1), _doc("t2", "a.pdf", 3), _doc("t3", "a.pdf", 1)]
        sources = rag.extract_sources(docs)
        assert len(sources) == 1
        assert sorted(sources[0]["pages"]) == [1, 3]

    def test_multiple_sources(self):
        docs = [_doc("t1", "a.pdf", 1), _doc("t2", "b.pdf", 2)]
        sources = rag.extract_sources(docs)
        assert len(sources) == 2
        src_names = {s["source"] for s in sources}
        assert src_names == {"a.pdf", "b.pdf"}

    def test_empty_docs_returns_empty_list(self):
        assert rag.extract_sources([]) == []

    def test_doc_without_page_metadata(self):
        doc = SimpleNamespace(page_content="t", metadata={"source": "x.pdf"})
        sources = rag.extract_sources([doc])
        assert sources[0]["pages"] == []

    def test_doc_without_source_metadata_defaults_to_unknown(self):
        doc = SimpleNamespace(page_content="t", metadata={})
        sources = rag.extract_sources([doc])
        assert sources[0]["source"] == "unknown"

    def test_pages_are_sorted(self):
        docs = [_doc("t", "a.pdf", 5), _doc("t", "a.pdf", 2), _doc("t", "a.pdf", 8)]
        sources = rag.extract_sources(docs)
        assert sources[0]["pages"] == [2, 5, 8]


# ---------------------------------------------------------------------------
# history_to_messages
# ---------------------------------------------------------------------------


class TestHistoryToMessages:
    def test_alternating_user_assistant(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        messages = rag.history_to_messages(history)
        assert len(messages) == 3
        from langchain_core.messages import AIMessage, HumanMessage

        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], HumanMessage)

    def test_empty_history(self):
        assert rag.history_to_messages([]) == []

    def test_only_user_messages(self):
        history = [{"role": "user", "content": "q1"}]
        messages = rag.history_to_messages(history)
        assert len(messages) == 1
        assert messages[0].content == "q1"

    def test_only_assistant_messages(self):
        history = [{"role": "assistant", "content": "a1"}]
        messages = rag.history_to_messages(history)
        assert len(messages) == 1
        assert messages[0].content == "a1"

    def test_content_preserved_exactly(self):
        history = [{"role": "user", "content": "Special chars: <>&\"'}"}]
        messages = rag.history_to_messages(history)
        assert messages[0].content == "Special chars: <>&\"'}"


# ---------------------------------------------------------------------------
# rerank_documents
# ---------------------------------------------------------------------------


class TestRerankDocuments:
    def _fake_reranker(self, scores):
        return SimpleNamespace(predict=lambda pairs: scores)

    def test_top_n_less_than_docs(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        reranker = self._fake_reranker([0.1, 0.9, 0.5])
        result = rag.rerank_documents("q", docs, top_n=2, reranker=reranker)
        assert len(result) == 2
        assert result[0].page_content == "b"

    def test_top_n_greater_than_docs_returns_all(self):
        docs = [_doc("a"), _doc("b")]
        reranker = self._fake_reranker([0.5, 0.3])
        result = rag.rerank_documents("q", docs, top_n=10, reranker=reranker)
        assert len(result) == 2

    def test_equal_scores_preserve_original_order(self):
        docs = [_doc("first"), _doc("second")]
        reranker = self._fake_reranker([0.5, 0.5])
        result = rag.rerank_documents("q", docs, top_n=2, reranker=reranker)
        assert [d.page_content for d in result] == ["first", "second"]

    def test_negative_scores_handled(self):
        docs = [_doc("bad"), _doc("worse")]
        reranker = self._fake_reranker([-0.8, -0.2])
        result = rag.rerank_documents("q", docs, top_n=1, reranker=reranker)
        assert result[0].page_content == "worse"

    def test_empty_docs_returns_empty(self):
        reranker = self._fake_reranker([])
        assert rag.rerank_documents("q", [], top_n=5, reranker=reranker) == []

    def test_single_doc_returns_single(self):
        docs = [_doc("only")]
        reranker = self._fake_reranker([0.7])
        result = rag.rerank_documents("q", docs, top_n=5, reranker=reranker)
        assert len(result) == 1

    def test_top_n_zero_returns_empty(self):
        docs = [_doc("a"), _doc("b")]
        reranker = self._fake_reranker([0.9, 0.1])
        result = rag.rerank_documents("q", docs, top_n=0, reranker=reranker)
        assert result == []
