"""Tests for domain/rag.py -- pure RAG logic: formatting, sources, history, reranking.

Complements test_rag_chain.py with additional edge cases.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import infrastructure.ml.rag as rag  # noqa: E402
from infrastructure.ml.rag import classify_question_breadth  # noqa: E402
from langchain_core.language_models import FakeListChatModel  # noqa: E402

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

    def test_with_scored_pairs_adds_max_score(self):
        docs = [_doc("t1", "a.pdf", 1), _doc("t2", "a.pdf", 3), _doc("t3", "b.pdf", 1)]
        scored = [(docs[0], 0.5), (docs[1], 0.9), (docs[2], 0.3)]
        sources = rag.extract_sources(scored)
        by_name = {s["source"]: s for s in sources}
        assert by_name["a.pdf"]["max_score"] == 0.9
        assert by_name["b.pdf"]["max_score"] == 0.3

    def test_with_scored_pairs_sorted_by_max_score(self):
        docs = [_doc("t1", "low.pdf", 1), _doc("t2", "high.pdf", 1)]
        scored = [(docs[0], 0.2), (docs[1], 0.95)]
        sources = rag.extract_sources(scored)
        assert sources[0]["source"] == "high.pdf"
        assert sources[1]["source"] == "low.pdf"

    def test_without_scores_no_max_score_key(self):
        docs = [_doc("t", "a.pdf", 1)]
        sources = rag.extract_sources(docs)
        assert "max_score" not in sources[0]


# ---------------------------------------------------------------------------
# filter_cited_sources
# ---------------------------------------------------------------------------


class TestFilterCitedSources:
    def test_keeps_cited_sources(self):
        sources = [
            {"source": "a.pdf", "pages": [1]},
            {"source": "b.pdf", "pages": [2]},
            {"source": "c.pdf", "pages": [3]},
        ]
        answer = "Согласно [1], нужно X. Также [3] указывает на Y."
        result = rag.filter_cited_sources(answer, sources)
        assert [s["source"] for s in result] == ["a.pdf", "c.pdf"]

    def test_no_citations_returns_all(self):
        sources = [{"source": "a.pdf", "pages": [1]}, {"source": "b.pdf", "pages": [2]}]
        answer = "Ответ без ссылок."
        result = rag.filter_cited_sources(answer, sources)
        assert len(result) == 2

    def test_empty_sources(self):
        assert rag.filter_cited_sources("answer [1]", []) == []

    def test_citation_out_of_range_returns_all_as_fallback(self):
        sources = [{"source": "a.pdf", "pages": [1]}]
        answer = "См. [5] для деталей."
        result = rag.filter_cited_sources(answer, sources)
        assert len(result) == 1

    def test_multiple_citations_same_source(self):
        sources = [{"source": "a.pdf", "pages": [1]}, {"source": "b.pdf", "pages": [2]}]
        answer = "[1] и [1] оба указывают на a.pdf"
        result = rag.filter_cited_sources(answer, sources)
        assert len(result) == 1
        assert result[0]["source"] == "a.pdf"


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
        result = asyncio.run(rag.rerank_documents("q", docs, top_n=2, reranker=reranker))
        assert len(result) == 2
        assert isinstance(result[0], tuple)
        assert result[0][0].page_content == "b"
        assert result[0][1] == 0.9

    def test_top_n_greater_than_docs_returns_all(self):
        docs = [_doc("a"), _doc("b")]
        reranker = self._fake_reranker([0.5, 0.3])
        result = asyncio.run(rag.rerank_documents("q", docs, top_n=10, reranker=reranker))
        assert len(result) == 2

    def test_equal_scores_preserve_original_order(self):
        docs = [_doc("first"), _doc("second")]
        reranker = self._fake_reranker([0.5, 0.5])
        result = asyncio.run(rag.rerank_documents("q", docs, top_n=2, reranker=reranker))
        assert [d.page_content for d, _ in result] == ["first", "second"]

    def test_negative_scores_handled(self):
        docs = [_doc("bad"), _doc("worse")]
        reranker = self._fake_reranker([-0.8, -0.2])
        result = asyncio.run(rag.rerank_documents("q", docs, top_n=1, reranker=reranker))
        assert result[0][0].page_content == "worse"

    def test_empty_docs_returns_empty(self):
        reranker = self._fake_reranker([])
        assert asyncio.run(rag.rerank_documents("q", [], top_n=5, reranker=reranker)) == []

    def test_single_doc_returns_single(self):
        docs = [_doc("only")]
        reranker = self._fake_reranker([0.7])
        result = asyncio.run(rag.rerank_documents("q", docs, top_n=5, reranker=reranker))
        assert len(result) == 1

    def test_top_n_zero_returns_empty(self):
        docs = [_doc("a"), _doc("b")]
        reranker = self._fake_reranker([0.9, 0.1])
        result = asyncio.run(rag.rerank_documents("q", docs, top_n=0, reranker=reranker))
        assert result == []

    def test_min_score_filters_low_scores(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        reranker = self._fake_reranker([0.9, 0.3, 0.1])
        result = asyncio.run(rag.rerank_documents("q", docs, top_n=3, reranker=reranker, min_score=0.5))
        assert len(result) == 1
        assert result[0][0].page_content == "a"

    def test_score_gap_ratio_filters_far_from_top(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        reranker = self._fake_reranker([1.0, 0.05, 0.01])
        result = asyncio.run(rag.rerank_documents("q", docs, top_n=3, reranker=reranker, score_gap_ratio=0.1))
        assert len(result) == 1
        assert result[0][0].page_content == "a"

    def test_min_score_and_gap_ratio_combined(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        reranker = self._fake_reranker([1.0, 0.8, 0.01])
        result = asyncio.run(
            rag.rerank_documents("q", docs, top_n=3, reranker=reranker, min_score=0.5, score_gap_ratio=0.5)
        )
        # gap_ratio cutoff = 1.0 * 0.5 = 0.5; min_score = 0.5
        # a: 1.0 >= 0.5 and >= 0.5 → keep
        # b: 0.8 >= 0.5 and >= 0.5 → keep
        # c: 0.01 < 0.5 → filtered by min_score
        assert len(result) == 2


# ---------------------------------------------------------------------------
# classify_question_breadth
# ---------------------------------------------------------------------------


class TestClassifyQuestionBreadth:
    def test_simple_question_is_narrow(self):
        assert classify_question_breadth("Какой пароль?") == "narrow"

    def test_podrobno_is_broad(self):
        assert classify_question_breadth("Расскажи подробно про маркировку") == "broad"

    def test_rasskazhi_pro_is_broad(self):
        assert classify_question_breadth("Расскажи про систему безопасности") == "broad"

    def test_poryadok_polucheniya_is_broad(self):
        assert classify_question_breadth("Порядок получения кода маркировки") == "broad"

    def test_sistema_is_broad(self):
        assert classify_question_breadth("Система маркировки товаров") == "broad"

    def test_obyasni_vse_is_broad(self):
        assert classify_question_breadth("Объясни всё про пароли") == "broad"

    def test_kak_rabotaet_is_broad(self):
        assert classify_question_breadth("Как работает шифрование?") == "broad"

    def test_case_insensitive(self):
        assert classify_question_breadth("ПОДРОБНО про безопасность") == "broad"

    def test_plain_factual_is_narrow(self):
        assert classify_question_breadth("Какой срок действия пароля?") == "narrow"

    def test_empty_string(self):
        assert classify_question_breadth("") == "narrow"

    def test_kakie_isklyucheniya_is_narrow(self):
        assert classify_question_breadth("Какие есть исключения из правила?") == "narrow"

    def test_kak_ubititsya_is_narrow(self):
        assert classify_question_breadth("Как убедиться что всё верно?") == "narrow"

    def test_chto_takoe_is_narrow(self):
        assert classify_question_breadth("Что такое ЭТТН?") == "narrow"

    def test_kakoy_srok_deystviya_is_narrow(self):
        assert classify_question_breadth("Какой срок действия пароля?") == "narrow"


# ---------------------------------------------------------------------------
# needs_decomposition
# ---------------------------------------------------------------------------


class TestNeedsDecomposition:
    def test_simple_question_no_decomposition(self):
        from domain.services.rag_policy import needs_decomposition

        assert needs_decomposition("Что такое ЭТТН?") is False

    def test_compound_with_and(self):
        from domain.services.rag_policy import needs_decomposition

        assert needs_decomposition("Расскажи про ЭТТН и про маркировку") is True

    def test_compare_question(self):
        from domain.services.rag_policy import needs_decomposition

        assert needs_decomposition("Сравни тарифы А и Б") is True

    def test_how_and_what(self):
        from domain.services.rag_policy import needs_decomposition

        assert needs_decomposition("Как оформить заказ и что для этого нужно?") is True

    def test_empty_string(self):
        from domain.services.rag_policy import needs_decomposition

        assert needs_decomposition("") is False

    def test_single_topic_no_decomposition(self):
        from domain.services.rag_policy import needs_decomposition

        assert needs_decomposition("Подробно расскажи про безопасность") is False


# ---------------------------------------------------------------------------
# check_relevance (mocked LLM)
# ---------------------------------------------------------------------------


class TestCheckRelevance:
    def test_empty_docs_returns_false(self):
        result = asyncio.run(rag.check_relevance(None, "question", []))
        assert result == (False, "Нет документов для проверки")

    def test_relevant_answer_detected(self):
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.is_relevant = True
        mock_result.reason = "Context is sufficient"
        mock_client.chat.completions.create.return_value = mock_result

        with patch.object(rag, "_get_rag_instructor_client", return_value=mock_client):
            with patch("infrastructure.ml.llm_schemas.RelevanceCheck"):
                from config import settings

                settings.llm_model = "test-model"
                result = asyncio.run(rag.check_relevance(None, "question", [_doc("some context")]))
                assert result[0] is True

    def test_irrelevant_answer_detected(self):
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.is_relevant = False
        mock_result.reason = "Context lacks relevant information"
        mock_client.chat.completions.create.return_value = mock_result

        with patch.object(rag, "_get_rag_instructor_client", return_value=mock_client):
            with patch("infrastructure.ml.llm_schemas.RelevanceCheck"):
                from config import settings

                settings.llm_model = "test-model"
                result = asyncio.run(rag.check_relevance(None, "question", [_doc("some context")]))
                assert result[0] is False


# ---------------------------------------------------------------------------
# decompose_question (mocked LLM)
# ---------------------------------------------------------------------------


class TestDecomposeQuestion:
    def test_single_line_returns_original(self):
        llm = FakeListChatModel(responses=["Один подвопрос"])
        result = asyncio.run(rag.decompose_question(llm, "complex question"))
        assert result == ["complex question"]

    def test_multi_line_returns_list(self):
        llm = FakeListChatModel(responses=["Подвопрос 1\nПодвопрос 2\nПодвопрос 3"])
        result = asyncio.run(rag.decompose_question(llm, "complex question"))
        assert len(result) == 3
        assert "Подвопрос 1" in result


# ---------------------------------------------------------------------------
# _docx_table_to_markdown
# ---------------------------------------------------------------------------


class TestDocxTableToMarkdown:
    def test_simple_table(self):
        from infrastructure.ml.ingestion import _docx_table_to_markdown

        class MockCell:
            def __init__(self, text):
                self.text = text

        class MockRow:
            def __init__(self, cells):
                self.cells = [MockCell(c) for c in cells]

        class MockTable:
            def __init__(self, rows):
                self.rows = [MockRow(r) for r in rows]

        table = MockTable([["Name", "Value"], ["A", "1"], ["B", "2"]])
        result = _docx_table_to_markdown(table)
        assert "| Name | Value |" in result
        assert "|---|---|" in result
        assert "| A | 1 |" in result
        assert "| B | 2 |" in result

    def test_empty_table(self):
        from infrastructure.ml.ingestion import _docx_table_to_markdown

        class MockTable:
            rows = []

        assert _docx_table_to_markdown(MockTable()) == ""

    def test_pipes_in_cells_escaped(self):
        from infrastructure.ml.ingestion import _docx_table_to_markdown

        class MockCell:
            def __init__(self, text):
                self.text = text

        class MockRow:
            def __init__(self, cells):
                self.cells = [MockCell(c) for c in cells]

        class MockTable:
            def __init__(self, rows):
                self.rows = [MockRow(r) for r in rows]

        table = MockTable([["Header"], ["value|with|pipes"]])
        result = _docx_table_to_markdown(table)
        assert "value\\|with\\|pipes" in result


# ---------------------------------------------------------------------------
# _pymupdf_table_to_markdown
# ---------------------------------------------------------------------------


class TestPyMuPDFTableToMarkdown:
    def test_simple_table(self):
        from infrastructure.ml.ingestion import _pymupdf_table_to_markdown

        class MockTable:
            def extract(self):
                return [["Name", "Value"], ["A", "1"], ["B", "2"]]

        result = _pymupdf_table_to_markdown(MockTable())
        assert "| Name | Value |" in result
        assert "|---|---|" in result

    def test_extract_failure_returns_empty(self):
        from infrastructure.ml.ingestion import _pymupdf_table_to_markdown

        class MockTable:
            def extract(self):
                raise Exception("not supported")

        assert _pymupdf_table_to_markdown(MockTable()) == ""


# ---------------------------------------------------------------------------
# answer_cache helpers
# ---------------------------------------------------------------------------


class TestAnswerCacheHelpers:
    def test_same_scope_same_hash(self):
        from infrastructure.ml.answer_cache import compute_visibility_scope_hash

        h1 = compute_visibility_scope_hash("internal", [1, 2])
        h2 = compute_visibility_scope_hash("internal", [1, 2])
        assert h1 == h2

    def test_different_scope_different_hash(self):
        from infrastructure.ml.answer_cache import compute_visibility_scope_hash

        h1 = compute_visibility_scope_hash("internal", [1])
        h2 = compute_visibility_scope_hash("client", [1])
        assert h1 != h2

    def test_different_groups_different_hash(self):
        from infrastructure.ml.answer_cache import compute_visibility_scope_hash

        h1 = compute_visibility_scope_hash("internal", [1, 2])
        h2 = compute_visibility_scope_hash("internal", [1, 3])
        assert h1 != h2

    def test_question_hash_deterministic(self):
        from infrastructure.ml.answer_cache import compute_question_hash

        h1 = compute_question_hash("Что такое ЭТТН?")
        h2 = compute_question_hash("Что такое ЭТТН?")
        assert h1 == h2

    def test_question_hash_case_insensitive(self):
        from infrastructure.ml.answer_cache import compute_question_hash

        h1 = compute_question_hash("ЧТО ТАКОЕ ЭТТН?")
        h2 = compute_question_hash("что такое Эттн?")
        assert h1 == h2
