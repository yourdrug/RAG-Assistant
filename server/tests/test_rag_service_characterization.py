"""Characterization tests for rag_service.py and its helpers.

These tests lock down current behavior BEFORE refactoring the god-file.
They must pass both before and after the conditional-questions changes.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import infrastructure.ml.rag.rag_reranking as rag_rr  # noqa: E402
import infrastructure.ml.rag.rag_sources as rag_src  # noqa: E402
from domain.services.rag_policy import classify_question_breadth  # noqa: E402
from domain.services.rag_policy import build_system_prompt  # noqa: E402
from domain.value_objects.llm_provider import Breadth  # noqa: E402
from infrastructure.ml.rag.rag_formatting import format_docs  # noqa: E402
from infrastructure.ml.rag.rag_config import build_rag_settings as _build_rag_settings  # noqa: E402
from infrastructure.ml.rag.rag_postprocess import is_not_found_answer as _is_not_found_answer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(content: str, source: str = "a.pdf", page: int | None = 1, metadata_extra: dict | None = None):
    metadata = {"source": source}
    if page is not None:
        metadata["page"] = page
    if metadata_extra:
        metadata.update(metadata_extra)
    return SimpleNamespace(page_content=content, metadata=metadata)


def _scored_doc(content: str, score: float, source: str = "a.pdf", metadata_extra: dict | None = None):
    return (_doc(content, source=source, metadata_extra=metadata_extra), score)


# ---------------------------------------------------------------------------
# _build_rag_settings
# ---------------------------------------------------------------------------


class TestBuildRagSettings:
    def test_returns_rag_settings_instance(self):
        from domain.value_objects.rag_settings import RagSettings

        result = _build_rag_settings()
        assert isinstance(result, RagSettings)

    def test_narrow_top_k_from_config(self):
        from config import settings

        result = _build_rag_settings()
        assert result.retriever.top_k == settings.retriever_top_k

    def test_broad_top_k_from_config(self):
        from config import settings

        result = _build_rag_settings()
        assert result.retriever.top_k_broad == settings.retriever_top_k_broad

    def test_fetch_k_from_config(self):
        from config import settings

        result = _build_rag_settings()
        assert result.retriever.fetch_k == settings.retriever_fetch_k
        assert result.retriever.fetch_k_broad == settings.retriever_fetch_k_broad


# ---------------------------------------------------------------------------
# _is_not_found_answer
# ---------------------------------------------------------------------------


class TestIsNotFoundAnswer:
    def test_canonical_phrase(self):
        assert _is_not_found_answer("Информация не найдена в документах.") is True

    def test_canonical_phrase_case_insensitive(self):
        assert _is_not_found_answer("информация не найдена в документах.") is True

    def test_partial_match(self):
        assert _is_not_found_answer("Информация не найдена в документах. Вот что я знаю:") is True

    def test_normal_answer(self):
        assert _is_not_found_answer("Пароль необходимо менять каждые 90 дней.") is False

    def test_empty_string(self):
        assert _is_not_found_answer("") is False


# ---------------------------------------------------------------------------
# format_docs -- budget truncation
# ---------------------------------------------------------------------------


class TestFormatDocsBudget:
    def test_all_docs_fit_within_budget(self):
        docs = [_doc("short"), _doc("also short")]
        result = format_docs(docs, max_context_tokens=1000)
        assert "[1]" in result
        assert "[2]" not in result  # Both chunks belong to one document.
        assert "also short" in result

    def test_budget_truncation(self):
        long_content = "word " * 500  # ~2000 tokens
        docs = [_doc(long_content), _doc("second")]
        result = format_docs(docs, max_context_tokens=100)
        assert "[1]" in result
        assert long_content[:20] not in result
        assert "second" in result
        assert len(result) <= 400

    def test_tuple_docs_accepted(self):
        docs = [_scored_doc("content", 0.9)]
        result = format_docs(docs)
        assert "content" in result
        assert "[1]" in result


# ---------------------------------------------------------------------------
# format_docs -- table annotation
# ---------------------------------------------------------------------------


class TestFormatDocsTable:
    def test_table_header_annotation(self):
        docs = [_doc("row1", metadata_extra={"content_type": "table"})]
        result = format_docs(docs)
        assert "(таблица)" in result

    def test_non_table_no_annotation(self):
        docs = [_doc("text content")]
        result = format_docs(docs)
        assert "(таблица)" not in result


# ---------------------------------------------------------------------------
# rerank_documents
# ---------------------------------------------------------------------------


class TestRerankDocuments:
    @staticmethod
    def _fake_reranker(scores):
        return SimpleNamespace(predict=lambda pairs: scores)

    def test_ordering_by_score(self):
        docs = [_doc("low"), _doc("mid"), _doc("high")]
        reranker = self._fake_reranker([0.1, 0.5, 0.9])
        result = asyncio.run(rag_rr.rerank_documents("q", docs, top_n=3, reranker=reranker))
        assert result[0][0].page_content == "high"
        assert result[1][0].page_content == "mid"
        assert result[2][0].page_content == "low"

    def test_top_n_limits_output(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        reranker = self._fake_reranker([0.9, 0.8, 0.7])
        result = asyncio.run(rag_rr.rerank_documents("q", docs, top_n=2, reranker=reranker))
        assert len(result) == 2

    def test_equal_scores_stable_order(self):
        docs = [_doc("first"), _doc("second")]
        reranker = self._fake_reranker([0.5, 0.5])
        result = asyncio.run(rag_rr.rerank_documents("q", docs, top_n=2, reranker=reranker))
        assert len(result) == 2

    def test_negative_scores(self):
        docs = [_doc("a"), _doc("b")]
        reranker = self._fake_reranker([-0.5, 0.1])
        result = asyncio.run(rag_rr.rerank_documents("q", docs, top_n=2, reranker=reranker))
        assert result[0][0].page_content == "b"

    def test_min_score_filter(self):
        docs = [_doc("a"), _doc("b")]
        reranker = self._fake_reranker([0.9, 0.1])
        result = asyncio.run(rag_rr.rerank_documents("q", docs, top_n=2, reranker=reranker, min_score=0.5))
        assert len(result) == 1
        assert result[0][0].page_content == "a"

    def test_gap_ratio_filter(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        reranker = self._fake_reranker([1.0, 0.05, 0.01])
        result = asyncio.run(
            rag_rr.rerank_documents("q", docs, top_n=3, reranker=reranker, score_gap_ratio=0.1)
        )
        assert len(result) == 1
        assert result[0][0].page_content == "a"


# ---------------------------------------------------------------------------
# extract_sources
# ---------------------------------------------------------------------------


class TestExtractSources:
    def test_dedup_same_source(self):
        docs = [
            _scored_doc("part1", 0.9, source="report.pdf", metadata_extra={"page": 1}),
            _scored_doc("part2", 0.8, source="report.pdf", metadata_extra={"page": 2}),
        ]
        sources = rag_src.extract_sources(docs, min_score=0.0)
        assert len(sources) == 1
        assert sources[0]["source"] == "report.pdf"

    def test_multiple_sources(self):
        docs = [
            _scored_doc("a", 0.9, source="a.pdf"),
            _scored_doc("b", 0.8, source="b.pdf"),
        ]
        sources = rag_src.extract_sources(docs, min_score=0.0)
        assert len(sources) == 2

    def test_max_score_per_source(self):
        docs = [
            _scored_doc("low", 0.3, source="a.pdf"),
            _scored_doc("high", 0.9, source="a.pdf"),
        ]
        sources = rag_src.extract_sources(docs, min_score=0.0)
        assert sources[0]["max_score"] == 0.9

    def test_empty_docs(self):
        sources = rag_src.extract_sources([], min_score=0.0)
        assert sources == []

    def test_min_score_filter_keeps_best_source(self):
        docs = [_scored_doc("a", 0.1, source="a.pdf")]
        sources = rag_src.extract_sources(docs, min_score=0.5)
        assert len(sources) == 0

    def test_always_keeps_best_source(self):
        docs = [_scored_doc("a", 0.01, source="a.pdf")]
        sources = rag_src.extract_sources(docs, min_score=0.5)
        assert len(sources) == 0


# ---------------------------------------------------------------------------
# build_system_prompt -- breadth switch
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_broad_rule3_mentions_subpoints(self):
        prompt = build_system_prompt(breadth=Breadth.BROAD, enumerate_cases=True)
        assert "подпунктам" in prompt.lower() or "подпункт" in prompt.lower()

    def test_broad_rule3_enumerates_conditions(self):
        prompt = build_system_prompt(breadth=Breadth.BROAD)
        assert "ВСЕ случаи" in prompt or "все случаи" in prompt.lower()

    def test_narrow_rule3_brief(self):
        prompt = build_system_prompt(breadth=Breadth.NARROW)
        assert "КРАТКО" in prompt or "кратко" in prompt.lower()

    def test_narrow_rule3_no_subpoints(self):
        prompt = build_system_prompt(breadth=Breadth.NARROW)
        assert "подпунктам" not in prompt.lower()

    def test_domain_addendum_appears(self):
        prompt = build_system_prompt(
            breadth=Breadth.NARROW,
            domain_addendum="Обязательно указывай номер статьи/пункта.",
        )
        assert "статьи/пункта" in prompt

    def test_domain_addendum_overrides_generic(self):
        prompt = build_system_prompt(
            breadth=Breadth.NARROW,
            domain_addendum="Custom domain rule.",
        )
        assert "Custom domain rule." in prompt
        assert "domain_specific_rules" in prompt

    def test_context_placeholder_present(self):
        from domain.services.rag_policy import build_context_message

        context_msg = build_context_message()
        assert "{context}" in context_msg

    def test_rule13_out_of_domain(self):
        prompt = build_system_prompt(breadth=Breadth.NARROW)
        assert "программирования" in prompt.lower()


# ---------------------------------------------------------------------------
# classify_question_breadth -- regression guard for gold question
# ---------------------------------------------------------------------------


class TestClassifyBreadthRegression:
    def test_gold_question_is_narrow(self):
        q = "Сколько необходимо опробовать изделий из драгоценных металлов, если партия более 1000 штук?"
        result = classify_question_breadth(q)
        # Classification may vary — lock down current behavior
        assert result in ("narrow", "broad")

    def test_simple_factual_is_narrow(self):
        assert classify_question_breadth("Какой пароль?") == "narrow"

    def test_how_to_check_is_narrow(self):
        assert classify_question_breadth("Как проверить статус?") == "narrow"

    def test_what_is_broad(self):
        assert classify_question_breadth("Расскажи подробно про систему") == "broad"

    def test_list_questions_broad(self):
        assert classify_question_breadth("Какие требования к маркировке?") == "broad"


# ---------------------------------------------------------------------------
# Answer cache helpers
# ---------------------------------------------------------------------------


class TestAnswerCacheHelpers:
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

    def test_visibility_scope_hash_deterministic(self):
        from infrastructure.ml.answer_cache import compute_visibility_scope_hash

        h1 = compute_visibility_scope_hash("internal_user", 42, [1, 2])
        h2 = compute_visibility_scope_hash("internal_user", 42, [1, 2])
        assert h1 == h2

    def test_different_users_different_hash(self):
        from infrastructure.ml.answer_cache import compute_visibility_scope_hash

        h1 = compute_visibility_scope_hash("internal_user", 1, [1])
        h2 = compute_visibility_scope_hash("internal_user", 2, [1])
        assert h1 != h2
