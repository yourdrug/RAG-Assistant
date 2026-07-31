"""
Тесты для чистой логики domain/rag.py: форматирование контекста, извлечение
источников, конвертация истории и реранк (с замоканным CrossEncoder — без
реальной загрузки bge-reranker-v2-m3).
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import infrastructure.ml.rag as rag  # noqa: E402


def _doc(content: str, source: str = "docs/a.pdf"):
    return SimpleNamespace(page_content=content, metadata={"source": source})


def test_format_docs_numbers_and_separates_chunks():
    docs = [_doc("текст 1", "a.pdf"), _doc("текст 2", "b.pdf")]
    out = rag.format_docs(docs)
    assert "[1] a.pdf" in out
    assert "[2] b.pdf" in out
    assert "---" in out


def test_format_docs_accepts_scored_pairs():
    docs = [_doc("текст 1", "a.pdf"), _doc("текст 2", "b.pdf")]
    scored = [(docs[0], 0.9), (docs[1], 0.3)]
    out = rag.format_docs(scored)
    assert "[1] a.pdf" in out
    assert "[2] b.pdf" in out
    assert "текст 1" in out


def test_extract_sources_deduplicates_by_source():
    docs = [_doc("t1", "a.pdf"), _doc("t2", "a.pdf"), _doc("t3", "b.pdf")]
    sources = rag.extract_sources(docs)
    assert [s["source"] for s in sources] == ["a.pdf", "b.pdf"]


def test_extract_sources_with_scores_adds_max_score():
    docs = [_doc("t1", "a.pdf"), _doc("t2", "a.pdf"), _doc("t3", "b.pdf")]
    scored = [(docs[0], 0.5), (docs[1], 0.9), (docs[2], 0.3)]
    sources = rag.extract_sources(scored)
    by_name = {s["source"]: s for s in sources}
    assert by_name["a.pdf"]["max_score"] == 0.9
    assert by_name["b.pdf"]["max_score"] == 0.3


def test_extract_sources_sorted_by_max_score():
    docs = [_doc("t1", "low.pdf"), _doc("t2", "high.pdf")]
    scored = [(docs[0], 0.2), (docs[1], 0.95)]
    sources = rag.extract_sources(scored)
    assert sources[0]["source"] == "high.pdf"
    assert sources[1]["source"] == "low.pdf"


def test_history_to_messages_maps_roles():
    history = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "привет!"},
    ]
    messages = rag.history_to_messages(history)
    assert messages[0].content == "привет"
    assert messages[1].content == "привет!"


def test_rerank_documents_orders_by_score_and_returns_pairs():
    docs = [_doc("нерелевантно"), _doc("релевантно"), _doc("средне")]

    fake_reranker = SimpleNamespace(predict=lambda pairs: [0.1, 0.9, 0.5])

    result = asyncio.run(rag.rerank_documents("вопрос", docs, top_n=2, reranker=fake_reranker))

    assert len(result) == 2
    assert isinstance(result[0], tuple)
    assert result[0][0].page_content == "релевантно"
    assert result[0][1] == 0.9
    assert result[1][0].page_content == "средне"
    assert result[1][1] == 0.5


def test_rerank_documents_empty_input_returns_empty():
    fake_reranker = SimpleNamespace(predict=lambda pairs: [])
    assert asyncio.run(rag.rerank_documents("вопрос", [], top_n=5, reranker=fake_reranker)) == []


def test_rerank_documents_min_score_filter():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    fake_reranker = SimpleNamespace(predict=lambda pairs: [0.9, 0.3, 0.1])
    result = asyncio.run(
        rag.rerank_documents("q", docs, top_n=3, reranker=fake_reranker, min_score=0.5)
    )
    assert len(result) == 1
    assert result[0][0].page_content == "a"


def test_rerank_documents_score_gap_ratio_filter():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    fake_reranker = SimpleNamespace(predict=lambda pairs: [1.0, 0.05, 0.01])
    result = asyncio.run(
        rag.rerank_documents("q", docs, top_n=3, reranker=fake_reranker, score_gap_ratio=0.1)
    )
    # cutoff = 1.0 * 0.1 = 0.1 → only "a" (1.0) passes; "b" (0.05) and "c" (0.01) are below
    assert len(result) == 1
    assert result[0][0].page_content == "a"


def test_filter_cited_sources_keeps_cited():
    sources = [
        {"source": "a.pdf", "pages": [1]},
        {"source": "b.pdf", "pages": [2]},
        {"source": "c.pdf", "pages": [3]},
    ]
    answer = "Согласно документу [1], нужно сделать X. Также [3] указывает на Y."
    result = rag.filter_cited_sources(answer, sources)
    assert [s["source"] for s in result] == ["a.pdf", "c.pdf"]


def test_filter_cited_sources_no_citations_returns_all():
    sources = [{"source": "a.pdf", "pages": [1]}, {"source": "b.pdf", "pages": [2]}]
    answer = "Ответ без ссылок на источники."
    result = rag.filter_cited_sources(answer, sources)
    assert len(result) == 2


def test_filter_cited_sources_empty_sources():
    assert rag.filter_cited_sources("answer [1]", []) == []
