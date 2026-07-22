"""
Тесты для чистой логики domain/rag.py: форматирование контекста, извлечение
источников, конвертация истории и реранк (с замоканным CrossEncoder — без
реальной загрузки bge-reranker-v2-m3).
"""

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


def test_extract_sources_deduplicates_by_source():
    docs = [_doc("t1", "a.pdf"), _doc("t2", "a.pdf"), _doc("t3", "b.pdf")]
    sources = rag.extract_sources(docs)
    assert [s["source"] for s in sources] == ["a.pdf", "b.pdf"]


def test_history_to_messages_maps_roles():
    history = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "привет!"},
    ]
    messages = rag.history_to_messages(history)
    assert messages[0].content == "привет"
    assert messages[1].content == "привет!"


def test_rerank_documents_orders_by_score_and_truncates():
    docs = [_doc("нерелевантно"), _doc("релевантно"), _doc("средне")]

    fake_reranker = SimpleNamespace(predict=lambda pairs: [0.1, 0.9, 0.5])

    top = rag.rerank_documents("вопрос", docs, top_n=2, reranker=fake_reranker)

    assert [d.page_content for d in top] == ["релевантно", "средне"]


def test_rerank_documents_empty_input_returns_empty():
    fake_reranker = SimpleNamespace(predict=lambda pairs: [])
    assert rag.rerank_documents("вопрос", [], top_n=5, reranker=fake_reranker) == []
