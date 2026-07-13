"""
Тесты для чистой логики rag_chain.py: форматирование контекста, извлечение
источников, конвертация истории и реранк (с замоканным CrossEncoder — без
реальной загрузки bge-reranker-v2-m3).
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag_chain  # noqa: E402


def _doc(content: str, source: str = "docs/a.pdf"):
    return SimpleNamespace(page_content=content, metadata={"source": source})


def test_format_docs_numbers_and_separates_chunks():
    docs = [_doc("текст 1", "a.pdf"), _doc("текст 2", "b.pdf")]
    out = rag_chain.format_docs(docs)
    assert "[1] a.pdf" in out
    assert "[2] b.pdf" in out
    assert "---" in out


def test_extract_sources_deduplicates_by_source():
    docs = [_doc("t1", "a.pdf"), _doc("t2", "a.pdf"), _doc("t3", "b.pdf")]
    sources = rag_chain.extract_sources(docs)
    assert [s["source"] for s in sources] == ["a.pdf", "b.pdf"]


def test_history_to_messages_maps_roles():
    history = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "привет!"},
    ]
    messages = rag_chain.history_to_messages(history)
    assert messages[0].content == "привет"
    assert messages[1].content == "привет!"


def test_rerank_documents_orders_by_score_and_truncates():
    docs = [_doc("нерелевантно"), _doc("релевантно"), _doc("средне")]

    fake_reranker = SimpleNamespace(predict=lambda pairs: [0.1, 0.9, 0.5])

    with patch.object(rag_chain, "get_reranker", return_value=fake_reranker):
        top = rag_chain.rerank_documents("вопрос", docs, top_n=2)

    assert [d.page_content for d in top] == ["релевантно", "средне"]


def test_rerank_documents_empty_input_returns_empty():
    assert rag_chain.rerank_documents("вопрос", [], top_n=5) == []
