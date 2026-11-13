"""Regression coverage for faithful context delivery."""

import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
import pytest
from sqlalchemy.dialects.postgresql import dialect

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.ingestion.embed_text import context_prefix
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.evidence.table_context import row_matches_query
from domain.shared.breadcrumb import shorten_breadcrumb
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.tokenizer import tokenize
from infrastructure.ml.ingestion.table_chunks import table_document_chunks
from infrastructure.ml.ingestion.text_chunks import text_document_chunks
from infrastructure.ml.rag.context_rendering import build_header, render_selected_docs
from infrastructure.ml.rag.evidence_focus import format_generation_messages
from infrastructure.ml.rag.generation_budget import prepare_generation_budget
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens
from infrastructure.ml.rag.retrieval_work import RetrievalWork
from infrastructure.repositories.chunk.chunk_search import (
    build_exact_search_stmt,
    build_provision_search_stmt,
)
from tests.test_rag_pipeline import make_rag


def test_numeric_table_key_requires_explicit_marker() -> None:
    row = "| 2 | Справка о доходах | 5 дней |"
    assert not row_matches_query(row, "Сколько дней на заявление, если прошло 2 недели?")
    assert row_matches_query(row, "Формат поля 2?")
    assert row_matches_query("| 1 | Заявление | 10 дней |", "Срок подачи заявления?")


def test_breadcrumb_keeps_specific_unit_in_header_and_prefix() -> None:
    path = "Раздел II " + "Общие положения " * 10 + " > Статья 31. Ставки налога"
    assert shorten_breadcrumb("A > B", 20) == "A > B"
    assert len(shorten_breadcrumb("X" * 100, 20)) == 20
    doc = Document(page_content="body", metadata={"section": path})
    assert "Статья 31" in build_header(doc, 1)
    assert "Статья 31" in context_prefix(doc.metadata, 550)


@pytest.mark.parametrize(
    "indices,versions,count", [((4, 5), (1, 1), 1), ((3, 5), (1, 1), 2), ((4, 5), (1, 2), 2)]
)
def test_order_overlap_and_version_boundaries(
    indices: tuple[int, int], versions: tuple[int, int], count: int
) -> None:
    shared = "Общий фрагмент, который должен встретиться только один раз."
    docs = [
        Document(
            page_content="Начало " + shared,
            metadata={"document_id": 1, "chunk_index": indices[0], "act_version_id": versions[0]},
        ),
        Document(
            page_content=shared + " Конец",
            metadata={"document_id": 1, "chunk_index": indices[1], "act_version_id": versions[1]},
        ),
    ]
    context, selected = render_selected_docs(list(reversed(docs)))
    assert context.count(shared) == count
    assert context.index("Начало") < context.index("Конец")
    assert {doc.metadata["citation_id"] for doc in selected} == {1}


def test_parent_scope_once_per_document() -> None:
    parent = "Общее условие для всех пунктов статьи."
    docs = [
        Document(
            page_content=f"Пункт {i}",
            metadata={"document_id": document_id, "chunk_index": i, "parent_units": [{"content": parent}]},
        )
        for document_id in (1, 2)
        for i in range(3)
    ]
    context, _ = render_selected_docs(docs)
    assert context.count(parent) == 2
    assert context.index(parent) < context.index("Пункт 0")


def test_real_ingestion_overlap() -> None:
    source = Document(
        page_content=" ".join(f"Содержание предложения номер {i}." for i in range(60)),
        metadata={"document_id": 1, "section": "Статья 31"},
    )
    chunks = text_document_chunks(source, 550, 200)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    first, second = chunks[:2]
    prefix = first.metadata["context_prefix"]
    first_body = first.page_content.removeprefix(prefix)
    second_body = second.page_content.removeprefix(prefix)
    overlap = next(
        size
        for size in range(min(len(first_body), len(second_body)), 29, -1)
        if first_body.endswith(second_body[:size])
    )
    context, _ = render_selected_docs([second, first])
    assert context.count(second_body[:overlap]) == 1


def test_compound_sparse_numbers_are_distinct() -> None:
    assert "3.2" in tokenize("пункт 3.2")
    assert "3.2" not in tokenize("пункт 2.3")
    assert {"3", "2"} <= set(tokenize("пункт 3.2"))


@pytest.mark.parametrize("separator", ["", "| --- | --- |\n"])
def test_markdown_first_data_row_is_not_a_header(separator: str) -> None:
    text = "| Документ | Срок |\n" + separator + "| Заявление | 10 дней |\n| Паспорт | 5 дней |"
    chunks = table_document_chunks(Document(page_content=text), 550, 1)
    assert "Заявление" not in chunks[0].metadata["table_header"]
    assert sum(chunk.page_content.count("Заявление") for chunk in chunks) == 1


def test_exact_sql_uses_metadata_and_line_start_without_numeric_continuation() -> None:
    stmt = build_provision_search_stmt("Подпункт 1.4", [], 42, 5)
    compiled = stmt.compile(dialect=dialect())  # type: ignore[no-untyped-call]
    assert "subpoint_num_number" in compiled.params.values()
    assert any(
        "1\\.4[.)][[:space:]]" in value for value in compiled.params.values() if isinstance(value, str)
    )
    assert "chunk_index" in str(compiled).split("ORDER BY")[1]


@pytest.mark.parametrize("query", ["в соответствии со статьей 5 настоящего Закона", "пункт 3", "2024"])
def test_user_exact_search_matches_the_literal_phrase(query: str) -> None:
    compiled = build_exact_search_stmt(query, [], None, 5).compile(dialect=dialect())  # type: ignore[no-untyped-call]
    params = [value for value in compiled.params.values() if isinstance(value, str)]
    where = str(compiled).split("WHERE", 1)[1]
    assert any(re.escape(query) in value for value in params)
    assert "context_metadata" not in where
    assert "false" not in where.lower()


def test_sparse_ranks_compound_reference_above_permuted_number() -> None:
    index = BM25Index(texts=["пункт 2.3 требования заявления", "пункт 3.2 требования заявления"])
    assert index.search("пункт 3.2")[0][0] == 1


@pytest.mark.parametrize("breadth", ["narrow", "broad"])
@pytest.mark.parametrize("with_history", [False, True])
def test_shared_generation_budget_counts_actual_base_prompt(breadth: str, with_history: bool) -> None:
    user = UserContext(7, UserKind.INTERNAL, UserRole.USER)
    history = (
        [HumanMessage(content="Предыдущий вопрос"), AIMessage(content="Предыдущий ответ")]
        if with_history
        else []
    )
    state = RetrievalWork(
        rag=make_rag(),
        t_pipeline_start=0,
        question="Какие документы нужны?",
        ctx=RagContext(user, summary="Обсуждали подачу заявления"),
        user=user,
        access_filter=None,
        retrieval_filter=None,
        breadth=Breadth(breadth),
        history_messages=history,
    )
    prompt, fitted, num_ctx, num_predict, remaining = prepare_generation_budget(state)
    messages = format_generation_messages(prompt, context="", history=fitted, question=state.question)
    assert remaining == num_ctx - num_predict - estimate_message_tokens(messages)
    assert fitted == history


def test_prompt_keeps_provision_heading_stripped_with_ingestion_prefix() -> None:
    heading = "Статья 12. Порядок подачи заявления"
    source = Document(
        page_content=heading + "\n" + " ".join(f"Положение номер {i} продолжается." for i in range(60)),
        metadata={"document_id": 1, "section": "Глава 2", "unit_heading_text": heading},
    )
    chunks = text_document_chunks(source, 550, 100)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    middle = chunks[2]
    assert heading in middle.metadata["context_prefix"]

    context, _ = render_selected_docs([middle])

    assert context.count(heading) == 1
    assert context.index(heading) < context.index("Положение номер")


@pytest.mark.parametrize(
    "query",
    ["значение поля наименования", "какое наименование указать", "заполнение по наименованию"],
)
def test_table_row_name_matches_inflected_query(query: str) -> None:
    assert row_matches_query("| 12 | Наименование | Строка |", query)


def test_table_row_name_does_not_match_unrelated_query() -> None:
    assert not row_matches_query("| 12 | Наименование | Строка |", "код страны происхождения")
    assert not row_matches_query("| 3 | Справка о доходах | 5 дней |", "срок подачи заявления")


@pytest.mark.parametrize("delimiter", ["|-|-|", "|:-:|--|", "| -- | :- |"])
def test_short_gfm_delimiter_row_stays_in_table_header(delimiter: str) -> None:
    rows = "\n".join(f"| {i} | значение {i} |" for i in range(6))
    doc = Document(page_content=f"| A | B |\n{delimiter}\n{rows}", metadata={})

    chunks = table_document_chunks(doc, 200, 2)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata["table_header"] == f"| A | B |\n{delimiter}"
        assert chunk.page_content.count(delimiter) == 1
