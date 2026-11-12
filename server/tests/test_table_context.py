"""Table identity, focused row continuation and bounded expansion contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from langchain.schema import Document
import pytest

from domain.identity.roles import UserKind
from domain.identity.user_context import UserContext
from domain.knowledge.chunk import TABLE_CONTEXT_MAX_CHUNKS, ChunkRecord, extract_chunk_context
from domain.knowledge.enums import PageContentType
from domain.rag.evidence.table_context import requests_table_overview
from infrastructure.ml.ingestion.sections import MAX_TABLE_INTRODUCTION_CHARS, Section, table_context_sections
from infrastructure.ml.ingestion.splitting import split_documents
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser
from infrastructure.ml.rag.prompt_budget import estimate_text_tokens
from infrastructure.ml.rag.rag_postprocess import enrich_with_neighbors
from infrastructure.ml.rag.table_context import focus_table_rows


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def two_tables(tmp_path: Path) -> list[Document]:
    path = tmp_path / "tables.md"
    rows = "\n".join(f"| {i} | value-{i} | condition-{i} |" for i in range(1, 21))
    path.write_text(
        "# Rules\n\n| Field | Format | Condition |\n|---|---|---|\n"
        + rows
        + "\n\nINTERMEDIATE TEXT\n\n| Other | Value |\n|---|---|\n| SECOND TABLE | forbidden |\n"
    )
    return split_documents(
        [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in LangchainDocumentParser().parse(path)
        ]
    )


def test_table_identity_survives_splitting_and_context_selection(two_tables: list[Document]) -> None:
    tables = [doc for doc in two_tables if doc.metadata.get("content_type") == PageContentType.TABLE.value]
    ids = [doc.metadata["table_id"] for doc in tables]
    assert len(set(ids)) == 2
    assert ids.count(ids[0]) > 1
    assert all(extract_chunk_context(doc.metadata)["table_id"] == doc.metadata["table_id"] for doc in tables)
    assert all("table_id" not in doc.metadata for doc in two_tables if doc not in tables)


def table_seed() -> Document:
    header = "| Field | Format | Condition |\n|---|---|---|"
    return Document(
        page_content="Scope: applies only to new documents.\n"
        + header
        + "\n| 18 | an..15 | unchanged |\n| 23 | an..19 | only after approval |",
        metadata={
            "document_id": 7,
            "chunk_index": 2,
            "content_hash": "seed",
            "content_type": PageContentType.TABLE.value,
            "table_id": "table-1",
            "table_header": header,
            "table_row_start": 1,
            "table_row_end": 2,
        },
    )


@pytest.mark.asyncio
async def test_target_field_keeps_columns_scope_and_split_row_conditions() -> None:
    seed = table_seed()
    continuation = ChunkRecord(
        chunk_id=3,
        document_id=7,
        filename="tables.md",
        chunk_index=3,
        content="Scope: applies only to new documents.\n"
        + seed.metadata["table_header"]
        + "\n| 23 | an..19 | except revoked approval |",
        content_hash="continuation",
        content_type=PageContentType.TABLE.value,
        context_metadata={**seed.metadata, "table_row_start": 2, "table_row_end": 2},
    )
    search = SimpleNamespace(get_table_batches=AsyncMock(return_value=[continuation]))
    user = UserContext(user_id=1, user_kind=UserKind.INTERNAL)
    result = await enrich_with_neighbors(
        [(seed, 0.9)], enumerate_cases=False, chunk_search=search, user=user, query="Формат поля 23?"
    )
    text = "\n".join(doc.page_content for doc, _ in result)
    assert "an..15" not in text
    assert "an..19" in text
    assert "only after approval" in text
    assert "except revoked approval" in text
    assert "applies only to new documents" in text
    assert "| Field | Format | Condition |" in text
    assert len(result[0][0].page_content) < len(seed.page_content)
    assert estimate_text_tokens(text) <= estimate_text_tokens(seed.page_content + "\n" + continuation.content)
    args = search.get_table_batches.call_args
    assert args.args == (7, "table-1")
    assert args.kwargs["row_start"] == args.kwargs["row_end"] == 2
    assert args.kwargs["user"] is user
    assert args.kwargs["limit"] == TABLE_CONTEXT_MAX_CHUNKS


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["Перечисли все поля таблицы", "Сравни строки таблицы"])
async def test_only_explicit_overview_fetches_whole_table(query: str) -> None:
    search = SimpleNamespace(get_table_batches=AsyncMock(return_value=[]))
    await enrich_with_neighbors(
        [(table_seed(), 0.9)],
        enumerate_cases=True,
        chunk_search=search,
        user=UserContext(user_id=1, user_kind=UserKind.INTERNAL),
        query=query,
    )
    assert search.get_table_batches.call_args.kwargs["row_start"] is None
    assert search.get_table_batches.call_args.kwargs["row_end"] is None


@pytest.mark.asyncio
async def test_conditional_rules_flag_does_not_expand_unrelated_rows() -> None:
    search = SimpleNamespace(get_table_batches=AsyncMock(return_value=[]))
    await enrich_with_neighbors(
        [(table_seed(), 0.9)],
        enumerate_cases=True,
        chunk_search=search,
        user=UserContext(user_id=1, user_kind=UserKind.INTERNAL),
        query="Формат поля 23 при одобрении?",
    )
    assert search.get_table_batches.call_args.kwargs["row_start"] == 2
    assert not requests_table_overview("Формат поля 23 при одобрении?")


@pytest.mark.asyncio
async def test_legacy_table_without_identity_does_not_expand() -> None:
    seed = table_seed()
    del seed.metadata["table_id"]
    search = SimpleNamespace(get_table_batches=AsyncMock())
    assert await enrich_with_neighbors(
        [(seed, 0.9)],
        enumerate_cases=True,
        chunk_search=search,
        user=UserContext(user_id=1, user_kind=UserKind.INTERNAL),
        query="Перечисли все поля",
    ) == [(seed, 0.9)]
    search.get_table_batches.assert_not_awaited()


@pytest.mark.asyncio
async def test_expansion_respects_context_budget() -> None:
    seed = table_seed()
    continuation = ChunkRecord(
        chunk_id=3,
        document_id=7,
        filename="tables.md",
        chunk_index=3,
        content="large continuation " * 100,
        content_hash="continuation",
    )
    search = SimpleNamespace(get_table_batches=AsyncMock(return_value=[continuation]))
    assert await enrich_with_neighbors(
        [(seed, 0.9)],
        enumerate_cases=True,
        chunk_search=search,
        query="Перечисли все поля",
        max_context_tokens=1,
    ) == [(seed, 0.9)]


def test_focused_field_keeps_following_editorial_note_and_exceptions() -> None:

    seed = table_seed()
    seed.page_content += "\n| (п. 23 в редакции изменения) | | |\n| 24 | n..8 | unrelated |"
    seed.metadata["table_row_end"] = 4
    focused = focus_table_rows(seed, "Формат поля 23?")
    assert "only after approval" in focused.page_content
    assert "п. 23 в редакции изменения" in focused.page_content
    assert "unrelated" not in focused.page_content
    assert "an..15" not in focused.page_content
    assert focused.metadata["table_row_start"] == 2
    assert focused.metadata["table_row_end"] == 3


@pytest.mark.parametrize(
    "query", ["Сравни формат поля 23 с предыдущей редакцией", "Как изменился формат поля 23?"]
)
def test_single_field_version_comparison_does_not_request_whole_table(query: str) -> None:
    assert not requests_table_overview(query)


def test_field_can_be_selected_by_name_without_number() -> None:

    seed = table_seed()
    seed.page_content = seed.metadata["table_header"] + (
        "\n| 18 | УНП грузоотправителя \u2039br\u203a an..15 | unchanged |"
        "\n| 23 | УНП \u2039br\u203a грузополучателя \u2039br\u203a an..19 | only after approval |"
    )
    focused = focus_table_rows(seed, "Какой формат поля «УНП грузополучателя»?")
    assert "an..15" not in focused.page_content
    assert "an..19" in focused.page_content
    assert "only after approval" in focused.page_content
    assert focused.metadata["table_row_start"] == focused.metadata["table_row_end"] == 2


def test_table_introduction_is_the_caption_and_lead_in_not_the_whole_chapter() -> None:
    """A heading-less DOCX chapter before a table must not become every row batch's scope."""
    chapter = "\n".join(f"{number}. Пункт главы с самостоятельным содержанием." for number in range(1, 400))
    lead_in = "Структура сообщения приведена в таблице 4.7:"
    caption = "Таблица 4.7 - структура и формат сообщения"
    sections = [
        Section(None, f"{chapter}\n{lead_in}\n{caption}"),
        Section(None, "| Реквизит | Формат |\n|---|---|\n| Номер | string |", PageContentType.TABLE),
    ]

    scope = table_context_sections(sections)[1].metadata["parent_units"][0]["content"]

    assert scope.endswith(f"{lead_in}\n{caption}")
    assert not any(line.startswith("1. ") for line in scope.splitlines())
    assert len(scope) <= MAX_TABLE_INTRODUCTION_CHARS
