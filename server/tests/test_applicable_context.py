"""Source links are recovered only inside an authorized document and version."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.documents import Document
import pytest

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.enums import PageContentType
from domain.rag.context import RagContext
from domain.rag.evidence.context import content_with_parent_scope
from infrastructure.domain_packs.belarus_law.evidence.applicable_context import enrich_applicable_context
from infrastructure.domain_packs.belarus_law.evidence.grounded_response import (
    extractive_timing_response,
    missing_field_response,
)


USER = RagContext(
    user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
).to_user_context()


pytestmark = pytest.mark.belarus
QUESTION = "Формат поля «Код» (позиция 18 таблицы 4.2, BLRDLN)?"


def source_chunk(index: int, text: str, table_id: str | None = None, version: int = 1) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=index,
        document_id=1,
        chunk_index=index,
        filename="Постановление N 71.docx",
        content=text,
        pack_metadata={"act_version_id": version},
        content_type=PageContentType.TABLE.value if table_id else None,
        context_metadata={"table_id": table_id} if table_id else {},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["", "gap", "version", "other_table"])
async def test_table_scope_requires_contiguous_matching_table_and_version(mutation: str) -> None:
    doc = Document(
        page_content="Теги ЭТН:\n| 18 | Код an..15 |",
        metadata={
            "source": "Постановление N 71.docx",
            "document_id": 1,
            "pack_metadata": {"act_version_id": 1},
            "chunk_index": 12,
            "table_id": "table-2",
            "content_type": PageContentType.TABLE.value,
        },
    )
    neighbors = [
        source_chunk(10, "Таблица 4.2 - структура сообщения BLRDLN"),
        source_chunk(11, "| 1 | Начало |", "table-2"),
        source_chunk(12, doc.page_content, "table-2"),
    ]

    if mutation == "gap":
        neighbors.pop(1)
    elif mutation == "version":
        neighbors[0].pack_metadata = {"act_version_id": 2}
    elif mutation == "other_table":
        neighbors[1].context_metadata["table_id"] = "table-3"

    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=neighbors))
    result = await enrich_applicable_context([(doc, 0.9)], QUESTION, search, USER)
    assert bool(missing_field_response(content_with_parent_scope(result[0][0]), QUESTION)) == bool(mutation)
    assert "parent_units" not in doc.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [1, 2])
async def test_timing_reference_requires_definition_and_appendix_in_same_version(version: int) -> None:
    question = "В каком режиме EDI-провайдер передаёт сведения о товарах, включённых в перечни, по N 71?"
    doc = Document(
        page_content="2.2. сведения, указанные в пункте 1 настоящего постановления:\n"
        "передаются EDI-провайдером в режиме реального времени;",
        metadata={
            "source": "Постановление N 71.docx",
            "document_id": 1,
            "pack_metadata": {"act_version_id": 1},
            "chunk_index": 4,
            "citation_id": 1,
        },
    )
    definition = source_chunk(1, "1. Определить состав сведений согласно приложению.")
    appendix = source_chunk(27, "Приложение\nк постановлению N 71\nСОСТАВ СВЕДЕНИЙ")
    category = source_chunk(30, "3. При передаче сведений о товарах, включенных в перечни:", version=version)
    search = SimpleNamespace(search_substring=AsyncMock(side_effect=[[definition], [appendix], [category]]))
    result = await enrich_applicable_context([(doc, 0.9)], question, search, USER)
    answer = extractive_timing_response(result, question)
    assert bool(answer) == (version == 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [1, 2])
async def test_amendment_target_link_is_explicit_and_same_version(version: int) -> None:
    question = "Изменился ли формат поля «Код» (позиция 18 таблицы 4.2 BLRDLN) постановлением N 71?"
    doc = Document(
        page_content="Таблица 4.2 BLRDLN\n| 18 | Код an..8 | М |",
        metadata={
            "source": "Постановление N 71.docx",
            "document_id": 1,
            "pack_metadata": {"act_version_id": 1},
            "chunk_index": 20,
            "citation_id": 1,
        },
    )
    link = source_chunk(
        10,
        "1. Внести в приложение к структуре, утвержденным постановлением N 58, следующие изменения:",
        version=version,
    )
    search = SimpleNamespace(search_substring=AsyncMock(return_value=[link]))
    result = await enrich_applicable_context([(doc, 0.9)], question, search, USER)
    assert result[0][0].metadata.get("verified_amended_acts") == (["58"] if version == 1 else None)
    assert "verified_amended_acts" not in doc.metadata


TIMING_QUESTION = "В каком режиме EDI-провайдер передаёт сведения о товарах, включённых в перечни, по N 71?"


def timing_doc() -> Document:
    return Document(
        page_content="2.2. сведения, указанные в пункте 1 настоящего постановления:\n"
        "передаются EDI-провайдером в режиме реального времени;",
        metadata={
            "source": "Постановление N 71.docx",
            "document_id": 1,
            "pack_metadata": {"act_version_id": 1},
            "chunk_index": 1,
            "citation_id": 1,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation, answered",
    [("", True), ("gap", False), ("late_heading", False), ("other_category", False)],
)
async def test_timing_reference_follows_split_appendix_heading(mutation: str, *, answered: bool) -> None:
    definition = source_chunk(0, "1. Определить состав сведений согласно приложению.")
    title = source_chunk(5, "СОГЛАСОВАНО\nПриложение\nк постановлению N 71")
    heading = "09.04.2021 N 71\nСОСТАВ\nСВЕДЕНИЙ\n1. Для проверки сведений о товарах, включенных в перечни:"
    continuation = source_chunk(6, heading)
    clauses = [
        source_chunk(8, "3. При передаче накладных при обороте товаров, включенных в перечни:"),
        source_chunk(10, "6. О товарах, включенных в перечни:\n7. При передаче сведений об отмене:"),
    ]

    if mutation == "gap":
        continuation = source_chunk(7, heading)
    elif mutation == "late_heading":
        continuation = source_chunk(6, "1. Сведения о товарах, включенных в перечни.\nСОСТАВ СВЕДЕНИЙ")
    elif mutation == "other_category":
        clauses.append(source_chunk(12, "8. При передаче сведений о товарах, не включенных в перечни:"))

    search = SimpleNamespace(
        search_substring=AsyncMock(side_effect=[[definition], [title], clauses]),
        get_neighbors=AsyncMock(return_value=[title, continuation]),
    )
    result = await enrich_applicable_context([(timing_doc(), 0.9)], TIMING_QUESTION, search, USER)
    answer = extractive_timing_response(result, TIMING_QUESTION)
    assert bool(answer) == answered

    if answered:
        assert answer is not None
        assert "в режиме реального времени" in answer


FIELD_QUESTION = (
    "Изменился ли формат поля «УНП грузоотправителя» (позиция 18 таблицы 4.2, BLRDLN) "
    "постановлением N 71 так же, как формат поля «УНП грузополучателя» (позиция 23)?"
)


@pytest.mark.parametrize(
    "body_table, found", [("4.2 - структура BLRDLN (ЭТН)", True), ("4.1 - BLRWBL", False)]
)
def test_field_row_is_bound_to_preceding_table_heading(body_table: str, *, found: bool) -> None:
    # Chunk metadata in the citation header names an earlier table of the same chunk.
    context = (
        "[2] Постановление N 71 | Таблица 4.1 - структура и формат сообщения BLRWBL (ЭТТН) | (таблица)\n"
        f"Таблица {body_table}\n"
        "| 18 | УНП грузоотправителя <br> an..15 | 2:М | <VATRegistrationNumber> | УНП |"
    )
    missing = missing_field_response(context, FIELD_QUESTION) or ""
    assert ("Целевая строка 18" in missing) != found
