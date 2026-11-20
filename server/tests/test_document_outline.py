"""A cited article or table absent from the named act is stated from the act's own outline."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
import pytest

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.document_outline import (
    CitedUnit,
    DocumentOutline,
    cited_units,
    compact_numbers,
    missing_unit_note,
)
from domain.rag.ranking import RETAINED_AS_REQUESTED
from infrastructure.domain_packs.belarus_law.evidence.profile import BelarusEvidenceProfile
from infrastructure.ml.evidence_profiles.profiles import evidence_profile_scope
from infrastructure.ml.rag.document_bounds import add_missing_unit_notes


LAW = "Закон Республики Беларусь от 28.12.2009 № 113-З (ред. от 16.03.2026) Об электронном документе.docx"
TNPA = "Постановление НАН Беларуси, Минфина РБ, МНС РБ от 18.12.2023 N 9753526.docx"
LAW_OUTLINE = DocumentOutline(articles=(*map(str, range(1, 34)), "26-1"), tables=())
TNPA_OUTLINE = DocumentOutline(articles=(), tables=("3.1", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7"))
USER = UserContext(user_id=7, user_kind=UserKind.INTERNAL, user_role=UserRole.USER, group_ids=())


class OutlineRepository:
    def __init__(self, outlines: dict[int, DocumentOutline]) -> None:
        self.outlines = outlines
        self.calls: list[tuple[int, UserContext]] = []

    async def get_document_outline(self, document_id: int, *, user: UserContext) -> DocumentOutline:
        self.calls.append((document_id, user))
        return self.outlines.get(document_id, DocumentOutline(articles=(), tables=()))


def chunk(text: str, source: str, document_id: int, **metadata: Any) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "source": source,
            "doc_title": source,
            "chunk_id": 1,
            **metadata,
        },
    )


def test_cited_units_read_articles_and_tables() -> None:
    assert cited_units("Что говорится в статье 40 Закона N 113-З?") == [CitedUnit("article", "40")]
    assert cited_units("Что описывает таблица 4.8 приложения 4?") == [CitedUnit("table", "4.8")]


def test_compact_numbers_keeps_suffixed_articles_apart() -> None:
    assert compact_numbers(["1", "2", "3", "5", "26-1"]) == "1\N{EN DASH}3, 5, 26-1"
    assert compact_numbers(["4.1", "4.2", "4.3", "4.7"]) == "4.1\N{EN DASH}4.3, 4.7"


def test_absent_article_note_lists_the_act_articles() -> None:
    note = missing_unit_note(CitedUnit("article", "40"), LAW_OUTLINE, "Закон № 113-З")

    assert note is not None
    assert "статьи 1\N{EN DASH}26, 26-1, 27\N{EN DASH}33" in note
    assert "Статьи 40 в этом документе нет" in note


def test_existing_or_unknown_units_give_no_note() -> None:
    assert missing_unit_note(CitedUnit("article", "26-1"), LAW_OUTLINE, "Закон") is None
    assert missing_unit_note(CitedUnit("article", "12"), LAW_OUTLINE, "Закон") is None
    # No outline of the kind: absence cannot be shown, a missed search proves nothing.
    assert missing_unit_note(CitedUnit("table", "1"), LAW_OUTLINE, "Закон") is None


def test_absent_table_is_compared_within_its_group() -> None:
    note = missing_unit_note(CitedUnit("table", "4.8"), TNPA_OUTLINE, "ТНПА")

    assert note is not None
    assert "таблицы 4.1\N{EN DASH}4.7" in note
    assert "3.1" not in note


@pytest.mark.asyncio
async def test_note_for_absent_article_comes_first_with_the_act_metadata() -> None:
    repository = OutlineRepository({18: LAW_OUTLINE})
    retained = [
        (chunk("Статья 11. Полномочия", LAW, 18, section="Статья 11", visibility="internal_public"), 0.4)
    ]

    with evidence_profile_scope(BelarusEvidenceProfile()):
        docs = await add_missing_unit_notes(
            retained, [], "Что говорится в статье 40 Закона Республики Беларусь N 113-З?", repository, USER
        )

    assert len(docs) == 2
    note, score = docs[0]
    assert "Статьи 40 в этом документе нет" in note.page_content
    assert score == 0.4
    assert note.metadata["document_id"] == 18
    assert note.metadata["visibility"] == "internal_public"
    assert note.metadata[RETAINED_AS_REQUESTED] is True
    assert "chunk_id" not in note.metadata
    assert note.metadata["section"] != "Статья 11"
    assert repository.calls == [(18, USER)]


@pytest.mark.asyncio
async def test_existing_article_or_unnamed_act_adds_nothing() -> None:
    repository = OutlineRepository({18: LAW_OUTLINE})
    retained = [(chunk("Статья 11. Полномочия", LAW, 18), 0.4)]

    with evidence_profile_scope(BelarusEvidenceProfile()):
        present = await add_missing_unit_notes(
            retained, [], "Что в статье 12 Закона N 113-З?", repository, USER
        )
        unnamed = await add_missing_unit_notes(retained, [], "Что говорится в статье 40?", repository, USER)

    assert present == retained
    assert unnamed == retained


@pytest.mark.asyncio
async def test_outline_is_read_only_for_documents_the_reader_was_given() -> None:
    """The act is resolved from authorized candidates; no other document is looked up."""
    repository = OutlineRepository({29: TNPA_OUTLINE})
    candidates = [chunk("Таблица 4.1", TNPA, 29)]

    with evidence_profile_scope(BelarusEvidenceProfile()):
        docs = await add_missing_unit_notes(
            [],
            candidates,
            "Что описывает таблица 4.8 приложения 4 (постановление N 9/75/35/26)?",
            repository,
            USER,
        )
        unseen = await add_missing_unit_notes(
            [], [], "Что описывает таблица 4.8 приложения 4 (постановление N 9/75/35/26)?", repository, USER
        )

    assert [doc.metadata["document_id"] for doc, _ in docs] == [29]
    assert "Таблицы 4.8 в этом документе нет" in docs[0][0].page_content
    assert unseen == []
    assert repository.calls == [(29, USER)]
