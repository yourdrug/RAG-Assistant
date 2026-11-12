"""Readable provision paths, bounded context and body-only lexical search."""

from langchain_core.documents import Document
import pytest

from domain.identity.roles import UserKind, UserRole
from domain.knowledge.access_control import get_visibility_conditions
from domain.knowledge.chunk import extract_chunk_context
from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.ranking import provision_identity
from domain.shared.hashing import content_hash
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.document_text import bm25_document_text
from infrastructure.ml.ingestion.table_chunks import table_document_chunks
from infrastructure.ml.ingestion.text_chunks import context_prefix
from infrastructure.repositories.vector.qdrant_ops import build_points
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME, encode_sparse
from tests.test_unit_packing import legal_chunks


def test_readable_path_preserves_titles_and_normalized_provision_numbers() -> None:
    text = (
        "Глава 1. Общие положения\nСтатья 1. Термины\n1. Условие действует. "
        + "Дополнительное предложение применяется к этому пункту. " * 55
    )
    chunks = legal_chunks(text)
    assert len(chunks) > 1
    assert {chunk.metadata["section"] for chunk in chunks} == {
        "Глава 1. Общие положения > Статья 1. Термины > п. 1"
    }
    assert all(chunk.metadata["article_number"] == "1" for chunk in chunks)
    assert all(chunk.metadata["point_number"] == "1" for chunk in chunks)

    for chunk in chunks:
        chunk.metadata.update(document_id=1, act_version_id=2)

    assert len({provision_identity(chunk) for chunk in chunks}) == 1


def test_legacy_sentence_path_keeps_the_same_provision_identity() -> None:
    docs = [
        Document(page_content="body", metadata={"document_id": 1, "section": section})
        for section in ("point 1 > subpoint_num 1.2", "point 1 > subpoint_num 1.2 > sentence")
    ]
    assert provision_identity(docs[0]) == provision_identity(docs[1])


def test_two_line_article_title_does_not_emit_a_bare_marker_chunk() -> None:
    chunks = legal_chunks(
        "Статья 9.\nПолномочия Совета Министров\n" + "Правительство определяет порядок обмена. " * 50
    )
    assert len(chunks) > 1
    assert all(chunk.metadata["body_text"] != "Статья 9." for chunk in chunks)
    assert all("Полномочия Совета Министров" in chunk.metadata["section"] for chunk in chunks)


def test_prefix_orders_path_and_keeps_only_nearest_substantive_introduction() -> None:
    metadata = {
        "doc_title": "Закон об обмене",
        "section": "Глава 1. Правила > Статья 2. Обязанности > п. 1 > подп. а)",
        "parent_units": [
            {"heading": "Глава 1. Правила", "content": "Глава 1. Правила\nДальнее условие."},
            {"heading": "Статья 2. Обязанности", "content": "Статья 2. Обязанности"},
            {"heading": "п. 1", "content": "1. Работодатель обязан:"},
        ],
    }
    prefix = context_prefix(metadata, 1000)
    assert prefix == (
        "Закон об обмене\nГлава 1. Правила > Статья 2. Обязанности > п. 1 > подп. а)\n"
        "1. Работодатель обязан:\n\n"
    )


def test_dense_scope_does_not_repeat_introduction_carried_by_the_prefix() -> None:
    intro = "В целях настоящего Закона стороны трудового договора обязаны соблюдать следующие требования:"
    chunks = legal_chunks(
        f"Глава 1. Общие положения\nСтатья 1. Обязанности сторон\n{intro}\n"
        "1. Работодатель обязан обеспечить условия. "
        + "Дополнительное требование к условиям труда работника. " * 12
        + "\n2. Работник обязан соблюдать дисциплину. "
        + "Дополнительное требование к дисциплине работника. " * 12
    )
    assert len(chunks) == 2

    for chunk in chunks:
        assert intro in chunk.metadata["context_prefix"]
        scoped = content_with_parent_scope(chunk)
        assert scoped.count(intro) == 1
        assert scoped.count("Статья 1. Обязанности сторон") == 1


def test_dense_scope_restores_heading_when_the_path_dropped_it() -> None:
    doc = Document(
        page_content="п. 1\nВступительное условие:\n\n1. Тело.",
        metadata={
            "parent_units": [
                {"heading": "Статья 2. Права", "content": "Статья 2. Права\nВступительное условие:"}
            ]
        },
    )
    assert content_with_parent_scope(doc) == "Статья 2. Права\n" + doc.page_content


def test_prompt_body_without_prefix_receives_the_complete_parent_scope() -> None:
    parent = "Статья 2. Права\nВступительное условие:"
    doc = Document(
        page_content="1. Тело.",
        metadata={"parent_units": [{"heading": "Статья 2. Права", "content": parent}]},
    )
    assert content_with_parent_scope(doc) == parent + "\n1. Тело."


def test_body_on_the_article_marker_line_is_not_used_as_its_title() -> None:
    sentence = "для которых установлены иные сроки вступления в силу и порядок применения. "
    chunks = legal_chunks(
        "Глава 5. Заключительные положения\nСтатья 31. Настоящий Закон вступает в силу "
        "через шесть месяцев после его официального опубликования, за исключением положений, "
        + sentence * 6
        + "\nСтатья 32. Совету Министров привести решения в соответствие с Законом.\n"
    )
    long_article = [chunk for chunk in chunks if chunk.metadata.get("article_number") == "31"]
    assert long_article
    assert {chunk.metadata["section"] for chunk in long_article} == {
        "Глава 5. Заключительные положения > Статья 31"
    }
    assert all(chunk.metadata["unit_heading_text"] == "Статья 31" for chunk in long_article)
    assert all("Настоящий Закон вступает в силу" in chunk.metadata["body_text"] for chunk in long_article)


def test_narrow_prefix_drops_distant_chapter_before_article() -> None:
    prefix = context_prefix(
        {"section": "Глава 1. " + "Дальнее название " * 20 + " > Статья 2. Обязанности > п. 1"},
        300,
    )
    assert "Глава 1" not in prefix
    assert "Статья 2. Обязанности > п. 1" in prefix


def test_prefix_does_not_repeat_headings_title_or_body_introduction() -> None:
    heading = "Статья 1. Термины"
    metadata = {
        "doc_title": "Закон",
        "section": f"Закон > Глава 1 > {heading}",
        "parent_units": [{"heading": "Глава 1", "content": "Глава 1"}],
    }
    content = context_prefix(metadata, 1000, body=heading + "\nТело нормы.") + heading + "\nТело нормы."
    assert content.count(heading) == 1
    assert content.count("Закон") == 1
    assert content.count("Глава 1") == 1


def test_long_document_title_is_bounded() -> None:
    prefix = context_prefix({"doc_title": "Длинное название " * 20}, 550)
    assert len(prefix.strip()) <= 80


def test_table_title_does_not_force_a_fitting_row_to_split() -> None:
    header = "| Номер | Значение |\n|---|---|"
    row = "| 1 | " + "значение " * 50 + "|"
    title = "Название документа " * 5
    doc = Document(page_content=header + "\n" + row, metadata={"doc_title": title})
    chunks = table_document_chunks(doc, 550, 15)
    assert len(chunks) == 1
    assert row in chunks[0].metadata["body_text"]
    assert chunks[0].metadata["context_prefix"] == ""
    assert chunks[0].metadata["doc_title"] == title


def test_table_keeps_document_title_when_the_row_fits() -> None:
    doc = Document(
        page_content="| Номер | Значение |\n|---|---|\n| 1 | Короткая строка |",
        metadata={"doc_title": "Правила"},
    )
    chunk = table_document_chunks(doc, 550, 15)[0]
    assert chunk.metadata["context_prefix"] == "Правила\n\n"


def test_large_introduction_stays_atomic_in_metadata_without_consuming_body_budget() -> None:
    intro = "Обязательное условие применяется при подаче заявления. " * 20
    parents = [{"content": intro}]
    metadata = {"doc_title": "Закон", "section": "Статья 1 > п. 1", "parent_units": parents}
    prefix = context_prefix(metadata, 1000)
    assert intro.strip() not in prefix
    assert len(prefix) <= 300
    assert parents[0]["content"] == intro


def test_body_only_bm25_preserves_hash_acl_and_snapshot_roundtrip() -> None:
    docs = [
        Document(
            page_content="Префикс\n\nПодача заявления",
            metadata={
                "context_prefix": "Префикс\n\n",
                "body_text": "Подача заявления",
                "visibility": "internal_private",
                "owner_id": owner,
            },
        )
        for owner in (42, 99)
    ]
    # Identical content in two ACL scopes still must be filtered before scoring.
    conditions = get_visibility_conditions(UserKind.INTERNAL, [], for_list=False, user_role=UserRole.USER)
    index = BM25Index.from_dict(BM25Index.from_documents(docs).to_dict())
    assert index.texts == ["Подача заявления"] * 2
    assert index.hashes == [content_hash(doc.page_content) for doc in docs]
    assert index.search("Префикс") == []
    hits = index.search_with_hashes(
        "заявления", visibility_conditions=conditions, user_id=42, user_group_ids=[]
    )
    assert len(hits) == 1
    assert hits[0][0] == content_hash(docs[0].page_content)
    assert (
        index.search_with_hashes("заявления", visibility_conditions=conditions, user_id=7, user_group_ids=[])
        == []
    )


@pytest.mark.parametrize("manual_edit", [False, True])
def test_qdrant_sparse_uses_body_and_full_payload_identity(*, manual_edit: bool) -> None:
    body = "Подача заявления"
    prefix = "Глава 1 > Статья 2\n\n"
    doc = Document(
        page_content=prefix + body,
        metadata={"chunk_id": 1, "context_prefix": prefix, "body_text": body},
    )

    if manual_edit:
        doc.page_content = "Отредактированный текст"

    expected = doc.page_content if manual_edit else body
    assert bm25_document_text(doc) == expected
    point = build_points([doc], [[1.0, 0.0]])[0]
    assert isinstance(point.vector, dict)
    assert point.vector[SPARSE_VECTOR_NAME] == encode_sparse(expected)
    assert point.payload is not None
    assert point.payload["page_content"] == doc.page_content
    assert extract_chunk_context(doc.metadata)["body_text"] == body
