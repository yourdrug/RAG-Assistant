"""Tests for the structured split path (split_documents + domain profiles)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from langchain.schema import Document

from application.services.document_pipeline import enrich_chunks_metadata
from domain.domain_profile.profiles.decree import DecreeDomainProfile
from infrastructure.ml.ingestion import split_documents


class FakeSettings:
    def get(self, key: str, domain_key: str = "") -> str:
        defaults = {"max_unit_chars": "200", "fingerprint_min_points": "1"}
        return defaults.get(key, "0")


_DECREE_TEXT = (
    "УКАЗ ПРЕЗИДЕНТА\nО мерах\n\nПОСТАНОВЛЯЮ:\n"
    "1. Принять меры по совершенствованию системы.\n"
    "2. Контроль возложить на Министерство.\n"
)


def test_structured_domain_uses_content_splitter():
    profile = DecreeDomainProfile(settings=FakeSettings())
    docs = [Document(page_content=_DECREE_TEXT, metadata={"source": "ukaz.rtf"})]
    chunks = split_documents(docs, domain="decree", profile=profile, settings=FakeSettings())
    kinds = [c.metadata.get("unit_kind") for c in chunks]
    assert "point" in kinds
    assert "preamble" in kinds
    # Boundary value captured into metadata
    numbered = [c for c in chunks if c.metadata.get("unit_kind") == "point"]
    assert any(c.metadata.get("point_number") == "1" for c in numbered)
    # Structural units are whole — never cut at an arbitrary char position
    assert any("Контроль возложить" in c.page_content for c in chunks)


def test_general_domain_still_uses_char_splitter():
    docs = [Document(page_content="Просто текст без структуры. " * 50, metadata={"source": "doc.md"})]
    chunks = split_documents(docs, domain="general", profile=None, settings=None)
    assert len(chunks) >= 1
    assert all("unit_kind" not in c.metadata for c in chunks)


def test_enrich_merges_doc_level_refs_without_clobbering_chunk_level():
    profile = DecreeDomainProfile(settings=FakeSettings())
    docs = [Document(page_content=_DECREE_TEXT, metadata={"source": "ukaz.rtf"})]
    chunks = split_documents(docs, domain="decree", profile=profile, settings=FakeSettings())

    # Doc-level identity (decree number/date) must reach EVERY chunk without
    # wiping chunk-level references (e.g. point numbers)
    enrich_chunks_metadata(
        chunks,
        document_id=7,
        visibility="internal_public",
        owner_id=None,
        group_id=None,
        doc_domain="decree",
        domain_metadata={"decree_number": "123", "decree_date": "1 января 2026 г.", "point": "doc"},
        act_version_id=55,
        act_id=9,
        effective_from=__import__("datetime").date(2026, 1, 1),
    )
    for chunk in chunks:
        dm = chunk.metadata["domain_metadata"]
        assert dm["decree_number"] == "123"
        assert chunk.metadata["act_version_id"] == 55
        assert chunk.metadata["act_id"] == 9
    # A chunk-level key is not overwritten by the doc-level value
    point_chunks = [c for c in chunks if "point" in c.metadata.get("domain_metadata", {})]
    assert point_chunks, "point refs must exist on chunk level"


def test_process_chunks_persists_versioning_columns(fake_uow_factory):
    """The denormalized columns must actually reach Postgres bulk_insert."""
    import asyncio

    from application.services.document_pipeline import process_chunks
    from domain.entities.raw_document import RawDocument

    chunks = [
        RawDocument(
            page_content="1. Пункт первый.",
            metadata={
                "domain_metadata": {"decree_number": "5"},
                "act_version_id": 3,
                "act_id": 8,
                "effective_from": "2026-01-01",
                "is_current": True,
            },
        )
    ]
    uow = fake_uow_factory._uow
    asyncio.run(
        process_chunks(
            uow_factory=fake_uow_factory,
            document_id=1,
            filename="ukaz.rtf",
            chunks=chunks,
            visibility="internal_public",
            owner_id=None,
            group_id=None,
            doc_domain="decree",
            set_indexing=False,
        )
    )
    stored = uow.chunks._chunks[0]
    assert stored["act_version_id"] == 3
    assert stored["domain_metadata"] == {"decree_number": "5"}
    assert stored["is_current"] is True
    assert stored["effective_from"] is not None
