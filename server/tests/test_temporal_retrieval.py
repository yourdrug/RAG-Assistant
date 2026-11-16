"""Source act info: the act number comes from document references, edition fields from the pack."""

from __future__ import annotations

from langchain.schema import Document
import pytest

from infrastructure.ml.rag.rag_sources import extract_sources


pytestmark = pytest.mark.generic_evidence


def test_neutral_profile_shows_the_act_number_but_no_pack_fields() -> None:
    docs = [
        (
            Document(
                page_content="text",
                metadata={
                    "source": "ukaz-123.rtf",
                    "document_id": 1,
                    "pack_metadata": {"effective_from": "2026-02-01", "effective_to": "2027-01-01"},
                    "domain_metadata": {"decree_number": "123"},
                },
            ),
            0.9,
        )
    ]
    sources = extract_sources(docs)
    assert sources[0]["act_number"] == "123"
    assert "effective_from" not in sources[0]
    assert "effective_to" not in sources[0]


def test_sources_without_act_info_unchanged() -> None:
    docs = [(Document(page_content="text", metadata={"source": "doc.md", "document_id": 2}), 0.8)]
    sources = extract_sources(docs)
    assert "act_number" not in sources[0]
    assert "effective_from" not in sources[0]
