"""Tests for temporal retrieval (with_temporal_filter) and source act info."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from langchain.schema import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue

from infrastructure.repositories.vector.acl import with_temporal_filter
from infrastructure.ml.rag import extract_sources


def _acl_filter() -> Filter:
    return Filter(
        should=[FieldCondition(key="metadata.visibility", match=MatchValue(value="internal_public"))]
    )


class TestWithTemporalFilter:
    def test_as_of_none_requires_current(self):
        result = with_temporal_filter(Filter(), None)
        cond = result.must[0]
        assert cond.key == "metadata.is_current"
        assert cond.match.value is True

    def test_as_of_date_combines_with_acl_filter(self):
        result = with_temporal_filter(_acl_filter(), date(2026, 1, 1))
        # ACL conditions + temporal condition both in must
        assert len(result.must) == 2
        temporal = result.must[1]
        # Two should-groups: effective_from <= date, effective_to > date
        assert len(temporal.must) == 2
        for group in temporal.must:
            # NULL dates always pass — IsNullCondition must be an alternative
            assert len(group.should) == 2

    def test_as_of_date_without_acl(self):
        result = with_temporal_filter(Filter(), date(2026, 6, 1))
        assert len(result.must) == 1
        group = result.must[0].must[0]
        ranges = [c for c in group.should if hasattr(c, "range")]
        assert ranges, "range condition for effective_from must be present"


class TestSourceActInfo:
    def test_sources_expose_act_number_and_effective_dates(self):
        docs = [
            (
                Document(
                    page_content="text",
                    metadata={
                        "source": "ukaz-123.rtf",
                        "document_id": 1,
                        "effective_from": "2026-02-01",
                        "effective_to": "2027-01-01",
                        "domain_metadata": {"decree_number": "123"},
                    },
                ),
                0.9,
            )
        ]
        sources = extract_sources(docs)
        assert sources[0]["act_number"] == "123"
        assert sources[0]["effective_from"] == "2026-02-01"
        assert sources[0]["effective_to"] == "2027-01-01"

    def test_sources_without_act_info_unchanged(self):
        docs = [
            (
                Document(page_content="text", metadata={"source": "doc.md", "document_id": 2}),
                0.8,
            )
        ]
        sources = extract_sources(docs)
        assert "act_number" not in sources[0]
        assert "effective_from" not in sources[0]
