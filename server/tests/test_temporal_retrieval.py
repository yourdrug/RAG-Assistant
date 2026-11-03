"""Tests for temporal retrieval (with_temporal_filter) and source act info."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from langchain.schema import Document
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    HasIdCondition,
    IsEmptyCondition,
    IsNullCondition,
    MatchValue,
    NestedCondition,
)

from infrastructure.ml.rag.rag_sources import extract_sources
from infrastructure.repositories.vector.acl import with_temporal_filter


type Condition = (
    FieldCondition | IsEmptyCondition | IsNullCondition | HasIdCondition | NestedCondition | Filter
)


def _acl_filter() -> Filter:
    return Filter(
        should=[FieldCondition(key="metadata.visibility", match=MatchValue(value="internal_public"))]
    )


def clauses(value: list[Condition] | Condition | None) -> list[Condition]:
    assert isinstance(value, list)
    return value


def filter_clause(value: list[Condition] | Condition | None, index: int) -> Filter:
    item = clauses(value)[index]
    assert isinstance(item, Filter)
    return item


def field_clause(value: list[Condition] | Condition | None, index: int) -> FieldCondition:
    item = clauses(value)[index]
    assert isinstance(item, FieldCondition)
    return item


class TestWithTemporalFilter:
    def test_current_query_uses_today_interval_and_current_fallback(self) -> None:
        result = with_temporal_filter(_acl_filter(), None)
        assert len(clauses(result.must)) == 2
        temporal = filter_clause(result.must, 1)
        assert len(clauses(temporal.must)) == 3
        start_range = field_clause(filter_clause(temporal.must, 0).should, 1).range
        end_range = field_clause(filter_clause(temporal.must, 1).should, 1).range
        assert isinstance(start_range, DatetimeRange)
        assert isinstance(end_range, DatetimeRange)
        assert start_range.lte == datetime.now(UTC).astimezone().date()
        assert end_range.gt == datetime.now(UTC).astimezone().date()
        eligibility = filter_clause(temporal.must, 2)
        current = field_clause(eligibility.should, 0)
        assert current.key == "metadata.is_current"
        assert isinstance(current.match, MatchValue)
        assert current.match.value is True

    def test_as_of_date_combines_with_acl_filter(self) -> None:
        result = with_temporal_filter(_acl_filter(), date(2026, 1, 1))
        assert len(clauses(result.must)) == 2
        temporal = filter_clause(result.must, 1)
        assert len(clauses(temporal.must)) == 3

        for group in clauses(temporal.must)[:2]:
            assert isinstance(group, Filter)
            assert len(clauses(group.should)) == 2
            assert isinstance(clauses(group.should)[0], IsEmptyCondition)

        current = field_clause(filter_clause(temporal.must, 2).should, 0)
        assert isinstance(current.match, MatchValue)
        assert current.match.value is True

    def test_as_of_date_without_acl(self) -> None:
        result = with_temporal_filter(_acl_filter(), date(2026, 6, 1))
        assert len(clauses(result.must)) == 2
        temporal = filter_clause(result.must, 1)
        group = filter_clause(temporal.must, 0)
        ranges = [c for c in clauses(group.should) if isinstance(c, FieldCondition) and c.range is not None]
        assert ranges, "range condition for effective_from must be present"

    def test_dated_noncurrent_versions_are_eligible_with_missing_date_fields_handled(self) -> None:
        result = with_temporal_filter(_acl_filter(), date(2024, 6, 1))
        temporal = filter_clause(result.must, 1)
        eligibility = filter_clause(temporal.must, 2)
        dated_alternatives = clauses(eligibility.should)[1:]
        assert all(isinstance(item, Filter) and item.must_not for item in dated_alternatives)


class TestSourceActInfo:
    def test_sources_expose_act_number_and_effective_dates(self) -> None:
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

    def test_sources_without_act_info_unchanged(self) -> None:
        docs = [
            (
                Document(page_content="text", metadata={"source": "doc.md", "document_id": 2}),
                0.8,
            )
        ]
        sources = extract_sources(docs)
        assert "act_number" not in sources[0]
        assert "effective_from" not in sources[0]
