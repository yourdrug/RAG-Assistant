"""Law pack edition rules: the edition in force passes the selection; one edition per act in context."""

from __future__ import annotations

from datetime import date
from typing import Any

from langchain_core.documents import Document
import pytest

from domain.domain_packs.belarus_law.acts.edition_metadata import edition_metadata
from domain.domain_packs.belarus_law.acts.models import ActVersion
from domain.knowledge.pack_metadata import PACK_METADATA_KEY, matches
from infrastructure.domain_packs.belarus_law.evidence.editions import (
    edition_selection,
    edition_source_fields,
    resolve_edition_conflicts,
)
from infrastructure.ml.rag.rag_sources import extract_sources


DAY = date(2025, 6, 1)


def edition(**fields: Any) -> dict[str, Any]:
    return edition_metadata(
        ActVersion(id=fields.pop("id", 1), act_id=fields.pop("act_id", 1), document_id=1, **fields)
    )


@pytest.mark.parametrize(
    "pack_metadata,selected",
    [
        ({}, True),
        (edition(), True),
        (edition(is_current=False), False),
        (edition(effective_from=date(2025, 6, 1)), True),
        (edition(effective_from=date(2025, 6, 2)), False),
        (edition(effective_from=date(2024, 1, 1), effective_to=date(2025, 6, 1), is_current=False), False),
        (edition(effective_from=date(2024, 1, 1), effective_to=date(2025, 6, 2), is_current=False), True),
        (edition(effective_to=date(2025, 6, 2), is_current=False), True),
    ],
)
def test_edition_in_force_on_the_question_date(pack_metadata: dict[str, Any], *, selected: bool) -> None:
    assert matches(edition_selection(DAY), pack_metadata) is selected


def test_current_questions_use_the_local_day_so_future_editions_start_without_reindexing() -> None:
    future = edition(effective_from=date(2025, 6, 2), is_current=False)
    assert not matches(edition_selection(None, current_day=DAY), future)
    assert matches(edition_selection(None, current_day=date(2025, 6, 2)), future)
    assert matches(edition_selection(date(2025, 7, 1), current_day=DAY), future)


def doc(version_id: int, effective_from: str | None, *, act_id: int = 1) -> Document:
    metadata = edition(
        id=version_id,
        act_id=act_id,
        effective_from=date.fromisoformat(effective_from) if effective_from else None,
    )
    return Document(
        page_content=str(version_id), metadata={"document_id": version_id, PACK_METADATA_KEY: metadata}
    )


def test_conflicting_editions_keep_every_chunk_of_the_newest_eligible_edition() -> None:
    old, new, new_second, other_act = (
        doc(1, "2020-01-01"),
        doc(2, "2024-01-01"),
        doc(2, "2024-01-01"),
        doc(3, "2010-01-01", act_id=2),
    )
    plain = Document(page_content="plain", metadata={"document_id": 9})
    resolved = resolve_edition_conflicts(
        [(old, 0.9), (plain, 0.8), (new, 0.7), (new_second, 0.6), (other_act, 0.5)]
    )
    assert [item[0] for item in resolved] == [plain, new, new_second, other_act]


def test_undated_editions_rank_below_dated_and_ties_keep_the_later_edition() -> None:
    assert [
        item[0].page_content
        for item in resolve_edition_conflicts([(doc(5, None), 1.0), (doc(4, "1999-01-01"), 0.5)])
    ] == ["4"]
    assert [
        item[0].page_content for item in resolve_edition_conflicts([(doc(7, None), 1.0), (doc(8, None), 0.5)])
    ] == ["8"]


def test_cited_sources_carry_edition_dates_from_the_law_pack() -> None:
    metadata = {
        "source": "ukaz-123.rtf",
        "document_id": 1,
        "domain_metadata": {"decree_number": "123"},
        PACK_METADATA_KEY: edition(effective_from=date(2026, 2, 1), effective_to=date(2027, 1, 1)),
    }
    sources = extract_sources([(Document(page_content="text", metadata=metadata), 0.9)])
    assert sources[0]["act_number"] == "123"
    assert sources[0]["effective_from"] == "2026-02-01"
    assert sources[0]["effective_to"] == "2027-01-01"


def test_sources_show_edition_dates_only() -> None:
    metadata = {PACK_METADATA_KEY: edition(effective_from=date(2026, 2, 1), effective_to=date(2027, 1, 1))}
    assert edition_source_fields(metadata) == {"effective_from": "2026-02-01", "effective_to": "2027-01-01"}
    assert edition_source_fields({PACK_METADATA_KEY: edition()}) == {}
    assert edition_source_fields({}) == {}
