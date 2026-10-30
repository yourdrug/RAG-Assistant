"""Generated edition timelines preserve intervals and repeatable corrections."""

from dataclasses import FrozenInstanceError, replace
from datetime import date

from hypothesis import given, strategies as st
import pytest

from domain.knowledge.acts.models import ActVersion
from domain.knowledge.acts.timeline import ActVersionTimeline


@st.composite
def edition_timelines(draw: st.DrawFn) -> tuple[ActVersionTimeline, date]:
    starts = draw(
        st.lists(st.dates(min_value=date(2000, 1, 1), max_value=date(2040, 1, 1)), min_size=1, max_size=15)
    )
    versions = tuple(ActVersion(index, 1, index, start) for index, start in enumerate(starts, 1))
    today = draw(st.dates(min_value=date(2000, 1, 1), max_value=date(2040, 1, 1)))
    timeline = ActVersionTimeline().rebuild(versions, dict.fromkeys(v.id for v in versions), today=today)
    return timeline, today


@given(edition_timelines())
def test_rebuild_has_at_most_one_current_and_nonoverlapping_intervals(
    case: tuple[ActVersionTimeline, date],
) -> None:
    timeline, today = case
    assert sum(v.is_current for v in timeline.versions) <= 1

    for index, version in enumerate(timeline.versions):
        version.validate_dates()

        if index + 1 < len(timeline.versions):
            next_start = timeline.versions[index + 1].effective_from
            assert next_start is not None
            assert version.effective_to is not None
            assert version.effective_to <= next_start

        if version.is_current:
            assert version.effective_from is not None
            assert version.effective_from <= today
            assert version.effective_to is None or today < version.effective_to


@given(edition_timelines())
def test_correction_is_idempotent(case: tuple[ActVersionTimeline, date]) -> None:
    timeline, today = case
    edited = timeline.versions[-1]
    first = timeline.correct(edited, end_provided=False, today=today)
    second = first.correct(edited, end_provided=False, today=today)
    assert first == second


@given(edition_timelines())
def test_removal_is_idempotent(case: tuple[ActVersionTimeline, date]) -> None:
    timeline, today = case
    removed_id = timeline.versions[0].id
    assert removed_id is not None
    first = timeline.remove(removed_id, today=today)
    assert first.remove(removed_id, today=today) == first


def test_versions_cannot_mutate_a_timeline_snapshot() -> None:
    version = ActVersion(1, 1, 1)
    timeline = ActVersionTimeline((version,))
    with pytest.raises(FrozenInstanceError):
        version.__setattr__("is_current", False)
    assert timeline.versions[0].is_current
    assert not replace(version, is_current=False).is_current


def test_unsaved_versions_still_have_a_single_current_edition() -> None:
    versions = (ActVersion(None, 1, 1, date(2024, 1, 1)), ActVersion(None, 1, 2, date(2025, 1, 1)))
    result = ActVersionTimeline().rebuild(versions, {None: None}, today=date(2026, 1, 1))
    assert [version.is_current for version in result.versions] == [False, True]
