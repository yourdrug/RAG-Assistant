"""Domain invariants for edition intervals and timeline corrections."""

from dataclasses import replace
from datetime import date

import pytest

from domain.entities.act_version import ActVersion
from domain.entities.act_version_timeline import ActVersionTimeline
from domain.shared.exceptions import EntityNotFound, ValidationError


def timeline() -> ActVersionTimeline:
    return ActVersionTimeline(
        (
            ActVersion(1, 10, 1, date(2024, 1, 1), date(2025, 1, 1), False),
            ActVersion(2, 10, 2, date(2025, 1, 1), None, True),
        )
    )


def test_correction_preserves_original_snapshot_and_closes_historical_gap():
    original = timeline()
    edited = replace(original.get_version(2), effective_from=date(2026, 1, 1))
    corrected = original.correct(edited, end_provided=False, today=date(2026, 6, 1))
    assert original.get_version(1).effective_to == date(2025, 1, 1)
    assert original.get_version(2).effective_from == date(2025, 1, 1)
    assert corrected.get_version(1).effective_to == date(2026, 1, 1)
    assert corrected.get_version(2).is_current
    edited.effective_from = date(2030, 1, 1)
    assert corrected.get_version(2).effective_from == date(2026, 1, 1)


def test_future_correction_demotes_before_promoting_previous_edition():
    original = timeline()
    edited = replace(original.get_version(2), effective_from=date(2028, 1, 1))
    corrected = original.correct(edited, end_provided=False, today=date(2026, 6, 1))
    changes = original.changed_versions(corrected.versions, edited_id=2)
    assert [(version.id, version.is_current) for version in changes] == [(2, False), (1, True)]


def test_same_date_supersession_keeps_empty_edition_valid():
    original = timeline()
    edited = replace(original.get_version(2), effective_from=date(2024, 1, 1))
    corrected = original.correct(edited, end_provided=False, today=date(2026, 6, 1))
    first = corrected.get_version(1)
    assert first.effective_to == first.effective_from
    assert not first.is_current and corrected.get_version(2).is_current
    first.validate_dates()


@pytest.mark.parametrize("expiry", [date(2025, 1, 1), date(2024, 1, 1)])
def test_explicit_nonpositive_interval_is_rejected_before_snapshot_changes(expiry):
    original = timeline()
    edited = replace(original.get_version(2), effective_to=expiry)
    with pytest.raises(ValidationError, match="later"):
        original.correct(edited, end_provided=True, today=date(2026, 6, 1))
    assert original.get_version(2).effective_to is None


def test_expired_version_cannot_remain_current():
    original = timeline()
    edited = replace(original.get_version(2), effective_to=date(2026, 1, 1))
    corrected = original.correct(edited, end_provided=True, today=date(2026, 6, 1))
    assert not any(version.is_current for version in corrected.versions)


def test_unknown_edition_cannot_be_corrected():
    with pytest.raises(EntityNotFound):
        timeline().correct(ActVersion(99, 10, 99), end_provided=False, today=date(2026, 6, 1))


def test_insertion_closes_previous_edition_without_mutating_snapshot():
    original = timeline()
    changed = original.insert_boundary(date(2026, 1, 1), today=date(2026, 6, 1))
    assert original.get_version(2).is_current
    assert original.get_version(2).effective_to is None
    assert changed.get_version(2).effective_to == date(2026, 1, 1)
    assert not changed.get_version(2).is_current


@pytest.mark.parametrize("removed_id, remaining_id, expected_end", [(2, 1, None), (1, 2, None)])
def test_removal_reopens_derived_boundaries_and_selects_current(removed_id, remaining_id, expected_end):
    original = timeline()
    remaining = original.remove(removed_id, today=date(2026, 6, 1))
    assert len(remaining.versions) == 1
    assert remaining.get_version(remaining_id).effective_to == expected_end
    assert remaining.get_version(remaining_id).is_current
    assert len(original.versions) == 2


def test_removal_preserves_explicit_expiry():
    original = ActVersionTimeline(
        (
            ActVersion(1, 10, 1, date(2024, 1, 1), date(2024, 6, 1), False),
            ActVersion(2, 10, 2, date(2025, 1, 1), None, True),
        )
    )
    remaining = original.remove(2, today=date(2026, 6, 1))
    assert remaining.get_version(1).effective_to == date(2024, 6, 1)
    assert not remaining.get_version(1).is_current


def test_removal_of_only_edition_leaves_empty_timeline():
    original = ActVersionTimeline((timeline().get_version(2),))
    assert original.remove(2, today=date(2026, 6, 1)).versions == ()


def test_removal_of_middle_edition_connects_remaining_neighbors():
    original = ActVersionTimeline(
        (
            *timeline().versions,
            ActVersion(3, 10, 3, date(2027, 1, 1), None, False),
        )
    )
    remaining = original.remove(2, today=date(2026, 6, 1))
    assert remaining.get_version(1).effective_to == date(2027, 1, 1)
    assert remaining.get_version(1).is_current
    assert not remaining.get_version(3).is_current
