"""Regulatory-act edition timeline: interval rules without repositories or I/O."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from domain.entities.act_version import ActVersion
from domain.exceptions import EntityNotFound


@dataclass(frozen=True)
class ActVersionTimeline:
    versions: tuple[ActVersion, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "versions", tuple(replace(version) for version in self.versions))

    def get_version(self, version_id: int | None) -> ActVersion:
        for version in self.versions:
            if version.id == version_id:
                return replace(version)
        raise EntityNotFound("ActVersion", str(version_id))

    def is_latest(self, effective_date: date | None, *, today: date) -> bool:
        if effective_date is None:
            return not any(version.is_current for version in self.versions)
        if effective_date > today:
            return False
        current_dates = [
            version.effective_from
            for version in self.versions
            if version.effective_from is not None and version.effective_from <= today
        ]
        return not current_dates or effective_date >= max(current_dates)

    def successor_date(self, effective_date: date | None) -> date | None:
        if effective_date is None:
            return None
        later_dates = [
            version.effective_from
            for version in self.versions
            if version.effective_from is not None and version.effective_from > effective_date
        ]
        return min(later_dates) if later_dates else None

    def insert_boundary(self, new_date: date | None, *, today: date) -> ActVersionTimeline:
        """Close preceding editions when inserting an upload or relinked edition."""
        versions = [replace(version) for version in self.versions]
        if new_date is None:
            return ActVersionTimeline(tuple(versions))
        new_is_current = self.is_latest(new_date, today=today)
        for version in versions:
            if version.effective_from is None or version.effective_from <= new_date:
                if version.effective_to is None or version.effective_to > new_date:
                    version.effective_to = new_date
            if new_is_current and version.is_current:
                version.is_current = False
        return ActVersionTimeline(tuple(versions))

    def changed_versions(
        self, candidates: tuple[ActVersion, ...], *, edited_id: int | None = None
    ) -> list[ActVersion]:
        previous = {version.id: version for version in self.versions}
        changed = [
            version
            for version in candidates
            if version.id == edited_id or version != previous.get(version.id)
        ]
        # Demote before promoting to preserve the single-current-edition constraint.
        return sorted(changed, key=lambda version: version.is_current)

    def correct(self, edited: ActVersion, *, end_provided: bool, today: date) -> ActVersionTimeline:
        """Rebuild adjacency after a date correction, preserving explicit expiry dates.

        Ends equal to the old successor's start are derived boundaries. Other ends
        represent explicit expiry and survive correction, capped by the successor.
        Equal starts use insertion order: the newest edition supersedes earlier ones.
        Returns copies so validation can finish before any repository writes.
        """
        previous = list(self.versions)
        self.get_version(edited.id)
        if end_provided:
            edited.validate_dates(explicit_end=edited.effective_to)

        def key(version: ActVersion) -> tuple[date, int]:
            return version.effective_from or date.min, version.id or 0

        old_order = sorted(previous, key=key)
        explicit_ends: dict[int | None, date | None] = {}
        for index, version in enumerate(old_order):
            successor = old_order[index + 1] if index + 1 < len(old_order) else None
            derived = successor is not None and version.effective_to == successor.effective_from
            explicit_ends[version.id] = None if derived else version.effective_to
        if end_provided:
            explicit_ends[edited.id] = edited.effective_to

        ordered = sorted(
            [replace(edited if version.id == edited.id else version) for version in previous], key=key
        )
        for index, version in enumerate(ordered):
            successor = ordered[index + 1] if index + 1 < len(ordered) else None
            boundary = successor.effective_from if successor is not None else None
            expiry = explicit_ends[version.id]
            version.effective_to = min(expiry, boundary) if expiry and boundary else expiry or boundary
            version.validate_dates()
            version.is_current = False

        eligible = [
            version
            for version in ordered
            if (version.effective_from is None or version.effective_from <= today)
            and (version.effective_to is None or today < version.effective_to)
        ]
        if eligible:
            eligible[-1].is_current = True
        return ActVersionTimeline(tuple(ordered))
