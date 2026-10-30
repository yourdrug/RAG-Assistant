"""Regulatory-act edition timeline: interval rules without repositories or I/O."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from domain.knowledge.acts.models import ActVersion
from domain.shared.exceptions import EntityNotFoundError


@dataclass(frozen=True)
class ActVersionTimeline:
    versions: tuple[ActVersion, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "versions", tuple(self.versions))

    def get_version(self, version_id: int | None) -> ActVersion:
        for version in self.versions:
            if version.id == version_id:
                return version

        msg = "ActVersion"
        raise EntityNotFoundError(msg, str(version_id))

    def is_latest(
        self, effective_date: date | None, *, today: date, effective_to: date | None = None
    ) -> bool:
        return self.inserting(effective_date, today=today, effective_to=effective_to)[1].is_current

    def inserting(
        self,
        effective_date: date | None,
        *,
        today: date,
        effective_to: date | None = None,
        candidate_id: int | None = None,
    ) -> tuple[ActVersionTimeline, ActVersion]:
        """Resolve uploads and transfers using the same ordering as corrections.

        Transfers retain their persisted ID; uploads sort after existing IDs.
        """
        if candidate_id is None:
            candidate_id = max((version.id or 0 for version in self.versions), default=0) + 1

        candidate = ActVersion(candidate_id, None, 0, effective_date, effective_to)
        ends = self.explicit_ends()
        ends[candidate_id] = effective_to
        rebuilt = self.rebuild((*self.versions, candidate), ends, today=today)
        inserted = rebuilt.get_version(candidate_id)
        previous = tuple(version for version in rebuilt.versions if version.id != candidate_id)
        return ActVersionTimeline(previous), inserted

    def successor_date(self, effective_date: date | None) -> date | None:
        if effective_date is None:
            return None

        later_dates = [
            version.effective_from
            for version in self.versions
            if version.effective_from is not None and version.effective_from > effective_date
        ]
        return min(later_dates) if later_dates else None

    def insert_boundary(
        self, new_date: date | None, *, today: date, effective_to: date | None = None
    ) -> ActVersionTimeline:
        """Apply the same interval and current-edition rules used by correction."""
        return self.inserting(new_date, today=today, effective_to=effective_to)[0]

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
        self.get_version(edited.id)

        if end_provided:
            edited.validate_dates(explicit_end=edited.effective_to)

        explicit_ends = self.explicit_ends()

        if end_provided:
            explicit_ends[edited.id] = edited.effective_to

        versions = tuple(edited if version.id == edited.id else version for version in self.versions)
        return self.rebuild(versions, explicit_ends, today=today)

    @staticmethod
    def order_key(version: ActVersion) -> tuple[date, int]:
        return version.effective_from or date.min, version.id or 0

    def explicit_ends(self) -> dict[int | None, date | None]:
        """Separate expiry dates from boundaries derived from the old adjacency."""
        old_order = sorted(self.versions, key=self.order_key)
        ends: dict[int | None, date | None] = {}

        for index, version in enumerate(old_order):
            successor = old_order[index + 1] if index + 1 < len(old_order) else None
            derived = successor is not None and version.effective_to == successor.effective_from
            ends[version.id] = None if derived else version.effective_to

        return ends

    def remove(self, version_id: int, *, today: date) -> ActVersionTimeline:
        """Reconnect remaining editions after unlinking, preserving explicit expiry."""
        if not any(version.id == version_id for version in self.versions):
            return self

        remaining = tuple(version for version in self.versions if version.id != version_id)
        return self.rebuild(remaining, self.explicit_ends(), today=today)

    def rebuild(
        self,
        versions: tuple[ActVersion, ...],
        explicit_ends: dict[int | None, date | None],
        *,
        today: date,
    ) -> ActVersionTimeline:
        ordered = sorted(versions, key=self.order_key)

        for index, original in enumerate(ordered):
            version = original
            successor = ordered[index + 1] if index + 1 < len(ordered) else None
            boundary = successor.effective_from if successor is not None else None
            expiry = explicit_ends[version.id]
            version = replace(
                version,
                effective_to=min(expiry, boundary) if expiry and boundary else expiry or boundary,
                is_current=False,
            )
            version.validate_dates()
            ordered[index] = version

        eligible = [
            version
            for version in ordered
            if (version.effective_from is None or version.effective_from <= today)
            and (version.effective_to is None or today < version.effective_to)
        ]

        if eligible:
            current = eligible[-1]
            ordered = [
                replace(version, is_current=True) if version is current else version for version in ordered
            ]

        return ActVersionTimeline(tuple(ordered))
