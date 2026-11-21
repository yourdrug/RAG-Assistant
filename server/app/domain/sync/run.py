"""One reconciliation of a channel with its storage, and what happened to each file."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from domain.shared.exceptions import BusinessRuleViolationError


class SyncRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    # Some files failed; the others were applied.
    PARTIAL = "partial"
    # Nothing was applied: unreadable snapshot, lost access.
    FAILED = "failed"
    # Superseded by a newer run or a settings change.
    CANCELLED = "cancelled"
    # The responsible user or the client owner no longer has the required rights.
    REJECTED = "rejected"


TERMINAL_RUN_STATUSES = frozenset(
    {
        SyncRunStatus.SUCCEEDED,
        SyncRunStatus.PARTIAL,
        SyncRunStatus.FAILED,
        SyncRunStatus.CANCELLED,
        SyncRunStatus.REJECTED,
    }
)


class SyncTrigger(StrEnum):
    MANUAL = "manual"
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"


class FileOutcomeKind(StrEnum):
    ADDED = "added"
    UPDATED = "updated"
    RENAMED = "renamed"
    DELETED = "deleted"
    UNCHANGED = "unchanged"
    EXCLUDED = "excluded"
    CONFLICT = "conflict"
    FAILED = "failed"


# Kinds repeated for most files on every run: stored as counts, not per path.
TALLIED_KINDS = frozenset({FileOutcomeKind.UNCHANGED, FileOutcomeKind.EXCLUDED})


@dataclass(frozen=True)
class FileOutcome:
    path: str
    kind: FileOutcomeKind
    detail: str | None = None


@dataclass
class SyncRun:
    channel_id: int
    # The channel generation the run was started for; its writes are fenced by it.
    generation: int
    trigger: SyncTrigger = SyncTrigger.MANUAL
    id: int | None = None
    # Webhook delivery that requested the run; a redelivery never starts another run.
    delivery_id: str | None = None
    status: SyncRunStatus = SyncRunStatus.QUEUED
    # The storage revision read by the run (a commit for GitHub).
    storage_revision: str | None = None
    error: str | None = None
    # Changes and problems by path; tallied kinds only as counts in ``tallies``.
    outcomes: tuple[FileOutcome, ...] = ()
    tallies: dict[str, int] = field(default_factory=dict)
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_finished(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES

    def counts(self) -> dict[str, int]:
        counts = Counter(outcome.kind.value for outcome in self.outcomes)
        counts.update(self.tallies)
        return dict(counts)

    def start(self, storage_revision: str | None = None) -> None:
        self.require(SyncRunStatus.QUEUED)
        self.status = SyncRunStatus.RUNNING
        self.storage_revision = storage_revision
        self.started_at = datetime.now(UTC)

    def complete(self, outcomes: tuple[FileOutcome, ...]) -> None:
        self.require(SyncRunStatus.RUNNING)
        self.outcomes = outcomes
        failed = any(outcome.kind == FileOutcomeKind.FAILED for outcome in outcomes)
        self.finish(SyncRunStatus.PARTIAL if failed else SyncRunStatus.SUCCEEDED)

    def fail(self, error: str) -> None:
        self.require(SyncRunStatus.QUEUED, SyncRunStatus.RUNNING)
        self.error = error
        self.finish(SyncRunStatus.FAILED)

    def reject(self, reason: str) -> None:
        self.require(SyncRunStatus.QUEUED, SyncRunStatus.RUNNING)
        self.error = reason
        self.finish(SyncRunStatus.REJECTED)

    def cancel(self) -> None:
        if not self.is_finished:
            self.finish(SyncRunStatus.CANCELLED)

    def require(self, *statuses: SyncRunStatus) -> None:
        if self.status not in statuses:
            msg = f"Sync run is {self.status.value}, expected {' or '.join(s.value for s in statuses)}"
            raise BusinessRuleViolationError(msg)

    def finish(self, status: SyncRunStatus) -> None:
        self.status = status
        self.finished_at = datetime.now(UTC)
