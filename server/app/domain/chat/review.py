"""Administrative review of a journal entry (an answer attempt)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from domain.shared.exceptions import BusinessRuleViolationError, ValidationError


MAX_REVIEW_NOTE_CHARS = 2000


class ReviewStatus(StrEnum):
    NEW = "new"
    BENCHMARK_CANDIDATE = "benchmark_candidate"
    IN_BENCHMARK = "in_benchmark"
    # The needed document is missing from the corpus.
    CONTENT_GAP = "content_gap"
    # The document exists, the answer is still bad.
    PIPELINE_ISSUE = "pipeline_issue"
    WONT_FIX = "wont_fix"


@dataclass
class ChatLogReview:
    chat_log_id: int
    status: ReviewStatus = ReviewStatus.NEW
    note: str | None = None
    reviewer_id: int | None = None
    benchmark_question_id: int | None = None
    update_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.status = ReviewStatus(self.status)

    def set_status(self, status: ReviewStatus, note: str | None, reviewer_id: int) -> None:
        """Manual triage; ``in_benchmark`` is reached only by transferring the question."""
        if status == ReviewStatus.IN_BENCHMARK:
            msg = "in_benchmark is set by transferring the question to a benchmark dataset"
            raise ValidationError(msg, field="status")

        text = (note or "").strip()

        if len(text) > MAX_REVIEW_NOTE_CHARS:
            msg = f"note must be at most {MAX_REVIEW_NOTE_CHARS} characters"
            raise ValidationError(msg, field="note")

        self.status = status
        self.note = text or None
        self.reviewer_id = reviewer_id
        self.update_date = datetime.now(UTC)

    def ensure_not_transferred(self) -> None:
        if self.benchmark_question_id is not None:
            msg = "The journal entry has already been transferred to a benchmark dataset"
            raise BusinessRuleViolationError(msg)

    def mark_in_benchmark(self, question_id: int, reviewer_id: int) -> None:
        self.ensure_not_transferred()
        self.status = ReviewStatus.IN_BENCHMARK
        self.benchmark_question_id = question_id
        self.reviewer_id = reviewer_id
        self.update_date = datetime.now(UTC)
