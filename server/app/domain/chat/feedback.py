"""User reaction to a delivered assistant answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from domain.shared.exceptions import ValidationError


MAX_FEEDBACK_COMMENT_CHARS = 1000


class FeedbackRating(StrEnum):
    UP = "up"
    DOWN = "down"


class FeedbackReason(StrEnum):
    NO_ANSWER = "no_answer"
    WRONG_FACT = "wrong_fact"
    IRRELEVANT_SOURCES = "irrelevant_sources"
    INCOMPLETE = "incomplete"
    OUTDATED = "outdated"
    TOO_LONG = "too_long"
    OTHER = "other"


def validate_feedback_comment(comment: str | None) -> str | None:
    """Return the trimmed comment as typed by the user, or ``None`` when empty."""
    text = (comment or "").strip()

    if len(text) > MAX_FEEDBACK_COMMENT_CHARS:
        msg = f"comment must be at most {MAX_FEEDBACK_COMMENT_CHARS} characters"
        raise ValidationError(msg, field="comment")

    return text or None


@dataclass
class MessageFeedback:
    message_id: int
    user_id: int
    rating: FeedbackRating
    reasons: frozenset[FeedbackReason] = frozenset()
    # Stored after PII redaction and HTML escaping; the length rule applies to the typed text.
    comment: str | None = None
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    update_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.rating = FeedbackRating(self.rating)
        self.reasons = frozenset(FeedbackReason(reason) for reason in self.reasons)

        if self.reasons and self.rating != FeedbackRating.DOWN:
            msg = "reasons are allowed only for a negative rating"
            raise ValidationError(msg, field="reasons")
