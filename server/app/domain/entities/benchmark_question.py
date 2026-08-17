"""BenchmarkQuestion domain entity — test question for RAG quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class BenchmarkQuestion:
    question: str = ""
    expected_answer: str | None = None
    source_hint: str | None = None
    tags: list[str] | None = None
    dataset: str = "main"
    is_active: bool = True
    created_by: int | None = None
    notes: str | None = None
    id: int | None = None
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
