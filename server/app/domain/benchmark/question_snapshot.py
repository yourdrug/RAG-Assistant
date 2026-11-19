"""Versioned snapshot of benchmark questions and its reconciliation with the stored set.

Questions are edited in the database; the snapshot is a reviewed copy committed with the
pack. Reconciliation matches questions by dataset and case id, which every snapshot question
must have (the text is editable, so it cannot identify a question across installations);
it creates missing ones, updates changed ones and never deletes:
a question that exists only in the database is reported, because it may be a fresh edit
that has not been exported yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from collections.abc import Iterable

    from domain.benchmark.question import BenchmarkQuestion


# Fields owned by the snapshot; id, author and creation date belong to one installation.
SNAPSHOT_FIELDS: tuple[str, ...] = (
    "dataset",
    "case_id",
    "question",
    "expected_answer",
    "source_hint",
    "annotations",
    "history",
    "as_of_date",
    "tags",
    "variant_of",
    "question_style",
    "notes",
    "is_active",
)


type QuestionKey = tuple[str, str]


def question_key(question: BenchmarkQuestion) -> QuestionKey:
    if not question.case_id:
        where = f"id {question.id}" if question.id is not None else repr(question.question[:80])
        message = (
            f"Benchmark question {where} in {question.dataset} has no case_id; assign one before syncing"
        )
        raise ValidationError(message)

    return question.dataset, question.case_id


def snapshot_values(question: BenchmarkQuestion) -> dict[str, object]:
    return {name: getattr(question, name) for name in SNAPSHOT_FIELDS}


def in_snapshot_scope(question: BenchmarkQuestion, dataset_prefix: str) -> bool:
    """Datasets of one pack are named `<pack>-<purpose>`."""
    return question.dataset.startswith(dataset_prefix)


@dataclass(frozen=True)
class QuestionUpdate:
    question_id: int
    key: QuestionKey
    changes: dict[str, object]


@dataclass(frozen=True)
class QuestionSyncPlan:
    create: tuple[BenchmarkQuestion, ...]
    update: tuple[QuestionUpdate, ...]
    unchanged: int
    # Present in the database, absent from the snapshot: left untouched.
    database_only: tuple[BenchmarkQuestion, ...]

    @property
    def in_sync(self) -> bool:
        return not self.create and not self.update and not self.database_only


def index_by_key(questions: Iterable[BenchmarkQuestion], origin: str) -> dict[QuestionKey, BenchmarkQuestion]:
    indexed: dict[QuestionKey, BenchmarkQuestion] = {}

    for question in questions:
        key = question_key(question)

        if key in indexed:
            message = f"Duplicate benchmark question {key[0]}/{key[1]} in the {origin}"
            raise ValidationError(message)

        indexed[key] = question

    return indexed


def plan_question_sync(
    snapshot: Iterable[BenchmarkQuestion], stored: Iterable[BenchmarkQuestion]
) -> QuestionSyncPlan:
    """What applying the snapshot changes in the stored questions of the same scope."""
    wanted = index_by_key(snapshot, "snapshot")
    current = index_by_key(stored, "database")
    create: list[BenchmarkQuestion] = []
    update: list[QuestionUpdate] = []
    unchanged = 0

    for key, question in wanted.items():
        existing = current.get(key)

        if existing is None:
            create.append(question)
            continue

        target = snapshot_values(question)
        changes = {name: value for name, value in target.items() if getattr(existing, name) != value}

        if not changes:
            unchanged += 1
            continue

        if existing.id is None:
            message = f"Stored benchmark question {key[0]}/{key[1]} has no id"
            raise ValidationError(message)

        update.append(QuestionUpdate(existing.id, key, changes))

    database_only = tuple(question for key, question in current.items() if key not in wanted)
    return QuestionSyncPlan(tuple(create), tuple(update), unchanged, database_only)
