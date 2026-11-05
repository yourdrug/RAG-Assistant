"""Load the active database dataset once for benchmark evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from application.uow import UnitOfWorkFactory


async def load_benchmark_questions(uow_factory: UnitOfWorkFactory, dataset: str) -> list[dict[str, Any]]:
    """Return all active cases, preserving database IDs and evaluation annotations."""
    async with uow_factory.create() as uow:
        questions = await uow.benchmark_questions.list_active(dataset=dataset)

    if not questions:
        message = f"No active benchmark questions found for dataset '{dataset}'"
        raise ValueError(message)

    return [
        {
            "id": q.id,
            "question": q.question,
            "expected_answer": q.expected_answer,
            "source_hint": q.source_hint,
            "annotations": q.annotations,
            "case_id": q.case_id,
            "variant_of": q.variant_of,
            "question_style": q.question_style,
            "tags": q.tags or [],
            "dataset": q.dataset,
            "history": q.history,
            "as_of_date": q.as_of_date.isoformat() if q.as_of_date else None,
        }
        for q in questions
    ]
