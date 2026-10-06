"""Load the active database dataset once for benchmark evaluation."""

from application.ports.unit_of_work_factory import UnitOfWorkFactory


async def load_benchmark_questions(uow_factory: UnitOfWorkFactory, dataset: str) -> list[dict]:
    """Return all active cases, preserving database IDs and evaluation annotations."""
    async with uow_factory.create() as uow:
        questions = await uow.benchmark_questions.list_active(dataset=dataset)

    if not questions:
        raise ValueError(f"No active benchmark questions found for dataset '{dataset}'")

    return [
        {
            "id": q.id,
            "question": q.question,
            "expected_answer": q.expected_answer,
            "source_hint": q.source_hint,
            "annotations": q.annotations,
            "tags": q.tags or [],
        }
        for q in questions
    ]
