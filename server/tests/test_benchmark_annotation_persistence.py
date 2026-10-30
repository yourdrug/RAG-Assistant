"""Every persistence entry point validates annotations before modifying the session."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.benchmark.question import BenchmarkQuestion
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_question_repository import (
    SQLAlchemyBenchmarkQuestionRepository,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "bulk_create", "update"])
async def test_invalid_annotations_never_reach_session(operation: str) -> None:
    session = MagicMock()
    session.execute = AsyncMock()
    repository = SQLAlchemyBenchmarkQuestionRepository(session)
    question = BenchmarkQuestion(question="Q")
    question.annotations = {"expected_refusal": "invalid"}
    action: Awaitable[object]

    if operation == "update":
        action = repository.update(1, annotations=question.annotations)
    elif operation == "create":
        action = repository.create(question)
    else:
        action = repository.bulk_create([question])

    with pytest.raises(ValueError, match="boolean"):
        await action
    session.execute.assert_not_called()
    session.add.assert_not_called()
    session.add_all.assert_not_called()
