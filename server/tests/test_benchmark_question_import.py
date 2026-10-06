"""Regression tests for active state in benchmark question JSON imports."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from application.services.benchmark_services import BenchmarkQuestionService
from domain.entities.benchmark_question import BenchmarkQuestion
from domain.value_objects.benchmark_dataset import BenchmarkDataset
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_question_repository import (
    SQLAlchemyBenchmarkQuestionRepository,
)
from presentation.api.helpers import question_create_to_dto
from presentation.api.routes.benchmark_admin import export_questions, import_questions
from presentation.api.schemas.benchmark import BenchmarkQuestionCreate, BenchmarkQuestionsImportRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [True, False, None])
async def test_create_preserves_active_state(fake_uow_factory, state):
    payload = {"question": "Из скольких частей состоит электронный документ?"}
    if state is not None:
        payload["is_active"] = state
    body = BenchmarkQuestionCreate.model_validate(payload)
    service = BenchmarkQuestionService(fake_uow_factory)

    created = await service.create(question_create_to_dto(body), created_by=1)

    assert created.is_active is (True if state is None else state)
    assert created.created_by == 1


@pytest.mark.asyncio
async def test_import_persists_active_inactive_and_default_questions(fake_uow_factory):
    annotations = {"required_facts": [["двух", "две"]], "expected_refusal": False}
    body = BenchmarkQuestionsImportRequest.model_validate(
        {
            "questions": [
                {"question": "Active", "is_active": True, "annotations": annotations},
                {"question": "Inactive", "is_active": False},
                {"question": "Default"},
            ]
        }
    )
    session = MagicMock()
    session.flush = AsyncMock()
    fake_uow_factory._uow.benchmark_questions = SQLAlchemyBenchmarkQuestionRepository(session)

    result = await import_questions(
        body, admin=SimpleNamespace(id=1), service=BenchmarkQuestionService(fake_uow_factory)
    )

    assert result.imported == 3
    stored = session.add_all.call_args.args[0]
    assert [question.is_active for question in stored] == [True, False, True]
    assert stored[0].annotations == annotations
    assert all(question.created_by == 1 for question in stored)
    session.flush.assert_awaited_once()
    assert fake_uow_factory._uow._committed


@pytest.mark.asyncio
async def test_export_retains_active_state_for_reimport():
    service = SimpleNamespace(
        export=AsyncMock(
            return_value=[
                BenchmarkQuestion(question="Active", is_active=True),
                BenchmarkQuestion(question="Inactive", is_active=False),
            ]
        )
    )

    exported = await export_questions(
        dataset=BenchmarkDataset.MAIN.value, admin=SimpleNamespace(id=1), service=service
    )
    imported = BenchmarkQuestionsImportRequest.model_validate({"questions": exported})

    assert [question.is_active for question in imported.questions] == [True, False]


@pytest.mark.parametrize("extra", [{"unexpected": True}, {"is_active": None}])
def test_import_still_rejects_unknown_fields_and_null_active_state(extra):
    with pytest.raises(ValidationError):
        BenchmarkQuestionsImportRequest.model_validate({"questions": [{"question": "Q", **extra}]})
