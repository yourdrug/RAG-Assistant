"""Journal review: triage statuses, conversation context, benchmark transfer and scope."""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime, timedelta
import inspect
import io
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

from fastapi import HTTPException
from fastapi.routing import APIRoute
import pytest

from application.chat import ChatLogService
from application.chat.review import BenchmarkTransfer
from domain.benchmark.question import BenchmarkQuestion
from domain.chat.chat_log import ChatLog
from domain.chat.message import Message, MessageRole
from domain.chat.review import ReviewStatus
from domain.identity.roles import UserKind, UserRole
from domain.rag.outcome import AnswerOutcome
from domain.shared.exceptions import (
    BusinessRuleViolationError,
    EntityNotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from presentation.api.routes import admin_chat_logs
from presentation.api.routes.admin_chat_logs import (
    ChatLogQuery,
    chat_log_stats,
    export_chat_logs,
    get_chat_log,
    review_chat_log,
    transfer_to_benchmark,
)
from presentation.api.schemas import BenchmarkTransferRequest, ChatLogReviewRequest, CurrentUser
from tests.fakes import FakeUnitOfWork, FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from domain.chat.repositories import ChatLogFilter


OWNER = 1
CURATOR = 10
ADMIN = 20


class Redactor:
    def redact(self, text: str) -> str:
        return text.replace("+375291234567", "[PHONE]")

    def scan_and_redact(self, text: str) -> tuple[str, list[str]]:
        return self.redact(text), []


def user(user_id: int, role: UserRole) -> CurrentUser:
    return CurrentUser(
        id=user_id, email=f"u{user_id}@example.test", role=role, kind=UserKind.INTERNAL, is_active=True
    )


def scope(managed: list[int]) -> Mock:
    return Mock(
        get_scope=AsyncMock(
            return_value={"managed_client_ids": [], "managed_internal_ids": managed, "managed_group_ids": []}
        )
    )


async def journal() -> tuple[FakeUnitOfWork, ChatLogService]:
    """Conversation of three turns; the third is the reviewed answer, plus a stranger's entry."""
    uow = FakeUnitOfWork()
    conversation = await uow.conversations.create(OWNER)
    assert conversation.id is not None
    started = datetime.now(UTC) - timedelta(minutes=10)

    for index, (role, content) in enumerate(
        [
            (MessageRole.USER, "first &lt;question&gt; +375291234567"),
            (MessageRole.ASSISTANT, "first answer"),
            (MessageRole.USER, "second question"),
            (MessageRole.ASSISTANT, "second answer"),
            (MessageRole.USER, "reviewed question"),
            (MessageRole.ASSISTANT, "reviewed answer"),
        ]
    ):
        await uow.messages.save(
            Message(
                conversation_id=conversation.id,
                role=role,
                content=content,
                creation_date=started + timedelta(seconds=index),
            )
        )

    await uow.chat_logs.save(
        ChatLog(
            user_id=OWNER,
            conversation_id=conversation.id,
            message_id=6,
            question="reviewed question +375291234567",
            answer="reviewed answer",
            sources=[{"source": "Law A"}, {"source": "Law B"}, {"source": "Law A"}],
            space_id=1,
            pack_key="belarus_law",
            as_of_date=date(2026, 1, 2),
            outcome=AnswerOutcome.NOT_FOUND,
        )
    )
    await uow.chat_logs.save(ChatLog(user_id=99, question="foreign", space_id=1, pack_key="belarus_law"))
    await uow.benchmark_questions.create(BenchmarkQuestion(question="existing", dataset="belarus_law-quick"))
    service = ChatLogService(
        FakeUnitOfWorkFactory(uow), pii_redactor=Redactor(), chat_settings=Mock(history_window=2)
    )
    return uow, service


@pytest.mark.asyncio
async def test_curator_sets_triage_status_but_not_in_benchmark() -> None:
    uow, service = await journal()
    curator = user(CURATOR, UserRole.CURATOR)

    review = await review_chat_log(
        1,
        ChatLogReviewRequest(status=ReviewStatus.BENCHMARK_CANDIDATE, note=" check the date "),
        curator,
        service,
        scope([OWNER]),
    )
    assert review.status == ReviewStatus.BENCHMARK_CANDIDATE
    assert review.note == "check the date"
    assert review.reviewer_id == CURATOR

    with pytest.raises(ValidationError):
        await review_chat_log(
            1, ChatLogReviewRequest(status=ReviewStatus.IN_BENCHMARK), curator, service, scope([OWNER])
        )

    # Entries of users outside the curator's scope are not found, not forbidden.
    with pytest.raises(EntityNotFoundError):
        await review_chat_log(
            2, ChatLogReviewRequest(status=ReviewStatus.WONT_FIX), curator, service, scope([OWNER])
        )

    assert uow.fake_chat_logs.reviews[1].status == ReviewStatus.BENCHMARK_CANDIDATE
    assert 2 not in uow.fake_chat_logs.reviews


def test_benchmark_transfer_requires_benchmark_capability() -> None:
    routes = [
        route
        for route in admin_chat_logs.router.routes
        if isinstance(route, APIRoute) and route.path == "/admin/chat-logs/{log_id}/benchmark-question"
    ]
    assert {method for route in routes for method in route.methods} == {"GET", "POST"}

    for route in routes:
        checks = [
            dependency.call
            for dependency in route.dependant.dependencies
            if dependency.call is not None and dependency.call.__name__ == "check_capability"
        ]
        assert len(checks) == 1

        with pytest.raises(PermissionDeniedError):
            checks[0](user(CURATOR, UserRole.CURATOR))

        assert checks[0](user(ADMIN, UserRole.ADMIN)).id == ADMIN


@pytest.mark.asyncio
async def test_card_shows_previous_turns_within_history_window() -> None:
    _uow, service = await journal()
    card = await get_chat_log(1, user(CURATOR, UserRole.CURATOR), service, scope([OWNER]))

    # The answer's own question is the entry itself; the window holds two turns.
    assert [turn["content"] for turn in card["context"]] == ["second question", "second answer"]

    with pytest.raises(HTTPException, match="404"):
        await get_chat_log(2, user(CURATOR, UserRole.CURATOR), service, scope([OWNER]))


@pytest.mark.asyncio
async def test_transfer_creates_inactive_redacted_question_once() -> None:
    uow, service = await journal()
    service.reviews.chat_settings = Mock(history_window=4)
    draft = await service.reviews.draft(1, None)

    assert draft.question == "reviewed question [PHONE]"
    assert draft.history[0] == {"role": "user", "content": "first <question> [PHONE]"}
    assert draft.datasets == ["belarus_law-main", "belarus_law-quick"]
    assert draft.source_hint == "Law A; Law B"

    question = await service.reviews.transfer(
        1,
        BenchmarkTransfer(
            dataset="belarus_law-main",
            question="edited question +375291234567",
            history=[{"role": "user", "content": "edited +375291234567"}],
        ),
        ADMIN,
        None,
    )

    assert question.dataset == "belarus_law-main"
    assert question.question == "edited question [PHONE]"
    assert question.history == [{"role": "user", "content": "edited [PHONE]"}]
    assert question.is_active is False
    assert question.as_of_date == date(2026, 1, 2)
    assert question.source_hint == "Law A; Law B"
    review = uow.fake_chat_logs.reviews[1]
    assert review.status == ReviewStatus.IN_BENCHMARK
    assert review.benchmark_question_id == question.id

    with pytest.raises(BusinessRuleViolationError):
        await service.reviews.transfer(
            1, BenchmarkTransfer(dataset="belarus_law-main", question="again"), ADMIN, None
        )

    assert [q.question for q in uow.fake_benchmark_questions.created] == [
        "existing",
        "edited question [PHONE]",
    ]


@pytest.mark.asyncio
async def test_transfer_rejects_foreign_pack_dataset_and_out_of_scope_entry() -> None:
    uow, service = await journal()

    with pytest.raises(ValidationError):
        await service.reviews.transfer(1, BenchmarkTransfer(dataset="main", question="q"), ADMIN, None)

    with pytest.raises(EntityNotFoundError):
        await service.reviews.transfer(
            2, BenchmarkTransfer(dataset="belarus_law-main", question="q"), ADMIN, (OWNER,)
        )

    assert len(uow.fake_benchmark_questions.created) == 1
    assert (
        uow.fake_chat_logs.reviews.get(1) is None
        or uow.fake_chat_logs.reviews[1].benchmark_question_id is None
    )


@pytest.mark.asyncio
async def test_transfer_creates_active_question_with_reviewed_reference_fields() -> None:
    uow, service = await journal()
    body = BenchmarkTransferRequest(
        dataset="belarus_law-quick",
        question="Review this",
        expected_answer="Contact +375291234567",
        source_hint="Law C",
        as_of_date="",
        annotations={"required_facts": [["Contact +375291234567", "Call the office"]]},
        tags=["regression", "from-journal", "regression"],
        notes="Confirmed with +375291234567",
        is_active=True,
    )
    response = await transfer_to_benchmark(1, body, user(ADMIN, UserRole.ADMIN), service, scope([]))

    question = uow.fake_benchmark_questions.created[-1]
    assert response == {"question_id": question.id, "dataset": "belarus_law-quick", "is_active": True}
    assert question.expected_answer == "Contact [PHONE]"
    assert question.source_hint == "Law C"
    assert question.as_of_date is None
    assert question.annotations == {"required_facts": [["Contact [PHONE]", "Call the office"]]}
    assert question.tags == ["from-journal", "regression"]
    assert question.notes == "chat_log:1\nConfirmed with [PHONE]"
    assert uow.fake_chat_logs.reviews[1].benchmark_question_id == question.id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_answer, annotations, message",
    [
        ("  ", None, "An expected answer is required"),
        ("Reference", {"required_facts": [3]}, "nonempty phrases"),
    ],
)
async def test_transfer_rejects_invalid_active_question_before_creating_it(
    expected_answer: str, annotations: dict[str, list[int]] | None, message: str
) -> None:
    uow, service = await journal()

    with pytest.raises(ValidationError) as error:
        await service.reviews.transfer(
            1,
            BenchmarkTransfer(
                dataset="belarus_law-main",
                question="Review this",
                is_active=True,
                expected_answer=expected_answer,
                annotations=annotations,
            ),
            ADMIN,
            None,
        )

    assert message in str(error.value.as_dict())
    assert len(uow.fake_benchmark_questions.created) == 1
    assert 1 not in uow.fake_chat_logs.reviews or uow.fake_chat_logs.reviews[1].benchmark_question_id is None


@pytest.mark.asyncio
async def test_stats_and_export_use_curator_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    uow, service = await journal()
    requested: list[ChatLogFilter] = []

    async def stats(filters: ChatLogFilter) -> dict[str, object]:
        requested.append(filters)
        return {"total": 1}

    async def list_logs(filters: ChatLogFilter) -> list[ChatLog]:
        requested.append(filters)
        logs = [
            log
            for log in uow.fake_chat_logs.logs
            if filters.user_ids is None or log.user_id in filters.user_ids
        ]
        return logs[filters.offset : filters.offset + filters.limit]

    monkeypatch.setattr(uow.fake_chat_logs, "stats", stats)
    monkeypatch.setattr(uow.fake_chat_logs, "list_logs", list_logs)
    curator = user(CURATOR, UserRole.CURATOR)

    assert await chat_log_stats(
        options=ChatLogQuery(needs_review=True),
        current_user=curator,
        service=service,
        assignment_service=scope([OWNER]),
    ) == {"total": 1}
    assert requested[-1].user_ids == (OWNER, CURATOR)
    assert requested[-1].needs_review is True

    response = await export_chat_logs(
        options=ChatLogQuery(),
        export_format="csv",
        max_rows=5000,
        current_user=curator,
        service=service,
        assignment_service=scope([OWNER]),
    )
    body = bytes(response.body).decode()
    header, *rows = body.strip().splitlines()
    assert "rating" in header.split(",")
    assert "review_status" in header.split(",")
    assert "trace" not in header
    assert len(rows) == 1
    assert "foreign" not in body


def test_export_fields_exclude_nested_objects() -> None:
    assert "feedback" not in admin_chat_logs.EXPORT_FIELDS
    assert "review" not in admin_chat_logs.EXPORT_FIELDS
    assert inspect.iscoroutinefunction(export_chat_logs)


@pytest.mark.asyncio
async def test_error_entry_context_survives_later_turns() -> None:
    uow, service = await journal()
    asked = datetime.now(UTC) - timedelta(minutes=5)
    await uow.chat_logs.save(
        ChatLog(
            user_id=OWNER,
            conversation_id=1,
            question="failed question",
            space_id=1,
            pack_key="belarus_law",
            outcome=AnswerOutcome.ERROR,
            creation_date=asked,
        )
    )
    error_id = len(uow.fake_chat_logs.logs)

    # The user keeps talking after the failed attempt.
    for index in range(40):
        await uow.messages.save(
            Message(
                conversation_id=1,
                role=MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT,
                content=f"later {index}",
                creation_date=asked + timedelta(seconds=index + 1),
            )
        )

    card = await get_chat_log(error_id, user(ADMIN, UserRole.ADMIN), service, scope([]))
    assert [turn["content"] for turn in card["context"]] == ["reviewed question", "reviewed answer"]
    draft = await service.reviews.draft(error_id, None)
    assert [turn["content"] for turn in draft.history] == ["reviewed question", "reviewed answer"]


@pytest.mark.parametrize("text", ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "\tx", "\rx", "  =HYPERLINK()"])
def test_csv_cells_neutralize_formula_prefixes(text: str) -> None:
    assert admin_chat_logs.csv_cell(text) == f"'{text}"


def test_csv_cells_keep_plain_text_and_numbers() -> None:
    assert admin_chat_logs.csv_cell("обычный вопрос") == "обычный вопрос"
    assert admin_chat_logs.csv_cell(-5) == -5
    assert admin_chat_logs.csv_cell(None) is None


@pytest.mark.asyncio
async def test_csv_export_writes_neutralized_question(monkeypatch: pytest.MonkeyPatch) -> None:
    uow, service = await journal()
    uow.fake_chat_logs.logs[0].question = "=1+1"

    async def list_logs(filters: ChatLogFilter) -> list[ChatLog]:
        return uow.fake_chat_logs.logs[:1][filters.offset : filters.offset + filters.limit]

    monkeypatch.setattr(uow.fake_chat_logs, "list_logs", list_logs)
    response = await export_chat_logs(
        options=ChatLogQuery(),
        export_format="csv",
        max_rows=10,
        current_user=user(ADMIN, UserRole.ADMIN),
        service=service,
        assignment_service=scope([]),
    )
    [record] = list(csv.DictReader(io.StringIO(bytes(response.body).decode())))
    assert record["question"] == "'=1+1"
