"""User reactions: own answers only, invariants, upsert and admin journal scope."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError as PydanticValidationError
import pytest

from application.chat import ChatLogService, MessageFeedbackService
from domain.chat.chat_log import ChatLog
from domain.chat.feedback import FeedbackRating, FeedbackReason, MessageFeedback
from domain.chat.message import Message, MessageRole
from domain.identity.roles import UserKind, UserRole
from domain.shared.exceptions import EntityNotFoundError, ValidationError
from presentation.api.routes.admin_chat_logs import ChatLogQuery, list_chat_logs
from presentation.api.routes.chat import clear_message_feedback, rate_message
from presentation.api.routes.conversations import get_conversation_history
from presentation.api.schemas import CurrentUser, FeedbackRequest
from tests.fakes import FakeUnitOfWork, FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from domain.chat.repositories import ChatLogFilter


OWNER = 1
CLIENT = 2
STRANGER = 3


class Redactor:
    def redact(self, text: str) -> str:
        return text.replace("owner@example.test", "[EMAIL]")

    def scan_and_redact(self, text: str) -> tuple[str, list[str]]:
        return self.redact(text), []


async def seeded() -> tuple[FakeUnitOfWork, dict[str, int]]:
    uow = FakeUnitOfWork()
    ids = {}

    for name, user_id in (("owner", OWNER), ("client", CLIENT)):
        conversation = await uow.conversations.create(user_id)
        assert conversation.id is not None
        question = Message(conversation_id=conversation.id, role=MessageRole.USER, content="question")
        answer = Message(conversation_id=conversation.id, role=MessageRole.ASSISTANT, content="answer")
        await uow.messages.save(question)
        await uow.messages.save(answer)
        assert question.id is not None
        assert answer.id is not None
        ids[f"{name}_question"] = question.id
        ids[f"{name}_answer"] = answer.id

    return uow, ids


def curator_scope(managed: list[int]) -> Mock:
    return Mock(
        get_scope=AsyncMock(
            return_value={"managed_client_ids": [], "managed_internal_ids": managed, "managed_group_ids": []}
        )
    )


def user(user_id: int, kind: UserKind = UserKind.INTERNAL, role: UserRole = UserRole.USER) -> CurrentUser:
    return CurrentUser(id=user_id, email=f"user{user_id}@example.test", role=role, kind=kind, is_active=True)


@pytest.mark.asyncio
async def test_owner_rates_own_answer_and_repeated_rating_updates_one_row() -> None:
    uow, ids = await seeded()
    service = MessageFeedbackService(FakeUnitOfWorkFactory(uow), Redactor())

    await service.rate(ids["owner_answer"], OWNER, FeedbackRating.UP)
    updated = await service.rate(
        ids["owner_answer"],
        OWNER,
        FeedbackRating.DOWN,
        [FeedbackReason.WRONG_FACT, FeedbackReason.OUTDATED],
        "  wrong <b>date</b>, mail owner@example.test  ",
    )

    assert list(uow.fake_message_feedback.items) == [(ids["owner_answer"], OWNER)]
    assert updated.rating == FeedbackRating.DOWN
    assert updated.reasons == {FeedbackReason.WRONG_FACT, FeedbackReason.OUTDATED}
    # Redacted like the journal question, then escaped like the stored conversation question.
    assert updated.comment == "wrong &lt;b&gt;date&lt;/b&gt;, mail [EMAIL]"


@pytest.mark.asyncio
async def test_client_rates_own_answer_on_the_same_terms() -> None:
    uow, ids = await seeded()
    response = await rate_message(
        message_id=ids["client_answer"],
        body=FeedbackRequest(rating=FeedbackRating.UP),
        current_user=user(CLIENT, UserKind.CLIENT),
        service=MessageFeedbackService(FakeUnitOfWorkFactory(uow), Redactor()),
    )

    assert response.rating == FeedbackRating.UP
    assert (ids["client_answer"], CLIENT) in uow.fake_message_feedback.items


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_key, user_id",
    [
        ("client_answer", OWNER),  # someone else's answer
        ("owner_question", OWNER),  # own message, but not an answer
        ("missing", OWNER),
    ],
)
async def test_foreign_user_and_missing_messages_are_indistinguishable_404(
    message_key: str, user_id: int
) -> None:
    uow, ids = await seeded()
    service = MessageFeedbackService(FakeUnitOfWorkFactory(uow), Redactor())
    message_id = ids.get(message_key, 999)

    with pytest.raises(EntityNotFoundError):
        await service.rate(message_id, user_id, FeedbackRating.UP)

    with pytest.raises(EntityNotFoundError):
        await clear_message_feedback(message_id=message_id, current_user=user(user_id), service=service)

    assert uow.fake_message_feedback.items == {}


@pytest.mark.asyncio
async def test_admin_cannot_rate_an_answer_in_someone_elses_conversation() -> None:
    uow, ids = await seeded()
    service = MessageFeedbackService(FakeUnitOfWorkFactory(uow), Redactor())

    with pytest.raises(EntityNotFoundError):
        await service.rate(ids["owner_answer"], STRANGER, FeedbackRating.DOWN)


def test_reasons_require_a_negative_rating() -> None:
    with pytest.raises(ValidationError):
        MessageFeedback(
            message_id=1, user_id=1, rating=FeedbackRating.UP, reasons=frozenset({FeedbackReason.OTHER})
        )

    feedback = MessageFeedback(
        message_id=1, user_id=1, rating=FeedbackRating.DOWN, reasons=frozenset({FeedbackReason.OTHER})
    )
    assert feedback.reasons == {FeedbackReason.OTHER}


@pytest.mark.asyncio
async def test_reasons_with_up_and_long_comment_are_rejected_before_writing() -> None:
    uow, ids = await seeded()
    service = MessageFeedbackService(FakeUnitOfWorkFactory(uow), Redactor())

    with pytest.raises(ValidationError):
        await service.rate(ids["owner_answer"], OWNER, FeedbackRating.UP, [FeedbackReason.INCOMPLETE])

    with pytest.raises(ValidationError):
        await service.rate(ids["owner_answer"], OWNER, FeedbackRating.DOWN, comment="x" * 1001)

    with pytest.raises(PydanticValidationError):
        FeedbackRequest(rating=FeedbackRating.DOWN, comment="x" * 1001)

    with pytest.raises(PydanticValidationError):
        FeedbackRequest.model_validate({"rating": "down", "reasons": ["unknown"]})

    assert uow.fake_message_feedback.items == {}


@pytest.mark.asyncio
async def test_repeated_click_clears_and_history_returns_only_own_reaction() -> None:
    uow, ids = await seeded()
    feedback = MessageFeedbackService(FakeUnitOfWorkFactory(uow), Redactor())
    await feedback.rate(ids["owner_answer"], OWNER, FeedbackRating.DOWN, [FeedbackReason.NO_ANSWER])
    conversations = Mock(
        get_history=AsyncMock(return_value=await uow.messages.list_page(1, limit=10)),
    )

    history = await get_conversation_history(
        conversation_id=1,
        limit=100,
        before_id=None,
        current_user=user(OWNER),
        service=conversations,
        feedback_service=feedback,
    )
    answer = next(message for message in history.messages if message.id == ids["owner_answer"])
    assert answer.feedback is not None
    assert answer.feedback.reasons == [FeedbackReason.NO_ANSWER]

    # An administrator reading the conversation sees no reaction of their own.
    admin_view = await get_conversation_history(
        conversation_id=1,
        limit=100,
        before_id=None,
        current_user=user(STRANGER, role=UserRole.ADMIN),
        service=conversations,
        feedback_service=feedback,
    )
    assert all(message.feedback is None for message in admin_view.messages)

    await clear_message_feedback(message_id=ids["owner_answer"], current_user=user(OWNER), service=feedback)
    assert uow.fake_message_feedback.items == {}


@pytest.mark.asyncio
async def test_curator_journal_shows_reactions_only_within_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    uow, ids = await seeded()
    owner_log = ChatLog(id=1, user_id=OWNER, message_id=ids["owner_answer"], space_id=1)
    foreign_log = ChatLog(id=2, user_id=CLIENT, message_id=ids["client_answer"], space_id=1)
    await uow.message_feedback.upsert(
        MessageFeedback(message_id=ids["owner_answer"], user_id=OWNER, rating=FeedbackRating.DOWN)
    )
    await uow.message_feedback.upsert(
        MessageFeedback(message_id=ids["client_answer"], user_id=CLIENT, rating=FeedbackRating.UP)
    )
    requested: list[ChatLogFilter] = []

    async def list_logs(filters: ChatLogFilter) -> list[ChatLog]:
        requested.append(filters)
        return [
            log
            for log in (owner_log, foreign_log)
            if filters.user_ids is None or log.user_id in filters.user_ids
        ]

    async def count_logs(filters: ChatLogFilter) -> int:
        return len(await list_logs(filters))

    monkeypatch.setattr(uow.fake_chat_logs, "list_logs", list_logs)
    monkeypatch.setattr(uow.fake_chat_logs, "count_logs", count_logs)
    response = await list_chat_logs(
        options=ChatLogQuery(rating="down"),
        current_user=user(10, role=UserRole.CURATOR),
        service=ChatLogService(FakeUnitOfWorkFactory(uow)),
        assignment_service=curator_scope([OWNER]),
    )

    assert requested[-1].user_ids == (OWNER, 10)
    assert requested[-1].rating == "down"
    assert [(entry.id, entry.feedback.rating if entry.feedback else None) for entry in response.logs] == [
        (1, FeedbackRating.DOWN)
    ]
