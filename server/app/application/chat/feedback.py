"""Rate a delivered assistant answer: only the owner of the conversation may do it."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from domain.chat.feedback import MessageFeedback, validate_feedback_comment
from domain.chat.message import MessageRole
from domain.shared.exceptions import EntityNotFoundError


if TYPE_CHECKING:
    from collections.abc import Iterable

    from application.chat.ports import PIIRedactorPort
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.chat.feedback import FeedbackRating, FeedbackReason


class MessageFeedbackService:
    def __init__(self, uow_factory: UnitOfWorkFactory, pii_redactor: PIIRedactorPort) -> None:
        self.uow_factory = uow_factory
        self.pii_redactor = pii_redactor

    async def rate(
        self,
        message_id: int,
        user_id: int,
        rating: FeedbackRating,
        reasons: Iterable[FeedbackReason] = (),
        comment: str | None = None,
    ) -> MessageFeedback:
        text = validate_feedback_comment(comment)
        feedback = MessageFeedback(
            message_id=message_id,
            user_id=user_id,
            rating=rating,
            reasons=frozenset(reasons),
            # Stored like a question in the conversation: redacted, then escaped.
            comment=html.escape(self.pii_redactor.redact(text)) if text else None,
        )

        async with self.uow_factory.create(master=True) as uow:
            await self.require_own_answer(uow, message_id, user_id)
            return await uow.message_feedback.upsert(feedback)

    async def clear(self, message_id: int, user_id: int) -> None:
        async with self.uow_factory.create(master=True) as uow:
            await self.require_own_answer(uow, message_id, user_id)
            await uow.message_feedback.delete(message_id, user_id)

    async def for_messages(self, message_ids: list[int], user_id: int) -> dict[int, MessageFeedback]:
        async with self.uow_factory.create() as uow:
            items = await uow.message_feedback.list_for_messages(message_ids, user_id)
            return {item.message_id: item for item in items}

    @staticmethod
    async def require_own_answer(uow: UnitOfWork, message_id: int, user_id: int) -> None:
        """Every failed check is the same 404: never reveal that someone else's message exists."""
        message = await uow.messages.get_by_id(message_id)

        if (
            message is None
            or message.role != MessageRole.ASSISTANT
            or await uow.conversations.get_owner_id(message.conversation_id) != user_id
        ):
            msg = "Message"
            raise EntityNotFoundError(msg, message_id)
