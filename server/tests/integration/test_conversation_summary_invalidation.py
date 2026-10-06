"""PostgreSQL contract for invalidating summaries after document deletion."""

import pytest
from sqlalchemy import null, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.value_objects.message_role import MessageRole
from infrastructure.database.models import ConversationModel, MessageModel, UserModel
from infrastructure.repositories.conversation.sqlalchemy_conversation_repository import (
    SQLAlchemyConversationRepository,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_clear_summaries_only_for_assistant_citations(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    cases = [
        (MessageRole.ASSISTANT, [{"document_id": 7}, {"document_id": 7}], "summary", None),
        (MessageRole.ASSISTANT, [{"document_id": 8}, {"document_id": 7}], "summary", None),
        (MessageRole.ASSISTANT, [{"document_id": 8}], "summary", "summary"),
        (MessageRole.USER, [{"document_id": 7}], "summary", "summary"),
        (MessageRole.ASSISTANT, [], "summary", "summary"),
        (MessageRole.ASSISTANT, None, "summary", "summary"),
        (MessageRole.ASSISTANT, null(), "summary", "summary"),
        (MessageRole.ASSISTANT, [{"title": "legacy citation"}], "summary", "summary"),
        (MessageRole.ASSISTANT, [{"document_id": 7}], None, None),
    ]
    async with sessions.begin() as session:
        user = UserModel(email="summary-test@example.com", hashed_password="unused")
        session.add(user)
        await session.flush()
        conversations = []
        for role, sources, summary, expected in cases:
            conversation = ConversationModel(user_id=user.id, summary=summary)
            session.add(conversation)
            await session.flush()
            session.add(
                MessageModel(
                    conversation_id=conversation.id,
                    role=role.value,
                    content="citation test",
                    sources=sources,
                )
            )
            conversations.append((conversation.id, expected))
        await session.flush()

        repo = SQLAlchemyConversationRepository(session)
        assert await repo.clear_summaries_referencing(7) == 2
        assert await repo.clear_summaries_referencing(7) == 0

    async with sessions() as session:
        summaries = dict(
            (await session.execute(select(ConversationModel.id, ConversationModel.summary))).all()
        )
        assert summaries == dict(conversations)
