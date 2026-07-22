"""
Tests for chat use cases: SyncChat, StreamChat.
All dependencies are mocked — no real RAG, no real DB.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from application.dto.chat_dto import ChatCommand
from application.use_cases.chat.stream_chat import StreamChat
from application.use_cases.chat.sync_chat import SyncChat
from domain.entities.conversation import Conversation
from domain.entities.message import Message
from domain.value_objects.message_role import MessageRole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(history_window=8):
    return SimpleNamespace(history_window=history_window)


def _make_conversation(id=1, user_id=1):
    return Conversation(id=id, user_id=user_id)


# ---------------------------------------------------------------------------
# SyncChat
# ---------------------------------------------------------------------------


class TestSyncChat:
    def setup_method(self):
        self.conversation_repo = MagicMock()
        self.message_repo = MagicMock()
        self.rag_service = AsyncMock()
        self.settings = _mock_settings()
        self.use_case = SyncChat(
            self.conversation_repo,
            self.message_repo,
            self.rag_service,
            self.settings,
        )

    @pytest.mark.asyncio
    async def test_successful_chat(self):
        conv = _make_conversation(id=10, user_id=1)
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = [
            Message(role=MessageRole.USER, content="old q"),
            Message(role=MessageRole.ASSISTANT, content="old a"),
        ]
        self.rag_service.invoke.return_value = ("answer text", [{"source": "doc.pdf"}])

        result = await self.use_case.execute(
            command=ChatCommand(question="What is X?"),
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        )

        assert result.answer == "answer text"
        assert result.conversation_id == 10
        assert len(result.sources) == 1
        # Verify user message saved (check by inspecting call args)
        saved_messages = [c.args[0] for c in self.message_repo.save.call_args_list]
        user_msgs = [m for m in saved_messages if m.role == MessageRole.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "What is X?"
        assert user_msgs[0].conversation_id == 10

    @pytest.mark.asyncio
    async def test_new_conversation_creates_when_none(self):
        conv = _make_conversation(id=20, user_id=1)
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = []
        self.rag_service.invoke.return_value = ("hi", [])

        result = await self.use_case.execute(
            command=ChatCommand(question="hello", conversation_id=None),
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        )

        self.conversation_repo.get_or_create.assert_called_once_with(None, 1)
        assert result.conversation_id == 20

    @pytest.mark.asyncio
    async def test_existing_conversation_reused(self):
        conv = _make_conversation(id=30, user_id=1)
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = []
        self.rag_service.invoke.return_value = ("ok", [])

        result = await self.use_case.execute(
            command=ChatCommand(question="next", conversation_id=30),
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        )

        self.conversation_repo.get_or_create.assert_called_once_with(30, 1)

    @pytest.mark.asyncio
    async def test_history_trims_last_user_message(self):
        """If history ends with user message, it's trimmed to avoid duplication."""
        conv = _make_conversation()
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = [
            Message(role=MessageRole.USER, content="q1"),
            Message(role=MessageRole.ASSISTANT, content="a1"),
            Message(role=MessageRole.USER, content="q2"),  # last is user — should be trimmed
        ]
        self.rag_service.invoke.return_value = ("answer", [])

        await self.use_case.execute(
            command=ChatCommand(question="q2"),
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        )

        # History passed to RAG should have trimmed the last user message
        call_args = self.rag_service.invoke.call_args
        history = call_args.kwargs.get("history") or call_args[1].get("history", call_args[0][1] if len(call_args[0]) > 1 else None)
        # The last message should be the assistant message, not the user message
        if history:
            assert history[-1].role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_history_not_trimmed_when_ends_with_assistant(self):
        conv = _make_conversation()
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = [
            Message(role=MessageRole.USER, content="q1"),
            Message(role=MessageRole.ASSISTANT, content="a1"),
        ]
        self.rag_service.invoke.return_value = ("answer", [])

        await self.use_case.execute(
            command=ChatCommand(question="q2"),
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        )

        call_args = self.rag_service.invoke.call_args
        history = call_args.kwargs.get("history") or call_args[1].get("history", None)
        if history:
            assert len(history) == 2

    @pytest.mark.asyncio
    async def test_empty_history(self):
        conv = _make_conversation()
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = []
        self.rag_service.invoke.return_value = ("first answer", [])

        result = await self.use_case.execute(
            command=ChatCommand(question="first question"),
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        )

        assert result.answer == "first answer"

    @pytest.mark.asyncio
    async def test_rag_receives_user_context(self):
        conv = _make_conversation()
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = []
        self.rag_service.invoke.return_value = ("answer", [])

        await self.use_case.execute(
            command=ChatCommand(question="q"),
            user_id=42,
            user_kind="client",
            user_role="user",
            user_group_ids=[10, 20],
            assigned_client_ids=[30],
        )

        call_kwargs = self.rag_service.invoke.call_args.kwargs
        assert call_kwargs["user_id"] == 42
        assert call_kwargs["user_kind"] == "client"
        assert call_kwargs["user_group_ids"] == [10, 20]
        assert call_kwargs["assigned_client_ids"] == [30]


# ---------------------------------------------------------------------------
# StreamChat
# ---------------------------------------------------------------------------


class TestStreamChat:
    def setup_method(self):
        self.conversation_repo = MagicMock()
        self.message_repo = MagicMock()
        self.rag_service = AsyncMock()
        self.settings = _mock_settings()
        self.use_case = StreamChat(
            self.conversation_repo,
            self.message_repo,
            self.rag_service,
            self.settings,
        )

    @pytest.mark.asyncio
    async def test_stream_yields_text_chunks(self):
        conv = _make_conversation(id=10)
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = []

        async def fake_stream(**kwargs):
            yield "Hello "
            yield "world!"

        self.rag_service.stream = fake_stream

        chunks = []
        async for chunk in self.use_case.execute(
            question="test",
            conversation_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        ):
            chunks.append(chunk)

        # Should have text chunks + meta chunk
        text_chunks = [c for c in chunks if not c.startswith("\n__meta__:")]
        assert "".join(text_chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_saves_assistant_message(self):
        conv = _make_conversation(id=10)
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = []

        async def fake_stream(**kwargs):
            yield "answer"

        self.rag_service.stream = fake_stream

        async for _ in self.use_case.execute(
            question="q",
            conversation_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        ):
            pass

        # Find the assistant message save call
        saved_messages = [c.args[0] for c in self.message_repo.save.call_args_list]
        assistant_msgs = [m for m in saved_messages if m.role == MessageRole.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "answer"

    @pytest.mark.asyncio
    async def test_stream_parses_sources(self):
        conv = _make_conversation(id=10)
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = []

        sources_data = [{"source": "doc.pdf", "pages": [1]}]

        async def fake_stream(**kwargs):
            yield "answer"
            yield f"\n__sources__:{__import__('json').dumps(sources_data)}"

        self.rag_service.stream = fake_stream

        chunks = []
        async for chunk in self.use_case.execute(
            question="q",
            conversation_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        ):
            chunks.append(chunk)

        # Sources should be in the meta chunk
        meta_chunk = [c for c in chunks if c.startswith("\n__meta__:")]
        assert len(meta_chunk) == 1
        import json
        meta = json.loads(meta_chunk[0].replace("\n__meta__:", ""))
        assert meta["sources"] == sources_data

    @pytest.mark.asyncio
    async def test_stream_empty(self):
        conv = _make_conversation(id=10)
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = []

        async def fake_stream(**kwargs):
            if False:
                yield  # make it an async generator

        self.rag_service.stream = fake_stream

        chunks = []
        async for chunk in self.use_case.execute(
            question="q",
            conversation_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        ):
            chunks.append(chunk)

        # Should still have meta chunk
        assert len(chunks) == 1
        assert chunks[0].startswith("\n__meta__:")

    @pytest.mark.asyncio
    async def test_stream_malformed_sources_does_not_crash(self):
        """Malformed JSON in sources chunk is logged and ignored, not raised."""
        conv = _make_conversation(id=10)
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = []

        async def fake_stream(**kwargs):
            yield "answer"
            yield "\n__sources__:{invalid json!!!}"

        self.rag_service.stream = fake_stream

        chunks = []
        async for chunk in self.use_case.execute(
            question="q",
            conversation_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        ):
            chunks.append(chunk)

        # Should still produce answer + meta, sources should be empty list
        assert "answer" in chunks
        assert any("__meta__" in c for c in chunks)
        # Verify sources is empty (malformed JSON was caught)
        meta_chunk = [c for c in chunks if c.startswith("\n__meta__:")][0]
        import json
        meta = json.loads(meta_chunk.replace("\n__meta__:", ""))
        assert meta["sources"] == []

    @pytest.mark.asyncio
    async def test_stream_trims_history_like_sync(self):
        conv = _make_conversation()
        self.conversation_repo.get_or_create.return_value = conv
        self.message_repo.get_history.return_value = [
            Message(role=MessageRole.USER, content="q1"),
            Message(role=MessageRole.ASSISTANT, content="a1"),
            Message(role=MessageRole.USER, content="q2"),
        ]

        captured_kwargs = {}

        async def fake_stream(**kwargs):
            captured_kwargs.update(kwargs)
            yield "answer"

        self.rag_service.stream = fake_stream

        async for _ in self.use_case.execute(
            question="q2",
            conversation_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
            user_group_ids=[],
            assigned_client_ids=[],
        ):
            pass

        # Verify history was trimmed
        history = captured_kwargs["history"]
        assert history[-1].role == MessageRole.ASSISTANT
