"""Fake implementations for unit-testing application services.

These are lightweight in-memory substitutes for infrastructure ports,
allowing application-layer tests to run without Postgres, Qdrant, Redis,
or LLM.

Usage::

    from tests.fakes import FakeUnitOfWorkFactory, FakeChatRAGPort

    async def test_stream_chat():
        service = ChatService(
            uow_factory=FakeUnitOfWorkFactory(),
            rag_service=FakeChatRAGPort(),
        )
        ...
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from domain.entities.message import Message
from domain.value_objects.stream_events import SourcesEvent, TextChunk

# ---------------------------------------------------------------------------
# Fake UnitOfWork
# ---------------------------------------------------------------------------


class FakeConversationRepository:
    def __init__(self) -> None:
        self._convs: dict[int, dict] = {}
        self._next_id = 1

    async def get_or_create(self, conversation_id: int | None, user_id: int):
        if conversation_id and conversation_id in self._convs:
            return type("Conv", (), self._convs[conversation_id])()
        conv_id = self._next_id
        self._next_id += 1
        self._convs[conv_id] = {"id": conv_id, "user_id": user_id, "summary": None}
        return type("Conv", (), self._convs[conv_id])()

    async def get(self, conv_id: int):
        if conv_id in self._convs:
            return type("Conv", (), self._convs[conv_id])()
        return None

    async def get_owner_id(self, conv_id: int) -> int | None:
        conv = self._convs.get(conv_id)
        return conv["user_id"] if conv else None

    async def create(self, user_id: int):
        conv_id = self._next_id
        self._next_id += 1
        self._convs[conv_id] = {"id": conv_id, "user_id": user_id, "summary": None}
        return type("Conv", (), self._convs[conv_id])()

    async def save(self, conv) -> None:
        if hasattr(conv, "id"):
            self._convs[conv.id] = {
                "id": conv.id,
                "user_id": getattr(conv, "user_id", 0),
                "summary": getattr(conv, "summary", None),
            }

    async def list_by_user(self, user_id: int, limit: int = 50, offset: int = 0):
        return []


class FakeMessageRepository:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def save(self, msg: Message) -> None:
        self._messages.append(msg)

    async def get_history(self, conversation_id: int, window: int = 100) -> list[Message]:
        return [m for m in self._messages if m.conversation_id == conversation_id][-window:]


class FakeGroupRepository:
    async def get_user_group_ids(self, user_id: int) -> list[int]:
        return []

    async def list_all(self):
        return []

    async def list_by_ids(self, ids: list[int]):
        return []

    async def create(self, name: str) -> int:
        return 1

    async def list_members(self, group_id: int):
        return []

    async def add_user(self, user_id: int, group_id: int) -> None:
        pass

    async def remove_user(self, user_id: int, group_id: int) -> None:
        pass


class FakeChunkRepository:
    async def search_substring(self, **kwargs):
        return []

    async def get_all_contents(self) -> list[str]:
        return []


class FakeDocumentRepository:
    async def get_by_id(self, doc_id: int):
        return None

    async def list_all(self):
        return []


class FakeChatLogRepository:
    async def save(self, log) -> None:
        pass


class FakeUserRepository:
    async def get_by_id(self, user_id: int):
        return None


class FakeConfigParameterRepository:
    async def get_all(self):
        return []

    async def get_by_key(self, key: str):
        return None

    async def update_value(self, key: str, value: str) -> None:
        pass


class FakeBackgroundJobRepository:
    async def create(self, job):
        return job

    async def mark_running(self, job_id: int) -> None:
        pass

    async def mark_done(self, job_id: int) -> None:
        pass

    async def mark_failed(self, job_id: int, error: str) -> None:
        pass

    async def count_active(self) -> int:
        return 0

    async def delete_old(self, days: int) -> int:
        return 0

    async def list_recent(self, limit: int = 50, offset: int = 0):
        return []

    async def get_by_id(self, job_id: int):
        return None

    async def count_by_status(self) -> dict[str, int]:
        return {}

    async def recover_orphaned(self, timeout_minutes: int = 15) -> list[int]:
        return []


class FakeApiKeyRepository:
    async def create(self, key):
        return key

    async def list_for_user(self, user_id: int):
        return []

    async def revoke(self, key_id: int) -> None:
        pass

    async def get_active_client_by_hash(self, key_hash: str):
        return None

    async def touch_last_used(self, key_id: int) -> None:
        pass


class FakeBenchmarkQuestionRepository:
    async def list(self, **kwargs):
        return []

    async def count(self, **kwargs) -> int:
        return 0

    async def create(self, entity):
        entity.id = 1
        return entity

    async def update(self, question_id: int, **kwargs):
        return None

    async def delete(self, question_id: int) -> bool:
        return True

    async def bulk_create(self, entities) -> int:
        return len(entities)


class FakeBenchmarkSweepRepository:
    async def get_by_id(self, sweep_id: int):
        return None

    async def create(self, entity):
        entity.id = 1
        return entity

    async def update_status(self, sweep_id: int, status: str) -> None:
        pass

    async def list(self, limit: int = 50, offset: int = 0):
        return []

    async def count(self) -> int:
        return 0


class FakeBenchmarkRunRepository:
    async def get_by_id(self, run_id: int):
        return None

    async def get_by_ids(self, ids: list[int]):
        return []

    async def list(self, **kwargs):
        return []

    async def count(self, **kwargs) -> int:
        return 0


class FakeUnitOfWork:
    """In-memory UnitOfWork for unit tests."""

    def __init__(self) -> None:
        self.users = FakeUserRepository()
        self.conversations = FakeConversationRepository()
        self.messages = FakeMessageRepository()
        self.documents = FakeDocumentRepository()
        self.chunks = FakeChunkRepository()
        self.groups = FakeGroupRepository()
        self.api_keys = FakeApiKeyRepository()
        self.config_parameters = FakeConfigParameterRepository()
        self.background_jobs = FakeBackgroundJobRepository()
        self.chat_logs = FakeChatLogRepository()
        self.benchmark_questions = FakeBenchmarkQuestionRepository()
        self.benchmark_sweeps = FakeBenchmarkSweepRepository()
        self.benchmark_runs = FakeBenchmarkRunRepository()
        self._event_handlers: list = []
        self._committed = False
        self._rolled_back = False

    def on_event(self, handler) -> None:
        self._event_handlers.append(handler)

    async def publish_event(self, event: object) -> None:
        for handler in self._event_handlers:
            await handler(event)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._committed = True
        else:
            self._rolled_back = True


class FakeUnitOfWorkFactory:
    """In-memory UnitOfWorkFactory for unit tests."""

    def __init__(self) -> None:
        self._uow = FakeUnitOfWork()

    @asynccontextmanager
    async def create(self, master: bool = False) -> AsyncGenerator[FakeUnitOfWork, None]:
        yield self._uow


# ---------------------------------------------------------------------------
# Fake ChatRAGPort
# ---------------------------------------------------------------------------


class FakeChatRAGPort:
    """Returns a canned answer for testing ChatService."""

    def __init__(
        self,
        answer: str = "Test answer",
        sources: list[dict] | None = None,
        breadth: str = "narrow",
        domain: str = "general",
    ) -> None:
        self._answer = answer
        self._sources = sources or []
        self._breadth = breadth
        self._domain = domain

    async def stream(self, question: str, history: list, ctx):
        yield TextChunk(text=self._answer)
        yield SourcesEvent(sources=self._sources, confidence=0.9)

    async def invoke(self, question: str, history: list, ctx):
        from application.dto.chat_dto import RagResult

        return RagResult(
            answer=self._answer,
            sources=self._sources,
            breadth=self._breadth,
            domain=self._domain,
            retrieval_count=len(self._sources),
            reranker_score=None,
            model_used="fake-model",
        )


# ---------------------------------------------------------------------------
# Fake MLClientRegistry
# ---------------------------------------------------------------------------


class FakeMLClientRegistry:
    """Lightweight substitute for MLClientRegistry in unit tests."""

    def __init__(self, llm_response: str = "fake") -> None:
        self._llm_response = llm_response
        self._invalidated_llm = False
        self._invalidated_bm25 = False

    def llm(self):
        return self

    def llm_for_breadth(self, breadth: str):
        return self

    def embeddings(self):
        return self

    def reranker(self):
        return self

    def qdrant_client(self):
        return self

    def bm25_index(self):
        return None

    def vector_store(self):
        return self

    async def astream(self, messages):
        yield type("Chunk", (), {"content": self._llm_response})()

    def embed_query(self, text: str):
        return [0.0] * 384

    def invalidate_llm(self) -> None:
        self._invalidated_llm = True

    def invalidate_bm25(self) -> None:
        self._invalidated_bm25 = True

    def invalidate_embeddings(self) -> None:
        pass

    def invalidate_reranker(self) -> None:
        pass

    def invalidate_qdrant(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fake EventBus
# ---------------------------------------------------------------------------


class FakeEventBus:
    """In-memory event bus for testing."""

    def __init__(self) -> None:
        self._handlers: dict[type, list] = {}
        self._published: list[object] = []

    def subscribe(self, event_type: type, handler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: object) -> None:
        self._published.append(event)
        for handler in self._handlers.get(type(event), []):
            handler(event)
