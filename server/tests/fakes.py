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
from domain.entities.vector_outbox_entry import OutboxStatus, VectorOutboxEntry
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
    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._next_id = 1

    async def bulk_insert(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[str],
        owner_id: int | None = None,
        group_id: int | None = None,
        doc_domain: str = "general",
        content_hashes: list[str] | None = None,
    ) -> list[int]:
        # Remove existing chunks for this document
        self._chunks = [c for c in self._chunks if c["document_id"] != document_id]
        ids = []
        for _i, content in enumerate(chunks):
            chunk_id = self._next_id
            self._next_id += 1
            self._chunks.append({"id": chunk_id, "document_id": document_id, "content": content})
            ids.append(chunk_id)
        return ids

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


class FakeVectorOutboxRepository:
    """In-memory outbox repository for testing SAGA/Outbox pattern."""

    def __init__(self) -> None:
        self._entries: dict[int, VectorOutboxEntry] = {}
        self._next_id = 1
        self._notifications: list[dict] = []

    async def enqueue(self, entry: VectorOutboxEntry) -> VectorOutboxEntry:
        entry.id = self._next_id
        self._next_id += 1
        self._entries[entry.id] = entry
        self._notifications.append({"id": entry.id, "op": entry.operation.value})
        return entry

    async def claim_batch(self, worker_id: str, limit: int = 20) -> list[VectorOutboxEntry]:
        claimed = []
        for entry in list(self._entries.values()):
            if len(claimed) >= limit:
                break
            if entry.status in (OutboxStatus.PENDING, OutboxStatus.FAILED):
                entry.status = OutboxStatus.IN_PROGRESS
                claimed.append(entry)
        return claimed

    async def mark_done(self, entry_id: int) -> None:
        if entry_id in self._entries:
            self._entries[entry_id].status = OutboxStatus.DONE

    async def mark_failed(self, entry_id: int, error: str, backoff_seconds: int) -> None:
        if entry_id in self._entries:
            entry = self._entries[entry_id]
            entry.attempts += 1
            entry.last_error = error
            if entry.attempts >= entry.max_attempts:
                entry.status = OutboxStatus.DEAD_LETTER
            else:
                entry.status = OutboxStatus.FAILED

    async def mark_dead_letter(self, entry_id: int, error: str) -> None:
        if entry_id in self._entries:
            self._entries[entry_id].status = OutboxStatus.DEAD_LETTER
            self._entries[entry_id].last_error = error

    async def count_pending(self) -> int:
        return sum(
            1
            for e in self._entries.values()
            if e.status in (OutboxStatus.PENDING, OutboxStatus.FAILED, OutboxStatus.IN_PROGRESS)
        )

    async def recover_stuck(self, stuck_timeout_minutes: int = 5) -> int:
        recovered = 0
        for entry in self._entries.values():
            if entry.status == OutboxStatus.IN_PROGRESS:
                entry.status = OutboxStatus.FAILED
                entry.last_error = "Recovered from stuck in_progress"
                recovered += 1
        return recovered

    async def count_by_document(self, document_id: int) -> dict[str, int]:
        pending_statuses = {OutboxStatus.PENDING, OutboxStatus.FAILED, OutboxStatus.IN_PROGRESS}
        failed_statuses = {OutboxStatus.FAILED, OutboxStatus.DEAD_LETTER}

        pending = sum(
            1
            for e in self._entries.values()
            if e.aggregate_id == document_id and e.status in pending_statuses
        )
        failed = sum(
            1 for e in self._entries.values() if e.aggregate_id == document_id and e.status in failed_statuses
        )
        return {"pending": pending, "failed": failed}

    async def list_dead_letters(self, limit: int = 50) -> list[VectorOutboxEntry]:
        return [e for e in self._entries.values() if e.status == OutboxStatus.DEAD_LETTER][:limit]


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
        self.vector_outbox = FakeVectorOutboxRepository()
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

    def __init__(self, uow: FakeUnitOfWork | None = None) -> None:
        self._uow = uow or FakeUnitOfWork()

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
