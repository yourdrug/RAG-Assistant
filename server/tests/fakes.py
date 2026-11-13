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

import asyncio
from contextlib import asynccontextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, Unpack
from unittest.mock import AsyncMock, MagicMock

from application.ports.rate_limit import RateLimitDecision
from application.rag.stream_events import SourcesEvent, TextChunk
from application.uow.factory import UnitOfWorkFactory
from application.uow.unit_of_work import UnitOfWork
from domain.chat.conversation import Conversation
from domain.chat.repositories import DEFAULT_CHAT_LOG_FILTER
from domain.identity.api_key import ApiKey
from domain.identity.roles import UserKind, UserRole
from domain.identity.user import User
from domain.knowledge.access_control import can_view_document
from domain.knowledge.chunk import (
    TABLE_CONTEXT_MAX_CHUNKS,
    ChunkRecord,
    ChunkStats,
    extract_chunk_context,
)
from domain.knowledge.document import DocumentProcessingUpdate
from domain.knowledge.enums import DocumentStatus, DocumentVisibility, DomainKey, SearchMode
from domain.knowledge.index_sync import OutboxStatus
from domain.knowledge.repositories.chunk import (
    DEFAULT_CHUNK_INSERT_OPTIONS,
    DEFAULT_MANUAL_CHUNK_INSERT_OPTIONS,
)
from domain.rag.breadth import Breadth
from domain.rag.result import RagResult
from domain.shared.exceptions import EntityNotFoundError, ValidationError
from domain.shared.pagination import CursorPage, encode_cursor


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Sequence
    from datetime import date, datetime
    from types import TracebackType

    from application.ports.rate_limit import RateLimitPolicyName
    from application.rag.stream_events import StreamEvent
    from application.runtime.config import ConfigParameter
    from domain.benchmark.question import BenchmarkQuestion
    from domain.benchmark.run import BenchmarkRun
    from domain.benchmark.sweep import BenchmarkSweep
    from domain.chat.chat_log import ChatLog
    from domain.chat.conversation import ConversationListItem
    from domain.chat.message import Message
    from domain.chat.repositories import ChatLogFilter
    from domain.identity.api_key import ApiKeyClientInfo
    from domain.identity.group import GroupInfo, GroupMemberInfo
    from domain.identity.user_context import UserContext
    from domain.ingestion.repositories import IngestionRegistryEntry
    from domain.knowledge.acts.models import ActVersion, RegulatoryAct
    from domain.knowledge.chunk import (
        ChunkInsert,
    )
    from domain.knowledge.document import Document
    from domain.knowledge.document_nodes import DocumentNode
    from domain.knowledge.index_sync import VectorOutboxEntry
    from domain.knowledge.repositories.chunk import (
        ChunkInsertOptions,
        ManualChunkInsertOptions,
    )
    from domain.knowledge.repositories.document import DocumentFilter, DocumentStatusUpdate
    from domain.rag.context import RagContext
    from domain.runtime.jobs import BackgroundJob


# ---------------------------------------------------------------------------
# Fake UnitOfWork
# ---------------------------------------------------------------------------


class FakeConversationRepository:
    def __init__(self) -> None:
        self._convs: dict[int, dict[str, Any]] = {}
        self._next_id = 1

    async def get_or_create(self, conversation_id: int | None, user_id: int) -> Conversation:
        if conversation_id and conversation_id in self._convs:
            return Conversation(**self._convs[conversation_id])

        conversation_id = self._next_id
        self._next_id += 1
        self._convs[conversation_id] = {"id": conversation_id, "user_id": user_id, "summary": None}
        return Conversation(**self._convs[conversation_id])

    async def get_by_id(self, conversation_id: int) -> Conversation | None:
        data = self._convs.get(conversation_id)
        return Conversation(**data) if data else None

    async def get_for_update(self, conversation_id: int) -> Conversation | None:
        return await self.get_by_id(conversation_id)

    async def update_summary(self, conversation_id: int, summary: str | None) -> None:
        conversation = self._convs.get(conversation_id)

        if conversation is not None:
            conversation["summary"] = summary

    async def update_summary_if_unchanged(
        self, conversation_id: int, expected_summary: str | None, new_summary: str | None
    ) -> bool:
        conversation = self._convs.get(conversation_id)

        if conversation is None or conversation["summary"] != expected_summary:
            return False

        conversation["summary"] = new_summary
        return True

    async def get_owner_id(self, conversation_id: int) -> int | None:
        conversation = self._convs.get(conversation_id)
        return conversation["user_id"] if conversation else None

    async def create(self, user_id: int) -> Conversation:
        conversation_id = self._next_id
        self._next_id += 1
        self._convs[conversation_id] = {"id": conversation_id, "user_id": user_id, "summary": None}
        return Conversation(**self._convs[conversation_id])

    async def save(self, conversation: Conversation) -> Conversation:
        if conversation.id is None:
            conversation.id = self._next_id
            self._next_id += 1

        if conversation.id is not None:
            self._convs[conversation.id] = {
                "id": conversation.id,
                "user_id": getattr(conversation, "user_id", 0),
                "summary": getattr(conversation, "summary", None),
            }

        return conversation

    async def clear_summaries_referencing(self, document_id: int) -> int:
        """Clear summaries for conversations whose messages cite the given document_id.

        Fake implementation: clears ALL summaries (conservative, matches pre-fix behavior).
        """
        del document_id
        count = 0

        for conversation in self._convs.values():
            if conversation.get("summary") is not None:
                conversation["summary"] = None
                count += 1

        return count

    async def list_by_user(
        self, user_id: int, limit: int = 50, offset: int = 0
    ) -> list[ConversationListItem]:
        del user_id, limit, offset
        return []


class FakeMessageRepository:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def save(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self, conversation_id: int, window: int = 8) -> list[Message]:
        return [m for m in self._messages if m.conversation_id == conversation_id][-window:]

    async def list_page(
        self, conversation_id: int, limit: int, before_id: int | None = None
    ) -> list[Message]:
        messages = [
            m
            for m in self._messages
            if m.conversation_id == conversation_id
            and (before_id is None or (m.id is not None and m.id < before_id))
        ]
        return sorted(messages, key=lambda m: m.id or 0)[-limit:]


class FakeGroupRepository:
    async def get_user_group_ids(self, user_id: int) -> list[int]:
        del user_id
        return []

    async def list_all(self) -> list[GroupInfo]:
        return []

    async def list_by_ids(self, ids: list[int]) -> list[GroupInfo]:
        del ids
        return []

    async def create(self, name: str) -> int:
        del name
        return 1

    async def list_members(self, group_id: int) -> list[GroupMemberInfo]:
        del group_id
        return []

    async def add_user(self, user_id: int, group_id: int) -> None:
        pass

    async def remove_user(self, user_id: int, group_id: int) -> None:
        pass


class FakeChunkRepository:
    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self._next_id = 1
        self.document_nodes: dict[int, tuple[int | None, list[DocumentNode]]] = {}

    async def replace_document_nodes(
        self, document_id: int, act_version_id: int | None, nodes: list[DocumentNode]
    ) -> None:
        self.document_nodes[document_id] = (act_version_id, list(nodes))

    async def get_ids_by_document_id(self, document_id: int, *, for_update: bool = False) -> list[int]:
        del for_update
        return [c["id"] for c in self._chunks if c["document_id"] == document_id]

    async def get_by_ids(self, document_id: int, chunk_ids: list[int]) -> list[ChunkRecord]:
        rows, _ = await self.list_for_document(document_id, limit=len(self._chunks))
        ids = set(chunk_ids)
        return [row for row in rows if row.chunk_id in ids]

    async def iter_all_contents_with_acl(
        self, batch_size: int = 1000
    ) -> AsyncIterator[tuple[str, str, int | None, int | None]]:
        del batch_size

        for c in self._chunks:
            yield c["content"], c["visibility"], c["owner_id"], c["group_id"]

    async def bulk_insert(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[ChunkInsert],
        options: ChunkInsertOptions = DEFAULT_CHUNK_INSERT_OPTIONS,
    ) -> list[int]:
        owner_id = options.owner_id
        group_id = options.group_id
        doc_domain = options.doc_domain
        domain_metadata = options.domain_metadata
        act_version_id = options.act_version_id
        effective_from = options.effective_from
        effective_to = options.effective_to
        is_current = options.is_current
        self._chunks = [c for c in self._chunks if c["document_id"] != document_id]
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._next_id
            self._next_id += 1
            self._chunks.append(
                {
                    "id": chunk_id,
                    "document_id": document_id,
                    "content": chunk.content,
                    "embed_text": chunk.embed_text,
                    "domain_metadata": domain_metadata,
                    "act_version_id": act_version_id,
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                    "is_current": is_current,
                    "filename": filename,
                    "visibility": visibility,
                    "doc_domain": doc_domain,
                    "owner_id": owner_id,
                    "group_id": group_id,
                    "chunk_index": i,
                    "content_hash": chunk.content_hash,
                    "context_metadata": extract_chunk_context(chunk.context_metadata),
                }
            )
            ids.append(chunk_id)

        return ids

    async def set_current_by_act_version_ids(self, act_version_ids: list[int], *, is_current: bool) -> int:
        updated = 0

        for c in self._chunks:
            if c.get("act_version_id") in act_version_ids:
                c["is_current"] = is_current
                updated += 1

        return updated

    async def update_temporal_by_act_version_id(
        self, act_version_id: int, effective_from: date | None, effective_to: date | None
    ) -> int:
        updated = 0

        for c in self._chunks:
            if c.get("act_version_id") == act_version_id:
                c["effective_from"] = effective_from
                c["effective_to"] = effective_to
                updated += 1

        return updated

    async def search_substring(
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        as_of_date: date | None = None,
        document_ids: Sequence[int] | None = None,
    ) -> list[ChunkRecord]:
        del query, user, limit, mode, document_id, as_of_date, document_ids
        return []

    async def search_substring_page(
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        offset: int = 0,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        as_of_date: date | None = None,
    ) -> tuple[list[ChunkRecord], int]:
        del query, user, limit, offset, mode, document_id, as_of_date
        return [], 0

    async def list_for_document(
        self, document_id: int, limit: int = 50, offset: int = 0, content_hashes: list[str] | None = None
    ) -> tuple[list[ChunkRecord], int]:

        del content_hashes
        rows = [c for c in self._chunks if c["document_id"] == document_id]
        total = len(rows)
        rows = rows[offset : offset + limit]
        items = [
            ChunkRecord(
                chunk_id=c["id"],
                document_id=c["document_id"],
                filename=c.get("filename", ""),
                content=c.get("content", ""),
                embed_text=c.get("embed_text"),
                chunk_index=c.get("chunk_index", 0),
                visibility=c.get("visibility"),
                doc_domain=c.get("doc_domain", "general"),
                owner_id=c.get("owner_id"),
                group_id=c.get("group_id"),
                content_hash=c.get("content_hash"),
                context_metadata=c.get("context_metadata", {}),
            )
            for c in rows
        ]
        return items, total

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> int:
        count = 0

        for c in self._chunks:
            if c["document_id"] == document_id:
                c["filename"] = new_filename
                count += 1

        return count

    async def list_for_document_cursor(
        self,
        document_id: int,
        limit: int = 50,
        cursor: tuple[int, int] | None = None,
        direction: str = "next",
        content_hashes: list[str] | None = None,
    ) -> CursorPage[ChunkRecord]:

        rows = [c for c in self._chunks if c["document_id"] == document_id]
        rows.sort(key=lambda c: (c.get("chunk_index", 0), c["id"]))

        if content_hashes:
            rows = [c for c in rows if c.get("content_hash") in content_hashes]

        if direction == "next":
            if cursor is not None:
                ci, cid = cursor
                rows = [c for c in rows if (c.get("chunk_index", 0), c["id"]) > (ci, cid)]

            page = rows[:limit]
            has_extra = len(rows) > limit
            next_cur = encode_cursor(page[-1]["chunk_index"], page[-1]["id"]) if has_extra and page else None
            prev_cur = (
                encode_cursor(page[0]["chunk_index"], page[0]["id"]) if page and cursor is not None else None
            )
        else:
            if cursor is None:
                msg = "cursor is required when direction=prev"
                raise ValidationError(msg)

            ci, cid = cursor
            rows = [c for c in rows if (c.get("chunk_index", 0), c["id"]) < (ci, cid)]
            rows.sort(key=lambda c: (c.get("chunk_index", 0), c["id"]), reverse=True)
            page = rows[:limit]
            has_extra = len(rows) > limit
            page.reverse()
            prev_cur = encode_cursor(page[0]["chunk_index"], page[0]["id"]) if has_extra and page else None
            next_cur = encode_cursor(page[0]["chunk_index"], page[0]["id"]) if page else None

        items = [
            ChunkRecord(
                chunk_id=c["id"],
                document_id=c["document_id"],
                filename=c.get("filename", ""),
                content=c.get("content", ""),
                embed_text=c.get("embed_text"),
                chunk_index=c.get("chunk_index", 0),
                visibility=c.get("visibility"),
                owner_id=c.get("owner_id"),
                group_id=c.get("group_id"),
                content_hash=c.get("content_hash"),
                context_metadata=c.get("context_metadata", {}),
            )
            for c in page
        ]
        return CursorPage(items=items, next_cursor=next_cur, prev_cursor=prev_cur)

    async def get_by_id(self, chunk_id: int) -> ChunkRecord | None:
        for row in self._chunks:
            if row["id"] == chunk_id:
                items, _ = await self.list_for_document(row["document_id"], limit=len(self._chunks))
                return next((item for item in items if item.chunk_id == chunk_id), None)

        return None

    async def get_max_chunk_index(self, document_id: int) -> int:
        return max(
            (row["chunk_index"] for row in self._chunks if row["document_id"] == document_id), default=-1
        )

    async def update_content(
        self,
        chunk_id: int,
        content: str,
        content_hash: str,
        edited_at: datetime,
        edited_by: int,
        *,
        embed_text: str | None = None,
    ) -> None:
        for row in self._chunks:
            if row["id"] == chunk_id:
                row.update(
                    content=content,
                    content_hash=content_hash,
                    edited_at=edited_at,
                    edited_by=edited_by,
                    embed_text=embed_text,
                )

    async def insert_one(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        filename: str,
        visibility: str,
        *,
        doc_domain: str,
        options: ManualChunkInsertOptions = DEFAULT_MANUAL_CHUNK_INSERT_OPTIONS,
    ) -> int:
        chunk_id = self._next_id
        self._next_id += 1
        self._chunks.append(
            {
                "id": chunk_id,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "content": content,
                "filename": filename,
                "visibility": visibility,
                "doc_domain": doc_domain,
                "owner_id": options.owner_id,
                "group_id": options.group_id,
                "content_hash": options.content_hash,
                "context_metadata": options.context_metadata or {},
                "embed_text": options.embed_text,
            }
        )
        return chunk_id

    async def delete_one(self, chunk_id: int) -> None:
        self._chunks = [row for row in self._chunks if row["id"] != chunk_id]

    async def delete_by_document_id(self, document_id: int) -> None:
        self._chunks = [row for row in self._chunks if row["document_id"] != document_id]

    async def find_duplicate_by_hash(
        self, document_id: int, content_hash: str, exclude_chunk_id: int | None = None
    ) -> ChunkRecord | None:
        items, _ = await self.list_for_document(document_id, limit=len(self._chunks))
        return next(
            (
                item
                for item in items
                if item.content_hash == content_hash and item.chunk_id != exclude_chunk_id
            ),
            None,
        )

    async def get_document_stats(self, document_id: int) -> ChunkStats:
        rows = [row for row in self._chunks if row["document_id"] == document_id]
        return ChunkStats(len(rows), sum(len(row["content"]) for row in rows))

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: set[str] | None = None,
        *,
        user: UserContext,
    ) -> list[ChunkRecord]:
        msg = "Neighbor retrieval requires a dedicated ACL-aware test repository"
        raise NotImplementedError(msg)

    async def get_table_batches(
        self,
        document_id: int,
        table_id: str,
        exclude_hashes: set[str] | None = None,
        *,
        user: UserContext,
        limit: int = TABLE_CONTEXT_MAX_CHUNKS,
        row_start: int | None = None,
        row_end: int | None = None,
    ) -> list[ChunkRecord]:
        msg = "Table retrieval requires a dedicated ACL-aware test repository"
        raise NotImplementedError(msg)

    async def get_parent_context(
        self,
        document_id: int,
        node_id: str,
        *,
        user: UserContext,
        act_version_id: int | None,
        max_chars: int,
    ) -> list[ChunkRecord]:
        msg = "Parent retrieval requires a dedicated ACL-aware test repository"
        raise NotImplementedError(msg)


class FakeDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[int, Document] = {}
        self._next_id = 1

    async def get_by_id(self, document_id: int, *, for_update: bool = False) -> Document | None:
        del for_update
        return self._documents.get(document_id)

    async def get_by_ids(self, document_ids: list[int]) -> list[Document]:
        return [
            self._documents[document_id]
            for document_id in set(document_ids)
            if document_id in self._documents
        ]

    async def list_all(self, limit: int = 200, offset: int = 0) -> list[Document]:
        return list(self._documents.values())[offset : offset + limit]

    async def list_after_id(
        self, after_id: int, limit: int, *, doc_domain: str | None = None
    ) -> list[Document]:
        matching = [
            document
            for document_id, document in sorted(self._documents.items())
            if document_id > after_id and (doc_domain is None or document.doc_domain == doc_domain)
        ]
        return matching[:limit]

    async def list_visible(self, filters: DocumentFilter) -> list[Document]:
        visible = [d for d in self._documents.values() if can_view_document(d, filters.user)]
        page = filters.pagination
        return visible[page.offset : page.offset + page.limit]

    async def save(self, document: Document) -> Document:
        document.id = self._next_id
        self._next_id += 1
        self._documents[document.id] = document
        return document

    async def find_ingestion_documents(self, keys: list[str]) -> list[Document]:
        return [d for d in self._documents.values() if d.ingestion_key in keys]

    async def list_ingestion_documents(self, prefix: str) -> list[Document]:
        return [
            d
            for d in self._documents.values()
            if d.ingestion_key is not None and d.ingestion_key.startswith(prefix)
        ]

    async def find_active_slots_by_filenames(
        self, filenames: list[str], *, owner_id: int | None = None, group_id: int | None = None
    ) -> list[Document]:
        return [
            d
            for d in self._documents.values()
            if getattr(d, "filename", None) in filenames
            and getattr(d, "owner_id", None) == owner_id
            and getattr(d, "group_id", None) == group_id
        ]

    async def find_active_slot(
        self, owner_id: int | None, filename: str, group_id: int | None, *, for_update: bool = False
    ) -> Document | None:
        del for_update

        for d in self._documents.values():
            if (
                getattr(d, "filename", None) == filename
                and getattr(d, "owner_id", None) == owner_id
                and getattr(d, "group_id", None) == group_id
            ):
                return d

        return None

    async def update_status(self, document_id: int, **changes: Unpack[DocumentStatusUpdate]) -> bool:
        document = self._documents.get(document_id)

        if document is None:
            return False

        expected_version = changes.get("expected_version")

        if expected_version is not None and document.version != expected_version:
            return False

        status = changes.get("status")
        document.apply_processing_update(
            DocumentProcessingUpdate(
                status=DocumentStatus(status) if status is not None else None,
                error=changes.get("error"),
                chunks=changes.get("chunks"),
                chars=changes.get("chars"),
                warning=changes.get("warning"),
                quality_score=changes.get("quality_score"),
            )
        )
        document.version += 1
        return True

    async def mark_done_if_indexing(self, document_id: int) -> bool:
        document = self._documents.get(document_id)

        if document is None or document.status != DocumentStatus.INDEXING:
            return False

        document.status = DocumentStatus.DONE
        return True

    async def mark_stuck_processing_failed(self) -> list[int]:
        return []

    async def reconcile_indexing_documents(self) -> list[int]:
        return []

    async def set_domain(self, document_id: int, doc_domain: str) -> None:
        document = self._documents.get(document_id)

        if document is not None:
            document.doc_domain = DomainKey(doc_domain)

    async def set_acl(
        self, document_id: int, visibility: DocumentVisibility, owner_id: int | None, group_id: int | None
    ) -> None:
        document = self._documents.get(document_id)

        if document is None:
            entity = "Document"
            raise EntityNotFoundError(entity, document_id)

        document.visibility = visibility
        document.owner_id = owner_id
        document.group_id = group_id
        document.version += 1

    async def set_source_path(self, document_id: int, source_path: str) -> None:
        document = self._documents.get(document_id)

        if document is not None:
            document.source_path = source_path

    async def delete(self, document_id: int) -> None:
        self._documents.pop(document_id, None)

    async def update_filename(self, document_id: int, new_filename: str, new_source_path: str) -> None:
        document = self._documents.get(document_id)

        if document is not None:
            document.filename = new_filename
            document.source_path = new_source_path

    async def list_distinct_filenames(self, search: str | None = None, limit: int = 100) -> list[str]:
        names = list({d.filename for d in self._documents.values() if d.filename})

        if search:
            names = [n for n in names if search.lower() in n.lower()]

        return sorted(names)[:limit]

    async def list_warned(self, quality_threshold: float = 0.3) -> list[Document]:
        return [
            d
            for d in self._documents.values()
            if d.warning_message or (d.quality_score is not None and d.quality_score < quality_threshold)
        ]

    async def set_has_manual_edits(self, document_id: int, *, value: bool) -> None:
        self._documents[document_id].has_manual_edits = value

    async def update_chunk_stats(self, document_id: int, chunks: int, chars: int) -> None:
        document = self._documents[document_id]
        document.chunks, document.chars = chunks, chars

    async def delete_internal_documents(self) -> int:
        ids = [key for key, d in self._documents.items() if d.visibility != DocumentVisibility.CLIENT_PRIVATE]

        for key in ids:
            del self._documents[key]

        return len(ids)


class FakeChatLogRepository:
    async def save(self, log: ChatLog) -> None:
        pass

    async def list_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> list[ChatLog]:
        del filters
        return []

    async def count_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> int:
        del filters
        return 0


class FakeVectorOutboxRepository:
    """In-memory outbox repository for testing SAGA/Outbox pattern."""

    def __init__(self) -> None:
        self._entries: dict[int, VectorOutboxEntry] = {}
        self._next_id = 1
        self._notifications: list[dict[str, Any]] = []

    async def enqueue(self, entry: VectorOutboxEntry) -> VectorOutboxEntry:
        entry.id = self._next_id
        self._next_id += 1
        self._entries[entry.id] = entry
        self._notifications.append({"id": entry.id, "op": entry.operation.value})
        return entry

    async def claim_batch(self, worker_id: str, limit: int = 20) -> list[VectorOutboxEntry]:
        del worker_id
        claimed: list[VectorOutboxEntry] = []

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
        del backoff_seconds

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
        del stuck_timeout_minutes
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

    async def count_by_documents(self, document_ids: list[int]) -> dict[int, dict[str, int]]:
        return {did: await self.count_by_document(did) for did in document_ids}

    async def get_failed_details(self, document_id: int) -> list[dict[str, object]]:
        return [
            {
                "operation": entry.operation.value,
                "attempts": entry.attempts,
                "max_attempts": entry.max_attempts,
                "last_error": entry.last_error,
            }
            for entry in self._entries.values()
            if entry.aggregate_id == document_id
            and entry.status in (OutboxStatus.FAILED, OutboxStatus.DEAD_LETTER)
        ]

    async def get_failed_details_by_documents(
        self, document_ids: list[int]
    ) -> dict[int, list[dict[str, object]]]:
        return {did: await self.get_failed_details(did) for did in document_ids}


class FakeUserRepository:
    def __init__(self) -> None:
        self._users: dict[int, User] = {}
        self._next_id = 1

    async def get_by_id(self, user_id: int) -> User | None:
        return self._users.get(user_id)

    async def list_all(self, limit: int | None = None, offset: int = 0) -> list[User]:
        del limit, offset
        return list(self._users.values())

    def add_user(self, user_id: int, email: str = "user@test.com", kind: str = "internal") -> User:

        u = User(id=user_id, email=email, kind=UserKind(kind))
        self._users[user_id] = u
        return u

    async def get_by_ids(self, user_ids: list[int]) -> list[User]:
        return [user for key, user in self._users.items() if key in user_ids]

    async def get_by_email(self, email: str) -> User | None:
        return next((u for u in self._users.values() if u.email == email), None)

    async def save(self, user: User) -> User:
        if user.id is None:
            user.id = self._next_id
            self._next_id += 1

        self._users[user.id] = user
        return user

    async def ensure_admin(self, email: str, hashed_password: str, role: str, kind: str) -> None:
        if await self.get_by_email(email) is None:
            await self.save(
                User(email=email, hashed_password=hashed_password, role=UserRole(role), kind=UserKind(kind))
            )

    async def exists_admin(self) -> bool:
        return any(u.role == UserRole.ADMIN for u in self._users.values())

    async def count_all(self) -> int:
        return len(self._users)

    async def set_active(self, user_id: int, *, is_active: bool) -> bool:
        user = self._users.get(user_id)

        if user is None:
            return False

        user.set_active(is_active=is_active)
        return True

    async def update_role(self, user_id: int, role: str) -> bool:
        user = self._users.get(user_id)

        if user is None:
            return False

        user.change_role(UserRole(role))
        return True


class FakeConfigParameterRepository:
    async def lock_configuration(self) -> None:
        pass

    def __init__(self) -> None:
        self._params: dict[tuple[str, str | None], ConfigParameter] = {}

    async def get_all(self) -> list[ConfigParameter]:
        return list(self._params.values())

    async def get_by_key(self, key: str) -> ConfigParameter | None:
        return self._params.get((key, None))

    async def get_by_key_and_domain(self, key: str, domain_key: str | None = None) -> ConfigParameter | None:
        return self._params.get((key, domain_key))

    async def update_value(self, key: str, value: str, domain_key: str | None = None) -> None:
        p = self._params.get((key, domain_key))

        if p:
            p.value = value

    async def update_category(self, key: str, category: str, domain_key: str | None = None) -> None:
        pass

    async def save(self, entity: ConfigParameter) -> None:
        self._params[(entity.key, getattr(entity, "domain_key", None))] = entity

    async def upsert(self, entity: ConfigParameter) -> None:
        self._params.setdefault((entity.key, getattr(entity, "domain_key", None)), entity)

    async def count(self) -> int:
        return len(self._params)

    async def delete_by_key_and_domain(self, key: str, domain_key: str | None = None) -> None:
        self._params.pop((key, domain_key), None)


class FakeBackgroundJobRepository:
    def __init__(self) -> None:
        self.jobs: dict[int, BackgroundJob] = {}

    async def create(self, job: BackgroundJob) -> BackgroundJob:
        job.id = len(self.jobs) + 1
        self.jobs[job.id] = job
        return job

    async def pending_dispatch(self, limit: int = 100) -> list[BackgroundJob]:
        return [
            job for job in self.jobs.values() if job.status == "pending" and job.dispatch_payload is not None
        ][:limit]

    async def mark_running(self, job_id: int) -> None:
        pass

    async def mark_done(self, job_id: int) -> None:
        pass

    async def mark_failed(self, job_id: int, error: str) -> None:
        pass

    async def touch_heartbeat(self, job_id: int) -> None:
        pass

    async def count_active(self) -> int:
        return 0

    async def delete_old(self, days: int) -> int:
        del days
        return 0

    async def list_recent(self, limit: int = 50, offset: int = 0) -> list[BackgroundJob]:
        del limit, offset
        return []

    async def get_by_id(self, job_id: int) -> BackgroundJob | None:
        del job_id
        return None

    async def count_by_status(self) -> dict[str, int]:
        return {}

    async def recover_orphaned(self, timeout_minutes: int = 15) -> list[int]:
        del timeout_minutes
        return []

    async def fail_stale_pending(self, timeout_minutes: int) -> list[BackgroundJob]:
        del timeout_minutes
        return []


class FakeApiKeyRepository:
    async def create(self, user_id: int, key_hash: str, key_prefix: str, name: str | None = None) -> ApiKey:
        return ApiKey(user_id=user_id, key_hash=key_hash, key_prefix=key_prefix, name=name)

    async def list_for_user(self, user_id: int) -> list[ApiKey]:
        del user_id
        return []

    async def revoke(self, api_key_id: int, user_id: int | None = None) -> bool:
        del api_key_id, user_id
        return False

    async def get_active_client_by_hash(self, key_hash: str) -> ApiKeyClientInfo | None:
        del key_hash
        return None

    async def touch_last_used(self, api_key_id: int) -> None:
        pass


class FakeBenchmarkQuestionRepository:
    async def list_active(self, *, dataset: str) -> list[BenchmarkQuestion]:
        del dataset
        return []

    async def list_items(self, **kwargs: object) -> list[BenchmarkQuestion]:
        del kwargs
        return []

    async def count(
        self,
        *,
        dataset: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        is_active: bool | None = None,
    ) -> int:
        del dataset, tag, search, is_active
        return 0

    async def create(self, question: BenchmarkQuestion) -> BenchmarkQuestion:
        question.id = 1
        return question

    async def update(self, question_id: int, **fields: object) -> BenchmarkQuestion | None:
        del question_id, fields
        return None

    async def delete(self, question_id: int) -> bool:
        del question_id
        return True

    async def bulk_create(self, questions: list[BenchmarkQuestion]) -> int:
        return len(questions)

    async def get_by_id(self, question_id: int) -> BenchmarkQuestion | None:
        del question_id
        return None

    async def get_datasets(self) -> list[str]:
        return []

    async def count_by_dataset(self) -> dict[str, int]:
        return {}


class FakeBenchmarkSweepRepository:
    async def set_job_id(self, sweep_id: int, job_id: int) -> None:
        self.job_ids[sweep_id] = job_id

    def __init__(self) -> None:
        self.progress: dict[int, tuple[int, int]] = {}
        self.job_ids: dict[int, int] = {}
        self.best_run_ids: dict[int, int] = {}
        self.checkpoints: dict[tuple[int, str], object] = {}

    async def load_checkpoint(self, sweep_id: int, key: str) -> object | None:

        return deepcopy(self.checkpoints.get((sweep_id, key)))

    async def save_checkpoint(self, sweep_id: int, key: str, value: object) -> None:

        self.checkpoints[sweep_id, key] = deepcopy(value)

    async def delete_checkpoints(self, sweep_id: int) -> None:
        self.checkpoints = {key: value for key, value in self.checkpoints.items() if key[0] != sweep_id}

    async def update_progress(self, sweep_id: int, evaluated: int, total: int) -> None:
        self.progress[sweep_id] = (evaluated, total)

    async def get_by_id(self, sweep_id: int, *, for_update: bool = False) -> BenchmarkSweep | None:
        del sweep_id, for_update
        return None

    async def create(self, sweep: BenchmarkSweep) -> BenchmarkSweep:
        sweep.id = 1
        return sweep

    async def update_status(self, sweep_id: int, status: str) -> None:
        pass

    async def list_items(self, *, limit: int = 50, offset: int = 0) -> list[BenchmarkSweep]:
        del limit, offset
        return []

    async def count(self) -> int:
        return 0

    async def has_active(self) -> bool:
        return False

    async def fail_dead_jobs(self) -> int:
        return 0

    async def requeue(self, sweep_id: int) -> bool:
        del sweep_id
        return False

    async def increment_evaluated(self, sweep_id: int) -> None:
        evaluated, total = self.progress.get(sweep_id, (0, 0))
        self.progress[sweep_id] = (evaluated + 1, total)

    async def set_best_run(self, sweep_id: int, run_id: int) -> None:
        self.best_run_ids[sweep_id] = run_id

    async def cleanup_checkpoints(self, cutoff: datetime) -> int:
        del cutoff
        return 0


class FakeBenchmarkRunRepository:
    def __init__(self) -> None:
        self.runs: dict[int, BenchmarkRun] = {}

    async def create(self, run: BenchmarkRun) -> BenchmarkRun:
        run.id = len(self.runs) + 1
        self.runs[run.id] = run
        return run

    async def save_for_sweep(self, run: BenchmarkRun) -> BenchmarkRun:
        existing = next(
            (
                r
                for r in self.runs.values()
                if r.sweep_id == run.sweep_id and r.config_json == run.config_json
            ),
            None,
        )

        if existing:
            run.id = existing.id

            if run.id is not None:
                self.runs[run.id] = run

            return run

        return await self.create(run)

    async def get_by_id(self, run_id: int) -> BenchmarkRun | None:
        del run_id
        return None

    async def get_by_ids(self, ids: list[int]) -> list[BenchmarkRun]:
        del ids
        return []

    async def list_items(self, **kwargs: object) -> list[BenchmarkRun]:
        del kwargs
        return []

    async def count(self, *, sweep_id: int | None = None, dataset: str | None = None) -> int:
        del sweep_id, dataset
        return 0

    async def get_by_job_id(self, job_id: int) -> BenchmarkRun | None:
        return next((r for r in self.runs.values() if r.job_id == job_id), None)

    async def get_by_legacy_hash(self, digest: str) -> BenchmarkRun | None:
        return next((r for r in self.runs.values() if r.legacy_hash == digest), None)

    async def save_once(self, run: BenchmarkRun) -> BenchmarkRun:
        existing = await self.get_by_job_id(run.job_id) if run.job_id is not None else None

        if existing is None and run.legacy_hash is not None:
            existing = await self.get_by_legacy_hash(run.legacy_hash)

        return existing if existing is not None else await self.create(run)

    async def get_baseline(self, current: BenchmarkRun) -> BenchmarkRun | None:
        del current
        return None

    async def export_header(self, run_id: int) -> dict[str, Any] | None:
        msg = "Benchmark export is not implemented by the in-memory test repository"
        raise NotImplementedError(msg)

    async def result_page(self, run_id: int, *, offset: int, limit: int) -> list[dict[str, Any]]:
        run = self.runs.get(run_id)
        return (run.per_question_results or [])[offset : offset + limit] if run else []

    async def get_sweep_run_ids(self, sweep_id: int, limit: int) -> list[int]:
        return [key for key, r in self.runs.items() if r.sweep_id == sweep_id][:limit]

    async def set_artifact(self, run_id: int, report_format: str, metadata: dict[str, Any]) -> None:
        self.runs[run_id].artifacts[report_format] = metadata

    async def get_latest(self, dataset: str | None = None) -> BenchmarkRun | None:
        return next(
            (r for r in reversed(self.runs.values()) if dataset is None or r.dataset == dataset), None
        )


class FakeRegulatoryActRepository:
    def __init__(self) -> None:
        self._acts: dict[int, RegulatoryAct] = {}
        self._next_id = 1

    async def find_by_type_and_number(
        self,
        act_type: str,
        act_number: str | None,
        act_date: date | None = None,
        visibility_scope: str = "internal_public",
    ) -> RegulatoryAct | None:
        for act in self._acts.values():
            if (
                act.act_type == act_type
                and act.act_number == act_number
                and act.act_date == act_date
                and act.visibility_scope == visibility_scope
            ):
                return act

        return None

    async def update_act_date(self, act_id: int, act_date: date) -> None:
        self._acts[act_id].act_date = act_date

    async def get_by_id(self, act_id: int) -> RegulatoryAct | None:
        return self._acts.get(act_id)

    async def save(self, act: RegulatoryAct) -> RegulatoryAct:
        act.id = self._next_id
        self._next_id += 1
        self._acts[act.id] = act
        return act

    async def list_all(self) -> list[RegulatoryAct]:
        return list(self._acts.values())


class FakeActVersionRepository:
    def __init__(self) -> None:
        self._versions: dict[int, ActVersion] = {}
        self._next_id = 1

    async def create(self, version: ActVersion) -> ActVersion:
        version = replace(version, id=self._next_id)
        self._next_id += 1
        self._versions[self._next_id - 1] = version
        return version

    async def get_by_id(self, version_id: int) -> ActVersion | None:
        return self._versions.get(version_id)

    async def get_by_document_id(self, document_id: int, *, for_update: bool = False) -> ActVersion | None:
        del for_update

        for v in self._versions.values():
            if v.document_id == document_id:
                return v

        return None

    async def unset_current(self, act_id: int) -> None:
        for v in self._versions.values():
            if v.act_id == act_id and v.is_current and v.id is not None:
                self._versions[v.id] = replace(v, is_current=False)

    async def list_by_act(self, act_id: int, *, for_update: bool = False) -> list[ActVersion]:
        del for_update
        return [v for v in self._versions.values() if v.act_id == act_id]

    async def list_pending_review(self) -> list[ActVersion]:
        return [v for v in self._versions.values() if v.date_source == "extracted" or v.act_id is None]

    async def list_page(self, limit: int, offset: int) -> tuple[list[ActVersion], int]:
        versions = sorted(self._versions.values(), key=lambda item: item.id or 0)
        return versions[offset : offset + limit], len(versions)

    async def update(self, version: ActVersion) -> None:
        if version.id is not None and version.id in self._versions:
            self._versions[version.id] = version


class FakeAssignmentRepository:
    def __init__(self) -> None:
        self._user_assignments: dict[
            tuple[int, int], dict[str, int]
        ] = {}  # (manager_id, target_user_id) -> {assigned_by}
        self._group_assignments: dict[
            tuple[int, int], dict[str, int]
        ] = {}  # (manager_id, group_id) -> {assigned_by}
        self._user_kinds: dict[int, str] = {}  # user_id -> kind (for filtering)

    def set_user_kind(self, user_id: int, kind: str) -> None:
        self._user_kinds[user_id] = kind

    async def get_managed_user_ids(self, manager_id: int) -> list[int]:
        return [t for (c, t) in self._user_assignments if c == manager_id]

    async def get_managed_client_ids(self, manager_id: int) -> list[int]:
        return [
            t for (c, t) in self._user_assignments if c == manager_id and self._user_kinds.get(t) == "client"
        ]

    async def get_managed_internal_ids(self, manager_id: int) -> list[int]:
        return [
            t
            for (c, t) in self._user_assignments
            if c == manager_id and self._user_kinds.get(t) == "internal"
        ]

    async def get_managed_group_ids(self, manager_id: int) -> list[int]:
        return [g for (c, g) in self._group_assignments if c == manager_id]

    async def assign_user(self, manager_id: int, target_user_id: int, assigned_by: int) -> None:
        self._user_assignments[(manager_id, target_user_id)] = {"assigned_by": assigned_by}

    async def unassign_user(self, manager_id: int, target_user_id: int) -> None:
        self._user_assignments.pop((manager_id, target_user_id), None)

    async def assign_group(self, manager_id: int, group_id: int, assigned_by: int) -> None:
        self._group_assignments[(manager_id, group_id)] = {"assigned_by": assigned_by}

    async def unassign_group(self, manager_id: int, group_id: int) -> None:
        self._group_assignments.pop((manager_id, group_id), None)

    async def clear_all_for_manager(self, manager_id: int) -> None:
        self._user_assignments = {k: v for k, v in self._user_assignments.items() if k[0] != manager_id}
        self._group_assignments = {k: v for k, v in self._group_assignments.items() if k[0] != manager_id}

    async def list_managers_for_group(self, group_id: int) -> list[int]:
        return [c for (c, g) in self._group_assignments if g == group_id]


class FakeIngestionRegistryRepository:
    """In-memory ingestion registry for unit tests."""

    def __init__(self) -> None:
        self._entries: dict[str, IngestionRegistryEntry] = {}

    async def get(self, filename: str) -> IngestionRegistryEntry | None:
        return self._entries.get(filename)

    async def upsert(self, entry: IngestionRegistryEntry) -> None:
        self._entries[entry.ingestion_key or entry.filename] = entry

    async def delete(self, filename: str) -> None:
        self._entries.pop(filename, None)

    async def list_all(self) -> dict[str, IngestionRegistryEntry]:
        return dict(self._entries)

    async def is_already_indexed(self, filename: str, file_hash: str) -> bool:
        entry = self._entries.get(filename)
        return entry is not None and entry.file_hash == file_hash


class FakeUnitOfWork(UnitOfWork):
    """In-memory UnitOfWork for unit tests."""

    def __init__(self) -> None:
        self.fake_users = self.users = FakeUserRepository()
        self.fake_conversations = self.conversations = FakeConversationRepository()
        self.fake_messages = self.messages = FakeMessageRepository()
        self.fake_documents = self.documents = FakeDocumentRepository()
        self.fake_chunks = self.chunks = FakeChunkRepository()
        self.fake_groups = self.groups = FakeGroupRepository()
        self.fake_api_keys = self.api_keys = FakeApiKeyRepository()
        self.fake_config_parameters = self.config_parameters = FakeConfigParameterRepository()
        self.fake_background_jobs = self.background_jobs = FakeBackgroundJobRepository()
        self.fake_chat_logs = self.chat_logs = FakeChatLogRepository()
        self.fake_benchmark_questions = self.benchmark_questions = FakeBenchmarkQuestionRepository()
        self.fake_benchmark_sweeps = self.benchmark_sweeps = FakeBenchmarkSweepRepository()
        self.fake_benchmark_runs = self.benchmark_runs = FakeBenchmarkRunRepository()
        self.fake_vector_outbox = self.vector_outbox = FakeVectorOutboxRepository()
        self.fake_regulatory_acts = self.regulatory_acts = FakeRegulatoryActRepository()
        self.fake_act_versions = self.act_versions = FakeActVersionRepository()
        self.fake_assignments = self.assignments = FakeAssignmentRepository()
        self.fake_ingestion_registry = self.ingestion_registry = FakeIngestionRegistryRepository()
        self._event_handlers: list[Callable[[object], Coroutine[Any, Any, None]]] = []
        self._committed = False
        self._rolled_back = False

    def on_event(self, handler: Callable[[object], Coroutine[Any, Any, None]]) -> None:
        self._event_handlers.append(handler)

    async def publish_event(self, event: object) -> None:
        for handler in self._event_handlers:
            await handler(event)

    async def __aenter__(self) -> Self:
        self._snapshots = {
            name: deepcopy({k: v for k, v in repo.__dict__.items() if not callable(v)})
            for name, repo in self.__dict__.items()
            if not name.startswith("_") and hasattr(repo, "__dict__")
        }
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self._committed = True
        else:
            self._rolled_back = True

            for name, snapshot in self._snapshots.items():
                repo = getattr(self, name)

                for key in list(repo.__dict__):
                    if not callable(repo.__dict__[key]):
                        del repo.__dict__[key]

                repo.__dict__.update(snapshot)


class FakeUnitOfWorkFactory(UnitOfWorkFactory):
    """In-memory UnitOfWorkFactory for unit tests."""

    def __init__(self, uow: FakeUnitOfWork | None = None) -> None:
        self._uow = uow or FakeUnitOfWork()
        self._document_index_locks: dict[int, asyncio.Lock] = {}

    @asynccontextmanager
    async def document_index_lock(self, document_id: int) -> AsyncIterator[None]:
        async with self._document_index_locks.setdefault(document_id, asyncio.Lock()):
            yield

    @asynccontextmanager
    async def ingestion_lock(self) -> AsyncIterator[None]:
        yield

    @asynccontextmanager
    async def create(self, master: bool = False) -> AsyncGenerator[FakeUnitOfWork, None]:
        # A new transaction wrapper shares the committed in-memory repositories.
        # This double models serial transactions; concurrency tests use PostgreSQL.
        del master
        uow = FakeUnitOfWork()

        for name, repository in self._uow.__dict__.items():
            if not name.startswith("_"):
                setattr(uow, name, repository)

        uow._event_handlers = list(self._uow._event_handlers)
        try:
            async with uow:
                yield uow
        finally:
            self._uow._committed = uow._committed
            self._uow._rolled_back = uow._rolled_back


# ---------------------------------------------------------------------------
# Fake ChatRAGPort
# ---------------------------------------------------------------------------


class FakeChatRAGPort:
    """Returns a canned answer for testing ChatService."""

    def __init__(
        self,
        answer: str = "Test answer",
        sources: list[dict[str, Any]] | None = None,
        breadth: str = "narrow",
        domain: str = "general",
    ) -> None:
        self._answer = answer
        self._sources = sources or []
        self._breadth = Breadth(breadth)
        self._domain = DomainKey(domain)

    async def stream(
        self, question: str, history: list[Message], ctx: RagContext
    ) -> AsyncIterator[StreamEvent]:
        del question, history, ctx
        yield TextChunk(text=self._answer)
        yield SourcesEvent(sources=self._sources, confidence=0.9)

    async def invoke(self, question: str, history: list[Message], ctx: RagContext) -> RagResult:

        del question, history, ctx
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


class FakeTimeoutSemaphore:
    """Fake semaphore that supports timeout semantics for integration-style tests."""

    def __init__(self, value: int = 8, timeout: float = 30.0) -> None:
        self._sem = asyncio.Semaphore(value)
        self._timeout = timeout

    async def acquire(self) -> bool:
        await asyncio.wait_for(self._sem.acquire(), timeout=self._timeout)
        return True

    def release(self) -> None:
        self._sem.release()

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


class FakeRateLimiterPort:
    """In-memory RateLimiterPort: records checks, denies after a fixed quota."""

    def __init__(self, allow: int | None = None) -> None:
        self._remaining = allow
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def check(self, policy: RateLimitPolicyName, principal: str) -> RateLimitDecision:

        self.calls.append((policy.value, principal))

        if self._remaining is not None:
            if self._remaining <= 0:
                return RateLimitDecision(allowed=False, retry_after_sec=60, limit=1)

            self._remaining -= 1

        return RateLimitDecision(allowed=True, limit=10)

    async def aclose(self) -> None:
        self.closed = True


class FakeMLClientRegistry:
    """Lightweight substitute for MLClientRegistry in unit tests."""

    def __init__(self, llm_response: str = "fake") -> None:
        self._llm_response = llm_response
        self._invalidated_llm = False
        self._invalidated_bm25 = False
        self._auxiliary_semaphore = FakeTimeoutSemaphore(4)
        self._generation_semaphore = FakeTimeoutSemaphore(12)
        self._qdrant_search_semaphore = FakeTimeoutSemaphore(8)
        self._bm25_search_semaphore = FakeTimeoutSemaphore(8)
        self._reranker_semaphore = FakeTimeoutSemaphore(8)
        self._embedding_semaphore = FakeTimeoutSemaphore(16)
        self._ingestion_semaphore = FakeTimeoutSemaphore(8)
        self._qdrant_write_semaphore = FakeTimeoutSemaphore(8)

    def llm(self) -> Self:
        return self

    def llm_for_breadth(self, breadth: str) -> Self:
        del breadth
        return self

    def fast_llm(self) -> Self:
        return self

    def embeddings(self) -> Self:
        return self

    def reranker(self) -> Self:
        return self

    def qdrant_client(self) -> Self:
        return self

    def bm25_index(self) -> None:
        return None

    def bm25_update(self) -> nullcontext[None]:
        return nullcontext()

    async def ensure_bm25_loaded(self) -> None:
        return None

    def vector_store(self) -> Self:
        return self

    @property
    def instructor_client(self) -> MagicMock:

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(is_relevant=True, reason="fake", is_sufficient=True, reasoning="fake")
        )
        return mock_client

    @property
    def auxiliary_semaphore(self) -> FakeTimeoutSemaphore:
        return self._auxiliary_semaphore

    @property
    def generation_semaphore(self) -> FakeTimeoutSemaphore:
        return self._generation_semaphore

    @property
    def qdrant_search_semaphore(self) -> FakeTimeoutSemaphore:
        return self._qdrant_search_semaphore

    @property
    def bm25_search_semaphore(self) -> FakeTimeoutSemaphore:
        return self._bm25_search_semaphore

    @property
    def reranker_semaphore(self) -> FakeTimeoutSemaphore:
        return self._reranker_semaphore

    @property
    def embedding_semaphore(self) -> FakeTimeoutSemaphore:
        return self._embedding_semaphore

    @property
    def ingestion_semaphore(self) -> FakeTimeoutSemaphore:
        return self._ingestion_semaphore

    @property
    def qdrant_write_semaphore(self) -> FakeTimeoutSemaphore:
        return self._qdrant_write_semaphore

    async def astream(self, messages: object) -> AsyncIterator[SimpleNamespace]:
        del messages
        yield SimpleNamespace(content=self._llm_response)

    def embed_query(self, text: str) -> list[float]:
        del text
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
        self._handlers: dict[type, list[Callable[[Any], None]]] = {}
        self._published: list[object] = []

    def subscribe(self, event_type: type, handler: Callable[[Any], None]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: object) -> None:
        self._published.append(event)

        for handler in self._handlers.get(type(event), []):
            handler(event)


def require_present[T](value: T | None) -> T:
    """Verify a fixture lookup/identity exists before checking its other properties."""
    assert value is not None
    return value
