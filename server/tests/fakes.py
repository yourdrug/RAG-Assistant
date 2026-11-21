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
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, Unpack, cast
from unittest.mock import AsyncMock, MagicMock

from application.ports.rate_limit import RateLimitDecision
from application.rag.stream_events import SourcesEvent, TextChunk
from application.spaces.ports import DomainPackNotLoadedError
from application.uow.factory import UnitOfWorkFactory
from application.uow.unit_of_work import UnitOfWork
from domain.chat.conversation import Conversation
from domain.chat.repositories import DEFAULT_CHAT_LOG_FILTER
from domain.chat.review import ChatLogReview
from domain.identity.api_key import ApiKey
from domain.identity.roles import UserKind, UserRole
from domain.identity.user import User
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.knowledge.access_control import can_view_document
from domain.knowledge.chunk import (
    TABLE_CONTEXT_MAX_CHUNKS,
    ChunkRecord,
    ChunkStats,
    extract_chunk_context,
)
from domain.knowledge.document import DocumentProcessingUpdate
from domain.knowledge.enums import DocumentStatus, DocumentVisibility, DomainKey, SearchMode, SourceType
from domain.knowledge.index_sync import OutboxStatus
from domain.knowledge.repositories.chunk import (
    DEFAULT_CHUNK_INSERT_OPTIONS,
    DEFAULT_MANUAL_CHUNK_INSERT_OPTIONS,
)
from domain.knowledge.repositories.document import DocumentTotals
from domain.knowledge.revision import RevisionSwitch, StagedRevision
from domain.knowledge.space import KnowledgeSpace
from domain.rag.breadth import Breadth
from domain.rag.document_outline import DocumentOutline
from domain.rag.result import RagResult
from domain.runtime.jobs import IndexingQueueSnapshot, JobConcurrencyLimitReachedError
from domain.shared.exceptions import EntityNotFoundError, UniqueConstraintError, ValidationError
from domain.shared.pagination import CursorPage, encode_cursor


# The space a new installation starts with (migration 0013).
DEFAULT_SPACE_ID = 1

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Sequence
    from types import TracebackType

    from application.indexing.pack_hooks import DocumentPackHooks
    from application.ports.rate_limit import RateLimitPolicyName
    from application.rag.stream_events import StreamEvent
    from application.runtime.config import ConfigParameter
    from domain.benchmark.agreement import HumanLabel, LabelMetric
    from domain.benchmark.policy import DatasetPolicy, JudgeCalibration
    from domain.benchmark.question import BenchmarkQuestion
    from domain.benchmark.run import BenchmarkRun
    from domain.benchmark.sweep import BenchmarkSweep
    from domain.chat.chat_log import ChatLog
    from domain.chat.conversation import ConversationListItem
    from domain.chat.feedback import MessageFeedback
    from domain.chat.message import Message
    from domain.chat.repositories import ChatLogFilter
    from domain.identity.api_key import ApiKeyClientInfo
    from domain.identity.group import GroupInfo, GroupMemberInfo
    from domain.identity.user_context import UserContext
    from domain.ingestion.repositories import IngestionRegistryEntry
    from domain.knowledge.chunk import (
        ChunkInsert,
    )
    from domain.knowledge.document import Document, SyncBinding
    from domain.knowledge.document_nodes import DocumentNode
    from domain.knowledge.index_sync import VectorOutboxEntry
    from domain.knowledge.pack_metadata import MetadataCondition
    from domain.knowledge.repositories.chunk import (
        ChunkInsertOptions,
        ManualChunkInsertOptions,
    )
    from domain.knowledge.repositories.document import DocumentFilter, DocumentStatusUpdate
    from domain.knowledge.revision import SourceVersion
    from domain.rag.context import RagContext
    from domain.runtime.jobs import BackgroundJob, ConcurrencyLimit
    from domain.sync.channel import SyncChannel
    from domain.sync.run import SyncRun


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

    async def assign_space(self, conversation_id: int, space_id: int) -> int:
        data = self._convs[conversation_id]

        if data.get("space_id") is None:
            data["space_id"] = space_id

        return cast("int", data["space_id"])

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
        message.id = len(self._messages) + 1
        self._messages.append(message)

    async def get_by_id(self, message_id: int) -> Message | None:
        return next((m for m in self._messages if m.id == message_id), None)

    async def get_history(self, conversation_id: int, window: int = 8) -> list[Message]:
        return [m for m in self._messages if m.conversation_id == conversation_id][-window:]

    async def list_page(
        self,
        conversation_id: int,
        limit: int,
        before_id: int | None = None,
        created_before: datetime | None = None,
    ) -> list[Message]:
        messages = [
            m
            for m in self._messages
            if m.conversation_id == conversation_id
            and (before_id is None or (m.id is not None and m.id < before_id))
            and (created_before is None or m.creation_date < created_before)
        ]
        return sorted(messages, key=lambda m: m.id or 0)[-limit:]


class FakeMessageFeedbackRepository:
    def __init__(self) -> None:
        self.items: dict[tuple[int, int], MessageFeedback] = {}

    async def upsert(self, feedback: MessageFeedback) -> MessageFeedback:
        previous = self.items.get((feedback.message_id, feedback.user_id))
        stored = replace(feedback, creation_date=previous.creation_date) if previous else feedback
        self.items[(feedback.message_id, feedback.user_id)] = stored
        return stored

    async def delete(self, message_id: int, user_id: int) -> bool:
        return self.items.pop((message_id, user_id), None) is not None

    async def list_for_messages(
        self, message_ids: list[int], user_id: int | None = None
    ) -> list[MessageFeedback]:
        return [
            item
            for (message_id, owner), item in self.items.items()
            if message_id in message_ids and (user_id is None or owner == user_id)
        ]


class FakeGroupRepository:
    # Groups every internal user belongs to; see ``admit_all_users``.
    member_of: tuple[int, ...] = ()

    async def get_user_group_ids(self, user_id: int) -> list[int]:
        del user_id
        return list(self.member_of)

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


def fake_chunk_record(c: dict[str, Any]) -> ChunkRecord:
    return ChunkRecord(
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
        section=c.get("section"),
        pack_metadata=c.get("pack_metadata") or {},
        context_metadata=c.get("context_metadata", {}),
    )


async def publish_revision(uow: FakeUnitOfWork, document_id: int, token: str) -> None:
    """Publish a staged revision and record its source version, as the dispatcher does."""
    switch = await uow.chunks.publish_staged_revision(document_id, token)

    if switch is not None and switch.source_version is not None:
        await uow.documents.set_storage_version(
            document_id, switch.source_version.version, switch.source_version.revision
        )


async def publish_staged_revisions(factory: FakeUnitOfWorkFactory) -> None:
    """Publish every staged revision, as the dispatcher does once its points are indexed."""
    chunks = factory._uow.fake_chunks

    for document_id in list(chunks._staged):
        staged = await chunks.get_staged_revision(document_id)

        if staged is not None:
            await publish_revision(factory._uow, document_id, staged.token)


class FakeChunkRepository:
    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self._next_id = 1
        self.document_nodes: dict[int, list[DocumentNode]] = {}
        # document_id -> {"token", "rows", "nodes"} of the staged revision.
        self._staged: dict[int, dict[str, Any]] = {}

    def chunk_rows(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[ChunkInsert],
        options: ChunkInsertOptions,
    ) -> list[dict[str, Any]]:
        rows = []

        for i, chunk in enumerate(chunks):
            rows.append(
                {
                    "id": self._next_id,
                    "document_id": document_id,
                    "content": chunk.content,
                    "embed_text": chunk.embed_text,
                    "domain_metadata": options.domain_metadata,
                    "pack_metadata": dict(options.pack_metadata or {}),
                    "filename": filename,
                    "visibility": visibility,
                    "doc_domain": options.doc_domain,
                    "owner_id": options.owner_id,
                    "group_id": options.group_id,
                    "chunk_index": i,
                    "content_hash": chunk.content_hash,
                    "context_metadata": extract_chunk_context(chunk.context_metadata),
                    "space_id": options.space_id,
                    "section": chunk.section,
                }
            )
            self._next_id += 1

        return rows

    async def stage_revision(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[ChunkInsert],
        *,
        nodes: list[DocumentNode],
        options: ChunkInsertOptions = DEFAULT_CHUNK_INSERT_OPTIONS,
        source_version: SourceVersion | None = None,
    ) -> StagedRevision:
        if document_id in self._staged:
            message = "discard the staged revision first"
            raise AssertionError(message)

        rows = self.chunk_rows(document_id, filename, visibility, chunks, options)
        token = f"rev-{rows[0]['id'] if rows else self._next_id}"
        self._staged[document_id] = {
            "token": token,
            "rows": rows,
            "nodes": list(nodes),
            "source_version": source_version,
        }
        return StagedRevision(document_id, token, tuple(row["id"] for row in rows), source_version)

    async def get_staged_revision(self, document_id: int) -> StagedRevision | None:
        staged = self._staged.get(document_id)

        if staged is None:
            return None

        return StagedRevision(
            document_id,
            staged["token"],
            tuple(row["id"] for row in staged["rows"]),
            staged.get("source_version"),
        )

    async def staged_source_versions(self, document_ids: list[int]) -> dict[int, SourceVersion]:
        return {
            document_id: staged["source_version"]
            for document_id, staged in self._staged.items()
            if document_id in document_ids and staged.get("source_version") is not None
        }

    async def get_staged_by_ids(
        self, document_id: int, token: str, chunk_ids: list[int]
    ) -> list[ChunkRecord]:
        staged = self._staged.get(document_id)

        if staged is None or staged["token"] != token:
            return []

        ids = set(chunk_ids)
        return [fake_chunk_record(row) for row in staged["rows"] if row["id"] in ids]

    async def discard_staged_revision(self, document_id: int) -> list[int]:
        staged = self._staged.pop(document_id, None)
        return [row["id"] for row in staged["rows"]] if staged else []

    async def publish_staged_revision(self, document_id: int, token: str) -> RevisionSwitch | None:
        staged = self._staged.get(document_id)

        if staged is None or staged["token"] != token:
            return None

        del self._staged[document_id]
        replaced = tuple(c["id"] for c in self._chunks if c["document_id"] == document_id)
        self._chunks = [c for c in self._chunks if c["document_id"] != document_id] + staged["rows"]
        self.document_nodes[document_id] = staged["nodes"]
        return RevisionSwitch(
            tuple(row["id"] for row in staged["rows"]), replaced, staged.get("source_version")
        )

    async def replace_document_nodes(self, document_id: int, nodes: list[DocumentNode]) -> None:
        self.document_nodes[document_id] = list(nodes)

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

    async def list_space_filenames(self, space_id: int) -> dict[int, str]:
        return {c["document_id"]: c["filename"] for c in self._chunks if c.get("space_id") == space_id}

    async def list_document_chunks(self, document_ids: list[int]) -> list[ChunkRecord]:
        rows = []

        for document_id in sorted(set(document_ids)):
            chunks, _ = await self.list_for_document(document_id, limit=len(self._chunks))
            rows.extend(sorted(chunks, key=lambda chunk: chunk.chunk_index))

        return rows

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
                    "pack_metadata": dict(options.pack_metadata or {}),
                    "filename": filename,
                    "visibility": visibility,
                    "doc_domain": doc_domain,
                    "owner_id": owner_id,
                    "group_id": group_id,
                    "chunk_index": i,
                    "content_hash": chunk.content_hash,
                    "context_metadata": extract_chunk_context(chunk.context_metadata),
                    "space_id": options.space_id,
                }
            )
            ids.append(chunk_id)

        return ids

    async def set_pack_metadata(self, document_id: int, pack_metadata: dict[str, Any]) -> int:
        updated = 0

        for c in self._staged.get(document_id, {}).get("rows", []):
            c["pack_metadata"] = dict(pack_metadata)

        for c in self._chunks:
            if c["document_id"] == document_id:
                c["pack_metadata"] = dict(pack_metadata)
                updated += 1

        return updated

    async def get_pack_metadata(self, document_id: int) -> dict[str, Any] | None:
        rows = sorted(
            (c for c in self._chunks if c["document_id"] == document_id),
            key=lambda c: c.get("chunk_index", 0),
        )
        return dict(rows[0].get("pack_metadata") or {}) if rows else None

    async def search_substring(  # noqa: PLR0913 - mirrors the repository's search options
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        selection: MetadataCondition | None = None,
        space_id: int | None = None,
        document_ids: Sequence[int] | None = None,
    ) -> list[ChunkRecord]:
        del query, user, limit, mode, document_id, selection, space_id, document_ids
        return []

    async def search_substring_page(  # noqa: PLR0913 - mirrors the repository's search options
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        offset: int = 0,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        selection: MetadataCondition | None = None,
        space_id: int | None = None,
    ) -> tuple[list[ChunkRecord], int]:
        del query, user, limit, offset, mode, document_id, selection, space_id
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
                pack_metadata=c.get("pack_metadata") or {},
                context_metadata=c.get("context_metadata", {}),
            )
            for c in rows
        ]
        return items, total

    async def set_acl(
        self, document_id: int, visibility: str, owner_id: int | None, group_id: int | None
    ) -> None:
        rows = [*self._chunks, *self._staged.get(document_id, {}).get("rows", [])]

        for c in rows:
            if c["document_id"] == document_id:
                c.update(visibility=visibility, owner_id=owner_id, group_id=group_id)

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> int:
        count = 0

        for c in self._staged.get(document_id, {}).get("rows", []):
            c["filename"] = new_filename

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
                "pack_metadata": dict(options.pack_metadata or {}),
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
        max_chars: int,
    ) -> list[ChunkRecord]:
        msg = "Parent retrieval requires a dedicated ACL-aware test repository"
        raise NotImplementedError(msg)

    async def get_document_outline(self, document_id: int, *, user: UserContext) -> DocumentOutline:
        del document_id, user
        return DocumentOutline(articles=(), tables=())


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
        visible = [
            d
            for d in self._documents.values()
            if can_view_document(d, filters.user)
            and filters.space_id in (None, d.space_id)
            and filters.status in (None, d.status)
        ]
        page = filters.pagination
        return visible[page.offset : page.offset + page.limit]

    async def summarize_visible(self, filters: DocumentFilter) -> DocumentTotals:
        by_status: dict[str, int] = {}
        chunks = 0

        for doc in self._documents.values():
            if (
                can_view_document(doc, filters.user)
                and filters.space_id in (None, doc.space_id)
                and filters.status in (None, doc.status)
            ):
                by_status[doc.status.value] = by_status.get(doc.status.value, 0) + 1
                chunks += doc.chunks or 0

        return DocumentTotals(by_status=by_status, chunks=chunks)

    async def save(self, document: Document) -> Document:
        document.id = self._next_id
        self._next_id += 1
        # Like the NOT NULL column: tests that do not choose a space use the starting one.
        document.space_id = document.space_id if document.space_id is not None else DEFAULT_SPACE_ID
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
        self,
        filenames: list[str],
        *,
        owner_id: int | None = None,
        group_id: int | None = None,
        space_id: int | None = None,
    ) -> list[Document]:
        return [
            d
            for d in self._documents.values()
            if getattr(d, "filename", None) in filenames
            and getattr(d, "sync", None) is None
            and getattr(d, "owner_id", None) == owner_id
            and getattr(d, "group_id", None) == group_id
            and space_id in (None, d.space_id)
        ]

    async def find_active_slot(
        self,
        owner_id: int | None,
        filename: str,
        group_id: int | None,
        *,
        space_id: int | None = None,
        for_update: bool = False,
    ) -> Document | None:
        del for_update

        for d in self._documents.values():
            if (
                getattr(d, "filename", None) == filename
                and getattr(d, "sync", None) is None
                and getattr(d, "owner_id", None) == owner_id
                and getattr(d, "group_id", None) == group_id
                and space_id in (None, d.space_id)
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

    async def set_sync_binding(self, document_id: int, binding: SyncBinding | None) -> None:
        if document_id in self._documents:
            self._documents[document_id].sync = binding

    async def set_storage_version(self, document_id: int, version: str | None, revision: str | None) -> None:
        document = self._documents.get(document_id)

        if document is not None and document.sync is not None:
            document.sync = replace(document.sync, storage_version=version, storage_revision=revision)

    async def list_channel_documents(self, channel_id: int) -> list[Document]:
        return [d for d in self._documents.values() if d.sync is not None and d.sync.channel_id == channel_id]

    async def unlink_document(self, document_id: int, source_path: str | None) -> None:
        document = self._documents[document_id]
        document.sync = None
        document.ingestion_key = None
        document.source_type = SourceType.FILE.value
        document.source_path = source_path or ""

    async def delete_internal_documents(self) -> int:
        ids = [
            key
            for key, d in self._documents.items()
            if d.visibility != DocumentVisibility.CLIENT_PRIVATE and d.sync is None
        ]

        for key in ids:
            del self._documents[key]

        return len(ids)


class FakeChatLogRepository:
    def __init__(self) -> None:
        self.logs: list[ChatLog] = []
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.traces: dict[int, dict[str, Any]] = {}
        self.reviews: dict[int, ChatLogReview] = {}

    async def save(self, log: ChatLog) -> None:
        log.id = len(self.logs) + 1
        self.logs.append(log)

    async def save_trace(self, log_id: int, data: dict[str, Any]) -> None:
        self.traces[log_id] = data

    async def get_trace(self, log_id: int) -> dict[str, Any] | None:
        return self.traces.get(log_id)

    async def delete_expired(self, trace_days: int, log_days: int, limit: int = 1000) -> int:
        del trace_days, log_days, limit
        return 0

    async def save_snapshot(self, config_hash: str, data: dict[str, Any]) -> None:
        self.snapshots.setdefault(config_hash, data)

    async def get_snapshot(self, config_hash: str) -> dict[str, Any] | None:
        return self.snapshots.get(config_hash)

    async def get_by_id(self, log_id: int, user_ids: tuple[int, ...] | None = None) -> ChatLog | None:
        return next(
            (log for log in self.logs if log.id == log_id and (user_ids is None or log.user_id in user_ids)),
            None,
        )

    async def list_reviews(self, log_ids: list[int]) -> list[ChatLogReview]:
        return [self.reviews[log_id] for log_id in log_ids if log_id in self.reviews]

    async def lock_review(self, log_id: int) -> ChatLogReview:
        return deepcopy(self.reviews.setdefault(log_id, ChatLogReview(chat_log_id=log_id)))

    async def save_review(self, review: ChatLogReview) -> None:
        self.reviews[review.chat_log_id] = deepcopy(review)

    async def stats(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> dict[str, Any]:
        del filters
        return {"total": len(self.logs)}

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
        self.locked_by: dict[int, str] = {}

    async def enqueue(self, entry: VectorOutboxEntry) -> VectorOutboxEntry:
        entry.id = self._next_id
        self._next_id += 1
        self._entries[entry.id] = entry
        self._notifications.append({"id": entry.id, "op": entry.operation.value})
        return entry

    async def claim_batch(self, worker_id: str, limit: int = 20) -> list[VectorOutboxEntry]:
        claimed: list[VectorOutboxEntry] = []

        for entry in list(self._entries.values()):
            if len(claimed) >= limit:
                break

            if entry.status in (OutboxStatus.PENDING, OutboxStatus.FAILED):
                entry.status = OutboxStatus.IN_PROGRESS

                if entry.id is not None:
                    self.locked_by[entry.id] = worker_id

                claimed.append(entry)

        return claimed

    async def release_abandoned(self, worker_prefix: str, current_worker_id: str) -> int:
        released = 0

        for entry_id, worker_id in list(self.locked_by.items()):
            entry = self._entries[entry_id]

            if (
                entry.status == OutboxStatus.IN_PROGRESS
                and worker_id.startswith(worker_prefix)
                and worker_id != current_worker_id
            ):
                entry.status = OutboxStatus.PENDING
                del self.locked_by[entry_id]
                released += 1

        return released

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

    async def assign_client_spaces(self, user_id: int, space_ids: frozenset[int]) -> None:
        self._users[user_id].assign_client_spaces(space_ids)


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

    async def check_admission(self, principal: object) -> None:
        del principal

    async def cancel(self, job_id: int) -> bool:
        job = self.jobs[job_id]

        if job.status not in {"pending", "running"}:
            return False

        job.status = type(job.status)("cancelled")
        job.finished_at = datetime.now(UTC)
        job.attempt_id = None
        job.lease_until = None
        return True

    async def save_progress(self, job_id: int, progress: dict[str, str]) -> None:
        self.jobs[job_id].progress = dict(progress)

    async def find_latest_for(self, job_type: str, related_id: int) -> BackgroundJob | None:
        return next(
            (
                job
                for _, job in sorted(self.jobs.items(), reverse=True)
                if job.job_type == job_type and job.related_id == related_id
            ),
            None,
        )

    async def trace_context(self, job_id: int) -> dict[str, str] | None:
        del job_id
        return None

    async def create(self, job: BackgroundJob) -> BackgroundJob:
        job.id = len(self.jobs) + 1
        self.jobs[job.id] = job
        return job

    async def pending_dispatch(self, limit: int = 100) -> list[BackgroundJob]:
        return [
            job for job in self.jobs.values() if job.status == "pending" and job.dispatch_payload is not None
        ][:limit]

    async def claim(
        self, job_id: int, attempt_id: str, *, lease_seconds: int, limit: ConcurrencyLimit | None = None
    ) -> int | None:
        del lease_seconds
        job = self.jobs.get(job_id)

        if job is None or job.status != "pending":
            return None

        if limit is not None:
            running = sum(
                1
                for other in self.jobs.values()
                if other.job_type in limit.job_types and other.status == "running"
            )

            if limit.max_running is not None and running >= limit.max_running:
                message = "concurrency group full"
                raise JobConcurrencyLimitReachedError(message)

            if limit.max_per_principal is not None:
                principal = (job.dispatch_payload or {}).get("cost_principal")
                running = sum(
                    other.status == "running"
                    and other.job_type in limit.job_types
                    and (other.dispatch_payload or {}).get("cost_principal") == principal
                    for other in self.jobs.values()
                )

                if running >= limit.max_per_principal:
                    message = "principal concurrency group full"
                    raise JobConcurrencyLimitReachedError(message)

        job.mark_running()
        job.attempt += 1
        job.attempt_id = attempt_id
        return job.attempt

    async def renew_lease(self, job_id: int, attempt_id: str, *, lease_seconds: int) -> bool:
        del lease_seconds
        return self.jobs[job_id].attempt_id == attempt_id

    async def finish_attempt(
        self, job_id: int, attempt_id: str, *, error: str | None = None, retry_seconds: float | None = None
    ) -> bool:
        job = self.jobs[job_id]

        if job.attempt_id != attempt_id:
            return False

        if retry_seconds is not None:
            job.status = type(job.status)("pending")
        elif error is not None:
            await self.mark_failed(job_id, error)
        else:
            await self.mark_done(job_id)

        return True

    async def lock_execution(self, job_id: int, attempt_id: str) -> None:
        if self.jobs[job_id].attempt_id != attempt_id:
            message = "lease lost"
            raise RuntimeError(message)

    async def record_dispatch(self, job_id: int, *, published: bool) -> None:
        if published:
            self.jobs[job_id].dispatch_attempts += 1

    async def mark_running(self, job_id: int) -> None:
        self.jobs[job_id].mark_running()

    async def mark_done(self, job_id: int) -> None:
        self.jobs[job_id].mark_done()

    async def mark_failed(self, job_id: int, error: str) -> None:
        self.jobs[job_id].mark_failed(error)

    async def touch_heartbeat(self, job_id: int) -> None:
        pass

    async def count_active(self) -> int:
        return 0

    async def indexing_queue_snapshot(self) -> IndexingQueueSnapshot:
        return IndexingQueueSnapshot()

    async def delete_old(self, days: int, limit: int = 1000) -> int:
        del days, limit
        return 0

    async def list_recent(
        self, limit: int = 50, offset: int = 0, status: str | None = None
    ) -> list[BackgroundJob]:
        del limit, offset, status
        return []

    async def get_by_id(self, job_id: int) -> BackgroundJob | None:
        return self.jobs.get(job_id)

    async def count_by_status(self) -> dict[str, int]:
        return {}

    async def lock_expired(self, timeout_minutes: int, limit: int = 100) -> list[BackgroundJob]:
        del timeout_minutes
        now = datetime.now(UTC)
        return [
            job
            for job in self.jobs.values()
            if job.status == "running" and job.lease_until is not None and job.lease_until <= now
        ][:limit]

    async def release_expired(self, job_id: int, *, error: str, retry_seconds: float | None) -> None:
        job = self.jobs[job_id]
        job.attempt_id = None
        job.lease_until = None

        if retry_seconds is None:
            job.mark_failed(error)
        else:
            job.status = type(job.status)("pending")
            job.error_message = error

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
    def __init__(self) -> None:
        self.created: list[BenchmarkQuestion] = []

    async def list_active(self, *, dataset: str) -> list[BenchmarkQuestion]:
        return [question for question in self.created if question.dataset == dataset and question.is_active]

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
        self.created.append(question)
        question.id = len(self.created)
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
        return sorted({question.dataset for question in self.created})

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
        return self.runs.get(run_id)

    async def exists(self, run_id: int) -> bool:
        return run_id in self.runs

    async def get_review_run(self, run_id: int) -> BenchmarkRun | None:
        return await self.get_by_id(run_id)

    async def get_review_runs(self, ids: list[int]) -> list[BenchmarkRun]:
        return await self.get_by_ids(ids)

    async def get_result_question(self, run_id: int, position: int) -> dict[str, Any] | None:
        rows = await self.result_page(run_id, offset=position, limit=1)
        return rows[0] if rows else None

    async def get_by_ids(self, ids: list[int]) -> list[BenchmarkRun]:
        return [self.runs[run_id] for run_id in ids if run_id in self.runs]

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


class FakeBenchmarkQualityRepository:
    def __init__(self) -> None:
        self.policies: dict[str, DatasetPolicy] = {}
        self.labels: dict[tuple[int, str, str, int], HumanLabel] = {}
        self.calibrations: list[tuple[str, JudgeCalibration]] = []

    async def add_calibration(self, dataset: str, calibration: JudgeCalibration) -> JudgeCalibration:
        calibration = replace(calibration, id=len(self.calibrations) + 1)
        self.calibrations.append((dataset, calibration))
        return calibration

    async def list_calibrations(self, dataset: str, limit: int) -> list[JudgeCalibration]:
        found = [item for name, item in self.calibrations if name == dataset]
        return sorted(found, key=lambda item: (item.measured_at, item.id or 0), reverse=True)[:limit]

    async def get_policy(self, dataset: str) -> DatasetPolicy | None:
        return self.policies.get(dataset)

    async def list_policies(self) -> list[DatasetPolicy]:
        return [self.policies[key] for key in sorted(self.policies)]

    async def save_policy(self, policy: DatasetPolicy) -> DatasetPolicy:
        self.policies[policy.dataset] = policy
        return policy

    async def list_labels(self, run_id: int) -> list[HumanLabel]:
        return [label for label in self.labels.values() if label.run_id == run_id]

    async def save_label(self, label: HumanLabel) -> HumanLabel:
        key = (label.run_id, label.question_key, label.metric.value, label.labeler_id)
        existing = self.labels.get(key)
        label.id = existing.id if existing is not None else len(self.labels) + 1
        self.labels[key] = label
        return label

    async def delete_label(
        self, run_id: int, question_key: str, metric: LabelMetric, labeler_id: int
    ) -> bool:
        return self.labels.pop((run_id, question_key, metric.value, labeler_id), None) is not None


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


class FakeKnowledgeSpaceRepository:
    """In-memory knowledge spaces; starts with the space a new installation has."""

    def __init__(self) -> None:
        self._spaces: dict[int, KnowledgeSpace] = {
            DEFAULT_SPACE_ID: KnowledgeSpace(id=DEFAULT_SPACE_ID, name="Общее", pack_key="generic")
        }
        self._next_id = DEFAULT_SPACE_ID + 1
        self.documents: FakeDocumentRepository | None = None

    async def get_by_id(self, space_id: int, *, for_update: bool = False) -> KnowledgeSpace | None:
        del for_update
        space = self._spaces.get(space_id)
        return replace(space) if space is not None else None

    async def list_all(self) -> list[KnowledgeSpace]:
        return [replace(space) for _, space in sorted(self._spaces.items())]

    async def create(self, space: KnowledgeSpace) -> KnowledgeSpace:
        if any(existing.name == space.name for existing in self._spaces.values()):
            raise UniqueConstraintError(detail=f"duplicate space name {space.name}")

        space.id = self._next_id
        self._next_id += 1
        self._spaces[space.id] = replace(space)
        return space

    async def update(self, space: KnowledgeSpace) -> KnowledgeSpace:
        if space.id is None or space.id not in self._spaces:
            msg = f"Space {space.id} does not exist"
            raise ValueError(msg)

        if any(other.name == space.name and other.id != space.id for other in self._spaces.values()):
            raise UniqueConstraintError(detail=f"duplicate space name {space.name}")

        stored = self._spaces[space.id]
        self._spaces[space.id] = replace(stored, name=space.name, upload_group_ids=space.upload_group_ids)
        return space

    async def delete(self, space_id: int) -> None:
        self._spaces.pop(space_id, None)

    async def has_documents(self, space_id: int) -> bool:
        documents = self.documents._documents.values() if self.documents is not None else ()
        return any(document.space_id == space_id for document in documents)

    async def document_status_counts(self, space_id: int) -> dict[str, int]:
        documents = self.documents._documents.values() if self.documents is not None else ()
        counts: dict[str, int] = {}

        for document in documents:
            if document.space_id == space_id:
                counts[str(document.status)] = counts.get(str(document.status), 0) + 1

        return counts


UPLOADERS_GROUP_ID = 900


def admit_all_users(uow: FakeUnitOfWork, space_id: int = DEFAULT_SPACE_ID) -> None:
    """Make every internal user a member of a group admitted to fill the space."""
    uow.fake_groups.member_of = (UPLOADERS_GROUP_ID,)
    uow.fake_knowledge_spaces._spaces[space_id].admit_groups({UPLOADERS_GROUP_ID})


class FakeSyncChannelRepository:
    def __init__(self) -> None:
        self.channels: dict[int, SyncChannel] = {}
        self._next_id = 1

    async def get_by_id(self, channel_id: int, *, for_update: bool = False) -> SyncChannel | None:
        del for_update
        channel = self.channels.get(channel_id)
        return deepcopy(channel) if channel is not None else None

    async def list_by_space(self, space_id: int) -> list[SyncChannel]:
        return [deepcopy(c) for c in self.channels.values() if c.space_id == space_id]

    async def list_all(self) -> list[SyncChannel]:
        return [deepcopy(c) for _, c in sorted(self.channels.items())]

    async def find_by_location(self, instance: str, repository_id: int, branch: str) -> list[SyncChannel]:
        key = ("github", instance.lower(), repository_id, branch)
        return [deepcopy(c) for c in self.channels.values() if c.settings.location.key == key]

    async def add(self, channel: SyncChannel) -> SyncChannel:
        channel.id = self._next_id
        self._next_id += 1
        self.channels[channel.id] = deepcopy(channel)
        return channel

    async def update(self, channel: SyncChannel) -> None:
        if channel.id in self.channels:
            self.channels[channel.id] = deepcopy(channel)

    async def delete(self, channel_id: int) -> None:
        self.channels.pop(channel_id, None)


class FakeSyncRunRepository:
    def __init__(self) -> None:
        self.runs: dict[int, SyncRun] = {}
        self._next_id = 1

    async def add(self, run: SyncRun) -> SyncRun:
        run.id = self._next_id
        self._next_id += 1
        self.runs[run.id] = deepcopy(run)
        return run

    async def update(self, run: SyncRun) -> None:
        if run.id in self.runs:
            self.runs[run.id] = deepcopy(run)

    async def get_by_id(self, run_id: int) -> SyncRun | None:
        run = self.runs.get(run_id)
        return deepcopy(run) if run is not None else None

    async def list_unfinished(self, channel_id: int) -> list[SyncRun]:
        return [deepcopy(r) for r in self.runs.values() if r.channel_id == channel_id and not r.is_finished]

    async def find_by_delivery(self, channel_id: int, delivery_id: str) -> SyncRun | None:
        return next(
            (
                deepcopy(r)
                for r in self.runs.values()
                if r.channel_id == channel_id and r.delivery_id == delivery_id
            ),
            None,
        )

    async def list_for_channel(self, channel_id: int, limit: int = 20) -> list[SyncRun]:
        runs = sorted(
            (r for r in self.runs.values() if r.channel_id == channel_id), key=lambda r: -(r.id or 0)
        )
        return [deepcopy(run) for run in runs[:limit]]


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
        self.fake_message_feedback = self.message_feedback = FakeMessageFeedbackRepository()
        self.fake_benchmark_questions = self.benchmark_questions = FakeBenchmarkQuestionRepository()
        self.fake_benchmark_sweeps = self.benchmark_sweeps = FakeBenchmarkSweepRepository()
        self.fake_benchmark_runs = self.benchmark_runs = FakeBenchmarkRunRepository()
        self.fake_benchmark_quality = self.benchmark_quality = FakeBenchmarkQualityRepository()
        self.fake_vector_outbox = self.vector_outbox = FakeVectorOutboxRepository()
        self.fake_assignments = self.assignments = FakeAssignmentRepository()
        self.fake_ingestion_registry = self.ingestion_registry = FakeIngestionRegistryRepository()
        self.fake_knowledge_spaces = self.knowledge_spaces = FakeKnowledgeSpaceRepository()
        self.fake_knowledge_spaces.documents = self.fake_documents
        self.fake_sync_channels = self.sync_channels = FakeSyncChannelRepository()
        self.fake_sync_runs = self.sync_runs = FakeSyncRunRepository()
        # Domain pack repositories; kept off the instance dict so rollback restores their
        # repositories (registered as attributes by pack test fakes) in place.
        self._pack_repositories: tuple[object, ...] = ()
        self._event_handlers: list[Callable[[object], Coroutine[Any, Any, None]]] = []
        self._committed = False
        self._rolled_back = False

    @property
    def pack_repositories(self) -> tuple[object, ...]:  # type: ignore[override]  # Read-only view of the snapshot-safe slot.
        return self._pack_repositories

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
        uow._pack_repositories = self._uow._pack_repositories
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
    ) -> AsyncGenerator[StreamEvent]:
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
            sources_count=len(self._sources),
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


class FakePackCatalog:
    """Loaded packs for scenario tests: the neutral pack, plus any registries and hooks given."""

    def __init__(
        self,
        registries: dict[str, DomainProfileRegistry] | None = None,
        hooks: dict[str, DocumentPackHooks | None] | None = None,
    ) -> None:
        neutral = DomainProfileRegistry()
        neutral.register(GeneralDomainProfile())
        self._registries = {"generic": neutral, **(registries or {})}
        self._hooks = dict(hooks or {})

    def pack_keys(self) -> tuple[str, ...]:
        return tuple(self._registries)

    def domain_registry(self, pack_key: str) -> DomainProfileRegistry:
        if pack_key not in self._registries:
            raise DomainPackNotLoadedError(pack_key)

        return self._registries[pack_key]

    def document_hooks(self, pack_key: str) -> DocumentPackHooks | None:
        return self._hooks.get(pack_key)
