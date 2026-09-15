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
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from domain.entities.message import Message
from domain.entities.vector_outbox_entry import OutboxStatus, VectorOutboxEntry
from domain.repositories.ingestion_registry_repository import IngestionRegistryEntry
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

    async def get_by_id(self, conv_id: int):
        return await self.get(conv_id)

    async def get_for_update(self, conv_id: int):
        return await self.get(conv_id)

    async def update_summary(self, conv_id: int, summary: str | None) -> None:
        conv = self._convs.get(conv_id)
        if conv is not None:
            conv["summary"] = summary

    async def update_summary_if_unchanged(self, conv_id: int, expected_summary, new_summary) -> bool:
        conv = self._convs.get(conv_id)
        if conv is None or conv["summary"] != expected_summary:
            return False
        conv["summary"] = new_summary
        return True

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

    async def clear_summaries_referencing(self, document_id: int) -> int:
        """Clear summaries for conversations whose messages cite the given document_id.

        Fake implementation: clears ALL summaries (conservative, matches pre-fix behavior).
        """
        count = 0
        for conv in self._convs.values():
            if conv.get("summary") is not None:
                conv["summary"] = None
                count += 1
        return count

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
        domain_metadata: dict | None = None,
        act_version_id: int | None = None,
        effective_from=None,
        effective_to=None,
        is_current: bool = True,
        sections: list[str | None] | None = None,
        headings: list[str | None] | None = None,
        heading_levels: list[int | None] | None = None,
        content_types: list[str | None] | None = None,
        doc_titles: list[str | None] | None = None,
        doc_types: list[str | None] | None = None,
    ) -> list[int]:
        # Remove existing chunks for this document
        self._chunks = [c for c in self._chunks if c["document_id"] != document_id]
        ids = []
        for i, content in enumerate(chunks):
            chunk_id = self._next_id
            self._next_id += 1
            self._chunks.append(
                {
                    "id": chunk_id,
                    "document_id": document_id,
                    "content": content,
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
                    "content_hash": content_hashes[i] if content_hashes and i < len(content_hashes) else None,
                }
            )
            ids.append(chunk_id)
        return ids

    async def set_current_by_act_version_ids(self, act_version_ids: list[int], is_current: bool) -> int:
        updated = 0
        for c in self._chunks:
            if c.get("act_version_id") in act_version_ids:
                c["is_current"] = is_current
                updated += 1
        return updated

    async def update_temporal_by_act_version_id(
        self, act_version_id: int, effective_from, effective_to
    ) -> int:
        updated = 0
        for c in self._chunks:
            if c.get("act_version_id") == act_version_id:
                c["effective_from"] = effective_from
                c["effective_to"] = effective_to
                updated += 1
        return updated

    async def search_substring(self, **kwargs):
        return []

    async def get_all_contents(self) -> list[str]:
        return []

    async def list_for_document(
        self,
        document_id: int,
        limit: int = 50,
        offset: int = 0,
        content_hashes: list[str] | None = None,
    ) -> tuple[list, int]:
        from domain.repositories.chunk_repository import ChunkSearchResult

        rows = [c for c in self._chunks if c["document_id"] == document_id]
        total = len(rows)
        rows = rows[offset: offset + limit]
        items = [
            ChunkSearchResult(
                chunk_id=c["id"],
                document_id=c["document_id"],
                filename=c.get("filename", ""),
                content=c.get("content", ""),
                chunk_index=c.get("chunk_index", 0),
                visibility=c.get("visibility", ""),
                doc_domain=c.get("doc_domain", "general"),
                owner_id=c.get("owner_id"),
                group_id=c.get("group_id"),
                content_hash=c.get("content_hash"),
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
    ):
        from domain.utils import encode_cursor
        from domain.value_objects.cursor_page import CursorPage

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
                from domain.exceptions import ValidationError

                raise ValidationError("cursor is required when direction=prev")
            ci, cid = cursor
            rows = [c for c in rows if (c.get("chunk_index", 0), c["id"]) < (ci, cid)]
            rows.sort(key=lambda c: (c.get("chunk_index", 0), c["id"]), reverse=True)
            page = rows[:limit]
            has_extra = len(rows) > limit
            page.reverse()
            prev_cur = encode_cursor(page[0]["chunk_index"], page[0]["id"]) if has_extra and page else None
            next_cur = encode_cursor(page[0]["chunk_index"], page[0]["id"]) if page else None

        from domain.repositories.chunk_repository import ChunkSearchResult

        items = [
            ChunkSearchResult(
                chunk_id=c["id"],
                document_id=c["document_id"],
                filename=c.get("filename", ""),
                content=c.get("content", ""),
                chunk_index=c.get("chunk_index", 0),
                content_hash=c.get("content_hash"),
            )
            for c in page
        ]
        return CursorPage(items=items, next_cursor=next_cur, prev_cursor=prev_cur)


class FakeDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[int, object] = {}
        self._next_id = 1

    async def get_by_id(self, doc_id: int):
        return self._documents.get(doc_id)

    async def list_all(self, limit: int = 200, offset: int = 0):
        return list(self._documents.values())[offset : offset + limit]

    async def list_visible(
        self,
        *,
        user_kind: str = "internal",
        user_id: int = 0,
        group_ids: list[int] | None = None,
        user_role: str = "user",
        limit: int = 200,
        offset: int = 0,
        managed_client_ids: list[int] | None = None,
        managed_internal_ids: list[int] | None = None,
        managed_group_ids: list[int] | None = None,
    ):
        from domain.services.access_control import is_in_search_scope

        class _FakeCtx:
            def __init__(self):
                self.user_id = user_id
                self.user_kind = user_kind
                self.user_role = user_role
                self.group_ids = group_ids or []
                self.managed_client_ids = managed_client_ids or []
                self.managed_internal_ids = managed_internal_ids or []
                self.managed_group_ids = managed_group_ids or []

        ctx = _FakeCtx()
        visible = [d for d in self._documents.values() if is_in_search_scope(d, ctx)]
        return visible[offset : offset + limit]

    async def save(self, doc):
        doc.id = self._next_id
        self._next_id += 1
        self._documents[doc.id] = doc
        return doc

    async def find_active_slots_by_filenames(self, filenames: list[str]):
        return [d for d in self._documents.values() if getattr(d, "filename", None) in filenames]

    async def find_active_slot(self, owner_id, filename, group_id, for_update=False):
        for d in self._documents.values():
            if (
                getattr(d, "filename", None) == filename
                and getattr(d, "owner_id", None) == owner_id
                and getattr(d, "group_id", None) == group_id
            ):
                return d
        return None

    async def update_status(self, document_id: int, status: str, **kwargs) -> None:
        return None

    async def mark_done_if_indexing(self, document_id: int) -> bool:
        return True

    async def mark_stuck_processing_failed(self) -> list[int]:
        return []

    async def reconcile_indexing_documents(self) -> list[int]:
        return []

    async def set_source_path(self, document_id: int, source_path: str) -> None:
        doc = self._documents.get(document_id)
        if doc is not None:
            doc.source_path = source_path

    async def delete(self, document_id: int) -> None:
        self._documents.pop(document_id, None)

    async def update_filename(self, document_id: int, new_filename: str, new_source_path: str) -> None:
        doc = self._documents.get(document_id)
        if doc is not None:
            doc.filename = new_filename
            doc.source_path = new_source_path

    async def list_distinct_filenames(self, search: str | None = None, limit: int = 100) -> list[str]:
        names = list({d.filename for d in self._documents.values() if d.filename})
        if search:
            names = [n for n in names if search.lower() in n.lower()]
        return sorted(names)[:limit]


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
    def __init__(self) -> None:
        self._users: dict[int, object] = {}
        self._next_id = 1

    async def get_by_id(self, user_id: int):
        return self._users.get(user_id)

    async def list_all(self):
        return list(self._users.values())

    def add_user(self, user_id: int, email: str = "user@test.com", kind: str = "internal"):
        from types import SimpleNamespace

        u = SimpleNamespace(id=user_id, email=email, kind=kind)
        self._users[user_id] = u
        return u


class FakeConfigParameterRepository:
    def __init__(self):
        self._params: dict[tuple[str, str | None], object] = {}

    async def get_all(self):
        return list(self._params.values())

    async def get_by_key(self, key: str):
        return self._params.get((key, None))

    async def get_by_key_and_domain(self, key: str, domain_key=None):
        return self._params.get((key, domain_key))

    async def update_value(self, key: str, value: str, domain_key=None) -> None:
        p = self._params.get((key, domain_key))
        if p:
            p.value = value

    async def update_category(self, key: str, category: str) -> None:
        pass

    async def save(self, entity) -> None:
        self._params[(entity.key, getattr(entity, "domain_key", None))] = entity

    async def upsert(self, entity) -> None:
        self._params.setdefault((entity.key, getattr(entity, "domain_key", None)), entity)

    async def count(self) -> int:
        return len(self._params)


class FakeBackgroundJobRepository:
    async def create(self, job):
        return job

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
        return 0

    async def list_recent(self, limit: int = 50, offset: int = 0):
        return []

    async def get_by_id(self, job_id: int):
        return None

    async def count_by_status(self) -> dict[str, int]:
        return {}

    async def recover_orphaned(self, timeout_minutes: int = 15) -> list[int]:
        return []

    async def fail_stale_pending(self, timeout_minutes: int) -> list:
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

    async def has_active(self) -> bool:
        return False


class FakeBenchmarkRunRepository:
    async def get_by_id(self, run_id: int):
        return None

    async def get_by_ids(self, ids: list[int]):
        return []

    async def list(self, **kwargs):
        return []

    async def count(self, **kwargs) -> int:
        return 0


class FakeRegulatoryActRepository:
    def __init__(self):
        self._acts: dict[int, object] = {}
        self._next_id = 1

    async def find_by_type_and_number(self, act_type: str, act_number: str):
        for act in self._acts.values():
            if act.act_type == act_type and act.act_number == act_number:
                return act
        return None

    async def get_by_id(self, act_id: int):
        return self._acts.get(act_id)

    async def save(self, act):
        act.id = self._next_id
        self._next_id += 1
        self._acts[act.id] = act
        return act

    async def list_all(self):
        return list(self._acts.values())


class FakeActVersionRepository:
    def __init__(self):
        self._versions: dict[int, object] = {}
        self._next_id = 1

    async def create(self, version):
        version.id = self._next_id
        self._next_id += 1
        self._versions[version.id] = version
        return version

    async def get_by_id(self, version_id: int):
        return self._versions.get(version_id)

    async def get_by_document_id(self, document_id: int):
        for v in self._versions.values():
            if v.document_id == document_id:
                return v
        return None

    async def unset_current(self, act_id: int) -> None:
        for v in self._versions.values():
            if v.act_id == act_id and v.is_current:
                v.is_current = False

    async def list_by_act(self, act_id: int):
        return [v for v in self._versions.values() if v.act_id == act_id]

    async def list_pending_review(self):
        return [v for v in self._versions.values() if v.date_source == "extracted" or v.act_id is None]

    async def update(self, version) -> None:
        if version.id in self._versions:
            self._versions[version.id] = version


class FakeAssignmentRepository:
    def __init__(self) -> None:
        self._user_assignments: dict[
            tuple[int, int], dict
        ] = {}  # (curator_id, target_user_id) -> {assigned_by}
        self._group_assignments: dict[tuple[int, int], dict] = {}  # (curator_id, group_id) -> {assigned_by}
        self._user_kinds: dict[int, str] = {}  # user_id -> kind (for filtering)

    def set_user_kind(self, user_id: int, kind: str) -> None:
        self._user_kinds[user_id] = kind

    async def get_managed_user_ids(self, curator_id: int) -> list[int]:
        return [t for (c, t) in self._user_assignments if c == curator_id]

    async def get_managed_client_ids(self, curator_id: int) -> list[int]:
        return [
            t for (c, t) in self._user_assignments if c == curator_id and self._user_kinds.get(t) == "client"
        ]

    async def get_managed_internal_ids(self, curator_id: int) -> list[int]:
        return [
            t
            for (c, t) in self._user_assignments
            if c == curator_id and self._user_kinds.get(t) == "internal"
        ]

    async def get_managed_group_ids(self, curator_id: int) -> list[int]:
        return [g for (c, g) in self._group_assignments if c == curator_id]

    async def assign_user(self, curator_id: int, target_user_id: int, assigned_by: int) -> None:
        self._user_assignments[(curator_id, target_user_id)] = {"assigned_by": assigned_by}

    async def unassign_user(self, curator_id: int, target_user_id: int) -> None:
        self._user_assignments.pop((curator_id, target_user_id), None)

    async def assign_group(self, curator_id: int, group_id: int, assigned_by: int) -> None:
        self._group_assignments[(curator_id, group_id)] = {"assigned_by": assigned_by}

    async def unassign_group(self, curator_id: int, group_id: int) -> None:
        self._group_assignments.pop((curator_id, group_id), None)

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
        self._entries[entry.filename] = entry

    async def delete(self, filename: str) -> None:
        self._entries.pop(filename, None)

    async def list_all(self) -> dict[str, IngestionRegistryEntry]:
        return dict(self._entries)

    async def is_already_indexed(self, filename: str, file_hash: str) -> bool:
        entry = self._entries.get(filename)
        return entry is not None and entry.file_hash == file_hash


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
        self.regulatory_acts = FakeRegulatoryActRepository()
        self.act_versions = FakeActVersionRepository()
        self.assignments = FakeAssignmentRepository()
        self.ingestion_registry = FakeIngestionRegistryRepository()
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
        from domain.value_objects.rag_result import RagResult

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
        self._auxiliary_semaphore = asyncio.Semaphore(4)
        self._generation_semaphore = asyncio.Semaphore(12)
        self._qdrant_search_semaphore = asyncio.Semaphore(8)
        self._bm25_search_semaphore = asyncio.Semaphore(8)
        self._reranker_semaphore = asyncio.Semaphore(8)
        self._embedding_semaphore = asyncio.Semaphore(8)

    def llm(self):
        return self

    def llm_for_breadth(self, breadth: str):
        return self

    def fast_llm(self):
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

    @property
    def instructor_client(self):
        from unittest.mock import AsyncMock, MagicMock

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(is_relevant=True, reason="fake", is_sufficient=True, reasoning="fake")
        )
        return mock_client

    @property
    def auxiliary_semaphore(self):
        return self._auxiliary_semaphore

    @property
    def generation_semaphore(self):
        return self._generation_semaphore

    @property
    def qdrant_search_semaphore(self):
        return self._qdrant_search_semaphore

    @property
    def bm25_search_semaphore(self):
        return self._bm25_search_semaphore

    @property
    def reranker_semaphore(self):
        return self._reranker_semaphore

    @property
    def embedding_semaphore(self):
        return self._embedding_semaphore

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
