"""Retryable PostgreSQL journal, write barrier, and serving-process acknowledgements.

A session advisory lock fences Qdrant writes without a database transaction
around network probes. A short final transaction fences PostgreSQL mutations
and rechecks the corpus. Committed switches remain write-blocked until API and
worker have installed the same immutable resource generation.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
from functools import partial
import logging
import re
from time import monotonic
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from config import settings
from domain.shared.exceptions import ValidationError
from infrastructure.corpus import snapshot as corpus_snapshot
from infrastructure.corpus.clients import probe_active_corpus, raw_corpus_client, validate_corpus_endpoint
from infrastructure.corpus.locks import corpus_switch_guard, corpus_write_guard, validate_switch_version
from infrastructure.corpus.repository import CorpusPhase, CorpusRepository
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
from infrastructure.corpus.validation import corpus_digest, validate_corpus, validate_corpus_binding
from infrastructure.ml.config.embedding_identity import configured_embedding_identity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from infrastructure.corpus.repository import CorpusRow
    from infrastructure.database.database import DatabaseManager
    from infrastructure.events.postgres_config_listener import PostgresConfigListener
    from infrastructure.corpus.resources import CorpusResources

log = logging.getLogger("default")


class PostgresCorpusControl:
    def __init__(
        self,
        database: DatabaseManager,
        resources: CorpusResources,
        configuration_listener: PostgresConfigListener | None,
    ) -> None:
        self.database = database
        self.repository = CorpusRepository(database.get_write_session)
        self.write_guard = partial(corpus_write_guard, database)
        self.resources = resources
        self.listener = configuration_listener
        self.process_id = str(uuid4())
        self.role = "cli"
        self.task: asyncio.Task[None] | None = None
        self.prepared: BoundQdrantClient | None = None
        self.prepared_index = None
        self.error: str | None = None
        self.sync_lock = asyncio.Lock()
        self.last_sync: float | None = None

    async def read_state(self) -> CorpusRow | None:
        return await self.repository.read_state()

    async def chunks(self) -> list[CorpusRow]:
        return await self.repository.chunks()

    async def initial_snapshot(self) -> CorpusSnapshot:
        client = raw_corpus_client(settings.qdrant_url)
        try:
            aliases = await asyncio.to_thread(client.get_aliases)
            physical = next(
                (a.collection_name for a in aliases.aliases if a.alias_name == settings.collection_name),
                settings.collection_name,
            )
            rows = await self.chunks()
            names = {item.name for item in (await asyncio.to_thread(client.get_collections)).collections}

            if physical in names and (await asyncio.to_thread(client.get_collection, physical)).points_count:
                verified = await asyncio.to_thread(
                    validate_corpus,
                    client,
                    settings.qdrant_url,
                    settings.collection_name,
                    configured_embedding_identity(),
                    settings.embed_dim,
                    rows,
                )
                return replace(verified, generation=str(uuid4()))

            return CorpusSnapshot(
                settings.qdrant_url,
                settings.collection_name,
                physical,
                configured_embedding_identity(),
                str(uuid4()),
            )
        finally:
            await asyncio.to_thread(client.close)

    async def start(self, role: str) -> None:
        self.role = role
        state = await self.read_state()

        if state is None:
            snapshot = await self.initial_snapshot()
            await self.repository.initialize(snapshot)
            state = await self.read_state()

        if state is None:
            raise ValidationError("Corpus journal is not initialized")

        active = CorpusSnapshot(**state["active"])

        if (settings.qdrant_url, settings.collection_name) != (active.endpoint, active.alias):
            raise ValidationError("Saved Qdrant settings differ from the durable corpus journal")

        if active.embedding_identity != configured_embedding_identity():
            raise ValidationError("Active corpus embedding identity differs from the configured encoder")

        raw = raw_corpus_client(active.endpoint)
        try:
            completed = await self.repository.has_completed_chunks()
            await asyncio.to_thread(probe_active_corpus, raw, active, completed=completed)
            index = None
        except BaseException:
            await asyncio.to_thread(raw.close)
            raise
        self.install(BoundQdrantClient(raw, active, self.write_guard), index)
        await self.tick()
        self.task = asyncio.create_task(self.monitor(), name=f"corpus-control-{role}")

    def install(self, client: BoundQdrantClient, index: None) -> None:
        self.resources.install_corpus(client, index)
        corpus_snapshot.installed_corpus = client.snapshot
        # This synchronous publication is the sole owner of corpus settings.
        settings.qdrant_url = client.snapshot.endpoint
        settings.collection_name = client.snapshot.alias

    async def heartbeat(self, prepared: str | None = None, error: str | None = None) -> None:
        return await self.repository.heartbeat(
            self.process_id, self.role, self.resources.corpus_snapshot, prepared, error
        )

    async def prepare(
        self, candidate: CorpusSnapshot, expected_digest: str
    ) -> tuple[BoundQdrantClient, None]:
        rows = await self.chunks()

        if corpus_digest(rows) != expected_digest:
            raise ValidationError("PostgreSQL corpus changed during preparation")

        validate_corpus_endpoint(candidate.endpoint, settings.qdrant_url)
        raw = raw_corpus_client(candidate.endpoint)
        try:
            verified = await asyncio.to_thread(
                validate_corpus_binding,
                raw,
                candidate.endpoint,
                candidate.alias,
                candidate.embedding_identity,
                settings.embed_dim,
            )

            if verified.physical_collection != candidate.physical_collection:
                raise ValidationError("Qdrant alias changed during preparation")

            index = None
            return BoundQdrantClient(raw, candidate, self.write_guard), index
        except BaseException:
            await asyncio.to_thread(raw.close)
            raise

    async def tick(self) -> None:
        async with self.sync_lock:
            state = await self.read_state()

            if state is None:
                raise ValidationError("Corpus journal is not initialized")

            active = CorpusSnapshot(**state["active"])

            if self.resources.corpus_snapshot.generation != active.generation or self.error:
                # A replica may miss the entire switch, including its committing phase.
                await self.apply_candidate({**state, "phase": CorpusPhase.COMMITTING}, active)

                if self.error:
                    await self.heartbeat(error=self.error)
                    return

            candidate = CorpusSnapshot(**state["candidate"]) if state["candidate"] else None
            prepared_id = None

            if state["phase"] in {CorpusPhase.PREPARING, CorpusPhase.COMMITTING} and candidate is not None:
                prepared_id = await self.apply_candidate(state, candidate)
            else:
                self.error = None

                if self.prepared is not None:
                    await asyncio.to_thread(self.prepared.close)
                    self.prepared = self.prepared_index = None

            await self.heartbeat(prepared_id, self.error)
            await self.recover(state)
            self.last_sync = monotonic()

    async def apply_candidate(self, state: CorpusRow, candidate: CorpusSnapshot) -> str | None:
        prepared_id = None
        try:
            current = self.resources.corpus_snapshot

            if current.generation != candidate.generation:
                if self.prepared is None or self.prepared.snapshot != candidate:
                    if self.prepared is not None:
                        await asyncio.to_thread(self.prepared.close)

                    self.prepared = self.prepared_index = None

                    if state["phase"] == CorpusPhase.COMMITTING:
                        # The durable active journal is the admission proof; the old
                        # preparation digest can be stale after subsequent ingestion.
                        index = None
                        self.prepared = BoundQdrantClient(
                            raw_corpus_client(candidate.endpoint), candidate, self.write_guard
                        )
                        self.prepared_index = index
                    else:
                        self.prepared, self.prepared_index = await self.prepare(candidate, state["digest"])

                prepared_id = candidate.generation

                if state["phase"] == CorpusPhase.COMMITTING:
                    self.install(self.prepared, self.prepared_index)
                    self.prepared = self.prepared_index = None

                    if self.listener is not None:
                        await self.listener.synchronize()
            else:
                prepared_id = candidate.generation

                if self.error and self.listener is not None:
                    await self.listener.synchronize()

            self.error = None
        except Exception:
            self.error = "Corpus preparation/application failed"
            log.exception("Corpus preparation/application failed")
        return prepared_id

    async def participants(self) -> list[CorpusRow]:
        return await self.repository.participants()

    async def recover(self, state: CorpusRow) -> None:
        participants = await self.participants()
        live = [p for p in participants if p["alive"]]

        if state["phase"] == CorpusPhase.COMMITTING and state["candidate"]:
            generation = state["candidate"]["generation"]

            if {p["role"] for p in live} == {"api", "worker"} and all(
                p["applied_generation"] == generation and not p["error"] for p in live
            ):
                await self.repository.finish_commit(generation)
        elif state["phase"] == CorpusPhase.PREPARING and not any(
            p["process_id"] == state["owner"] and p["alive"] for p in live
        ):
            await self.repository.fail_orphan(state["owner"])

    async def monitor(self) -> None:
        while True:
            await asyncio.sleep(2)
            try:
                await self.tick()
            except Exception:
                self.error = "Corpus synchronization unavailable"
                log.exception("Corpus synchronization unavailable")

    async def is_ready(self) -> bool:
        if (
            self.task is None
            or self.task.done()
            or self.error
            or self.last_sync is None
            or monotonic() - self.last_sync >= 15
        ):
            return False

        try:
            state = await self.read_state()
        except Exception:
            log.exception("Corpus admission journal unavailable")
            return False
        current = self.resources.corpus_snapshot
        return bool(state and current is not None and current.generation == state["active"]["generation"])

    async def status(self) -> dict[str, Any]:
        state = await self.read_state()
        return {
            "phase": state["phase"] if state else "uninitialized",
            "active": state["active"] if state else None,
            "candidate": state["candidate"] if state else None,
            "error": state["error"] if state else None,
            "processes": [dict(p) for p in await self.participants()],
        }

    async def quiescent(self, session: AsyncSession) -> None:
        return await self.repository.quiescent(session)

    async def switch(self, endpoint: str, collection: str, expected_generation: str) -> dict[str, Any]:
        validate_corpus_endpoint(endpoint, settings.qdrant_url)

        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", collection):
            raise ValidationError("Invalid Qdrant collection or alias")

        guards = AsyncExitStack()
        committed = False
        try:
            state = await self.read_state()

            if state is None or state["phase"] not in {CorpusPhase.ACTIVE, CorpusPhase.FAILED}:
                raise ValidationError("A corpus switch is already pending; inspect its status")

            if state["active"]["generation"] != expected_generation:
                raise ValidationError("Corpus generation changed; refresh before switching")

            participants = [p for p in await self.participants() if p["alive"]]

            if {p["role"] for p in participants} != {"api", "worker"} or any(
                p["error"] or p["applied_generation"] != expected_generation for p in participants
            ):
                raise ValidationError("Both API and worker must be ready before switching")

            rows, config_rows = await self.repository.switch_inputs()
            raw = raw_corpus_client(endpoint)
            try:
                snapshot = await asyncio.to_thread(
                    validate_corpus,
                    raw,
                    endpoint,
                    collection,
                    configured_embedding_identity(),
                    settings.embed_dim,
                    rows,
                )
                snapshot = replace(snapshot, generation=str(uuid4()))
            finally:
                await asyncio.to_thread(raw.close)
            await guards.enter_async_context(corpus_switch_guard(self.database))
            latest_state = await self.read_state()
            validate_switch_version(latest_state, state, expected_generation)
            digest = corpus_digest(rows)
            await self.repository.begin_prepare(snapshot, digest, self.process_id)
            async with asyncio.timeout(15):
                await self.wait_prepared(snapshot.generation, {p["process_id"] for p in participants})
            await self.commit_switch(snapshot, digest, config_rows)
            committed = True

            for _ in range(30):
                await self.tick()
                state = await self.read_state()

                if state["phase"] == CorpusPhase.ACTIVE:
                    return await self.status()

                await asyncio.sleep(0.5)

            raise ValidationError(
                "Corpus committed; writes remain fenced until API/worker readiness recovers"
            )
        except BaseException:
            if not committed:
                await self.repository.fail_prepare(self.process_id)

            raise
        finally:
            await guards.aclose()

    async def commit_switch(self, snapshot: CorpusSnapshot, digest: str, config_rows: Sequence[Any]) -> None:
        return await self.repository.commit_switch(snapshot, digest, config_rows, self.process_id)

    async def wait_prepared(self, generation: str, required: set[str]) -> None:
        for _ in range(30):
            await self.tick()
            participants = {p["process_id"]: p for p in await self.participants()}

            if any(
                pid not in participants or not participants[pid]["alive"] or participants[pid]["error"]
                for pid in required
            ):
                raise ValidationError("Serving process unavailable or failed corpus preparation")

            if all(participants[pid]["prepared_generation"] == generation for pid in required):
                return

            await asyncio.sleep(0.5)

        raise ValidationError("Serving processes did not prepare the corpus before timeout")

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

        if self.prepared is not None:
            await asyncio.to_thread(self.prepared.close)
            self.prepared = None

        await self.repository.remove_process(self.process_id)
