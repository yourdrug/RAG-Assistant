"""PostgreSQL corpus journal, participant records and atomic switch commit."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from enum import StrEnum
import json
from typing import Any
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from domain.shared.exceptions import ValidationError
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.index_sync import OutboxStatus
from infrastructure.corpus.snapshot import CorpusSnapshot
from infrastructure.corpus.validation import corpus_digest


class CorpusPhase(StrEnum):
    ACTIVE = "active"
    FAILED = "failed"
    PREPARING = "preparing"
    COMMITTING = "committing"


CorpusRow = Mapping[str, Any]
SessionFactory = Callable[[], AsyncSession]
CHUNKS_SQL = """SELECT id, document_id, visibility, owner_id, group_id, content_hash,
                      doc_domain, content FROM chunks ORDER BY id"""


class CorpusRepository:
    def __init__(self, session_factory: SessionFactory) -> None:
        self.session_factory = session_factory

    async def read_state(self) -> CorpusRow | None:
        async with self.session_factory() as session:
            row = (
                (await session.execute(text("SELECT * FROM corpus_control WHERE id=1")))
                .mappings()
                .one_or_none()
            )
            return dict(row) if row is not None else None

    async def chunks(self) -> list[CorpusRow]:
        async with self.session_factory() as session:
            return [dict(row) for row in (await session.execute(text(CHUNKS_SQL))).mappings()]

    async def participants(self) -> list[CorpusRow]:
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    text("""SELECT *,
                heartbeat_at > now() - interval '15 seconds' AS alive
                FROM corpus_processes WHERE role IN ('api', 'worker')""")
                )
            ).mappings()
            return [dict(row) for row in rows]

    async def quiescent(self, session: AsyncSession) -> None:
        busy = (
            await session.execute(
                text("""SELECT
            EXISTS(SELECT 1 FROM documents WHERE status IN (:pending,:processing,:indexing))
            OR EXISTS(SELECT 1 FROM vector_store_outbox WHERE status <> :done)"""),
                {
                    "pending": DocumentStatus.PENDING.value,
                    "processing": DocumentStatus.PROCESSING.value,
                    "indexing": DocumentStatus.INDEXING.value,
                    "done": OutboxStatus.DONE.value,
                },
            )
        ).scalar_one()

        if busy:
            raise ValidationError("Finish or reconcile ingestion/outbox work before switching corpus")

    async def commit_switch(
        self, snapshot: CorpusSnapshot, digest: str, config_rows: Sequence[Any], process_id: str
    ) -> None:
        # No network calls in this short final transaction.
        async with self.session_factory() as session, session.begin():
            journal = (
                (
                    await session.execute(
                        text("SELECT phase, owner, candidate FROM corpus_control WHERE id=1 FOR UPDATE")
                    )
                )
                .mappings()
                .one()
            )

            if (
                journal["phase"] != CorpusPhase.PREPARING
                or journal["owner"] != process_id
                or not journal["candidate"]
                or journal["candidate"]["generation"] != snapshot.generation
            ):
                raise ValidationError("Corpus preparation journal changed before commit")

            await session.execute(
                text("""LOCK TABLE documents, chunks, vector_store_outbox,
                config_parameters IN SHARE ROW EXCLUSIVE MODE""")
            )
            await self.quiescent(session)
            latest = list((await session.execute(text(CHUNKS_SQL))).mappings())
            config_latest = (
                await session.execute(text("SELECT key, domain_key, value FROM config_parameters"))
            ).all()

            if corpus_digest(latest) != digest or set(config_latest) != set(config_rows):
                raise ValidationError("Configuration/corpus changed during probe; retry switching")

            for key, value in [("qdrant_url", snapshot.endpoint), ("collection_name", snapshot.alias)]:
                result = await session.execute(
                    text("""UPDATE config_parameters SET value=:value
                    WHERE key=:key AND domain_key IS NULL RETURNING key"""),
                    {"key": key, "value": value},
                )

                if len(result.all()) != 1:
                    raise ValidationError("Corpus configuration is not initialized")

            await session.execute(
                text("""UPDATE corpus_control SET phase=:phase,
                active=candidate, updated_at=now() WHERE id=1"""),
                {"phase": CorpusPhase.COMMITTING},
            )
            await session.execute(text("SELECT pg_notify('config_changed', '{}')"))

    async def heartbeat(
        self,
        process_id: str,
        role: str,
        active: CorpusSnapshot | None,
        prepared: str | None = None,
        error: str | None = None,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                text("""
                INSERT INTO corpus_processes (process_id, role, prepared_generation,
                                              applied_generation, error, heartbeat_at)
                VALUES (:id, :role, :prepared, :applied, :error, now())
                ON CONFLICT (process_id) DO UPDATE SET
                    prepared_generation=EXCLUDED.prepared_generation,
                    applied_generation=EXCLUDED.applied_generation, error=EXCLUDED.error,
                    heartbeat_at=now()
            """),
                {
                    "id": process_id,
                    "role": role,
                    "prepared": prepared,
                    "applied": active.generation if active else None,
                    "error": error,
                },
            )

    async def initialize(self, snapshot: CorpusSnapshot) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                text("""INSERT INTO corpus_control (id, phase, active)
                VALUES (1, :phase, CAST(:active AS jsonb)) ON CONFLICT DO NOTHING"""),
                {"active": json.dumps(asdict(snapshot)), "phase": CorpusPhase.ACTIVE},
            )

    async def has_completed_chunks(self) -> bool:
        async with self.session_factory() as session:
            return bool(
                (
                    await session.execute(
                        text("""SELECT EXISTS(SELECT 1 FROM chunks c
                JOIN documents d ON d.id=c.document_id WHERE d.status=:status)"""),
                        {"status": DocumentStatus.DONE.value},
                    )
                ).scalar_one()
            )

    async def switch_inputs(self) -> tuple[list[CorpusRow], Sequence[Any]]:
        async with self.session_factory() as session:
            await self.quiescent(session)
            rows: list[CorpusRow] = [
                dict(row) for row in (await session.execute(text(CHUNKS_SQL))).mappings()
            ]
            config_rows = (
                await session.execute(text("SELECT key, domain_key, value FROM config_parameters"))
            ).all()
            return rows, config_rows

    async def begin_prepare(self, snapshot: CorpusSnapshot, digest: str, owner: str) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                text("""UPDATE corpus_control SET phase=:phase,
                candidate=CAST(:candidate AS jsonb), digest=:digest, owner=:owner,
                error=NULL, updated_at=now() WHERE id=1"""),
                {
                    "candidate": json.dumps(asdict(snapshot)),
                    "digest": digest,
                    "owner": owner,
                    "phase": CorpusPhase.PREPARING,
                },
            )

    async def finish_commit(self, generation: str) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                text("""UPDATE corpus_control SET phase=:active, error=NULL,
                candidate=NULL, updated_at=now() WHERE id=1 AND phase=:committing
                AND active->>'generation'=:generation"""),
                {
                    "generation": generation,
                    "active": CorpusPhase.ACTIVE,
                    "committing": CorpusPhase.COMMITTING,
                },
            )

    async def fail_orphan(self, owner: str) -> None:
        await self.fail_prepare(owner, "Switch owner unavailable before commit")

    async def fail_prepare(self, owner: str, error: str = "Corpus switch failed before commit") -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                text("""UPDATE corpus_control SET phase=:failed, candidate=NULL,
                error=:error, updated_at=now()
                WHERE id=1 AND phase=:preparing AND owner=:owner"""),
                {
                    "owner": owner,
                    "error": error,
                    "failed": CorpusPhase.FAILED,
                    "preparing": CorpusPhase.PREPARING,
                },
            )

    async def remove_process(self, process_id: str) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                text("DELETE FROM corpus_processes WHERE process_id=:id"), {"id": process_id}
            )
