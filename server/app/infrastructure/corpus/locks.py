"""Session advisory guards fence Qdrant writes across corpus generations."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import text

from domain.shared.exceptions import ValidationError
from infrastructure.corpus.repository import CorpusPhase, CorpusRow
from infrastructure.corpus.snapshot import CorpusSnapshot
from infrastructure.database.database import DatabaseManager

# Stable across processes and releases: changing this lets old and new writers bypass each other.
CORPUS_SWITCH_ADVISORY_LOCK_KEY = 7319048273461


@contextmanager
def corpus_write_guard(database: DatabaseManager, snapshot: CorpusSnapshot) -> Iterator[None]:
    # The synchronous Qdrant adapter already runs this in its worker thread.
    with database.master_connection() as conn:
        conn.execute(text("SET statement_timeout = '5s'"))
        conn.execute(text("SELECT pg_advisory_lock_shared(:key)"), {"key": CORPUS_SWITCH_ADVISORY_LOCK_KEY})
        admitted = False
        try:
            row = (
                conn.execute(text("SELECT phase, active FROM corpus_control WHERE id=1"))
                .mappings()
                .one_or_none()
            )
            validate_write_admission(dict(row) if row is not None else None, snapshot)
            admitted = True
            yield
        finally:
            try:
                if admitted:
                    conn.execute(text("UPDATE corpus_control SET write_epoch=write_epoch+1 WHERE id=1"))
            finally:
                conn.execute(
                    text("SELECT pg_advisory_unlock_shared(:key)"), {"key": CORPUS_SWITCH_ADVISORY_LOCK_KEY}
                )


def validate_write_admission(row: CorpusRow | None, snapshot: CorpusSnapshot) -> None:
    if row is None or row["phase"] not in {CorpusPhase.ACTIVE, CorpusPhase.FAILED}:
        raise ValidationError("Corpus switch is pending; retry the write after readiness")

    if CorpusSnapshot(**row["active"]).namespace != snapshot.namespace:
        raise ValidationError("Corpus generation changed; retry using the current snapshot")


@asynccontextmanager
async def corpus_switch_guard(database: DatabaseManager) -> AsyncIterator[None]:
    # NullPool closes the physical connection even on acquisition failure or cancellation.
    async with database.async_master_connection() as conn:
        await conn.execute(text("SET statement_timeout = '5s'"))
        await conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": CORPUS_SWITCH_ADVISORY_LOCK_KEY})
        yield


def validate_switch_version(latest: CorpusRow, previous: CorpusRow, expected_generation: str) -> None:
    if (
        latest["active"]["generation"] != expected_generation
        or latest["phase"] not in {CorpusPhase.ACTIVE, CorpusPhase.FAILED}
        or latest["write_epoch"] != previous["write_epoch"]
    ):
        raise ValidationError("Corpus changed during validation; retry the switch")
