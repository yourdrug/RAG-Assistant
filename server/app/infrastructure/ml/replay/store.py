"""PostgreSQL storage of recorded LLM responses.

Recorded prompts contain private document text: the table is read only by benchmark runs,
and entries unused for ``retention_days`` are deleted by ``purge``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol
import zlib

from sqlalchemy import text

from infrastructure.ml.replay.recording import RecordedResponse


if TYPE_CHECKING:
    from infrastructure.database.database import DatabaseManager


class ReplayStore(Protocol):
    async def load(self, key: str) -> RecordedResponse | None: ...

    async def save(self, key: str, model: str | None, response: RecordedResponse) -> None: ...


class PostgresReplayStore:
    def __init__(self, database: DatabaseManager) -> None:
        self._database = database

    async def load(self, key: str) -> RecordedResponse | None:
        async with self._database.get_session(master=True) as session:
            row = (
                (
                    await session.execute(
                        text("""
                        UPDATE llm_replay_entries SET last_used_at = now()
                        WHERE key = :key
                        RETURNING status_code, headers, body, input_tokens, output_tokens
                    """),
                        {"key": key},
                    )
                )
                .mappings()
                .first()
            )
            await session.commit()

        if row is None:
            return None

        return RecordedResponse(
            status_code=row["status_code"],
            headers=json.loads(row["headers"]),
            body=zlib.decompress(row["body"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        )

    async def save(self, key: str, model: str | None, response: RecordedResponse) -> None:
        async with self._database.get_session(master=True) as session:
            await session.execute(
                text("""
                INSERT INTO llm_replay_entries
                    (key, model, status_code, headers, body, input_tokens, output_tokens)
                VALUES (:key, :model, :status_code, :headers, :body, :input_tokens, :output_tokens)
                ON CONFLICT (key) DO UPDATE SET
                    model = EXCLUDED.model,
                    status_code = EXCLUDED.status_code,
                    headers = EXCLUDED.headers,
                    body = EXCLUDED.body,
                    input_tokens = EXCLUDED.input_tokens,
                    output_tokens = EXCLUDED.output_tokens,
                    created_at = now(),
                    last_used_at = now()
            """),
                {
                    "key": key,
                    "model": model,
                    "status_code": response.status_code,
                    "headers": json.dumps(response.headers),
                    "body": zlib.compress(response.body),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
            await session.commit()

    async def purge(self, retention_days: int) -> int:
        async with self._database.get_session(master=True) as session:
            result = await session.execute(
                text(
                    "DELETE FROM llm_replay_entries WHERE last_used_at < now() - make_interval(days => :days)"
                ),
                {"days": retention_days},
            )
            await session.commit()

        return int(getattr(result, "rowcount", 0) or 0)
