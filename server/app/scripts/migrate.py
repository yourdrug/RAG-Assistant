"""Run Alembic migrations with an advisory lock.

Holds a synchronous psycopg connection open for the entire migration
duration so the advisory lock is released only on explicit unlock
(or when the process dies — Postgres closes the TCP connection and
auto-releases the session-level lock).

Usage:
    python scripts/migrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg
from alembic import command
from alembic.config import Config

from config import settings

LOCK_ID = 727271

ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"


def main() -> None:
    dsn = (
        f"postgresql://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            print("Acquiring advisory lock...")
            cur.execute("SELECT pg_advisory_lock(%s)", (LOCK_ID,))
            print("Lock acquired")

            try:
                cfg = Config(str(ALEMBIC_INI))
                command.upgrade(cfg, "head")
                print("Migrations completed.")
            except Exception:
                print("alembic upgrade head failed", file=sys.stderr)
                raise
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
                print("Lock released")


if __name__ == "__main__":
    main()
