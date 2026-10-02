"""Run Alembic migrations with an advisory lock.

Holds a synchronous psycopg connection open for the entire migration
duration so the advisory lock is released only on explicit unlock
(or when the process dies — Postgres closes the TCP connection and
auto-releases the session-level lock).

Usage:
    python scripts/migrate.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg
from alembic import command
from alembic.config import Config

from config import settings

logger = logging.getLogger("default")
LOCK_ID = 727271

ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"


def main() -> None:
    with psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname=settings.db_name,
        connect_timeout=10,
        autocommit=True,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = '60s'")
            logger.info("Acquiring advisory lock...")
            cur.execute("SELECT pg_advisory_lock(%s)", (LOCK_ID,))
            logger.info("Lock acquired")

            try:
                cfg = Config(str(ALEMBIC_INI))
                command.upgrade(cfg, "head")
                logger.info("Migrations completed.")
            except Exception:
                logger.exception("alembic upgrade head failed")
                raise
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
                logger.info("Lock released")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
