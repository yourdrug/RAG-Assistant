"""Run Alembic migrations with an advisory lock.

Holds a dedicated DatabaseManager connection open for the entire migration
duration so the advisory lock is released only on explicit unlock
(or when the process dies — Postgres closes the TCP connection and
auto-releases the session-level lock).

Usage:
    python scripts/migrate.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import text
from alembic import command
from alembic.config import Config

from infrastructure.database.database import DatabaseManager

logger = logging.getLogger("default")
LOCK_ID = 727271

ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"


def main() -> None:
    database = DatabaseManager()
    with database.master_connection(
        connect_args={"connect_timeout": 10, "options": "-c statement_timeout=0"}
    ) as connection:
        connection.execute(text("SET lock_timeout = '60s'"))
        logger.info("Acquiring advisory lock...")
        connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": LOCK_ID})
        logger.info("Lock acquired")
        try:
            cfg = Config(str(ALEMBIC_INI))
            command.upgrade(cfg, "head")
            logger.info("Migrations completed.")
        except Exception:
            logger.exception("alembic upgrade head failed")
            raise
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": LOCK_ID})
            logger.info("Lock released")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
