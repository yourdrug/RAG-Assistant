"""FastAPI dependency — yields DB session with commit/rollback boundary."""

from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy.orm import Session

from infrastructure.database.engine import SessionLocal

log = logging.getLogger("default")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
