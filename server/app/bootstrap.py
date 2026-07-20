"""bootstrap.py — Auto-create admin on startup."""

import logging

from config import settings
from infrastructure.auth import hash_password
from infrastructure.database import (
    SessionLocal,
    any_admin_exists,
    create_user,
)

logger = logging.getLogger("default")


def bootstrap_admin():
    db = SessionLocal()
    try:
        if any_admin_exists(db):
            return
        if not settings.admin_email or not settings.admin_password:
            logger.warning(
                "No admin exists and ADMIN_EMAIL/ADMIN_PASSWORD not set — "
                "you won't be able to log in. Set them in server/.env and restart."
            )
            return
        create_user(
            db,
            email=settings.admin_email,
            hashed_password=hash_password(settings.admin_password),
            role="admin",
        )
        logger.info("Admin created: %s", settings.admin_email)
    finally:
        db.close()
