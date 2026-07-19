"""bootstrap.py — Автоматическое создание admin при старте."""

import logging

from config import settings

logger = logging.getLogger("default")


def bootstrap_admin():
    from infrastructure.auth import hash_password
    from infrastructure.database import (
        SessionLocal,
        any_admin_exists,
        create_user,
    )

    db = SessionLocal()
    try:
        if any_admin_exists(db):
            return
        if not settings.admin_email or not settings.admin_password:
            logger.warning(
                "Нет ни одного admin, а ADMIN_EMAIL/ADMIN_PASSWORD не заданы — "
                "залогиниться будет некому. Задай их в server/.env и перезапусти."
            )
            return
        create_user(
            db,
            email=settings.admin_email,
            hashed_password=hash_password(settings.admin_password),
            role="admin",
        )
        logger.info("Создан admin: %s", settings.admin_email)
    finally:
        db.close()
