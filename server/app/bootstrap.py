"""bootstrap.py — Auto-create admin on startup."""

from __future__ import annotations

import logging

from config import settings
from infrastructure.auth.password_hasher import BCryptPasswordHasher
from infrastructure.database.engine import SessionLocal
from infrastructure.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository

logger = logging.getLogger("default")


def bootstrap_admin() -> None:
    db = SessionLocal()
    try:
        user_repo = SQLAlchemyUserRepository(db)
        if user_repo.exists_admin():
            return

        if not settings.admin_email or not settings.admin_password:
            logger.warning(
                "No admin exists and ADMIN_EMAIL/ADMIN_PASSWORD not set — "
                "you won't be able to log in. Set them in server/.env and restart."
            )
            return

        hasher = BCryptPasswordHasher()
        from domain.entities.user import User
        from domain.value_objects.roles import UserKind, UserRole

        user = User(
            email=settings.admin_email,
            hashed_password=hasher.hash(settings.admin_password),
            role=UserRole.ADMIN,
            kind=UserKind.INTERNAL,
        )
        user_repo.save(user)
        db.commit()
        logger.info("Admin created: %s", settings.admin_email)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
