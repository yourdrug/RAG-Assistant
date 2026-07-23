"""bootstrap.py — Auto-create admin on startup."""

from __future__ import annotations

import logging

from config import settings
from infrastructure.auth.password_hasher import BCryptPasswordHasher
from infrastructure.uow_factory import UnitOfWorkFactory

logger = logging.getLogger("default")


def bootstrap_admin() -> None:
    factory = UnitOfWorkFactory()
    with factory.create() as uow:
        if uow.users.exists_admin():
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
        uow.users.save(user)
        logger.info("Admin created: %s", settings.admin_email)
