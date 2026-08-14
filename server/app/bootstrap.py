"""bootstrap.py — Auto-create admin on startup (async, KinTree-style)."""

from __future__ import annotations

import logging

from application.ports.unit_of_work_factory import UnitOfWorkFactory
from config import settings
from domain.entities.user import User
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.auth.password_hasher import BCryptPasswordHasher

logger = logging.getLogger("default")


async def bootstrap_admin(uow_factory: UnitOfWorkFactory) -> None:
    async with uow_factory.create(master=True) as uow:
        if await uow.users.exists_admin():
            return

        if not settings.admin_email or not settings.admin_password:
            logger.warning(
                "No admin exists and ADMIN_EMAIL/ADMIN_PASSWORD not set — "
                "you won't be able to log in. Set them in server/.env and restart."
            )
            return

        hasher = BCryptPasswordHasher()
        user = User(
            email=settings.admin_email,
            hashed_password=hasher.hash(settings.admin_password),
            role=UserRole.ADMIN,
            kind=UserKind.INTERNAL,
        )
        await uow.users.save(user)
        logger.info("Admin created: %s", settings.admin_email)
