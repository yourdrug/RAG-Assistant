"""Reauthorize durable deletion and retry its cache invalidation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.identity.capabilities import Capability, get_role_capabilities
from domain.shared.exceptions import EntityNotFoundError, PermissionDeniedError


if TYPE_CHECKING:
    from application.documents.commands import DocumentCommandService
    from application.ports.action_logger import ActionLoggerPort
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.uow.factory import UnitOfWorkFactory


class DocumentDeletionJob:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        commands: DocumentCommandService,
        cache: CacheInvalidatorPort,
        audit: ActionLoggerPort,
    ) -> None:
        self.uow_factory = uow_factory
        self.commands = commands
        self.cache = cache
        self.audit = audit

    async def run(self, document_id: int, user_id: int) -> None:
        factory = self.uow_factory

        async with factory.create(master=True) as uow:
            doc = await uow.documents.get_by_id(document_id)
            user = await uow.users.get_by_id(user_id)

        if doc is not None:
            if user is None or not user.is_active:
                message = "Deletion requester is no longer active"
                raise PermissionDeniedError(message)

            if Capability.DOCUMENTS_MANAGE not in get_role_capabilities(user.role):
                raise PermissionDeniedError(Capability.DOCUMENTS_MANAGE.value)

            try:
                await self.commands.delete_document(document_id, user_id, user.role, user.kind)
            except EntityNotFoundError:
                # A duplicate job may have deleted it between lookup and command.
                async with factory.create(master=True) as uow:
                    if await uow.documents.get_by_id(document_id) is not None:
                        raise

        # Also runs on retries after the database deletion committed. The vector
        # outbox provides a durable retry boundary for subsequent invalidation.
        await self.cache.invalidate_by_document_ids([document_id], raise_on_error=True)
        self.audit("document.delete", user_id=user_id, details={"document_id": document_id})
