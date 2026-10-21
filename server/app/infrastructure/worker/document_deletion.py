"""Durable document deletion; permissions are checked again at execution time."""

from domain.identity.capabilities import Capability, get_role_capabilities
from domain.shared.exceptions import EntityNotFoundError, PermissionDeniedError
from infrastructure.worker.runtime import job


@job()
async def delete_document(ctx: dict, *, document_id: int, user_id: int, job_id: int) -> None:  # noqa: ARG001 — consumed by @job
    container = ctx["container"]
    infra = container.infrastructure
    factory = infra.db.uow_factory

    async with factory.create(master=True) as uow:
        doc = await uow.documents.get_by_id(document_id)
        user = await uow.users.get_by_id(user_id)
    if doc is not None:
        if user is None or not user.is_active:
            raise PermissionDeniedError("Deletion requester is no longer active")
        if Capability.DOCUMENTS_MANAGE not in get_role_capabilities(user.role):
            raise PermissionDeniedError(Capability.DOCUMENTS_MANAGE.value)
        try:
            await container.application.document_command_service.delete_document(
                document_id, user_id, user.role, user.kind
            )
        except EntityNotFoundError:
            # A duplicate job may have deleted it between lookup and command.
            async with factory.create(master=True) as uow:
                if await uow.documents.get_by_id(document_id) is not None:
                    raise
    # Also runs on retries after the database deletion committed. The vector
    # outbox provides a durable retry boundary for subsequent invalidation.
    await infra.services.cache_invalidator.invalidate_by_document_ids([document_id], raise_on_error=True)
    infra.services.action_logger("document.delete", user_id=user_id, details={"document_id": document_id})
