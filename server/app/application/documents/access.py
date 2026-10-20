"""Validate RAG dependencies against primary database document ACLs."""

from application.uow import UnitOfWorkFactory
from domain.identity.user_context import UserContext
from domain.services.access_control import is_in_search_scope


class DocumentAccessService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def allowed_document_ids(self, document_ids: list[int], user: UserContext) -> set[int]:
        if not document_ids:
            return set()

        # Never use a replica or an index for this security decision.
        async with self._uow_factory.create(master=True) as uow:
            documents = await uow.documents.get_by_ids(list(set(document_ids)))
            return {doc.id for doc in documents if doc.id is not None and is_in_search_scope(doc, user)}
