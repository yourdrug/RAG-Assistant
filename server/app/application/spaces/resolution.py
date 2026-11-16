"""Resolve the knowledge space of a request: explicit, or the only one that exists."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.shared.exceptions import EntityNotFoundError, ValidationError


if TYPE_CHECKING:
    from application.indexing.pack_hooks import DocumentPackHooks
    from application.spaces.ports import DomainPackCatalog
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.knowledge.document import Document
    from domain.knowledge.space import KnowledgeSpace, SpaceRef


SPACE_ENTITY = "KnowledgeSpace"


async def resolve_space(uow: UnitOfWork, space_id: int | None) -> KnowledgeSpace:
    """An explicit space must exist; without one, the only space of the installation is used."""
    if space_id is not None:
        space = await uow.knowledge_spaces.get_by_id(space_id)

        if space is None:
            raise EntityNotFoundError(SPACE_ENTITY, space_id)

        return space

    spaces = await uow.knowledge_spaces.list_all()

    if len(spaces) == 1:
        return spaces[0]

    msg = "space_id is required: choose a knowledge space" if spaces else "No knowledge space exists"
    raise ValidationError(msg, field="space_id")


async def default_space(uow_factory: UnitOfWorkFactory) -> KnowledgeSpace:
    """Space of a standalone run without one (benchmarks, sweeps): the only space."""
    async with uow_factory.create() as uow:
        return await resolve_space(uow, None)


async def default_space_ref(uow_factory: UnitOfWorkFactory) -> SpaceRef:
    return (await default_space(uow_factory)).ref


async def document_space(uow: UnitOfWork, document: Document) -> KnowledgeSpace:
    if document.space_id is None:
        msg = f"Document {document.id} has no knowledge space"
        raise RuntimeError(msg)

    space = await uow.knowledge_spaces.get_by_id(document.space_id)

    if space is None:
        raise EntityNotFoundError(SPACE_ENTITY, document.space_id)

    return space


class SpacePackHooks:
    """Pack hooks of a document chosen by the pack of the document's space."""

    def __init__(self, catalog: DomainPackCatalog) -> None:
        self._catalog = catalog

    async def for_document(self, uow: UnitOfWork, document: Document) -> DocumentPackHooks | None:
        space = await document_space(uow, document)
        return self._catalog.document_hooks(space.pack_key)
