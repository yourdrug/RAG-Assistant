"""Process each document with the rules of its space's pack."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.spaces.ports import DomainPackNotLoadedError
from application.spaces.resolution import document_space


if TYPE_CHECKING:
    from collections.abc import Mapping

    from application.documents.dto import ProcessDocumentCommand
    from application.documents.reindex import DocumentReprocessor
    from application.uow import UnitOfWorkFactory

log = logging.getLogger("default")


class SpaceDocumentProcessor:
    """Choose the document processor of the pack of the document's space.

    Classification, splitting and pack hooks then run only with that pack's profiles.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory, processors: Mapping[str, DocumentReprocessor]) -> None:
        self._uow_factory = uow_factory
        self._processors = dict(processors)

    async def process(self, command: ProcessDocumentCommand) -> None:
        async with self._uow_factory.create() as uow:
            document = await uow.documents.get_by_id(command.document_id)

            if document is None:
                log.info("Document %d already deleted; skipping processing", command.document_id)
                return

            space = await document_space(uow, document)

        processor = self._processors.get(space.pack_key)

        if processor is None:
            raise DomainPackNotLoadedError(space.pack_key)

        await processor.process(command)
