"""Domain pack hooks into the document lifecycle; the core calls them without interpreting results.

A pack may analyse a document before the persistence transaction, record its own state in that
transaction (for example an act edition), attach pack metadata to the document's chunks and
take part in deletion. Without a pack the core indexes documents with no pack metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.uow import UnitOfWork
    from domain.knowledge.document import Document


type DomainMetadata = dict[str, str | list[str]]


@dataclass(frozen=True)
class PreparedPackDocument:
    """Pack analysis of a document's text, computed before the persistence transaction."""

    domain_metadata: DomainMetadata | None = None
    warning: str | None = None
    # Pack-private, passed back to ``DocumentPackHooks.apply``.
    state: object | None = None


@dataclass(frozen=True)
class AppliedPackDocument:
    """What the pack recorded for a document inside the persistence transaction."""

    domain_metadata: DomainMetadata | None = None
    # Copied to every chunk of the document; see ``domain.knowledge.pack_metadata``.
    pack_metadata: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None
    # Pack-private, passed to ``DocumentPackHooks.after_commit``.
    state: object | None = None


class DocumentPackHooks(Protocol):
    async def prepare(self, doc_domain: str, full_text: str) -> PreparedPackDocument | None:
        """Analyse the text outside a transaction; ``None`` when the pack has nothing to record."""
        ...

    async def apply(
        self, uow: UnitOfWork, document_id: int, prepared: PreparedPackDocument
    ) -> AppliedPackDocument:
        """Record pack state in the caller's transaction, before the chunks are written."""
        ...

    async def after_commit(self, document_id: int, applied: AppliedPackDocument) -> None:
        """Run effects that must follow the commit (for example answer cache invalidation)."""
        ...

    async def chunk_metadata(self, uow: UnitOfWork, document_id: int) -> dict[str, Any]:
        """Pack metadata of a chunk added to the document, read under the pack's locks."""
        ...

    async def delete_document(
        self, uow: UnitOfWork, document_id: int, delete: Callable[[], Awaitable[None]]
    ) -> None:
        """Delete the document by calling ``delete`` and keep the pack's state consistent."""
        ...


class DocumentHooksResolver(Protocol):
    """Hooks of the pack that owns a document (the pack of the document's space)."""

    async def for_document(self, uow: UnitOfWork, document: Document) -> DocumentPackHooks | None: ...


@dataclass(frozen=True)
class FixedPackHooks:
    """One pack's hooks for every document: a single-pack scenario or test."""

    hooks: DocumentPackHooks | None

    async def for_document(self, uow: UnitOfWork, document: Document) -> DocumentPackHooks | None:
        del uow, document
        return self.hooks


async def apply_pack_document(
    hooks: DocumentPackHooks | None, uow: UnitOfWork, document_id: int, prepared: PreparedPackDocument | None
) -> AppliedPackDocument:
    if prepared is None:
        return AppliedPackDocument()

    if hooks is None:
        msg = "A prepared pack document requires the pack's document hooks"
        raise RuntimeError(msg)

    return await hooks.apply(uow, document_id, prepared)
