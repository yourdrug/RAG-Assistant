"""Act editions in the document lifecycle: the law pack's implementation of the core hooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from application.domain_packs.belarus_law.act_versioning.persistence import ActVersionPersistence
from application.domain_packs.belarus_law.act_versioning.repositories import act_repositories
from application.indexing.pack_hooks import AppliedPackDocument, PreparedPackDocument
from domain.domain_packs.belarus_law.acts.edition_metadata import edition_metadata


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.domain_packs.belarus_law.act_versioning.dto import VersioningPlan
    from application.domain_packs.belarus_law.act_versioning.service import ActVersioningService
    from application.uow import UnitOfWork
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.ingestion.profiles.registry import DomainProfileRegistry


@dataclass(frozen=True)
class PreparedEdition:
    profile: DocumentProfile
    plan: VersioningPlan


class UnexpectedPackStateError(TypeError):
    """A prepared document was not prepared by these hooks."""

    def __init__(self) -> None:
        super().__init__("The prepared document carries no act edition plan")


class ActEditionHooks:
    """Record the document's act edition with its chunks and keep timelines whole on deletion."""

    def __init__(self, service: ActVersioningService, domain_registry: DomainProfileRegistry) -> None:
        self._service = service
        self._domain_registry = domain_registry

    async def prepare(self, doc_domain: str, full_text: str) -> PreparedPackDocument | None:
        try:
            profile = self._domain_registry.get(doc_domain)
        except KeyError:
            return None

        plan = await self._service.prepare_document_versioning(profile, full_text)

        if plan is None:
            return None

        return PreparedPackDocument(plan.domain_metadata, plan.warning, PreparedEdition(profile, plan))

    async def apply(
        self, uow: UnitOfWork, document_id: int, prepared: PreparedPackDocument
    ) -> AppliedPackDocument:
        edition = prepared.state

        if not isinstance(edition, PreparedEdition):
            raise UnexpectedPackStateError

        result = await self._service.apply_versioning_plan(uow, edition.profile, document_id, edition.plan)
        return AppliedPackDocument(
            domain_metadata=result.domain_metadata,
            pack_metadata=result.pack_metadata,
            warning=result.warning,
            state=result.act_id,
        )

    async def after_commit(self, document_id: int, applied: AppliedPackDocument) -> None:
        # Answers citing any edition of the act may change; otherwise only this document's.
        if isinstance(applied.state, int):
            await self._service.invalidate_act_answers(applied.state)
        else:
            await self._service.invalidate_document_answers(document_id)

    async def chunk_metadata(self, uow: UnitOfWork, document_id: int) -> dict[str, Any]:
        # Hold the edition lock so a concurrent date correction cannot miss the new chunk.
        version = await act_repositories(uow).act_versions.get_by_document_id(document_id, for_update=True)
        return edition_metadata(version) if version is not None else {}

    async def delete_document(
        self, uow: UnitOfWork, document_id: int, delete: Callable[[], Awaitable[None]]
    ) -> None:
        """Delete the document and reconnect its act timeline in the same transaction."""
        version = await act_repositories(uow).act_versions.get_by_document_id(document_id)
        persistence = ActVersionPersistence(uow)
        timeline = None

        if version is not None and version.act_id is not None and version.id is not None:
            version, timeline = await persistence.load(version.id)

        # The cascade removes the current edition before another can be promoted.
        await delete()

        if timeline is not None and version is not None and version.id is not None:
            remaining = timeline.remove(version.id, today=datetime.now(UTC).astimezone().date())
            await persistence.save_changes(timeline, remaining.versions)
