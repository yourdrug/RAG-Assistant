"""Whether a space's corpus is searchable: every document indexed in the vector store.

Chunks reach PostgreSQL in the ingestion transaction and the vector store later, through the
outbox; a document becomes DONE only after its outbox work is finished. Chunk-level checks
(such as fragment reachability) read PostgreSQL and cannot tell that search is ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.knowledge.enums import DocumentStatus


if TYPE_CHECKING:
    from application.uow import UnitOfWorkFactory


@dataclass(frozen=True)
class CorpusReadiness:
    space_id: int
    counts: dict[str, int]

    @property
    def documents(self) -> int:
        return sum(self.counts.values())

    @property
    def in_progress(self) -> int:
        return sum(count for status, count in self.counts.items() if DocumentStatus(status).is_active)

    @property
    def failed(self) -> int:
        return self.counts.get(DocumentStatus.FAILED.value, 0)

    @property
    def ready(self) -> bool:
        return self.documents > 0 and self.in_progress == 0 and self.failed == 0

    def describe(self) -> str:
        return (
            f"space {self.space_id}: documents={self.documents} in_progress={self.in_progress} "
            f"failed={self.failed}"
        )


async def corpus_readiness(uow_factory: UnitOfWorkFactory, space_id: int) -> CorpusReadiness:
    async with uow_factory.create() as uow:
        counts = await uow.knowledge_spaces.document_status_counts(space_id)
    return CorpusReadiness(space_id, counts)
