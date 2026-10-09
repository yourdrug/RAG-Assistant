"""Evidence decisions independent of retrieval and generation frameworks."""

from dataclasses import dataclass
from enum import StrEnum


class EvidenceStatus(StrEnum):
    COMPLETE = 'complete'
    PARTIAL = 'partial'
    INSUFFICIENT = 'insufficient'
    CONFLICT = 'conflict'


@dataclass(frozen=True)
class EvidenceFragment:
    text: str
    source: int
    document_id: int | str | None
    version_id: int | None
    act_number: str | None
    chunk_index: int = 0
    linked_scope: str = ''


@dataclass(frozen=True)
class EvidenceClaim:
    label: str
    quote: str
    source: int


@dataclass(frozen=True)
class EvidenceAssessment:
    status: EvidenceStatus
    claims: tuple[EvidenceClaim, ...] = ()
    missing: tuple[str, ...] = ()
