"""Public indexing results."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import NotRequired, TypedDict

from application.uow import UnitOfWork


@dataclass(frozen=True)
class DomainClassification:
    """Classification result: domain key + optional ambiguity warning."""

    domain_key: str
    warning: str | None = None
    confidence: float = 1.0
    is_ambiguous: bool = False
    candidate_scores: dict[str, float] = field(default_factory=dict)
    used_registry: bool = False

    @property
    def scores_str(self) -> str:
        return ", ".join(f"{k}={v:.1f}" for k, v in self.candidate_scores.items() if not k.startswith("__"))

    @property
    def ambiguous_candidates(self) -> str:
        return ",".join(sorted(k for k in self.candidate_scores if not k.startswith("__"))) or self.domain_key


class VersionMetadataOptions(TypedDict, total=False):
    domain_metadata: Mapping[str, object] | None
    act_version_id: int | None
    act_id: int | None
    effective_from: date | str | None
    effective_to: date | str | None
    is_current: bool


class ChunkWriteOptions(TypedDict):
    owner_id: int | None
    group_id: int | None
    doc_domain: str
    set_indexing: NotRequired[bool]


class ChunkProcessingOptions(ChunkWriteOptions):
    _existing_uow: NotRequired[UnitOfWork | None]
