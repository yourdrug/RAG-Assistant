"""GeneralDomainProfile — fallback for unstructured documents.

No structural boundaries, no fingerprint, no versioning.
Uses token budgets with sentence boundaries and atomic instruction steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from domain.shared.profile_policy import AnswerPolicy


if TYPE_CHECKING:
    from domain.ingestion.profiles.types import (
        BoundaryLevel,
        ConfigDefault,
        EffectiveDateCandidate,
        ReferenceMatch,
    )


class GeneralDomainProfile:
    answer_policy = AnswerPolicy(None, fallback_to_full_corpus=False)
    query_patterns: tuple[str, ...] = ()
    query_priority = 0
    key = "general"
    display_name = "Общие документы"
    is_versioned = False
    resolution_strategy: Literal["version", "replace"] = "replace"

    def config_defaults(self) -> list[ConfigDefault]:
        return []

    def structural_fingerprint(self, _text: str) -> bool:
        return False

    def classify_score(self, _text: str) -> float:
        return 0.0

    def content_boundaries(self) -> list[BoundaryLevel]:
        return []

    def unit_label(self, _kind: str, _value: str) -> str:
        return ""

    def extract_references(self, _text: str) -> list[ReferenceMatch]:
        return []

    def extract_effective_date(self, _text: str) -> EffectiveDateCandidate | None:
        return None
