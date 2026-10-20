"""GeneralDomainProfile — fallback for unstructured documents.

No structural boundaries, no fingerprint, no versioning.
Uses char-based splitting (existing RecursiveCharacterTextSplitter).
"""

from __future__ import annotations

from domain.ingestion.profiles.types import (
    ConfigDefault,
    EffectiveDateCandidate,
    ReferenceMatch,
)


class GeneralDomainProfile:
    key = "general"
    display_name = "Общие документы"
    is_versioned = False

    def config_defaults(self) -> list[ConfigDefault]:
        return []

    def structural_fingerprint(self, text: str) -> bool:
        return False

    def classify_score(self, text: str) -> float:
        return 0.0

    def content_boundaries(self) -> list:
        return []

    def extract_references(self, text: str) -> list[ReferenceMatch]:
        return []

    def extract_effective_date(self, text: str) -> EffectiveDateCandidate | None:
        return None
