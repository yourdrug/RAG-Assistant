"""GeneralDomainProfile — fallback for unstructured documents.

No structural boundaries, no fingerprint, no versioning.
Uses token budgets with sentence boundaries and atomic instruction steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from domain.ingestion.chunking_policy import profile_policy
from domain.ingestion.tree_refinement import refine_tree
from domain.shared.profile_policy import AnswerPolicy


if TYPE_CHECKING:
    from domain.ingestion.chunking_policy import ChunkingPolicy
    from domain.ingestion.document_tree import BoundaryEventSink, DocNode
    from domain.ingestion.ports import DomainSettingsPort
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

    def refine(self, tree: DocNode, on_boundary_event: BoundaryEventSink | None = None) -> DocNode:
        return refine_tree(tree, self, on_boundary_event)

    def chunking_policy(self, settings: DomainSettingsPort | None = None) -> ChunkingPolicy:
        # Sentence overlap helps unstructured guides; structured profiles keep exact boundaries.
        return profile_policy(self.key, self.config_defaults(), settings, overlap_tokens=50)

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
