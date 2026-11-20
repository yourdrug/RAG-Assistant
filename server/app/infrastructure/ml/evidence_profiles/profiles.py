"""Request-scoped evidence rules; a domain pack supplies its own profile."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol, override


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from datetime import date

    from langchain_core.documents import Document

    from domain.identity.user_context import UserContext
    from domain.knowledge.pack_metadata import MetadataCondition
    from domain.rag.context import RagContext
    from domain.rag.evidence.assessment import EvidenceAssessment
    from domain.rag.ports import ChunkRetrievalRepository


# Neutral budget for exact reading excerpts appended to the generation prompt.
DEFAULT_FOCUS_CHARS = 3000


class EvidenceProfileNotSelectedError(RuntimeError):
    """Evidence rules were requested outside an explicit profile scope."""

    def __init__(self) -> None:
        super().__init__("Evidence profile is not selected; wrap the call in evidence_profile_scope()")


class EvidenceProfile(Protocol):
    key: str

    @property
    def version(self) -> str: ...

    @property
    def max_focus_chars(self) -> int: ...

    def chunk_selection(self, as_of_date: date | None) -> MetadataCondition | None:
        """Pack selection applied with ACL before top-k in every store; ``None`` keeps all."""
        ...

    def resolve_conflicts(self, docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        """Drop candidates superseded by another retrieved candidate (e.g. older editions)."""
        ...

    def source_fields(self, metadata: Mapping[str, Any]) -> dict[str, str]:
        """Pack fields shown with a cited source."""
        ...

    def reading_rules(self, question: str) -> list[str]: ...

    def reading_reminder(self, question: str) -> str: ...

    def search_queries(self, question: str) -> list[str]: ...

    def named_acts(self, question: str) -> tuple[str, ...]: ...

    def act_queries(self, question: str) -> list[str]: ...

    def anchors(self, context: str, question: str) -> list[str]: ...

    def contradicts_scope(self, question: str, text: str) -> bool: ...

    def requested_provision(self, question: str, text: str) -> bool:
        """Whether the text states the kind of provision the question asks for.

        Such reranked evidence is kept beyond the top_n window, never added.
        """
        ...

    def scoped_response(
        self, docs: list[tuple[Document, float]], context: str, question: str
    ) -> tuple[str | None, EvidenceAssessment | None]: ...

    async def augment_fields(
        self,
        question: str,
        candidates: list[Document],
        chunk_search: ChunkRetrievalRepository | None,
        ctx: RagContext,
    ) -> list[Document]: ...

    async def enrich_links(
        self,
        docs: list[tuple[Document, float]],
        question: str,
        chunk_search: ChunkRetrievalRepository | None,
        user: UserContext,
    ) -> list[tuple[Document, float]]: ...

    async def enrich_scope(
        self,
        docs: list[tuple[Document, float]],
        question: str,
        chunk_search: ChunkRetrievalRepository | None,
        user: UserContext,
    ) -> list[tuple[Document, float]]: ...


class GenericEvidenceProfile(EvidenceProfile):
    """Generic corpus policy: use retrieved evidence without corpus-specific inference."""

    key = "generic"

    @property
    def version(self) -> str:
        # Bumped when retrieval changes what answers are built from (topic document, outline notes).
        return "2"

    @property
    def max_focus_chars(self) -> int:
        return DEFAULT_FOCUS_CHARS

    @override
    def chunk_selection(self, as_of_date: date | None) -> MetadataCondition | None:
        return None

    @override
    def resolve_conflicts(self, docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        return docs

    @override
    def source_fields(self, metadata: Mapping[str, Any]) -> dict[str, str]:
        return {}

    @override
    def reading_rules(self, question: str) -> list[str]:
        return []

    @override
    def reading_reminder(self, question: str) -> str:
        return ""

    @override
    def search_queries(self, question: str) -> list[str]:
        return [question]

    @override
    def named_acts(self, question: str) -> tuple[str, ...]:
        return ()

    @override
    def act_queries(self, question: str) -> list[str]:
        return []

    @override
    def anchors(self, context: str, question: str) -> list[str]:
        return []

    @override
    def requested_provision(self, question: str, text: str) -> bool:
        return False

    @override
    def contradicts_scope(self, question: str, text: str) -> bool:
        return False

    @override
    def scoped_response(
        self, docs: list[tuple[Document, float]], context: str, question: str
    ) -> tuple[str | None, EvidenceAssessment | None]:
        return None, None

    @override
    async def augment_fields(
        self,
        question: str,
        candidates: list[Document],
        chunk_search: ChunkRetrievalRepository | None,
        ctx: RagContext,
    ) -> list[Document]:
        return candidates

    @override
    async def enrich_links(
        self,
        docs: list[tuple[Document, float]],
        question: str,
        chunk_search: ChunkRetrievalRepository | None,
        user: UserContext,
    ) -> list[tuple[Document, float]]:
        return docs

    @override
    async def enrich_scope(
        self,
        docs: list[tuple[Document, float]],
        question: str,
        chunk_search: ChunkRetrievalRepository | None,
        user: UserContext,
    ) -> list[tuple[Document, float]]:
        return docs


# No default: live requests select the configured profile, standalone tools pass one explicitly.
active_evidence_profile: ContextVar[EvidenceProfile] = ContextVar("evidence_profile")


def current_evidence_profile() -> EvidenceProfile:
    try:
        return active_evidence_profile.get()
    except LookupError as exc:
        raise EvidenceProfileNotSelectedError from exc


@contextmanager
def evidence_profile_scope(profile: EvidenceProfile) -> Iterator[None]:
    token = active_evidence_profile.set(profile)
    try:
        yield
    finally:
        active_evidence_profile.reset(token)
