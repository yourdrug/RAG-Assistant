"""Request-scoped evidence plugins; a new corpus can select or supply its own policy."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol, override

from infrastructure.ml.evidence_profiles.belarus import (
    applicable_context,
    field_evidence,
    focus,
    grounded_response,
    linked_context,
    predicates,
    queries,
    reading,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.documents import Document

    from domain.identity.user_context import UserContext
    from domain.rag.context import RagContext
    from domain.rag.evidence.assessment import EvidenceAssessment
    from domain.rag.ports import ChunkRetrievalRepository


class EvidenceProfile(Protocol):
    key: str

    @property
    def version(self) -> str: ...

    def reading_rules(self, question: str) -> list[str]: ...

    def reading_reminder(self, question: str) -> str: ...

    def search_queries(self, question: str) -> list[str]: ...

    def anchors(self, context: str, question: str) -> list[str]: ...

    def contradicts_scope(self, question: str, text: str) -> bool: ...

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
        return "1"

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
    def anchors(self, context: str, question: str) -> list[str]:
        return []

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


class BelarusEvidenceProfile(GenericEvidenceProfile):
    key = "belarus"

    @property
    def version(self) -> str:
        return focus.EVIDENCE_POLICY_VERSION

    @override
    def reading_rules(self, question: str) -> list[str]:
        return reading.evidence_reading_rules(question)

    @override
    def reading_reminder(self, question: str) -> str:
        return reading.evidence_reading_reminder(question)

    @override
    def search_queries(self, question: str) -> list[str]:
        return queries.evidence_search_queries(question)

    @override
    def anchors(self, context: str, question: str) -> list[str]:
        return focus.evidence_reading_map(context, question) + focus.reading_anchors(context, question)

    @override
    def contradicts_scope(self, question: str, text: str) -> bool:
        return predicates.contradicts_requested_scope(question, text)

    @override
    def scoped_response(
        self, docs: list[tuple[Document, float]], context: str, question: str
    ) -> tuple[str | None, EvidenceAssessment | None]:
        return grounded_response.scoped_evidence_response(docs, context, question)

    @override
    async def augment_fields(
        self,
        question: str,
        candidates: list[Document],
        chunk_search: ChunkRetrievalRepository | None,
        ctx: RagContext,
    ) -> list[Document]:
        return await field_evidence.augment_field_candidates(question, candidates, chunk_search, ctx)

    @override
    async def enrich_links(
        self,
        docs: list[tuple[Document, float]],
        question: str,
        chunk_search: ChunkRetrievalRepository | None,
        user: UserContext,
    ) -> list[tuple[Document, float]]:
        return await linked_context.enrich_linked_context(docs, question, chunk_search, user)

    @override
    async def enrich_scope(
        self,
        docs: list[tuple[Document, float]],
        question: str,
        chunk_search: ChunkRetrievalRepository | None,
        user: UserContext,
    ) -> list[tuple[Document, float]]:
        return await applicable_context.enrich_applicable_context(docs, question, chunk_search, user)


EVIDENCE_PROFILES: dict[str, EvidenceProfile] = {
    profile.key: profile for profile in (GenericEvidenceProfile(), BelarusEvidenceProfile())
}
# Preserve the existing corpus for standalone tools; live requests explicitly select a profile.
active_evidence_profile: ContextVar[EvidenceProfile] = ContextVar(
    "evidence_profile",
    default=EVIDENCE_PROFILES["belarus"],
)


@contextmanager
def evidence_profile_scope(profile: EvidenceProfile) -> Iterator[None]:
    token = active_evidence_profile.set(profile)
    try:
        yield
    finally:
        active_evidence_profile.reset(token)
