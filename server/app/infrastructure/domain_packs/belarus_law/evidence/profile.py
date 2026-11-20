"""Evidence rules of Belarusian regulatory and electronic transport documents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from domain.rag.act_references import act_focused_queries, extract_act_numbers
from infrastructure.domain_packs.belarus_law.evidence import (
    applicability,
    applicable_context,
    editions,
    field_evidence,
    focus,
    grounded_response,
    linked_context,
    predicates,
    queries,
    reading,
)
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile


if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date

    from langchain_core.documents import Document

    from domain.identity.user_context import UserContext
    from domain.knowledge.pack_metadata import MetadataCondition
    from domain.rag.context import RagContext
    from domain.rag.evidence.assessment import EvidenceAssessment
    from domain.rag.ports import ChunkRetrievalRepository


class BelarusEvidenceProfile(GenericEvidenceProfile):
    key = "belarus"

    @property
    def version(self) -> str:
        return focus.EVIDENCE_POLICY_VERSION

    @property
    @override
    def max_focus_chars(self) -> int:
        return focus.MAX_FOCUS_CHARS

    @override
    def chunk_selection(self, as_of_date: date | None) -> MetadataCondition | None:
        return editions.edition_selection(as_of_date)

    @override
    def resolve_conflicts(self, docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        return editions.resolve_edition_conflicts(docs)

    @override
    def source_fields(self, metadata: Mapping[str, Any]) -> dict[str, str]:
        return editions.edition_source_fields(metadata)

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
    def named_acts(self, question: str) -> tuple[str, ...]:
        return extract_act_numbers(question)

    @override
    def act_queries(self, question: str) -> list[str]:
        return act_focused_queries(question)

    @override
    def anchors(self, context: str, question: str) -> list[str]:
        return focus.evidence_reading_map(context, question) + focus.reading_anchors(context, question)

    @override
    def requested_provision(self, question: str, text: str) -> bool:
        if applicability.timing_requested(question) and applicability.timing_provision(text):
            return True

        classifier = applicability.approved_classifier(question)
        return classifier is not None and applicability.classifier_reference(classifier, text)

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
