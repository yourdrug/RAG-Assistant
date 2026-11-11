"""RAG reranking — cross-encoder reranking, deduplication, and section grouping."""

from collections import defaultdict
from collections.abc import Awaitable
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from application.search import HybridRetriever
from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.ranking import (
    RetrievalDocument,
    evidence_identity,
    prioritize_distinct_provisions,
    prioritize_referenced_acts,
)
from infrastructure.ml.evidence_profiles.profiles import active_evidence_profile
from infrastructure.ml.rag.benchmark_evidence import active_evidence, capture_exclusions, capture_stage
from infrastructure.ml.rag.context_selection import ExclusionReason, exclusion_record


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float] | Awaitable[list[float]]: ...


def deduplicate_docs[DocumentT: RetrievalDocument](docs: list[DocumentT]) -> list[DocumentT]:
    """Deduplicate retrieval candidates and preserve request-local diagnostics."""
    seen = set()
    selected: list[DocumentT] = []

    for doc in docs:
        identity = evidence_identity(doc)

        if identity in seen:
            if active_evidence.get() is not None:
                capture_exclusions([exclusion_record(doc, ExclusionReason.DUPLICATE, "retrieval")])

            continue

        seen.add(identity)
        selected.append(doc)

    return selected


def filter_scope_candidates[ItemT: RetrievalDocument | tuple[RetrievalDocument, float]](
    docs: list[ItemT], question: str, stage: str = "reranker"
) -> list[ItemT]:
    compatible = []

    for item in docs:
        doc = item[0] if isinstance(item, tuple) else item

        if active_evidence_profile.get().contradicts_scope(question, content_with_parent_scope(doc)):
            if active_evidence.get() is not None:
                capture_exclusions([exclusion_record(item, ExclusionReason.SCOPE_MISMATCH, stage)])
        else:
            compatible.append(item)

    return compatible


async def rerank_documents(
    question: str,
    docs: list[Document],
    top_n: int,
    reranker: Reranker | None = None,
    min_score: float | None = None,
    *,
    score_gap_ratio: float | None = None,
) -> list[tuple[Document, float]]:
    """Rerank, filter and retain distinct provisions within the top_n budget."""
    docs = filter_scope_candidates(docs, question)

    if not docs:
        capture_stage("reranked", [], score_kind="reranker")
        return []

    pairs = [(question, reranker_text(doc)) for doc in docs]

    if reranker is None:
        message = "A reranker client is required"
        raise RuntimeError(message)

    prediction = reranker.predict(pairs)
    scores = prediction if isinstance(prediction, list) else await prediction

    scored = sorted(zip(docs, scores, strict=True), key=lambda x: x[1], reverse=True)
    retriever = HybridRetriever()
    eligible = retriever.apply_rerank_filters(scored, min_score=min_score, score_gap_ratio=score_gap_ratio)
    ranked = prioritize_referenced_acts(
        prioritize_distinct_provisions(eligible), active_evidence_profile.get().named_acts(question), top_n
    )

    if active_evidence.get() is not None:
        capture_exclusions(
            [exclusion_record(item, ExclusionReason.RERANK_TOP_N, "reranker") for item in ranked[top_n:]]
        )

    capture_stage("reranked", ranked, score_kind="reranker")
    ranked = ranked[:top_n]

    selected = ranked

    if active_evidence.get() is not None:
        eligible_ids = {id(doc) for doc, _ in eligible}
        capture_exclusions(
            [
                exclusion_record(
                    item,
                    ExclusionReason.RERANK_THRESHOLD,
                    "reranker",
                    min_score=min_score,
                    score_gap_ratio=score_gap_ratio,
                )
                for item in scored
                if id(item[0]) not in eligible_ids
            ]
        )

    return selected


def group_by_section(docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """Группирует чанки по секции, сохраняя порядок по score внутри группы."""
    groups: dict[str, list[tuple[Document, float]]] = defaultdict(list)

    for doc, score in docs:
        section = doc.metadata.get("heading") or doc.metadata.get("section") or ""
        groups[section].append((doc, score))

    for group in groups.values():
        group.sort(key=lambda x: x[1], reverse=True)

    sorted_groups = sorted(groups.values(), key=lambda g: g[0][1], reverse=True)
    return [item for group in sorted_groups for item in group]


def reranker_text(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    filename = doc.metadata.get("filename", "")
    doc_name = filename or Path(source).name if source else ""
    prefix = f"[{doc_name}]" if doc_name else ""
    section = doc.metadata.get("heading") or doc.metadata.get("section", "")

    if section:
        prefix += f" ({section})"

    content = content_with_parent_scope(doc)
    return f"{prefix} {content}" if prefix else content
