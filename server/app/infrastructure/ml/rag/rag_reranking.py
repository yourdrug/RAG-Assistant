"""RAG reranking — cross-encoder reranking, deduplication, and section grouping."""

from pathlib import Path

from application.services.retrieval import HybridRetriever
from domain.utils import content_hash
from domain.services.retrieval_diversity import prioritize_distinct_provisions
from infrastructure.ml.rag.benchmark_evidence import active_evidence, capture_exclusions
from infrastructure.ml.rag.context_selection import ExclusionReason, exclusion_record


def deduplicate_docs(docs: list) -> list:
    """Deduplicate retrieval candidates and preserve request-local diagnostics."""
    seen = set()
    selected = []
    for doc in docs:
        identity = content_hash(doc.page_content)
        if identity in seen:
            if active_evidence.get() is not None:
                capture_exclusions([exclusion_record(doc, ExclusionReason.DUPLICATE, "retrieval")])
            continue
        seen.add(identity)
        selected.append(doc)
    return selected


async def rerank_documents(
    question: str,
    docs: list,
    top_n: int,
    reranker=None,
    min_score: float | None = None,
    score_gap_ratio: float | None = None,
) -> list[tuple]:
    """Rerank, filter and retain distinct provisions within the top_n budget."""
    if not docs:
        return []

    pairs = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "")
        doc_name = filename or Path(source).name if source else ""
        prefix = f"[{doc_name}]" if doc_name else ""
        section = doc.metadata.get("heading") or doc.metadata.get("section", "")
        if section:
            prefix += f" ({section})"
        content_with_prefix = f"{prefix} {doc.page_content}" if prefix else doc.page_content
        pairs.append((question, content_with_prefix))

    scores = reranker.predict(pairs)
    if hasattr(scores, "__await__"):
        scores = await scores

    scored = sorted(zip(docs, scores, strict=True), key=lambda x: x[1], reverse=True)
    retriever = HybridRetriever()
    eligible = retriever.apply_rerank_filters(scored, min_score=min_score, score_gap_ratio=score_gap_ratio)
    ranked = prioritize_distinct_provisions(eligible)
    if active_evidence.get() is not None:
        capture_exclusions(
            [exclusion_record(item, ExclusionReason.RERANK_TOP_N, "reranker") for item in ranked[top_n:]]
        )
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


def group_by_section(docs: list[tuple]) -> list[tuple]:
    """Группирует чанки по секции, сохраняя порядок по score внутри группы."""
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for doc, score in docs:
        section = doc.metadata.get("heading") or doc.metadata.get("section") or ""
        groups[section].append((doc, score))
    for group in groups.values():
        group.sort(key=lambda x: x[1], reverse=True)
    sorted_groups = sorted(groups.values(), key=lambda g: g[0][1], reverse=True)
    return [item for group in sorted_groups for item in group]
