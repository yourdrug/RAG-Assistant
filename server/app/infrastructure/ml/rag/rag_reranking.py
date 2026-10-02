"""RAG reranking — cross-encoder reranking, deduplication, and section grouping."""

from pathlib import Path

from domain.utils import deduplicate_docs  # noqa: F401


async def rerank_documents(
    question: str,
    docs: list,
    top_n: int,
    reranker=None,
    min_score: float | None = None,
    score_gap_ratio: float | None = None,
) -> list[tuple]:
    """Переранжировать кандидатов кросс-энкодером и вернуть top_n лучших."""
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

    ranked = sorted(zip(docs, scores, strict=True), key=lambda x: x[1], reverse=True)[:top_n]

    from application.services.retrieval import HybridRetriever

    retriever = HybridRetriever()
    return retriever.apply_rerank_filters(ranked, min_score=min_score, score_gap_ratio=score_gap_ratio)


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
