"""RAG reranking — cross-encoder reranking, deduplication, and section grouping."""

from pathlib import Path

from infrastructure.bm25.hybrid import content_hash


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

    ranked = sorted(zip(docs, scores, strict=False), key=lambda x: x[1], reverse=True)[:top_n]

    if min_score is not None:
        ranked = [(d, s) for d, s in ranked if s >= min_score]

    if score_gap_ratio is not None and ranked:
        top_score = ranked[0][1]
        cutoff = top_score * score_gap_ratio
        ranked = [(d, s) for d, s in ranked if s >= cutoff]

    return ranked


def deduplicate_docs(docs: list) -> list:
    """Remove near-duplicate chunks by content_hash to improve context diversity."""
    seen_hashes = set()
    unique_docs = []
    for doc in docs:
        h = content_hash(doc.page_content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)
    return unique_docs


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
