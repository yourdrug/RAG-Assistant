"""Pure policies for retrieval limits, hybrid weights and document context budgets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict, Unpack

from domain.knowledge.chunk import SENTENCE_UNIT_KIND
from domain.rag.breadth import Breadth
from domain.rag.classification import has_exact_reference
from domain.shared.exceptions import ValidationError
from domain.shared.hashing import content_hash


if TYPE_CHECKING:
    from collections.abc import Sequence

    from domain.rag.settings import RagSettings


class RetrievalDocument(Protocol):
    """Read-only content and metadata required by retrieval policies."""

    @property
    def page_content(self) -> str: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


class RetrievalParams(TypedDict):
    """Candidate limits and weights consumed by the retrieval pipeline."""

    fetch_k: int
    rerank_top_n: int
    effective_dense_weight: float
    effective_sparse_weight: float
    use_exact_ref_boost: bool


def compute_retrieval_params(
    breadth: Breadth,
    rag_settings: RagSettings,
    query: str,
    exact_ref_sparse_boost: float = 1.0,
) -> RetrievalParams:
    """Compute retrieval parameters based on breadth and settings.

    ``rerank_top_n`` always uses the broad candidate limit; final document
    selection is handled separately by ``select_final_top_k``. Only the
    sparse weight is boosted for exact structural references.

    Pure function — no infrastructure dependencies.
    """
    use_exact_ref_boost = has_exact_reference(query)

    fetch_k = (
        rag_settings.retriever.fetch_k_broad if breadth == Breadth.BROAD else rag_settings.retriever.fetch_k
    )
    rerank_top_n = rag_settings.retriever.top_k_broad

    effective_dense_weight = rag_settings.hybrid_search.dense_weight
    effective_sparse_weight = rag_settings.hybrid_search.sparse_weight

    if use_exact_ref_boost:
        effective_sparse_weight = rag_settings.hybrid_search.sparse_weight * exact_ref_sparse_boost

    return {
        "fetch_k": fetch_k,
        "rerank_top_n": rerank_top_n,
        "effective_dense_weight": effective_dense_weight,
        "effective_sparse_weight": effective_sparse_weight,
        "use_exact_ref_boost": use_exact_ref_boost,
    }


class ContextBudgetOptions(TypedDict, total=False):
    chars_per_token: int
    reserved_overhead: int
    min_context_tokens: int
    num_predict_narrow: int
    num_predict_broad: int


def compute_context_budget(
    breadth: Breadth,
    *,
    enumerate_cases: bool,
    history_chars: int,
    question_chars: int,
    num_ctx_narrow: int,
    num_ctx_broad: int,
    **options: Unpack[ContextBudgetOptions],
) -> int:
    """Compute maximum context tokens available for retrieved documents.

    Pure function — no infrastructure dependencies.
    """
    chars_per_token = options.get("chars_per_token", 4)
    reserved_overhead = options.get("reserved_overhead", 3000)
    min_context_tokens = options.get("min_context_tokens", 1000)
    num_predict_narrow = options.get("num_predict_narrow", 400)
    num_predict_broad = options.get("num_predict_broad", 2048)
    effective_breadth = Breadth.BROAD if enumerate_cases else breadth
    num_ctx = num_ctx_broad if effective_breadth == Breadth.BROAD else num_ctx_narrow
    num_predict = num_predict_broad if effective_breadth == Breadth.BROAD else num_predict_narrow
    reserved_chars = history_chars + question_chars + reserved_overhead
    reserved_for_system_and_history = max(reserved_chars // chars_per_token, 1500)
    return max(num_ctx - num_predict - reserved_for_system_and_history, min_context_tokens)


def select_final_top_k(
    breadth: Breadth,
    *,
    enumerate_cases: bool,
    rag_settings: RagSettings,
) -> int:
    """Select the final number of documents to keep after reranking.

    Pure function — no infrastructure dependencies.
    """
    if not enumerate_cases:
        return (
            rag_settings.retriever.top_k if breadth == Breadth.NARROW else rag_settings.retriever.top_k_broad
        )

    return rag_settings.retriever.top_k_broad


def prioritize_distinct_provisions[ItemT: RetrievalDocument | tuple[RetrievalDocument, float]](
    items: list[ItemT],
) -> list[ItemT]:
    """Stable first pass over provisions, followed by their remaining chunks.

    Accept documents or (document, score) pairs. Sentence splits share their
    parent's provision. Unstructured chunks and tables retain their own slots.
    Scope includes the document/version so unrelated provisions never collapse.
    No text is removed: remaining chunks can fill any unused budget.
    """
    seen = set()
    first: list[ItemT] = []
    remaining: list[ItemT] = []

    for item in items:
        doc = item[0] if isinstance(item, tuple) else item
        metadata = doc.metadata
        section = metadata.get("section") or metadata.get("heading")
        source = metadata.get("document_id") or metadata.get("source")

        if not section or not source or metadata.get("table_id"):
            first.append(item)
            continue

        parts = section.replace(" \u203a ", " > ").split(" > ")

        if parts[-1] == SENTENCE_UNIT_KIND:
            parts = parts[:-1]

        if not parts:
            first.append(item)
            continue

        key = (source, metadata.get("act_version_id"), tuple(parts))

        if key in seen:
            remaining.append(item)
        else:
            seen.add(key)
            first.append(item)

    return first + remaining


class ContextBudgetExceededError(ValidationError):
    """No complete retrieved chunk and its conditions fit the context window."""

    code = "context_budget_exceeded"

    def __init__(self) -> None:
        super().__init__(
            "Полный контекст найденных документов не помещается в лимит ответа. "
            "Уточните вопрос или выберите конкретный пункт документа.",
            errors={"code": self.code},
        )


def compute_reranker_score(sources: Sequence[object]) -> float | None:
    """Extract the maximum reranker score from a list of source dicts."""
    if not sources:
        return None

    scores = [s.get("max_score", 0) for s in sources if isinstance(s, dict)]
    return max(scores) if scores else None


def rrf_merge(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[str]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Takes (content_hash, score) pairs from each source.
    Returns merged list of content hashes sorted by descending RRF score.
    k=60 is the standard constant from the original RRF paper.
    """
    rrf_scores: dict[str, float] = {}

    for rank, (h, _score) in enumerate(dense_results):
        rrf_scores[h] = rrf_scores.get(h, 0.0) + dense_weight / (k + rank + 1)

    for rank, (h, _score) in enumerate(sparse_results):
        rrf_scores[h] = rrf_scores.get(h, 0.0) + sparse_weight / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [h for h, _score in merged]


def deduplicate_docs[DocumentT: RetrievalDocument](docs: list[DocumentT]) -> list[DocumentT]:
    """Remove near-duplicate chunks by content_hash to improve context diversity."""
    seen_hashes: set[str] = set()
    unique_docs: list[DocumentT] = []

    for doc in docs:
        h = content_hash(doc.page_content)

        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)

    return unique_docs
