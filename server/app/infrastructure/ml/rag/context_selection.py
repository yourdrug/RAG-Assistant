"""Budget selection independent of document/section presentation order."""

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

from langchain_core.documents import Document

from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.guardrails.sanitization import sanitize_for_prompt
from domain.rag.ranking import (
    CONTINUATION_SCORE_RATIO,
    MAX_PROVISION_CHUNKS,
    RetrievalDocument,
    evidence_identity,
    same_evidence_block,
)
from infrastructure.bm25.tokenizer import tokenize
from infrastructure.ml.rag.benchmark_evidence import snapshot_documents
from infrastructure.ml.rag.context_rendering import render_selected_docs


class ExclusionReason(StrEnum):
    BUDGET = "budget"
    RERANK_THRESHOLD = "reranker_threshold"
    DUPLICATE = "duplicate"
    OVERSIZED_BLOCK = "oversized_block"
    RERANK_TOP_N = "reranker_top_n"
    FINAL_TOP_K = "final_top_k"
    SCOPE_MISMATCH = "scope_mismatch"
    TEMPORAL_CONFLICT = "temporal_conflict"


# At most 15% of the relevance score range can be traded for new query coverage.
CONTEXT_COVERAGE_WEIGHT = 0.15


def exclusion_record(
    item: RetrievalDocument | tuple[RetrievalDocument, float],
    reason: ExclusionReason,
    stage: str,
    **details: object,
) -> dict[str, Any]:

    doc, score = item if isinstance(item, tuple) else (item, 0.0)
    return {
        **snapshot_documents([(doc, score)])[0],
        "score": float(score) if isinstance(item, tuple) else None,
        "reason": reason.value,
        "stage": stage,
        **details,
    }


def select_context[ItemT: Document | tuple[Document, float]](
    docs: Sequence[ItemT],
    budget: int,
    count_tokens: Callable[[str], int],
    query_parts: list[str],
    exclusions: list[dict[str, Any]] | None = None,
) -> tuple[str, list[ItemT]]:
    """Greedily pack relevance plus marginal lexical query-part coverage.

    Always try the highest reranker score first. Later candidates can receive a
    bounded bonus for query terms absent from the accepted context. Coverage is
    averaged per sub-query so a long question part cannot drown a short one.
    This is a lexical heuristic, not a claim of semantic evidence coverage.
    """
    if budget <= 0:
        message = "max_context_tokens must be positive"
        raise ValueError(message)

    remaining, parts = context_candidates(docs, query_parts, exclusions)
    covered: set[str] = set()
    score_range = max((entry[2] for entry in remaining), default=0.0) - min(
        (entry[2] for entry in remaining), default=0.0
    )
    # Equal scores still allow coverage to break ties, without altering relevance.
    coverage_weight = CONTEXT_COVERAGE_WEIGHT * score_range
    chosen: list[ItemT] = []
    context = ""
    selected: list[ItemT] = []

    while remaining:

        def priority(entry: tuple[int, ItemT, float, set[str]]) -> tuple[float, float, int]:
            index, _, score, terms = entry
            gain = (
                sum(len((terms & part) - covered) / len(part) for part in parts) / len(parts) if parts else 0
            )
            bonus = coverage_weight * gain if chosen else 0.0
            return score + bonus, gain if chosen else 0, -index

        entry = max(remaining, key=priority)
        remaining.remove(entry)
        _, item, _, terms = entry
        companions = evidence_companions(item, remaining, budget, count_tokens, chosen)
        trial_context, trial_selected = render_selected_docs(
            [*chosen, item, *(other[1] for other in companions)]
        )
        trial_tokens = count_tokens(trial_context)

        if trial_tokens <= budget:
            chosen.extend([item, *(other[1] for other in companions)])
            covered.update(terms, *(companion[3] for companion in companions))
            remaining[:] = [candidate for candidate in remaining if candidate not in companions]
            context, selected = trial_context, trial_selected
            continue

        if exclusions is not None:
            record_budget_exclusion(item, trial_tokens, budget, count_tokens, exclusions)

    return context, selected


def context_candidates[ItemT: Document | tuple[Document, float]](
    docs: Sequence[ItemT],
    query_parts: list[str],
    exclusions: list[dict[str, Any]] | None,
) -> tuple[list[tuple[int, ItemT, float, set[str]]], list[set[str]]]:
    ranked = sorted(
        enumerate(docs),
        key=lambda pair: -(float(pair[1][1]) if isinstance(pair[1], tuple) else 0.0),
    )
    parts = [set(tokenize(part)) for part in query_parts if part.strip()]
    parts = [part for part in parts if part]
    seen: set[tuple[object, ...]] = set()
    remaining = []

    for index, item in ranked:
        doc, score = item if isinstance(item, tuple) else (item, 0.0)
        scoped = sanitize_for_prompt(content_with_parent_scope(doc))
        identity = evidence_identity(doc)

        if identity in seen:
            if exclusions is not None:
                exclusions.append(exclusion_record(item, ExclusionReason.DUPLICATE, "context"))

            continue

        seen.add(identity)
        remaining.append((index, item, float(score), set(tokenize(scoped)) if parts else set()))

    return remaining, parts


def record_budget_exclusion(
    item: Document | tuple[Document, float],
    trial_tokens: int,
    budget: int,
    count_tokens: Callable[[str], int],
    exclusions: list[dict[str, Any]],
) -> None:
    single_context, _ = render_selected_docs([item])
    single_tokens = count_tokens(single_context)
    reason = ExclusionReason.OVERSIZED_BLOCK if single_tokens > budget else ExclusionReason.BUDGET
    exclusions.append(
        exclusion_record(
            item, reason, "context", tokens=trial_tokens, single_tokens=single_tokens, budget=budget
        )
    )


def evidence_companions[ItemT: Document | tuple[Document, float]](
    item: ItemT,
    remaining: list[tuple[int, ItemT, float, set[str]]],
    budget: int,
    count_tokens: Callable[[str], int],
    chosen: Sequence[ItemT] = (),
) -> list[tuple[int, ItemT, float, set[str]]]:
    """Group bounded provision continuations only when they fit with chosen evidence."""
    doc, score = item if isinstance(item, tuple) else (item, 0.0)
    companions = [
        other
        for other in remaining
        if same_evidence_block(doc, other[1][0] if isinstance(other[1], tuple) else other[1])
        and other[2] >= score * CONTINUATION_SCORE_RATIO
    ][: MAX_PROVISION_CHUNKS - 1]

    if not companions:
        return []

    block_context, _ = render_selected_docs([item, *(other[1] for other in companions)])
    # Leave room for independent provisions and question parts.
    block_cost = count_tokens(block_context) - count_tokens("")

    if block_cost > (budget - count_tokens("")) / 2:
        return []

    # Optional continuations must fit alongside the accepted context. Otherwise
    # let the higher-ranked fragment compete on its own before its tail.
    trial_context, _ = render_selected_docs([*chosen, item, *(other[1] for other in companions)])
    return companions if count_tokens(trial_context) <= budget else []
