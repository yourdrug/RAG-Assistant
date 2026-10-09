"""Budget selection independent of document/section presentation order."""

from enum import StrEnum

from domain.services.rag_policy import sanitize_for_prompt
from domain.utils import content_hash
from infrastructure.bm25.tokenizer import tokenize
from infrastructure.ml.rag.document_sources import document_identity
from infrastructure.ml.rag.rag_formatting import build_header, content_with_parent_scope, render_selected_docs


class ExclusionReason(StrEnum):
    BUDGET = "budget"
    RERANK_THRESHOLD = "reranker_threshold"
    DUPLICATE = "duplicate"
    OVERSIZED_BLOCK = "oversized_block"
    RERANK_TOP_N = "reranker_top_n"
    FINAL_TOP_K = "final_top_k"
    SCOPE_MISMATCH = "scope_mismatch"


# At most 15% of the relevance score range can be traded for new query coverage.
CONTEXT_COVERAGE_WEIGHT = 0.15


def exclusion_record(item, reason: ExclusionReason, stage: str, **details) -> dict:
    from infrastructure.ml.rag.benchmark_evidence import snapshot_documents

    doc, score = item if isinstance(item, tuple) else (item, 0.0)
    return {
        **snapshot_documents([(doc, score)])[0],
        "score": float(score) if isinstance(item, tuple) else None,
        "reason": reason.value,
        "stage": stage,
        **details,
    }


def select_context(docs, budget: int, count_tokens, query_parts: list[str], exclusions=None):
    """Greedily pack relevance plus marginal lexical query-part coverage.

    Always try the highest reranker score first. Later candidates can receive a
    bounded bonus for query terms absent from the accepted context. Coverage is
    averaged per sub-query so a long question part cannot drown a short one.
    This is a lexical heuristic, not a claim of semantic evidence coverage.
    """
    if budget <= 0:
        raise ValueError("max_context_tokens must be positive")
    ranked = sorted(
        enumerate(docs),
        key=lambda pair: -(float(pair[1][1]) if isinstance(pair[1], tuple) else 0.0),
    )
    parts = [set(tokenize(part)) for part in query_parts if part.strip()]
    parts = [part for part in parts if part]
    covered: set[str] = set()
    seen: set[tuple] = set()
    remaining = []
    for index, item in ranked:
        doc, score = item if isinstance(item, tuple) else (item, 0.0)
        scoped = sanitize_for_prompt(content_with_parent_scope(doc))
        # Equal text can have different legal scope/source metadata. Only merge
        # blocks with the same document, header and complete parent conditions.
        identity = (document_identity(doc), build_header(doc, 1), content_hash(scoped))
        if identity in seen:
            if exclusions is not None:
                exclusions.append(exclusion_record(item, ExclusionReason.DUPLICATE, "context"))
            continue
        seen.add(identity)
        remaining.append((index, item, float(score), set(tokenize(scoped)) if parts else set()))
    score_range = max((entry[2] for entry in remaining), default=0.0) - min(
        (entry[2] for entry in remaining), default=0.0
    )
    # Equal scores still allow coverage to break ties, without altering relevance.
    coverage_weight = CONTEXT_COVERAGE_WEIGHT * score_range
    chosen = []
    context, selected = "", []
    while remaining:

        def priority(entry):
            index, _, score, terms = entry
            gain = (
                sum(len((terms & part) - covered) / len(part) for part in parts) / len(parts) if parts else 0
            )
            bonus = coverage_weight * gain if chosen else 0.0
            return score + bonus, gain if chosen else 0, -index

        entry = max(remaining, key=priority)
        remaining.remove(entry)
        _, item, _, terms = entry
        trial_context, trial_selected = render_selected_docs([*chosen, item])
        trial_tokens = count_tokens(trial_context)
        if trial_tokens <= budget:
            chosen.append(item)
            covered.update(terms)
            context, selected = trial_context, trial_selected
            continue
        if exclusions is not None:
            single_context, _ = render_selected_docs([item])
            single_tokens = count_tokens(single_context)
            reason = ExclusionReason.OVERSIZED_BLOCK if single_tokens > budget else ExclusionReason.BUDGET
            exclusions.append(
                exclusion_record(
                    item, reason, "context", tokens=trial_tokens, single_tokens=single_tokens, budget=budget
                )
            )
    return context, selected
