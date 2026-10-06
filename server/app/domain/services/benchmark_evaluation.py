"""Deterministic evidence metrics and shared aggregation for benchmark results."""

import math
import re
from pathlib import PurePosixPath

from domain.services.benchmark_source_hints import matches_source_hints, parse_source_hints

EVIDENCE_METRICS = (
    "fragment_recall_at_k",
    "fragment_mrr",
    "fragment_ndcg_at_k",
    "retrieval_fact_coverage",
    "evidence_completion_rr",
    "context_fragment_recall",
    "context_fact_coverage",
    "context_condition_coverage",
    "answer_fact_coverage",
    "answer_condition_coverage",
    "refusal_score",
    "citation_support_score",
    "requirement_preservation_score",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def source_name(source: str) -> str:
    return PurePosixPath(source.replace("\\", "/")).name.casefold()


def matches_fragment(fragment: dict, document: dict) -> bool:
    metadata = document["metadata"]
    if source_name(fragment["source"]) != source_name(metadata.get("source", "")):
        return False
    if fragment.get("section") and normalize_text(fragment["section"]) != normalize_text(
        metadata.get("section", "")
    ):
        return False
    if fragment.get("pages"):
        pages = metadata.get("pages") or []
        if metadata.get("page") is not None:
            pages = [*pages, metadata["page"]]
        if metadata.get("page_start") is not None:
            pages = [
                *pages,
                *range(metadata["page_start"], metadata.get("page_end", metadata["page_start"]) + 1),
            ]
        if not set(fragment["pages"]).issubset(pages):
            return False
    return normalize_text(fragment["text"]) in normalize_text(document["content"])


def requirement_coverage(requirements: list, text: str) -> tuple[float | None, list]:
    normalized = normalize_text(text)
    missing = [
        group
        for group in requirements
        if not any(
            normalize_text(phrase) in normalized for phrase in ([group] if isinstance(group, str) else group)
        )
    ]
    return (1 - len(missing) / len(requirements) if requirements else None), missing


def evaluate_evidence(
    annotations: dict | None,
    retrieved: list[dict] | None,
    selected: list[dict] | None,
    context: str | None,
    answer: str,
) -> tuple[dict, dict]:
    metrics = dict.fromkeys(EVIDENCE_METRICS)
    diagnostics: dict = {"missing_fragments": None, "missing_context_fragments": None}
    if annotations is None:
        return metrics, diagnostics
    fragments = annotations.get("expected_fragments", [])
    if fragments and retrieved is not None:
        ranks = [
            next((rank for rank, doc in enumerate(retrieved, 1) if matches_fragment(f, doc)), None)
            for f in fragments
        ]
        found = [r for r in ranks if r is not None]
        metrics["fragment_recall_at_k"] = len(found) / len(fragments)
        metrics["fragment_mrr"] = 1 / min(found) if found else 0.0
        # Each expected fragment contributes once. Duplicate chunks cannot inflate nDCG.
        seen: set[int] = set()
        dcg = 0.0
        for rank, doc in enumerate(retrieved, 1):
            new = {i for i, f in enumerate(fragments) if i not in seen and matches_fragment(f, doc)}
            if new:
                dcg += 1 / math.log2(rank + 1)
                seen.update(new)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(retrieved), len(fragments)) + 1))
        metrics["fragment_ndcg_at_k"] = dcg / ideal if ideal else 0.0
        diagnostics["missing_fragments"] = [
            f for f, rank in zip(fragments, ranks, strict=True) if rank is None
        ]
    if fragments and selected is not None and context is not None:
        missing = [
            f
            for f in fragments
            if not any(matches_fragment(f, doc) for doc in selected)
            or normalize_text(f["text"]) not in normalize_text(context)
        ]
        metrics["context_fragment_recall"] = 1 - len(missing) / len(fragments)
        diagnostics["missing_context_fragments"] = missing
    for kind in ("fact", "condition"):
        requirements = annotations.get(f"required_{kind}s", [])
        for stage, text in (("context", context), ("answer", answer)):
            if text is not None:
                score, missing = requirement_coverage(requirements, text)
                metrics[f"{stage}_{kind}_coverage"] = score
                diagnostics[f"missing_{stage}_{kind}s"] = missing
    return metrics, diagnostics


def has_retrieval_labels(question: dict) -> bool:
    annotations = question.get("annotations") or {}
    return bool(
        parse_source_hints(question.get("source_hint"))
        or annotations.get("expected_fragments")
        or annotations.get("required_facts")
    )


def evaluate_retrieval(question: dict, documents: list[dict] | None) -> dict:
    """Shared A/B scoring; missing evidence never falls back to filename success.

    MRR is the first matching fragment (or legacy source). Completion RR
    measures the first prefix with a matching fragment and all required facts.
    Facts alone do not define the rank of an individually relevant fragment.
    """
    result = dict.fromkeys(
        (
            "hit_rate",
            "mrr",
            "source_hit_rate",
            "source_mrr",
            "fragment_recall_at_k",
            "fragment_mrr",
            "retrieval_fact_coverage",
            "evidence_completion_rr",
        )
    )
    annotations = question.get("annotations") or {}
    fragments = annotations.get("expected_fragments") or []
    facts = annotations.get("required_facts") or []
    result["annotated"] = bool(fragments or facts)
    result["labelled"] = has_retrieval_labels(question)
    hints = parse_source_hints(question.get("source_hint"))
    result["mrr_labelled"] = bool(fragments or (hints and not result["annotated"]))
    if documents is None:
        return result
    if hints:
        rank = next(
            (
                rank
                for rank, doc in enumerate(documents, 1)
                if matches_source_hints(doc["metadata"].get("source", ""), hints)
            ),
            None,
        )
        result["source_hit_rate"] = int(rank is not None)
        result["source_mrr"] = 1 / rank if rank else 0.0
    if not result["annotated"]:
        result["hit_rate"] = result["source_hit_rate"]
        result["mrr"] = result["source_mrr"]
        return result

    evidence, _ = evaluate_evidence(
        annotations, documents, None, "\n".join(doc["content"] for doc in documents), ""
    )
    result["fragment_recall_at_k"] = evidence["fragment_recall_at_k"]
    result["fragment_mrr"] = evidence["fragment_mrr"]
    result["retrieval_fact_coverage"] = evidence["context_fact_coverage"]
    result["mrr"] = evidence["fragment_mrr"]
    result["hit_rate"] = 0
    result["evidence_completion_rr"] = 0.0
    for rank in range(1, len(documents) + 1):
        prefix = documents[:rank]
        fragment_hit = not fragments or any(matches_fragment(f, doc) for f in fragments for doc in prefix)
        facts_hit = not facts or requirement_coverage(facts, "\n".join(d["content"] for d in prefix))[0] == 1
        source_hit = (
            bool(fragments)
            or not hints
            or any(matches_source_hints(doc["metadata"].get("source", ""), hints) for doc in prefix)
        )
        if fragment_hit and facts_hit and source_hit:
            result["hit_rate"] = 1
            result["evidence_completion_rr"] = 1 / rank
            break
    return result


def summarize_retrieval(results: list[dict]) -> dict:
    """Preserve annotation/source diagnostics and count unavailable labelled cases."""
    summary = {}
    for metric in ("hit_rate", "mrr", "source_hit_rate", "source_mrr"):
        values = [r.get("retriever_metrics", {}).get(metric) for r in results]
        available = [value for value in values if value is not None]
        key = "hit_rate" if metric == "hit_rate" else f"avg_{metric}"
        summary[key] = round(sum(available) / len(available), 4) if available else None
        summary[f"{metric}_evaluated_count"] = len(available)
    labelled = [r for r in results if r.get("retriever_metrics", {}).get("labelled")]
    annotated = [r for r in labelled if r["retriever_metrics"].get("annotated")]
    summary["retrieval_labelled_count"] = len(labelled)
    summary["hit_rate_expected_count"] = len(labelled)
    summary["mrr_expected_count"] = sum(bool(r["retriever_metrics"].get("mrr_labelled")) for r in results)
    summary["retrieval_annotated_count"] = len(annotated)
    summary["retrieval_source_only_count"] = len(labelled) - len(annotated)
    summary["retrieval_unavailable_count"] = sum(
        r["retriever_metrics"].get("hit_rate") is None for r in labelled
    )
    return summary


def summarize_evidence(results: list[dict]) -> dict:
    summary = {}
    for metric in EVIDENCE_METRICS:
        values = [r.get("evidence_metrics", {}).get(metric) for r in results]
        available = [v for v in values if v is not None]
        summary[f"avg_{metric}"] = round(sum(available) / len(available), 4) if available else None
        summary[f"{metric}_evaluated_count"] = len(available)
        if metric.startswith("fragment_") or metric == "context_fragment_recall":
            annotation = "expected_fragments"
        elif "fact" in metric:
            annotation = "required_facts"
        elif "condition" in metric:
            annotation = "required_conditions"
        else:
            continue
        summary[f"{metric}_expected_count"] = sum(
            bool((r.get("annotations") or {}).get(annotation)) for r in results
        )
    sizes = [
        r["evidence"]["context_chars"]
        for r in results
        if r.get("evidence", {}).get("context_chars") is not None
    ]
    summary["avg_context_chars"] = round(sum(sizes) / len(sizes), 1) if sizes else None
    return summary
