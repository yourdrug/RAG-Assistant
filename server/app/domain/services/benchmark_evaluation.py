"""Deterministic evidence metrics and shared aggregation for benchmark results."""

import math
import re
from pathlib import PurePosixPath

EVIDENCE_METRICS = (
    "fragment_recall_at_k",
    "fragment_mrr",
    "fragment_ndcg_at_k",
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


def _source_name(source: str) -> str:
    return PurePosixPath(source.replace("\\", "/")).name.casefold()


def _matches(fragment: dict, document: dict) -> bool:
    metadata = document["metadata"]
    if _source_name(fragment["source"]) != _source_name(metadata.get("source", "")):
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


def _coverage(requirements: list, text: str) -> tuple[float | None, list]:
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
            next((rank for rank, doc in enumerate(retrieved, 1) if _matches(f, doc)), None) for f in fragments
        ]
        found = [r for r in ranks if r is not None]
        metrics["fragment_recall_at_k"] = len(found) / len(fragments)
        metrics["fragment_mrr"] = 1 / min(found) if found else 0.0
        # Each expected fragment contributes once. Duplicate chunks cannot inflate nDCG.
        seen: set[int] = set()
        dcg = 0.0
        for rank, doc in enumerate(retrieved, 1):
            new = {i for i, f in enumerate(fragments) if i not in seen and _matches(f, doc)}
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
            if not any(_matches(f, doc) for doc in selected)
            or normalize_text(f["text"]) not in normalize_text(context)
        ]
        metrics["context_fragment_recall"] = 1 - len(missing) / len(fragments)
        diagnostics["missing_context_fragments"] = missing
    for kind in ("fact", "condition"):
        requirements = annotations.get(f"required_{kind}s", [])
        for stage, text in (("context", context), ("answer", answer)):
            if text is not None:
                score, missing = _coverage(requirements, text)
                metrics[f"{stage}_{kind}_coverage"] = score
                diagnostics[f"missing_{stage}_{kind}s"] = missing
    return metrics, diagnostics


def summarize_evidence(results: list[dict]) -> dict:
    summary = {}
    for metric in EVIDENCE_METRICS:
        values = [r.get("evidence_metrics", {}).get(metric) for r in results]
        available = [v for v in values if v is not None]
        summary[f"avg_{metric}"] = round(sum(available) / len(available), 4) if available else None
        summary[f"{metric}_evaluated_count"] = len(available)
    sizes = [
        r["evidence"]["context_chars"]
        for r in results
        if r.get("evidence", {}).get("context_chars") is not None
    ]
    summary["avg_context_chars"] = round(sum(sizes) / len(sizes), 1) if sizes else None
    return summary
