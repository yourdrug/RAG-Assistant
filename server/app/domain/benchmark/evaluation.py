"""Source matching, labelled retrieval scores and evidence coverage summaries."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
import re
from typing import Any

from domain.benchmark.question import ABSTENTION_TYPES, answer_type, evaluation_annotations


def parse_source_hints(source_hint: str | None) -> tuple[str, ...]:
    if source_hint is None:
        return ()

    return tuple(part.strip().lower() for part in source_hint.split(";") if part.strip())


def matches_source_hints(filename: str, hints: tuple[str, ...]) -> bool:
    return any(hint in filename.lower() for hint in hints)


DETERMINISTIC_EVIDENCE_METRICS = (
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
)

JUDGED_EVIDENCE_METRICS = (
    "refusal_score",
    "citation_support_score",
    "requirement_preservation_score",
)


EVIDENCE_METRICS = DETERMINISTIC_EVIDENCE_METRICS + JUDGED_EVIDENCE_METRICS


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def source_name(source: str) -> str:
    return PurePosixPath(source.replace("\\", "/")).name.casefold()


def fragment_identity_matches(fragment: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return all(
        field not in fragment or str(fragment[field]) == str(metadata.get(field))
        for field in ("document_id", "act_version_id")
    )


def fragment_pages_match(expected: list[Any], metadata: dict[str, Any]) -> bool:
    pages = list(metadata.get("pages") or [])

    if metadata.get("page") is not None:
        pages.append(metadata["page"])

    if metadata.get("page_start") is not None:
        end = metadata.get("page_end")
        pages.extend(range(metadata["page_start"], (end if end is not None else metadata["page_start"]) + 1))

    return set(expected).issubset(pages)


def matches_fragment(fragment: dict[str, Any], document: dict[str, Any]) -> bool:
    if "alternatives" in fragment:
        return any(matches_fragment(item, document) for item in fragment["alternatives"])

    metadata = document["metadata"]
    return (
        fragment_identity_matches(fragment, metadata)
        and source_name(fragment["source"]) == source_name(metadata.get("source", ""))
        and (
            not fragment.get("section")
            or normalize_text(fragment["section"]) == normalize_text(metadata.get("section", ""))
        )
        and (not fragment.get("pages") or fragment_pages_match(fragment["pages"], metadata))
        and normalize_text(fragment["text"]) in normalize_text(document["content"])
    )


def requirement_coverage(requirements: list[Any], text: str) -> tuple[float | None, list[Any]]:
    normalized = normalize_text(text)
    missing = [
        group
        for group in requirements
        if not any(
            normalize_text(phrase) in normalized for phrase in ([group] if isinstance(group, str) else group)
        )
    ]
    return (1 - len(missing) / len(requirements) if requirements else None), missing


def evaluate_fragment_ranks(
    fragments: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
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
    diagnostics["missing_fragments"] = [f for f, rank in zip(fragments, ranks, strict=True) if rank is None]


def evaluate_evidence(
    annotations: dict[str, Any] | None,
    retrieved: list[dict[str, Any]] | None,
    selected: list[dict[str, Any]] | None,
    context: str | None,
    answer: str,
    *,
    judge_scores: dict[str, float | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = dict.fromkeys(EVIDENCE_METRICS)

    if judge_scores is not None:
        metrics.update({key: judge_scores.get(key) for key in JUDGED_EVIDENCE_METRICS})

    diagnostics: dict[str, Any] = {"missing_fragments": None, "missing_context_fragments": None}

    if annotations is None:
        return metrics, diagnostics

    annotations = evaluation_annotations(annotations) or {}
    fragments = annotations.get("expected_fragments", [])

    if fragments and retrieved is not None:
        evaluate_fragment_ranks(fragments, retrieved, metrics, diagnostics)

    if fragments and selected is not None and context is not None:
        missing = [
            f
            for f in fragments
            if not any(
                matches_fragment(option, doc) and normalize_text(option["text"]) in normalize_text(context)
                for option in f.get("alternatives", [f])
                for doc in selected
            )
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


def has_retrieval_labels(question: dict[str, Any]) -> bool:
    annotations = evaluation_annotations(question.get("annotations")) or {}

    if answer_type(annotations) in ABSTENTION_TYPES:
        return False

    return bool(
        parse_source_hints(question.get("source_hint"))
        or annotations.get("expected_fragments")
        or annotations.get("required_facts")
    )


def evaluate_retrieval(question: dict[str, Any], documents: list[dict[str, Any]] | None) -> dict[str, Any]:
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
    annotations = evaluation_annotations(question.get("annotations")) or {}

    if answer_type(annotations) in ABSTENTION_TYPES:
        return {**result, "annotated": False, "labelled": False, "mrr_labelled": False}

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
    result.update(
        fragment_recall_at_k=evidence["fragment_recall_at_k"],
        fragment_mrr=evidence["fragment_mrr"],
        retrieval_fact_coverage=evidence["context_fact_coverage"],
        mrr=evidence["fragment_mrr"],
        hit_rate=0,
        evidence_completion_rr=0.0,
    )

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


def summarize_retrieval(results: list[dict[str, Any]]) -> dict[str, Any]:
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


def summarize_evidence(results: list[dict[str, Any]]) -> dict[str, Any]:
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
            summary[f"{metric}_expected_count"] = sum(bool(r.get("annotations")) for r in results)
            continue

        summary[f"{metric}_expected_count"] = sum(
            bool((evaluation_annotations(r.get("annotations")) or {}).get(annotation)) for r in results
        )

    sizes = [
        r["evidence"]["context_chars"]
        for r in results
        if r.get("evidence", {}).get("context_chars") is not None
    ]
    summary["avg_context_chars"] = round(sum(sizes) / len(sizes), 1) if sizes else None
    return summary


STAGE_NAMES = ("candidates", "reranked", "retained", "enriched", "prompt")


def stage_fragment_present(
    fragment: dict[str, Any], document: dict[str, Any], context: str | None = None
) -> bool:
    alternatives = fragment.get("alternatives", [fragment])
    return any(
        matches_fragment(option, {**document, "content": document.get("scoped_content", document["content"])})
        and (context is None or normalize_text(option["text"]) in normalize_text(context))
        for option in alternatives
    )


def score_evidence_stage(
    annotations: dict[str, Any] | None, stage: dict[str, Any], name: str
) -> tuple[dict[str, Any], set[int]]:
    annotations = evaluation_annotations(annotations) or {}
    fragments = annotations.get("expected_fragments") or []
    documents = stage["documents"]
    context = stage.get("context") if name == "prompt" else None
    found = {
        i
        for i, fragment in enumerate(fragments)
        if any(stage_fragment_present(fragment, document, context) for document in documents)
    }
    scored = {
        "status": stage["status"],
        "found": sorted(found) if fragments else None,
        "required_count": len(fragments) if fragments else None,
        "coverage": len(found) / len(fragments) if fragments else None,
        "all_evidence_present": len(found) == len(fragments) if fragments else None,
        "lost_evidence": [],
        "recovered_evidence": [],
        "mrr": None,
        "ndcg": None,
    }

    if fragments and stage.get("ordered"):
        ranked_metrics: dict[str, Any] = {}
        evaluate_fragment_ranks(fragments, documents, ranked_metrics, {})
        scored["mrr"] = ranked_metrics["fragment_mrr"]
        scored["ndcg"] = ranked_metrics["fragment_ndcg_at_k"]

    text = (
        context
        if context is not None
        else "\n".join(doc.get("scoped_content", doc["content"]) for doc in documents)
    )

    for kind in ("fact", "condition"):
        scored[f"lexical_{kind}_coverage"] = requirement_coverage(
            annotations.get(f"required_{kind}s") or [], text
        )[0]

    return scored, found


def evidence_transition_issues(
    fragments: list[Any], scored: dict[str, Any], attempt: dict[str, Any], transition: dict[str, Any]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    decision_stages = {
        "reranked": {"reranker"},
        "retained": {"reranker", "retrieval", "temporal"},
        "enriched": {"context"},
        "prompt": {"context"},
    }.get(transition["to"], set())

    for index in scored["lost_evidence"]:
        decisions = [
            item
            for item in attempt.get("exclusions", [])
            if item["stage"] in decision_stages and stage_fragment_present(fragments[index], item)
        ]
        issues.append({"kind": "evidence_lost", "fragment": index, **transition, "decisions": decisions})

    issues.extend(
        {"kind": "evidence_recovered", "fragment": index, **transition}
        for index in scored["recovered_evidence"]
    )
    return issues


def lexical_answer_omissions(
    annotations: dict[str, Any] | None, context: str | None, answer: str
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    if context is None:
        return issues

    for kind in ("fact", "condition"):
        requirements = (annotations or {}).get(f"required_{kind}s") or []
        _, missing = requirement_coverage(requirements, answer)
        issues.extend(
            {
                "kind": "lexical_answer_omission",
                "requirement_kind": kind,
                "requirement": requirement,
                "semantic_review_required": True,
            }
            for requirement in missing
            if requirement_coverage([requirement], context)[0] == 1
        )

    return issues


def evaluate_attempt(
    annotations: dict[str, Any] | None, attempt: dict[str, Any], index: int, previous: dict[str, Any]
) -> tuple[dict[str, Any], list[Any]]:
    evaluated: dict[str, Any] = {"attempt": index + 1, "stages": {}}
    issues: list[dict[str, Any]] = []
    fragments = (evaluation_annotations(annotations) or {}).get("expected_fragments") or []

    for name in STAGE_NAMES:
        stage = attempt.get("stages", {}).get(name)

        if stage is None or stage.get("status") != "executed":
            evaluated["stages"][name] = {
                "status": stage["status"] if stage else "not_executed",
                "coverage": None,
            }
            continue

        scored, found = score_evidence_stage(annotations, stage, name)

        if fragments:
            if previous.get("found") is not None:
                scored["lost_evidence"] = sorted(previous["found"] - found)
                scored["recovered_evidence"] = sorted(found - previous["found"])

            issues.extend(
                evidence_transition_issues(
                    fragments,
                    scored,
                    attempt,
                    {"attempt": index + 1, "from": previous.get("stage"), "to": name},
                )
            )

            if name == "candidates":
                issues.extend(
                    {"kind": "not_found_before_reranking", "fragment": i, "attempt": index + 1, "stage": name}
                    for i in sorted(set(range(len(fragments))) - found)
                )

            previous.update(found=found, stage=f"{index + 1}:{name}")

        evaluated["stages"][name] = scored

    return evaluated, issues


def evaluate_stage_evidence(
    annotations: dict[str, Any] | None, evidence: dict[str, Any] | None, answer: str
) -> dict[str, Any]:
    """Pure replay of observed stages. Missing stages are never reconstructed."""
    evidence = evidence or {}
    annotations = evaluation_annotations(annotations)
    result: dict[str, Any] = {
        "schema_version": evidence.get("schema_version", 1),
        "coverage_method": "lexical",
        "annotation_status": evidence.get("annotation_validation", {}).get("status", "unvalidated"),
        "attempts": [],
        "issues": [],
    }
    valid_annotations = None if result["annotation_status"] == "invalid_annotation" else annotations
    previous: dict[str, Any] = {}

    for index, attempt in enumerate(evidence.get("attempts") or []):
        evaluated, issues = evaluate_attempt(valid_annotations, attempt, index, previous)
        result["attempts"].append(evaluated)
        result["issues"].extend(issues)

    result["issues"].extend(lexical_answer_omissions(valid_annotations, evidence.get("context"), answer))
    result["answer_requirements"] = {}

    for kind in ("fact", "condition"):
        requirements = (valid_annotations or {}).get(f"required_{kind}s") or []
        coverage, missing = requirement_coverage(requirements, answer)
        result["answer_requirements"][kind] = {
            "method": "lexical",
            "coverage": coverage,
            "found_count": len(requirements) - len(missing) if requirements else None,
            "required_count": len(requirements) if requirements else None,
        }

    fragments = (valid_annotations or {}).get("expected_fragments") or []
    result["evidence_labels"] = fragments
    result["last_observed_stage"] = previous.get("stage")
    result["last_observed_missing"] = (
        sorted(set(range(len(fragments))) - previous["found"]) if previous.get("found") is not None else None
    )
    result["first_observed_loss"] = {}

    for issue in result["issues"]:
        if issue["kind"] in {"not_found_before_reranking", "evidence_lost"}:
            result["first_observed_loss"].setdefault(str(issue["fragment"]), issue)

    return result
