"""Offline context-budget ablations; no retrieval, generation or judge calls.

Run from server with PYTHONPATH=app and DATA_DIR pointing at a writable folder.
Input contains frozen runtime settings and per-question evidence/annotations.
Legacy evidence is reconstructed from retrieved + selected. This cannot recover
rejected neighbors; exact legacy context matches are reported explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from application.rag.dto import PreparedQuery, RagRequest, RetrievalResult
from domain.benchmark.evaluation import evaluate_evidence
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.enums import DocDomain
from domain.rag.breadth import Breadth
from domain.rag.classification import classify_query_domain
from domain.rag.context import RagContext
from domain.shared.hashing import content_hash
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence, active_evidence, snapshot_documents
from infrastructure.ml.rag.document_sources import group_by_document
from infrastructure.ml.rag.langchain_mapping import lc_documents, scored_values
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens
from infrastructure.ml.rag.rag_config import build_rag_settings
from infrastructure.ml.rag.rag_formatting import CHARS_PER_TOKEN, render_selected_docs
from infrastructure.ml.rag.rag_prompts import build_prompt, domain_prompt_addendum
from infrastructure.ml.rag.rag_reranking import group_by_section
from infrastructure.ml.rag.rag_steps import RagOperations


if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.prompts import ChatPromptTemplate

    from domain.rag.settings import RagSettings


type ScoredDocument = tuple[Document, float]


def chunk_key(item: ScoredDocument) -> str:
    doc = item[0]
    return content_hash(doc.metadata.get("source", "") + "\n" + doc.page_content)


def reconstruct_candidates(evidence: dict[str, Any]) -> list[ScoredDocument]:
    if evidence.get("prompt_candidates") is not None:
        saved = evidence["prompt_candidates"]
    else:
        saved = [*evidence["retrieved"], *evidence["selected"]]
    seen = set()
    candidates = []
    for item in saved:
        doc = Document(page_content=item["content"], metadata=item["metadata"])
        scored = (doc, item["score"])
        key = chunk_key(scored)
        if key not in seen:
            seen.add(key)
            candidates.append(scored)
    return candidates


def sequential_pack(
    candidates: list[ScoredDocument], budget: int, count_tokens: Callable[[str], int]
) -> tuple[str, list[ScoredDocument], list[str]]:
    chosen = []
    rejected = []
    for item in candidates:
        context, _ = render_selected_docs([*chosen, item])
        if count_tokens(context) <= budget:
            chosen.append(item)
        else:
            rejected.append(chunk_key(item))
    context, selected = render_selected_docs(chosen)
    return context, selected, rejected


def legacy_selection(
    candidates: list[ScoredDocument],
    prompt: ChatPromptTemplate,
    question: str,
    input_budget: int,
    *,
    grouped: bool,
    unified: bool,
) -> tuple[str, list[ScoredDocument], list[dict[str, Any]]]:
    """Frozen section/document-first packing and tail trimming, plus ablations."""
    if grouped:
        ordered = [item for group in group_by_document(group_by_section(candidates)) for item in group]
    else:
        ordered = sorted(candidates, key=lambda item: item[1], reverse=True)

    def full_tokens(context: str) -> int:
        return estimate_message_tokens(prompt.format_messages(context=context, history=[], question=question))

    context_budget = input_budget - full_tokens("")
    if context_budget <= 0:
        msg = "Frozen base prompt exhausts the input budget"
        raise ValueError(msg)
    count = full_tokens if unified else len
    budget = input_budget if unified else context_budget * CHARS_PER_TOKEN
    trace = []
    while True:
        context, selected, rejected = sequential_pack(ordered, budget, count)
        tokens = full_tokens(context)
        iteration = {
            "selected": [chunk_key(item) for item in selected],
            "formatter_excluded": rejected,
            "full_prompt_tokens": tokens,
            "tail_removed": None,
        }
        trace.append(iteration)
        if tokens <= input_budget:
            return context, selected, trace
        if not selected:
            msg = "No complete block fits the frozen prompt"
            raise ValueError(msg)
        iteration["tail_removed"] = chunk_key(selected[-1])
        ordered = selected[:-1]


def compact_exclusion(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk": content_hash(item["metadata"].get("source", "") + "\n" + item["content"]),
        **{key: value for key, value in item.items() if key not in {"metadata", "content"}},
    }


def replay_question(
    question: dict[str, Any], rag: RagSettings, domain_keys: tuple[str, ...]
) -> dict[str, Any]:
    evidence = question["evidence"]
    candidates = reconstruct_candidates(evidence)
    ctx = RagContext(1, UserKind.INTERNAL, UserRole.ADMIN)
    # Old snapshots did not save the flag. True reproduces all four target contexts.
    enumerate_cases = (evidence.get("prompt_budget") or {}).get("enumerate_cases", True)
    breadth = Breadth.BROAD if enumerate_cases else Breadth(question["breadth"])
    query_domain = classify_query_domain(question["question"])
    prompt = build_prompt(
        breadth,
        domain_addendum=domain_prompt_addendum(query_domain, ctx, domain_keys),
        enumerate_cases=enumerate_cases,
    )
    num_ctx = rag.llm_num_ctx_broad if breadth == Breadth.BROAD else rag.llm_num_ctx_narrow
    num_predict = rag.llm_num_predict_broad if breadth == Breadth.BROAD else rag.llm_num_predict_narrow
    variants = {}
    for name, grouped, unified in (
        ("legacy", True, False),
        ("relevance_legacy_counter", False, False),
        ("grouped_unified_counter", True, True),
        ("relevance_unified_counter", False, True),
    ):
        context, selected, trace = legacy_selection(
            candidates, prompt, question["question"], num_ctx - num_predict, grouped=grouped, unified=unified
        )
        variants[name] = {"context": context, "selected": selected, "trace": trace, "exclusions": []}

    request = RagRequest(question["question"], (), ctx, rag, 0, "replay", "")
    result = RetrievalResult(
        PreparedQuery(request, question["question"]),
        Breadth(question["breadth"]),
        query_domain,
        0,
        0,
        0,
        0,
        exact_ref_boost=False,
        candidates=(),
        documents=scored_values(candidates),
        sub_queries=(),
        enumerate_cases=enumerate_cases,
        average_similarity=0,
    )
    operations = RagOperations(None, domain_keys=domain_keys, document_access=None)
    captured = BenchmarkEvidence()
    token = active_evidence.set(captured)
    try:
        prepared = operations.build(result)
    finally:
        active_evidence.reset(token)
    variants["new"] = {
        "context": captured.context,
        "selected": lc_documents(prepared.documents),
        "exclusions": [compact_exclusion(item) for item in captured.exclusions],
        "trace": [],
    }
    for result in variants.values():
        selected = result.pop("selected")
        metrics, diagnostics = evaluate_evidence(
            question["annotations"],
            evidence["retrieved"],
            snapshot_documents(selected),
            result["context"],
            "",
        )
        result.update(
            selected=[chunk_key(item) for item in selected],
            selected_count=len(selected),
            input_tokens=estimate_message_tokens(
                prompt.format_messages(context=result["context"], history=[], question=question["question"])
            ),
            context_fragment_recall=metrics["context_fragment_recall"],
            context_fact_coverage=metrics["context_fact_coverage"],
            missing_context_fragments=diagnostics["missing_context_fragments"],
        )
    return {
        "id": question["id"],
        "legacy_context_exact_match": variants["legacy"]["context"] == evidence["context"],
        "input_budget": num_ctx - num_predict,
        "enumerate_cases": enumerate_cases,
        "candidates": [
            {"chunk": chunk_key(item), "score": float(item[1]), "source": item[0].metadata.get("source")}
            for item in candidates
        ],
        "variants": variants,
    }


def replay(snapshot: dict[str, Any]) -> dict[str, Any]:
    rag = build_rag_settings(SimpleNamespace(**snapshot["runtime"]))
    domain_keys = (DocDomain.LEGAL.value, DocDomain.GENERAL.value)
    return {
        "sweep_id": snapshot["sweep_id"],
        "run_id": snapshot["run_id"],
        "reconstruction": snapshot.get("reconstruction"),
        "questions": [replay_question(question, rag, domain_keys) for question in snapshot["questions"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = replay(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
