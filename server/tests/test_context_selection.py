"""Context packing regressions: priority, complete blocks and full-prompt budget."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import runpy
from unittest.mock import MagicMock

import pytest

from domain.exceptions import ContextBudgetExceededError
from infrastructure.ml.rag.benchmark_evidence import (
    BenchmarkEvidence,
    active_evidence,
    capture_prompt,
)
from infrastructure.ml.rag.context_selection import ExclusionReason
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens, estimate_text_tokens
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection, render_selected_docs
from infrastructure.ml.rag.rag_reranking import deduplicate_docs, rerank_documents
from infrastructure.ml.rag.rag_steps import step_build_context
from test_rag_improvements import doc, state_for


def test_relevance_selection_precedes_document_grouping():
    first = (doc("Most relevant evidence " * 5, 1), 0.99)
    weak = (doc("Low priority table rows " * 5, 1), 0.1)
    amendment = (doc("Important amendment " * 5, 2), 0.98)
    expected, _ = render_selected_docs([first, amendment])
    exclusions = []
    context, selected = format_docs_with_selection(
        [first, weak, amendment], estimate_text_tokens(expected), exclusions=exclusions
    )
    assert {d.page_content for d, _ in selected} == {first[0].page_content, amendment[0].page_content}
    assert context == expected
    assert [item["reason"] for item in exclusions] == [ExclusionReason.BUDGET]


def test_marginal_coverage_prioritizes_an_uncovered_question_part():
    docs = [
        (doc("registration passport", 1), 0.99),
        (doc("registration extra form", 1), 0.98),
        (doc("delivery invoice", 2), 0.97),
        (doc("unrelated material", 3), 0.1),
    ]

    # Only two blocks fit, regardless of their lengths.
    def counter(text):
        return text.count("same.pdf")

    _, selected = format_docs_with_selection(
        docs, 2, context_counter=counter, query_parts=["registration passport", "delivery invoice"]
    )
    assert [d.page_content for d, _ in selected] == ["registration passport", "delivery invoice"]


def test_rejected_oversized_block_does_not_mark_question_part_as_covered():
    docs = [
        (doc("registration", 1), 1.0),
        (doc("delivery invoice " * 100, 2), 0.99),
        (doc("registration form", 3), 0.98),
        (doc("delivery invoice", 4), 0.97),
        (doc("unrelated", 5), 0.0),
    ]
    _, selected = format_docs_with_selection(docs, 40, query_parts=["registration", "delivery invoice"])
    assert "delivery invoice" in [d.page_content for d, _ in selected]


def test_duplicate_and_oversized_decisions_preserve_complete_parent_scope():
    parent = "Only organizations meeting all conditions. " * 100
    oversized = doc("Submit the form.", 3)
    oversized.metadata["parent_units"] = [{"content": parent}]
    first = doc("Duplicate statement.", 1)
    duplicate = doc("Duplicate statement.", 1)
    exclusions = []
    context, selected = format_docs_with_selection(
        [(first, 1.0), (duplicate, 0.9), (oversized, 0.8)], 50, exclusions=exclusions
    )
    assert len(selected) == 1
    assert "Submit the form" not in context
    assert {item["reason"] for item in exclusions} == {
        ExclusionReason.DUPLICATE,
        ExclusionReason.OVERSIZED_BLOCK,
    }
    assert oversized.metadata["parent_units"][0]["content"] == parent


def test_equal_text_with_different_source_or_parent_scope_is_not_a_duplicate():
    first = doc("Required statement.", 1)
    other_source = doc("Required statement.", 2)
    scoped = doc("Required statement.", 1)
    scoped.metadata["parent_units"] = [{"content": "Only for legal entities."}]
    exclusions = []
    _, selected = format_docs_with_selection(
        [(first, 1.0), (other_source, 0.9), (scoped, 0.8)], exclusions=exclusions
    )
    assert len(selected) == 3
    assert exclusions == []


def test_cyrillic_context_and_messages_share_the_same_estimate():
    text = "Условия применения: организация и грузополучатель."
    from langchain_core.messages import HumanMessage

    assert estimate_message_tokens([HumanMessage(content=text)]) == 15 + estimate_text_tokens(text)
    context, _ = format_docs_with_selection([(doc(text * 20, 1), 1.0)], 100)
    assert context == ""


def test_pipeline_keeps_high_score_from_another_section_with_full_prompt_counter():
    docs = [
        (doc("Primary evidence " * 40, 1), 0.99),
        (doc("Unimportant rows " * 40, 1), 0.1),
        (doc("Amendment rule " * 40, 2), 0.98),
    ]
    docs[0][0].metadata["section"] = docs[1][0].metadata["section"] = "Table"
    docs[2][0].metadata["section"] = "Amendment"
    # Derive a window that fits exactly the top two complete blocks.
    reference = state_for([docs[0], docs[2]])
    messages, _ = step_build_context(reference, None)
    limit = estimate_message_tokens(messages) + reference.rag.llm_num_predict_narrow
    state = state_for(docs, rag=replace(reference.rag, llm_num_ctx_narrow=limit))
    evidence = BenchmarkEvidence()
    token = active_evidence.set(evidence)
    counter = MagicMock(side_effect=estimate_message_tokens)
    try:
        messages, retrieved = step_build_context(state, None, token_counter=counter)
    finally:
        active_evidence.reset(token)
    assert len(retrieved) == 3
    assert {d.metadata["document_id"] for d, _ in state._prompt_docs} == {1, 2}
    assert docs[1][0].page_content not in evidence.context
    assert evidence.prompt_budget["input_tokens"] + state.rag.llm_num_predict_narrow == limit
    assert len(evidence.prompt_candidates) == 3
    assert evidence.exclusions[0]["reason"] == ExclusionReason.BUDGET
    assert all(any(state.question in m.content for m in call.args[0]) for call in counter.call_args_list)
    assert all("citation_id" not in d.metadata for d, _ in docs)


def test_no_fitting_block_preserves_exclusion_before_raising():
    state = state_for([(doc("Very large block " * 10000, 1), 0.99)])
    evidence = BenchmarkEvidence()
    token = active_evidence.set(evidence)
    try:
        with pytest.raises(ContextBudgetExceededError):
            step_build_context(state, None)
    finally:
        active_evidence.reset(token)
    assert evidence.selected == []
    assert evidence.exclusions[0]["reason"] == ExclusionReason.OVERSIZED_BLOCK


@pytest.mark.asyncio
async def test_reranker_threshold_and_dedup_diagnostics_survive_prompt_capture():
    docs = [doc("Best", 1), doc("Below score gap", 2), doc("Below minimum", 3)]
    evidence = BenchmarkEvidence()
    token = active_evidence.set(evidence)
    try:
        assert len(deduplicate_docs([docs[0], docs[0]])) == 1
        selected = await rerank_documents(
            "Q",
            docs,
            3,
            MagicMock(predict=MagicMock(return_value=[0.9, 0.3, 0.1])),
            min_score=0.2,
            score_gap_ratio=0.5,
        )
        capture_prompt(selected, selected, "Best")
    finally:
        active_evidence.reset(token)
    assert len(evidence.exclusions) == 3
    assert [item["reason"] for item in evidence.exclusions] == [
        ExclusionReason.DUPLICATE,
        ExclusionReason.RERANK_THRESHOLD,
        ExclusionReason.RERANK_THRESHOLD,
    ]
    restored = BenchmarkEvidence(**asdict(evidence))
    assert restored == evidence
    assert BenchmarkEvidence(**{"retrieved": [], "selected": [], "context": ""}).exclusions == []


@pytest.fixture(scope="module")
def saved_snapshot_replay():
    root = Path(__file__).resolve().parents[2]
    script = root / "server/scripts/replay_context_snapshots.py"
    snapshot = root / "docs/benchmarks/context-budget-replay-2026-10-07/input.json"
    return runpy.run_path(str(script))["replay"](json.loads(snapshot.read_text(encoding="utf-8")))


@pytest.mark.parametrize("question_id", [136, 140, 154, 158])
def test_saved_snapshot_loss_is_reproduced_and_repaired(saved_snapshot_replay, question_id):
    question = next(q for q in saved_snapshot_replay["questions"] if q["id"] == question_id)
    assert question["legacy_context_exact_match"]
    legacy = question["variants"]["legacy"]
    rebuilt = question["variants"]["new"]
    assert legacy["context_fragment_recall"] < 1
    assert rebuilt["context_fragment_recall"] == 1
    assert rebuilt["input_tokens"] <= question["input_budget"]
    assert rebuilt["missing_context_fragments"] == []
