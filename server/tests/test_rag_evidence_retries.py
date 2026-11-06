"""Synthetic evidence regressions independent of benchmark questions or documents."""

import asyncio
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from langchain_core.documents import Document
import pytest

from application.rag.dto import PreparedQuery, RagRequest
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.chunk import ChunkRecord
from domain.rag.context import RagContext
from domain.rag.ranking import prioritize_distinct_provisions
from infrastructure.ml.clients.llm_schemas import RelevanceCheck, SufficiencyAssessment
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile, evidence_profile_scope
from infrastructure.ml.rag import rag_relevance, retrieval_pipeline
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence, active_evidence
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection
from infrastructure.ml.rag.rag_postprocess import fetch_context_neighbors
from infrastructure.ml.rag.rag_steps import RagOperations
from tests.test_rag_pipeline import make_rag


CTX = RagContext(UserContext(1, UserKind.INTERNAL, UserRole.USER))


def fragment(text: str, document_id: int = 1, **metadata: Any) -> Document:
    return Document(page_content=text, metadata={"document_id": document_id, **metadata})


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [{1, 2}, {2}])
async def test_retry_reauthorizes_and_reranks_complementary_evidence(
    monkeypatch: pytest.MonkeyPatch, allowed: set[int]
) -> None:
    old = fragment("first requirement", section="point 1")
    new = fragment("second requirement", 2, section="point 2")
    irrelevant = fragment("unrelated", 3)
    search = AsyncMock(side_effect=[([old, irrelevant], ["original"]), ([new], ["refinement"])])
    monkeypatch.setattr(retrieval_pipeline, "retrieve_with_decomposition", search)
    access = SimpleNamespace(allowed_document_ids=AsyncMock(side_effect=lambda ids, _user: set(ids)))
    predict = AsyncMock(
        side_effect=lambda pairs: [0.01 if "unrelated" in text else 0.99 for _, text in pairs]
    )
    ml = SimpleNamespace(
        reranker_semaphore=asyncio.Semaphore(1), reranker=lambda: SimpleNamespace(predict=predict)
    )
    ops = RagOperations(ml, document_access=access)
    request = RagRequest("original", (), CTX, make_rag(), 0, "test", "model")
    with evidence_profile_scope(GenericEvidenceProfile()):
        first = await ops.retrieve(PreparedQuery(request, "original"))
        access.allowed_document_ids.side_effect = lambda ids, _user: set(ids) & allowed
        second = await ops.retrieve(replace(first.prepared, query="refinement"), previous=first)

    assert second.prepared.query == "original"
    assert second.sub_queries == ("original", "refinement")
    assert {item.chunk.content for item in second.documents} == (
        {"first requirement", "second requirement"} if 1 in allowed else {"second requirement"}
    )
    assert {question for question, _ in predict.call_args.args[0]} == {"original"}
    assert all(item.chunk.content != "unrelated" for item in second.documents)
    assert any(call.args[0] == [1, 2, 3] for call in access.allowed_document_ids.call_args_list)


@pytest.mark.asyncio
async def test_retry_resolves_conflicting_versions_after_pooling(monkeypatch: pytest.MonkeyPatch) -> None:
    old = fragment("old obligation", act_id=7, act_version_id=1, effective_from="2020-01-01")
    new = fragment("new obligation", 2, act_id=7, act_version_id=2, effective_from="2025-01-01")
    monkeypatch.setattr(
        retrieval_pipeline,
        "retrieve_with_decomposition",
        AsyncMock(side_effect=[([old], ["original"]), ([new], ["refinement"])]),
    )
    access = SimpleNamespace(allowed_document_ids=AsyncMock(side_effect=lambda ids, _user: set(ids)))
    ml = SimpleNamespace(
        reranker_semaphore=asyncio.Semaphore(1),
        reranker=lambda: SimpleNamespace(predict=lambda pairs: [0.99] * len(pairs)),
    )
    ops = RagOperations(ml, document_access=access)
    request = RagRequest(
        "original", (), replace(CTX, as_of_date=date(2026, 1, 1)), make_rag(), 0, "test", "model"
    )
    with evidence_profile_scope(GenericEvidenceProfile()):
        first = await ops.retrieve(PreparedQuery(request, "original"))
        second = await ops.retrieve(replace(first.prepared, query="refinement"), previous=first)
    assert [item.chunk.content for item in second.documents] == ["new obligation"]


def test_strong_continuation_survives_weaker_distinct_provision_with_bounded_expansion() -> None:
    chunks = [fragment(f"category {i}", section="point 1", chunk_index=i) for i in range(7)]
    other = fragment("different provision", section="point 2")
    ranked = prioritize_distinct_provisions(
        [*((doc, 0.99 - i / 100) for i, doc in enumerate(chunks)), (other, 0.5)]
    )
    assert [doc for doc, _ in ranked[:4]] == [*chunks[:3], other]


def test_context_packs_a_bounded_block_and_other_document() -> None:
    head = fragment("UNIT eligible categories:", section="point 1", chunk_index=0)
    continuation = fragment("UNIT additional category", section="point 1", chunk_index=1)
    other = fragment("UNIT additional conditions", 2, section="point 2")
    costly = fragment("UNIT UNIT UNIT related discussion", 3, section="point 3")
    context, selected = format_docs_with_selection(
        [(head, 0.99), (costly, 0.98), (continuation, 0.97), (other, 0.96)],
        max_context_tokens=4,
        context_counter=lambda context: context.count("UNIT"),
        query_parts=["categories", "conditions"],
    )
    assert {doc.page_content for doc, _ in selected} == {
        head.page_content,
        continuation.page_content,
        other.page_content,
    }
    assert "additional category" in context


@pytest.mark.asyncio
async def test_text_neighbors_complete_only_same_provision_without_enumeration() -> None:
    anchor = fragment("categories:", section="point 1", chunk_index=4, act_version_id=2)
    matches = [
        ChunkRecord(5, 1, "rules", "continuation", 5, section="point 1", act_version_id=2),
        ChunkRecord(6, 1, "rules", "different rule", 6, section="point 2", act_version_id=2),
        ChunkRecord(7, 1, "rules", "older version", 5, section="point 1", act_version_id=1),
    ]
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=matches))
    result = await fetch_context_neighbors(
        anchor,
        search,
        CTX.user,
        full_table=False,
        enumerate_cases=False,
        hashes=set(),
        scopes=set(),
    )
    assert result == matches[:1]
    assert search.get_neighbors.call_args.kwargs["window"] == 1
    assert search.get_neighbors.call_args.kwargs["user"] is CTX.user


@pytest.mark.asyncio
async def test_both_judges_receive_generation_context_and_capture_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # More than the previous 2000-token cap, spread across independent sources.
    docs = [fragment(f"rule {i}: " + "условие " * 650, i + 1, section=f"point {i}") for i in range(3)]
    monkeypatch.setattr(
        retrieval_pipeline,
        "retrieve_with_decomposition",
        AsyncMock(return_value=(docs, ["rule", "условие"])),
    )
    access = SimpleNamespace(allowed_document_ids=AsyncMock(return_value={1, 2, 3}))
    ml = SimpleNamespace(
        reranker_semaphore=asyncio.Semaphore(1),
        auxiliary_semaphore=asyncio.Semaphore(1),
        reranker=lambda: SimpleNamespace(predict=lambda pairs: [0.99] * len(pairs)),
        instructor_client=SimpleNamespace(),
    )
    ops = RagOperations(ml, document_access=access)
    rag = make_rag(llm_num_ctx_narrow=24000, llm_num_predict_narrow=1000)
    rag = replace(rag, features=replace(rag.features, relevance_gate_enabled=True))
    request = RagRequest("rule conditions", (), CTX, rag, 0, "test", "model")
    calls = []

    async def structured(client: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)

        if kwargs["response_model"] is RelevanceCheck:
            return RelevanceCheck(is_relevant=False, reason="missing condition")

        return SufficiencyAssessment(
            is_sufficient=True, reasoning="all conditions present", suggested_refinement=""
        )

    monkeypatch.setattr(rag_relevance, "create_structured_response", structured)
    evidence = BenchmarkEvidence()
    token = active_evidence.set(evidence)
    try:
        with evidence_profile_scope(GenericEvidenceProfile()):
            result = await ops.retrieve(PreparedQuery(request, "rule conditions"))
            assert not await ops.relevant(result)
            assert (await ops.assess(result)).sufficient
            assert "prompt" not in evidence.attempts[-1]["stages"]
            context = ops.build(result)
    finally:
        active_evidence.reset(token)
    assert len(context.documents) == 3

    for call in calls:
        assert call["messages"][1]["content"].endswith(context.evidence_context)

    checks = evidence.attempts[-1]["checks"]
    assert [(check["stage"], check["context"]) for check in checks] == [
        ("relevance", context.evidence_context),
        ("sufficiency", context.evidence_context),
    ]
    assert checks[0]["result"]["reason"] == "missing condition"
