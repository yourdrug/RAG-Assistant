"""Regression coverage for profile isolation and domain mutation boundaries."""

from __future__ import annotations

import ast
import asyncio
from graphlib import TopologicalSorter
from pathlib import Path

import pytest

from domain.benchmark.evaluation import evaluate_evidence, summarize_evidence
from domain.benchmark.sweep import BenchmarkSweep, BenchmarkSweepStatus
from domain.identity import CuratorScope, UserContext, UserKind, UserRole
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.knowledge import Document, DocumentStatus, DomainKey
from domain.knowledge.document import DocumentProcessingUpdate
from domain.rag import RagContext
from domain.rag.guardrails.injection import has_blocking_injection, has_flagged_injection
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError
from domain.shared.profile_policy import AnswerPolicy
from infrastructure.domain_profile.answer_policy import classify_query_domain, policy_for
from infrastructure.ml.evidence_profiles.profiles import (
    EVIDENCE_PROFILES,
    active_evidence_profile,
    evidence_profile_scope,
)
from infrastructure.ml.rag.prompts import build_system_prompt


@pytest.mark.asyncio
async def test_evidence_profiles_are_task_local_and_generic_has_no_transport_rules() -> None:
    async def selected(key: str) -> tuple[str, list[str], str]:
        with evidence_profile_scope(EVIDENCE_PROFILES[key]):
            await asyncio.sleep(0)
            profile = active_evidence_profile.get()
            return (
                profile.key,
                profile.search_queries("Формат поля «Код» (позиция 18 таблицы 4.2)?"),
                build_system_prompt(question="Кто получил право?"),
            )

    generic, belarus = await asyncio.gather(selected("generic"), selected("belarus"))
    assert generic[0] == "generic"
    assert belarus[0] == "belarus"
    assert len(generic[1]) == 1
    assert "<norm_subject_reading>" not in generic[2]
    assert "<norm_subject_reading>" in belarus[2]
    assert active_evidence_profile.get().key == "belarus"


def test_custom_document_profile_controls_query_classification_and_answer_policy() -> None:
    class CustomProfile(GeneralDomainProfile):
        key = "custom"
        query_patterns: tuple[str, ...] = (r"custom topic",)
        query_priority = 100
        answer_policy = AnswerPolicy("Custom rule", fallback_to_full_corpus=True)

    registry = DomainProfileRegistry()
    registry.register(CustomProfile())
    assert classify_query_domain("custom topic", registry) == "custom"
    assert policy_for("custom", registry).rules == "Custom rule"
    assert classify_query_domain("Пункт указа №5") == "decree"


@pytest.mark.parametrize(
    "text",
    ["Как использовать curl для корпоративного API?", "Как выдать sudo?", "Что означает developer mode?"],
)
def test_tool_mentions_are_signals_without_blocking_valid_questions(text: str) -> None:
    assert not has_blocking_injection(text)
    assert has_flagged_injection(text)


def test_rag_access_snapshot_is_immutable_and_reuses_identity() -> None:
    groups = [1]
    user = UserContext(1, UserKind.INTERNAL, UserRole.CURATOR, tuple(groups)).with_curator_scope(
        CuratorScope(managed_group_ids=(2,))
    )
    context = RagContext(user=user)
    groups.append(99)
    assert context.user_group_ids == (1,)
    assert context.to_user_context() is user
    assert context.curator_scope == CuratorScope(managed_group_ids=(2,))


def test_domain_key_supports_extensions_and_rejects_invalid_identifiers() -> None:
    assert DomainKey("custom_corpus") == "custom_corpus"
    with pytest.raises(ValidationError):
        DomainKey("../private")


def test_terminal_document_cannot_be_failed_by_repository_update_policy() -> None:
    document = Document(status=DocumentStatus.DONE)
    with pytest.raises(BusinessRuleViolationError):
        document.apply_processing_update(
            DocumentProcessingUpdate(status=DocumentStatus.FAILED, error="late worker")
        )
    assert document.status == DocumentStatus.DONE


def test_sweep_cannot_overwrite_cancellation_with_worker_success() -> None:
    sweep = BenchmarkSweep(status=BenchmarkSweepStatus.CANCELLED)
    with pytest.raises(BusinessRuleViolationError):
        sweep.transition_to(BenchmarkSweepStatus.DONE)
    assert sweep.status == BenchmarkSweepStatus.CANCELLED


def test_semantic_judge_metrics_reach_evidence_summary() -> None:
    metrics, _ = evaluate_evidence(
        {},
        [],
        [],
        "",
        "answer",
        judge_scores={
            "refusal_score": 8.0,
            "citation_support_score": 9.0,
            "requirement_preservation_score": 7.0,
        },
    )
    summary = summarize_evidence([{"evidence_metrics": metrics}])
    assert summary["avg_refusal_score"] == 8.0
    assert summary["avg_citation_support_score"] == 9.0
    assert summary["avg_requirement_preservation_score"] == 7.0


def test_runtime_domain_import_graph_has_no_cycles() -> None:
    root = Path(__file__).resolve().parents[1] / "app"
    graph: dict[str, set[str]] = {}

    for path in (root / "domain").rglob("*.py"):
        parts = path.relative_to(root).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        imports: set[str] = set()

        def visit(nodes: list[ast.stmt], found: set[str]) -> None:
            for node in nodes:
                if isinstance(node, ast.If):
                    if ast.unparse(node.test) not in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
                        visit(node.body, found)

                    visit(node.orelse, found)
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("domain"):
                    found.add(node.module)
                elif isinstance(node, ast.Import):
                    found.update(alias.name for alias in node.names if alias.name.startswith("domain"))

        visit(ast.parse(path.read_text()).body, imports)
        graph[module] = imports

    tuple(TopologicalSorter(graph).static_order())


def test_failed_processing_update_keeps_default_failure_reason() -> None:
    document = Document(status=DocumentStatus.PROCESSING)
    document.apply_processing_update(DocumentProcessingUpdate(status=DocumentStatus.FAILED))
    assert document.error_message == "Document processing failed"
