"""Request-local evidence capture; enabled only by benchmark execution."""

from collections.abc import Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from domain.knowledge.chunk import extract_chunk_context
from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.ranking import RetrievalDocument
from domain.shared.hashing import content_hash


@dataclass
class BenchmarkEvidence:
    retrieved: list[dict[str, Any]] | None = None
    selected: list[dict[str, Any]] | None = None
    context: str | None = None
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    prompt_candidates: list[dict[str, Any]] | None = None
    prompt_budget: dict[str, Any] | None = None
    # Default to the legacy version when restoring an old checkpoint.
    schema_version: int = 1
    attempts: list[dict[str, Any]] = field(default_factory=list)
    request: dict[str, Any] | None = None
    answer: dict[str, Any] | None = None
    completion: str | None = None
    authorized: list[str] = field(default_factory=list)
    completion_details: dict[str, Any] | None = None


active_evidence: ContextVar[BenchmarkEvidence | None] = ContextVar("benchmark_evidence", default=None)


def snapshot_documents(
    docs: Sequence[RetrievalDocument | tuple[RetrievalDocument, float]],
) -> list[dict[str, Any]]:
    return deepcopy(
        [
            {
                "content": doc.page_content,
                "metadata": {
                    **extract_chunk_context(doc.metadata),
                    "source": doc.metadata.get("filename") or doc.metadata.get("source", ""),
                    **{
                        key: doc.metadata[key]
                        for key in (
                            "document_id",
                            "chunk_index",
                            "chunk_id",
                            "content_hash",
                            "citation_id",
                            "act_id",
                            "act_version_id",
                            "effective_from",
                            "effective_to",
                            "is_current",
                            "verified_timing_scope",
                            "verified_amended_acts",
                        )
                        if key in doc.metadata
                    },
                },
                "content_hash": content_hash(doc.page_content),
                "scoped_content": content_with_parent_scope(doc),
                "position": position,
                "score": float(score) if score is not None else None,
            }
            for position, (doc, score) in enumerate(
                (item if isinstance(item, tuple) else (item, None) for item in docs), 1
            )
        ]
    )


def capture_retrieval(docs: Sequence[RetrievalDocument | tuple[RetrievalDocument, float]]) -> None:
    evidence = active_evidence.get()

    if evidence is not None:
        evidence.retrieved = snapshot_documents(docs)
        # A relevance gate can terminate before prompt building: no context reaches the LLM.
        evidence.context = ""
        evidence.selected = []


def capture_exclusions(exclusions: list[dict[str, Any]]) -> None:
    evidence = active_evidence.get()

    if evidence is not None:
        # Pre-ACL merges/deduplication must never serialize unverified documents.
        safe = [
            item
            for item in exclusions
            if not evidence.attempts or evidence_identity(item) in evidence.authorized
        ]
        evidence.exclusions.extend(deepcopy(safe))

        if evidence.attempts:
            evidence.attempts[-1]["exclusions"].extend(deepcopy(safe))


def capture_prompt(
    docs: Sequence[RetrievalDocument | tuple[RetrievalDocument, float]],
    selected: Sequence[RetrievalDocument | tuple[RetrievalDocument, float]],
    context: str,
    *,
    exclusions: list[dict[str, Any]] | None = None,
    budget: dict[str, Any] | None = None,
    messages: Sequence[dict[str, Any]] | None = None,
    status: str = "executed",
) -> None:
    evidence = active_evidence.get()

    if evidence is not None:
        if evidence.retrieved is None:
            evidence.retrieved = snapshot_documents(docs)

        evidence.selected = snapshot_documents(selected)
        evidence.context = context
        evidence.prompt_candidates = snapshot_documents(docs)
        evidence.prompt_budget = budget

        # Replace earlier context decisions if a caller rebuilds the prompt.
        if not evidence.attempts:
            evidence.exclusions = [item for item in evidence.exclusions if item["stage"] != "context"]

        capture_exclusions(exclusions or [])
        capture_stage("prompt", selected, ordered=False, score_kind="reranker")

        if evidence.attempts:
            stage = evidence.attempts[-1]["stages"]["prompt"]
            stage["context"] = context
            stage["status"] = status
            stage["messages"] = deepcopy(messages)
            stage["budget"] = deepcopy(budget)


def evidence_identity(document: dict[str, Any]) -> str:
    metadata = document["metadata"]
    return "|".join(
        str(value)
        for value in (
            metadata.get("document_id"),
            metadata.get("act_version_id"),
            metadata.get("source"),
            document.get("content_hash") or content_hash(document["content"]),
        )
    )


def begin_attempt(question: str, query: str) -> None:
    evidence = active_evidence.get()

    if evidence is not None:
        evidence.schema_version = 2
        evidence.attempts.append(
            {
                "question": question,
                "query": query,
                "sub_queries": [],
                "stages": {},
                "exclusions": [],
                "origins": {},
            }
        )
        evidence.retrieved = None
        evidence.selected = None
        evidence.context = None
        evidence.prompt_candidates = None
        evidence.prompt_budget = None
        evidence.authorized = []


def record_origins(
    docs: Sequence[RetrievalDocument | tuple[RetrievalDocument, float]], origin: str, query: str | None = None
) -> None:
    evidence = active_evidence.get()

    if evidence is not None and evidence.attempts:
        origins = evidence.attempts[-1]["origins"]

        for document in snapshot_documents(docs):
            records = origins.setdefault(evidence_identity(document), [])
            record = {"kind": origin, "query": query}

            if record not in records and (origin != "candidate_augmentation" or not records):
                records.append(record)


def authorize_capture(docs: Sequence[RetrievalDocument | tuple[RetrievalDocument, float]]) -> None:
    evidence = active_evidence.get()

    if evidence is not None:
        for document in snapshot_documents(docs):
            identity = evidence_identity(document)

            if identity not in evidence.authorized:
                evidence.authorized.append(identity)


def capture_stage(
    stage: str,
    docs: Sequence[RetrievalDocument | tuple[RetrievalDocument, float]],
    *,
    ordered: bool = True,
    score_kind: str | None = None,
) -> None:
    evidence = active_evidence.get()

    if evidence is None or not evidence.attempts:
        return

    attempt = evidence.attempts[-1]
    documents = snapshot_documents(docs)

    for document in documents:
        document["score_kind"] = score_kind if document["score"] is not None else None

        if not ordered:
            document["position"] = None

        document["origins"] = deepcopy(attempt["origins"].get(evidence_identity(document), []))

        if document["origins"] and all(
            item["kind"] in {"neighbor", "linked_fragment", "scope_context"} for item in document["origins"]
        ):
            document["score"] = None
            document["score_kind"] = None

    if stage == "candidates":
        identities = {evidence_identity(document) for document in documents}
        attempt["origins"] = {key: value for key, value in attempt["origins"].items() if key in identities}

    attempt["stages"][stage] = {"status": "executed", "ordered": ordered, "documents": documents}


def capture_completion(reason: str) -> None:
    evidence = active_evidence.get()

    if evidence is not None:
        evidence.completion = reason


def capture_prompt_delivery(*, sent: bool, finish_reason: str | None = None) -> None:
    evidence = active_evidence.get()

    if evidence is not None:
        evidence.completion_details = {"generator_called": sent, "finish_reason": finish_reason}

        if evidence.attempts:
            prompt = evidence.attempts[-1]["stages"].get("prompt")

            if prompt is not None:
                prompt["status"] = "executed" if sent else "not_sent"


def capture_prepared_query(query: str) -> None:
    evidence = active_evidence.get()

    if evidence is not None and evidence.request is not None:
        evidence.request["search_query"] = query
