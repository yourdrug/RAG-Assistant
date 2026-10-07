"""Request-local evidence capture; enabled only by benchmark execution."""

from contextvars import ContextVar
from dataclasses import dataclass, field

from domain.value_objects.chunk_context import extract_chunk_context


@dataclass
class BenchmarkEvidence:
    retrieved: list[dict] | None = None
    selected: list[dict] | None = None
    context: str | None = None
    exclusions: list[dict] = field(default_factory=list)
    prompt_candidates: list[dict] | None = None
    prompt_budget: dict | None = None


active_evidence: ContextVar[BenchmarkEvidence | None] = ContextVar("benchmark_evidence", default=None)


def snapshot_documents(docs: list) -> list[dict]:
    return [
        {
            "content": doc.page_content,
            "metadata": {
                **extract_chunk_context(doc.metadata),
                "source": doc.metadata.get("filename") or doc.metadata.get("source", ""),
                **{
                    key: doc.metadata[key]
                    for key in ("document_id", "chunk_index", "content_hash", "citation_id")
                    if key in doc.metadata
                },
            },
            "score": float(score),
        }
        for doc, score in docs
    ]


def capture_retrieval(docs: list) -> None:
    evidence = active_evidence.get()
    if evidence is not None:
        evidence.retrieved = snapshot_documents(docs)
        # A relevance gate can terminate before prompt building: no context reaches the LLM.
        evidence.context = ""
        evidence.selected = []


def capture_exclusions(exclusions: list[dict]) -> None:
    evidence = active_evidence.get()
    if evidence is not None:
        evidence.exclusions.extend(exclusions)


def capture_prompt(docs: list, selected: list, context: str, *, exclusions=None, budget=None) -> None:
    evidence = active_evidence.get()
    if evidence is not None:
        if evidence.retrieved is None:
            evidence.retrieved = snapshot_documents(docs)
        evidence.selected = snapshot_documents(selected)
        evidence.context = context
        evidence.prompt_candidates = snapshot_documents(docs)
        evidence.prompt_budget = budget
        # Replace earlier context decisions if a caller rebuilds the prompt.
        evidence.exclusions = [item for item in evidence.exclusions if item["stage"] != "context"]
        evidence.exclusions.extend(exclusions or [])
