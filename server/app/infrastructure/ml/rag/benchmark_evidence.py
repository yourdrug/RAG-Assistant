"""Request-local evidence capture; enabled only by benchmark execution."""

from contextvars import ContextVar
from dataclasses import dataclass

from domain.value_objects.chunk_context import extract_chunk_context


@dataclass
class BenchmarkEvidence:
    retrieved: list[dict] | None = None
    selected: list[dict] | None = None
    context: str | None = None


active_evidence: ContextVar[BenchmarkEvidence | None] = ContextVar("benchmark_evidence", default=None)


def snapshot_documents(docs: list) -> list[dict]:
    return [
        {
            "content": doc.page_content,
            "metadata": {
                **extract_chunk_context(doc.metadata),
                "source": doc.metadata.get("filename") or doc.metadata.get("source", ""),
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


def capture_prompt(docs: list, selected: list, context: str) -> None:
    evidence = active_evidence.get()
    if evidence is not None:
        if evidence.retrieved is None:
            evidence.retrieved = snapshot_documents(docs)
        evidence.selected = snapshot_documents(selected)
        evidence.context = context
