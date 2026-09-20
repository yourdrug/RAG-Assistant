"""RAG pipeline — shared state dataclass for the RAG flow.

``RagPipelineState`` holds all mutable state threaded through pipeline
steps.  The ``RagService.stream()`` method uses it as a structured
container instead of passing 20+ local variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from domain.value_objects.chat_context import ChatContext
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.rag_settings import RagSettings
from domain.value_objects.user_context import UserContext

if TYPE_CHECKING:
    from application.ports.pii_redactor import PIIRedactorPort


@dataclass
class RagPipelineState:
    """Mutable state threaded through pipeline steps."""

    # ── Immutable inputs ────────────────────────────────────────────────
    rag: RagSettings
    t_pipeline_start: float
    question: str
    ctx: ChatContext
    user: UserContext
    access_filter: object
    retrieval_filter: object
    req_id: str = ""
    visibility_conditions: list = field(default_factory=list)

    # ── History ─────────────────────────────────────────────────────────
    history_messages: list = field(default_factory=list)

    # ── Mutable pipeline state ──────────────────────────────────────────
    query_for_search: str = ""
    breadth: Breadth | None = None
    fetch_k: int = 0
    rerank_top_n: int = 0
    effective_dense_weight: float = 0.0
    effective_sparse_weight: float = 0.0
    use_exact_ref_boost: bool = False
    query_domain: str = ""
    sub_queries: list[str] = field(default_factory=list)
    enumerate_cases: bool = False
    candidates: list = field(default_factory=list)
    docs: list = field(default_factory=list)
    avg_sim: float = 0.0

    # ── Output ──────────────────────────────────────────────────────────
    full_answer: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    usage_report: object | None = None
    ttft_sec: float | None = None

    # ── Internal (set during generation step) ──────────────────────────
    _last_chunk: object | None = None
    _grouped_docs: list = field(default_factory=list)

    # ── PII ─────────────────────────────────────────────────────────────
    pii_redactor: "PIIRedactorPort | None" = None

    # ── Cache ───────────────────────────────────────────────────────────
    q_hash: str = ""
    vis_hash: str = ""

    # ── Control ─────────────────────────────────────────────────────────
    terminal: bool = False
