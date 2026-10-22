"""Adapter-local retrieval inputs and results; never cross application ports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage

from domain.identity.user_context import UserContext
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.settings import RagSettings


@dataclass
class RetrievalWork:
    """Inputs and working collections for a single retrieval operation."""

    # ── Immutable inputs ────────────────────────────────────────────────
    rag: RagSettings
    t_pipeline_start: float
    question: str
    ctx: RagContext
    user: UserContext
    access_filter: object
    retrieval_filter: object
    req_id: str = ""
    visibility_conditions: list[dict[str, Any]] = field(default_factory=list)

    # ── History ─────────────────────────────────────────────────────────
    history_messages: list[BaseMessage] = field(default_factory=list)

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
    candidates: list[Document] = field(default_factory=list)
    docs: list[tuple[Document, float]] = field(default_factory=list)
    avg_sim: float = 0.0
