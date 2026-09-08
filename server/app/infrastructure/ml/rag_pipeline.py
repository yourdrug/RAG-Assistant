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

if TYPE_CHECKING:
    pass


@dataclass
class RagPipelineState:
    """Mutable state threaded through pipeline steps."""

    rag: RagSettings
    t_pipeline_start: float
    query_for_search: str
    question: str
    history_messages: list
    ctx: ChatContext
    user: dict
    access_filter: object
    retrieval_filter: object
    breadth: Breadth | None = None
    fetch_k: int = 0
    top_k: int = 0
    effective_dense_weight: float = 0.0
    effective_sparse_weight: float = 0.0
    use_exact_ref_boost: bool = False
    query_domain: str = ""
    candidates: list = field(default_factory=list)
    docs: list = field(default_factory=list)
    avg_sim: float = 0.0
    full_answer: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    usage_report: object | None = None
    q_hash: str = ""
    vis_hash: str = ""
    terminal: bool = False
