"""Shared retrieval and authorization stages for initial search and Self-RAG."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.rag.classification import should_enumerate_cases
from infrastructure.ml.rag.document_access import filter_documents
from infrastructure.ml.rag.helpers import apply_exact_search, rerank_and_enrich, retrieve_with_decomposition


if TYPE_CHECKING:
    from application.documents import DocumentAccessPort
    from application.ports.ml_clients import MLClientPort
    from domain.rag.ports import ChunkRetrievalRepository
    from infrastructure.ml.rag.retrieval_work import RetrievalWork


class RetrievalPipeline:
    def __init__(
        self,
        ml_clients: MLClientPort,
        chunk_search: ChunkRetrievalRepository | None,
        document_access: DocumentAccessPort,
    ) -> None:
        self.ml_clients = ml_clients
        self.chunk_search = chunk_search
        self.document_access = document_access

    async def run(self, query: str, state: RetrievalWork) -> RetrievalWork:
        candidates, state.sub_queries = await retrieve_with_decomposition(query, state, self.ml_clients)

        if state.use_exact_ref_boost:
            await apply_exact_search(query, candidates, state.user, state.ctx, self.chunk_search)

        state.candidates = await filter_documents(candidates, self.document_access, state.user)
        state.enumerate_cases = should_enumerate_cases([doc.page_content for doc in state.candidates])
        state.docs, _, state.avg_sim = await rerank_and_enrich(
            query,
            state.candidates,
            state,
            self.ml_clients,
            self.chunk_search,
            document_access=self.document_access,
        )
        return state
