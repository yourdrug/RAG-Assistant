"""Load sweep datasets and cache retrieval candidates with benchmark ACL."""

from __future__ import annotations

from application.benchmark import load_benchmark_questions
from application.benchmark.sweep_dto import DenseCache, SparseCache, CandidateCache
from infrastructure.ml.rag.rag_steps import chunk_value
from langchain_core.documents import Document as LCDocument
from copy import deepcopy
from application.uow.factory import UnitOfWorkFactory
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.access_control import get_visibility_conditions

from infrastructure.benchmark.sweep.reranking import cache_reranker_scores
from infrastructure.benchmark.sweep.candidates import cache_candidates
from application.benchmark.sweep_ports import ShouldCancel
from infrastructure.ml.clients import factories
from infrastructure.repositories.vector.acl import build_qdrant_filter


class SweepDataSource:
    def __init__(self, uow_factory: UnitOfWorkFactory, ml_clients) -> None:
        self._uow_factory = uow_factory
        self._ml_clients = ml_clients

    async def load_questions(self, dataset: str) -> list[dict]:
        return await load_benchmark_questions(self._uow_factory, dataset)

    async def cache_candidates(
        self, questions: list[dict], max_fetch_k: int
    ) -> tuple[DenseCache, SparseCache, CandidateCache]:
        access_filter = build_qdrant_filter(
            user={"id": 0, "kind": UserKind.INTERNAL, "role": UserRole.ADMIN}, group_ids=[]
        )
        dense, sparse, candidates = await cache_candidates(
            questions,
            max_fetch_k,
            ml_clients=self._ml_clients,
            access_filter=access_filter,
            visibility_conditions=get_visibility_conditions(
                user_kind=UserKind.INTERNAL,
                group_ids=[],
                for_list=False,
                user_role=UserRole.ADMIN,
            ),
            user_id=0,
            user_group_ids=[],
        )
        values = {key: chunk_value(doc) for key, doc in candidates.items()}
        return (
            {query: [(key, score, values[key]) for key, score, _ in hits] for query, hits in dense.items()},
            sparse,
            values,
        )

    async def cache_reranker_scores(
        self,
        questions: list[dict],
        dense: dict,
        sparse: dict,
        candidates: dict,
        should_cancel: ShouldCancel | None,
    ) -> dict:
        if not candidates:
            return {}

        candidates = {
            key: LCDocument(page_content=chunk.content, metadata=deepcopy(chunk.metadata))
            for key, chunk in candidates.items()
        }
        if self._ml_clients is not None:
            reranker = self._ml_clients.reranker()
            return await cache_reranker_scores(questions, dense, sparse, candidates, reranker, should_cancel)

        reranker = factories.create_reranker()
        try:
            return await cache_reranker_scores(questions, dense, sparse, candidates, reranker, should_cancel)
        finally:
            await reranker.close()
