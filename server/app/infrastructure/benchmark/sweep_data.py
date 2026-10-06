"""Load sweep datasets and cache retrieval candidates with benchmark ACL."""

from __future__ import annotations


from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.value_objects.roles import UserKind, UserRole

from application.services.benchmark_dataset import load_benchmark_questions
from infrastructure.benchmark.sweep_scoring import cache_candidates
from infrastructure.repositories.vector.acl import build_qdrant_filter


class SweepDataSource:
    def __init__(self, uow_factory: UnitOfWorkFactory, ml_clients) -> None:
        self._uow_factory = uow_factory
        self._ml_clients = ml_clients

    async def load_questions(self, dataset: str) -> list[dict]:
        return await load_benchmark_questions(self._uow_factory, dataset)

    async def cache_candidates(self, questions: list[dict], max_fetch_k: int) -> tuple[dict, dict, dict]:
        access_filter = build_qdrant_filter(
            user={"id": 0, "kind": UserKind.INTERNAL, "role": UserRole.ADMIN}, group_ids=[]
        )
        return await cache_candidates(
            questions, max_fetch_k, ml_clients=self._ml_clients, access_filter=access_filter
        )
