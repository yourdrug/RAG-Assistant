"""Load sweep datasets and cache retrieval candidates with benchmark ACL."""

from __future__ import annotations

import asyncio
import logging

from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.value_objects.roles import UserKind, UserRole

from infrastructure.benchmark.runner import load_questions
from infrastructure.benchmark.sweep_scoring import cache_candidates
from infrastructure.benchmark.sweep_settings import SweepSettingsPort
from infrastructure.repositories.vector.acl import build_qdrant_filter

logger = logging.getLogger("default")


class SweepDataSource:
    def __init__(self, uow_factory: UnitOfWorkFactory, ml_clients, runtime: SweepSettingsPort) -> None:
        self._uow_factory = uow_factory
        self._ml_clients = ml_clients
        self._runtime = runtime

    async def load_questions(self, dataset: str, questions_path: str | None) -> list[dict]:
        try:
            async with self._uow_factory.create() as uow:
                questions = await uow.benchmark_questions.list_items(
                    dataset=dataset, is_active=True, limit=1000
                )
            if questions:
                return [
                    {
                        "question": q.question,
                        "expected_answer": q.expected_answer,
                        "source_hint": q.source_hint,
                        "tags": q.tags or [],
                    }
                    for q in questions
                ]
        except Exception:
            # CLI file-backed datasets remain usable when PostgreSQL is unavailable.
            logger.warning("Failed to load questions from DB; using file dataset", exc_info=True)
        return await asyncio.to_thread(load_questions, questions_path or self._runtime.questions_path)

    async def cache_candidates(self, questions: list[dict], max_fetch_k: int) -> tuple[dict, dict, dict]:
        access_filter = build_qdrant_filter(
            user={"id": 0, "kind": UserKind.INTERNAL, "role": UserRole.ADMIN}, group_ids=[]
        )
        return await cache_candidates(
            questions, max_fetch_k, ml_clients=self._ml_clients, access_filter=access_filter
        )
