"""BenchmarkQuestionRepository — persistence protocol for benchmark test questions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.entities.benchmark_question import BenchmarkQuestion


@runtime_checkable
class BenchmarkQuestionRepository(Protocol):
    async def get_by_id(self, question_id: int) -> BenchmarkQuestion | None: ...

    async def list_items(
        self,
        *,
        dataset: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        is_active: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[BenchmarkQuestion]: ...

    async def count(
        self,
        *,
        dataset: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        is_active: bool | None = None,
    ) -> int: ...

    async def create(self, question: BenchmarkQuestion) -> BenchmarkQuestion: ...

    async def update(self, question_id: int, **fields) -> BenchmarkQuestion | None: ...

    async def delete(self, question_id: int) -> bool: ...

    async def bulk_create(self, questions: list[BenchmarkQuestion]) -> int: ...

    async def get_datasets(self) -> list[str]: ...

    async def count_by_dataset(self) -> dict[str, int]: ...
