"""Read-side vector payload lookup."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStoreRepository(Protocol):
    async def generate_embeddings(self, text: str) -> list[float]: ...
    async def get_point_payload(self, point_id: int, access_filter: Any) -> dict | None: ...
