"""Client assignment repository interface -- maps internal users to external client accounts."""

from __future__ import annotations

from typing import Protocol

from domain.value_objects.query_results import ClientAssignmentInfo


class ClientAssignmentRepository(Protocol):
    async def assign(self, internal_user_id: int, client_user_id: int, assigned_by: int) -> None: ...
    async def unassign(self, internal_user_id: int, client_user_id: int) -> None: ...
    async def get_assigned_client_ids(self, internal_user_id: int) -> list[int]: ...
    async def list_for_client(self, client_user_id: int) -> list[ClientAssignmentInfo]: ...
