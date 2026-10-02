"""Authoritative document authorization for evidence and cached answers."""

from typing import Protocol

from domain.value_objects.user_context import UserContext


class DocumentAccessPort(Protocol):
    async def allowed_document_ids(self, document_ids: list[int], user: UserContext) -> set[int]: ...
