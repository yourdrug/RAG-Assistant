"""Token Provider Protocol — abstracts token creation for the domain layer."""

from __future__ import annotations

from typing import Protocol


class ITokenProvider(Protocol):
    def create_token(self, user_id: int, role: str) -> str: ...
    def decode_token(self, token: str) -> dict: ...
